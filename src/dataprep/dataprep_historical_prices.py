# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from bcb import sgs

from src.portfolio_config import load_config

CONFIG = load_config()
PATHS = CONFIG.paths

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def split_batches(lst, batch_size=8):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def load_screened_tickers() -> list[str]:
    stocks = pd.read_csv(PATHS.screened_stocks_path)
    fiis = pd.read_csv(PATHS.screened_fiis_path)
    tickers = (
        pd.concat([stocks["ticker"], fiis["ticker"]])
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return tickers


def download_prices(tickers: list[str]) -> pd.DataFrame:
    all_prices = []
    for batch in split_batches(tickers, batch_size=CONFIG.historical_batch_size):
        print("Baixando batch:", batch)
        try:
            data = yf.download(
                batch,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=False,
                progress=False,
            )
            all_prices.append(data)
        except Exception as exc:
            print("Erro no batch:", batch)
            print(exc)
        time.sleep(CONFIG.historical_pause_seconds)

    if not all_prices:
        raise RuntimeError("Nenhum batch de preços foi baixado com sucesso.")
    return pd.concat(all_prices, axis=1)


def prices_to_panel(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    dfs = []
    for ticker in tickers:
        if ticker not in prices.columns.get_level_values(0):
            continue
        tmp = prices[ticker][["Open", "High", "Low", "Close"]].copy()
        tmp.columns = ["open", "high", "low", "close"]
        tmp["ticker"] = ticker
        tmp["date"] = tmp.index
        dfs.append(tmp)

    if not dfs:
        raise RuntimeError("Não foi possível converter preços para painel longo.")

    prices_long = pd.concat(dfs).reset_index(drop=True)
    prices_long["price_mean"] = prices_long[["open", "high", "low", "close"]].mean(axis=1)

    df_prices = prices_long.pivot_table(index="date", columns="ticker", values="price_mean")
    return df_prices.sort_index()


def download_ibov() -> pd.Series:
    ibov = yf.download("^BVSP", interval="1d", progress=False)
    if isinstance(ibov.columns, pd.MultiIndex):
        ibov = ibov[("Close", "^BVSP")]
    else:
        ibov = ibov["Close"]
    return ibov.rename("ibov_close")


def download_cdi() -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * CONFIG.historical_cdi_years)

    last_exc = None

    for attempt in range(1, CONFIG.historical_cdi_max_retries + 1):
        try:
            print(
                f"Baixando CDI | tentativa {attempt}/{CONFIG.historical_cdi_max_retries} | "
                f"{start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}"
            )

            cdi = sgs.get(
                12,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

            if cdi is None or cdi.empty:
                raise ValueError("A API do BCB retornou um dataframe vazio para o CDI.")

            cdi.columns = ["cdi_percent"]
            cdi["cdi_return"] = cdi["cdi_percent"] / 100

            if cdi[["cdi_return"]].dropna().empty:
                raise ValueError("A série de CDI veio sem valores válidos.")

            return cdi[["cdi_return"]]

        except Exception as exc:
            last_exc = exc
            print(f"[WARN] Falha ao baixar CDI: {exc}")

            if attempt < CONFIG.historical_cdi_max_retries:
                print(f"[INFO] Aguardando {CONFIG.historical_cdi_retry_sleep_seconds}s para nova tentativa...")
                time.sleep(CONFIG.historical_cdi_retry_sleep_seconds)

    raise RuntimeError(
        f"Falha ao baixar CDI após {CONFIG.historical_cdi_max_retries} tentativas."
    ) from last_exc


def apply_resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    freq = freq.upper()
    if freq == "D":
        return df
    return df.resample(freq).last()


def main():
    tickers = load_screened_tickers()
    print("Total tickers:", len(tickers))

    prices = download_prices(tickers)
    df_prices = prices_to_panel(prices, tickers)
    cdi = download_cdi()
    ibov = download_ibov()

    df = df_prices.join(cdi, how="left").join(ibov, how="left").sort_index()
    df[["cdi_return", "ibov_close"]] = df[["cdi_return", "ibov_close"]].ffill()
    df = apply_resample(df, CONFIG.historical_freq)
    df = df.reset_index()

    df.to_csv(PATHS.historical_prices_path, index=False)
    print(f"Dataset salvo em: {PATHS.historical_prices_path}")


if __name__ == "__main__":
    main()
