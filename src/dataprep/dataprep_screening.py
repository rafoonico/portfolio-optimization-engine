# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from yahooquery import Ticker

from src.portfolio_config import load_config

CONFIG = load_config()
PATHS = CONFIG.paths


fii = pd.read_excel(PATHS.input_titles_path, sheet_name="fii", header=0)
stock = pd.read_excel(PATHS.input_titles_path, sheet_name="stock", header=0)


fii_codes = fii["code"].dropna().astype(str).str.strip().str.upper()
stock_roots = stock["code"].dropna().astype(str).str.strip().str.upper()


def check_fii_tickers(codes, suffixes=("11", "12", "13", "14")):
    s = pd.Series(list(codes), dtype="string").dropna().str.strip().str.upper()
    s = s.str.replace(r"\.SA$", "", regex=True)

    has_num = s.str.match(r".*\d+$")
    roots = s[~has_num]
    already = s[has_num]

    candidates = list(already) + [f"{r}{suf}" for r in roots for suf in suffixes]
    candidates_sa = [c + ".SA" for c in candidates]

    t = Ticker(candidates_sa)
    price = t.price

    if isinstance(price, list):
        price = {d.get("symbol"): d for d in price if isinstance(d, dict) and d.get("symbol")}

    valid_syms = set()
    if isinstance(price, dict):
        for sym, payload in price.items():
            if isinstance(payload, dict) and payload.get("regularMarketPrice") is not None:
                valid_syms.add(sym)

    def pick_best_for_root(root):
        for suf in suffixes:
            sym = f"{root}{suf}.SA"
            if sym in valid_syms:
                return sym
        return None

    chosen = []
    invalid_roots = []

    for c in already:
        sym = f"{c}.SA"
        if sym in valid_syms:
            chosen.append(c)
        else:
            invalid_roots.append(c)

    for r in roots:
        best = pick_best_for_root(r)
        if best:
            chosen.append(best)
        else:
            invalid_roots.append(r)

    details = pd.DataFrame({"input_code": list(s)})
    return sorted(set(chosen)), sorted(set(invalid_roots)), details


def expand_stock_roots_to_tickers(roots, suffixes=("3", "4", "5", "6")):
    roots = [str(r).strip().upper() for r in roots if pd.notna(r) and str(r).strip()]
    candidates = [f"{r}{s}.SA" for r in roots for s in suffixes]

    t = Ticker(candidates)
    price = t.price

    if isinstance(price, list):
        price = {d.get("symbol"): d for d in price if isinstance(d, dict) and d.get("symbol")}

    if not isinstance(price, dict):
        return []

    valid_rows = []
    for sym, payload in price.items():
        if not isinstance(payload, dict):
            continue
        p = payload.get("regularMarketPrice")
        if p is None:
            continue
        vol = payload.get("regularMarketVolume")
        valid_rows.append({"symbol": sym, "price": p, "vol": vol if vol is not None else 0})

    if not valid_rows:
        return []

    dfv = pd.DataFrame(valid_rows)
    dfv["root"] = dfv["symbol"].str.replace(r"\.SA$", "", regex=True).str.slice(0, 4)
    dfv["vol"] = pd.to_numeric(dfv["vol"], errors="coerce").fillna(0)

    chosen = (
        dfv.sort_values(["root", "vol"], ascending=[True, False])
        .groupby("root", as_index=False)
        .head(1)["symbol"]
        .tolist()
    )
    return chosen


def run_screener(ticker_list, tipo="acao", debug=True):
    def log(msg):
        if debug:
            print(msg)

    log("=" * 70)
    log(f"🔎 INICIANDO SCREENER | Tipo: {tipo} | tickers: {len(ticker_list)}")
    log("📡 Consultando YahooQuery...")

    t = Ticker(ticker_list)
    summary = pd.DataFrame(t.summary_detail).T
    key_stats = pd.DataFrame(t.key_stats).T
    financial = pd.DataFrame(t.financial_data).T

    log(f"✔ summary: {summary.shape} | key_stats: {key_stats.shape} | financial: {financial.shape}")

    df = pd.concat([summary, key_stats, financial], axis=1)
    df = df.reset_index().rename(columns={"index": "ticker"})
    log(f"📊 DF consolidado: {df.shape}")

    need_num = [
        "dividendYield",
        "averageDailyVolume10Day",
        "marketCap",
        "priceToBook",
        "enterpriseToEbitda",
        "enterpriseValue",
        "ebitda",
        "currentPrice",
        "regularMarketPreviousClose",
        "regularMarketOpen",
        "regularMarketDayLow",
        "regularMarketDayHigh",
    ]

    log("🔢 Convertendo colunas numéricas...")
    for c in need_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    log("💰 Construindo preço...")
    if "currentPrice" in df.columns and df["currentPrice"].notna().any():
        df["price_used"] = df["currentPrice"]
    else:
        price_cols = ["regularMarketPreviousClose", "regularMarketOpen", "regularMarketDayLow", "regularMarketDayHigh"]
        have_price = [c for c in price_cols if c in df.columns]
        df["price_used"] = df[have_price].mean(axis=1) if have_price else np.nan

    log("💧 Calculando liquidez_proxy = volume10d * preço...")
    if {"averageDailyVolume10Day", "price_used"}.issubset(df.columns):
        df["liquidity_proxy"] = df["averageDailyVolume10Day"] * df["price_used"]
    else:
        df["liquidity_proxy"] = np.nan

    log("🏗 Construindo EV/EBITDA...")
    if "enterpriseToEbitda" in df.columns and df["enterpriseToEbitda"].notna().any():
        df["ev_ebitda_used"] = df["enterpriseToEbitda"]
    else:
        if {"enterpriseValue", "ebitda"}.issubset(df.columns):
            df["ev_ebitda_used"] = df["enterpriseValue"] / df["ebitda"]
        else:
            df["ev_ebitda_used"] = np.nan

    if tipo == "acao":
        dy_min = 0.06
        liq_min = 5_000_000
        ev_ebitda_max = 10
        mcap_min = 1_000_000_000
        mask = (
            (df["dividendYield"] > dy_min)
            & (df["liquidity_proxy"] > liq_min)
            & (df["marketCap"] > mcap_min)
            & (df["ev_ebitda_used"] > 0)
            & (df["ev_ebitda_used"] < ev_ebitda_max)
        )
        df_filtered = df[mask]
    elif tipo == "fii":
        dy_min = 0.08
        liq_min = 1_000_000
        pvp_max = 1.05
        mask = (df["dividendYield"] > dy_min) & (df["liquidity_proxy"] > liq_min)
        if "priceToBook" in df.columns and df["priceToBook"].notna().any():
            mask &= df["priceToBook"] < pvp_max
        df_filtered = df[mask]
    else:
        raise ValueError("tipo deve ser 'acao' ou 'fii'")

    log(f"✅ Linhas após filtro: {len(df_filtered)}")
    return df_filtered.sort_values("dividendYield", ascending=False)


def main():
    valid_fiis, invalid_fiis, _ = check_fii_tickers(fii_codes)
    valid_stocks = expand_stock_roots_to_tickers(stock_roots.tolist())

    print("Válidos FIIs:", len(valid_fiis))
    print("Inválidos FIIs:", len(invalid_fiis))
    print("Exemplos inválidos:", invalid_fiis[:20])
    print("Ações válidas:", len(valid_stocks))

    screened_stocks = run_screener(valid_stocks, tipo="acao", debug=CONFIG.screening_debug)
    screened_fiis = run_screener(valid_fiis, tipo="fii", debug=CONFIG.screening_debug)

    screened_stocks.to_csv(PATHS.screened_stocks_path, index=False)
    screened_fiis.to_csv(PATHS.screened_fiis_path, index=False)

    print(f"Arquivo gerado: {PATHS.screened_stocks_path}")
    print(f"Arquivo gerado: {PATHS.screened_fiis_path}")


if __name__ == "__main__":
    main()
