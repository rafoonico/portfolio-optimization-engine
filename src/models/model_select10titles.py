# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum
from scipy.stats import kurtosis, norm, skew

from src.portfolio_config import load_config

CONFIG = load_config()
PATHS = CONFIG.paths


def annualize_return(r: pd.Series, periods_per_year: int = 252) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    compounded = (1 + r).prod()
    return compounded ** (periods_per_year / len(r)) - 1


def annualize_vol(r: pd.Series, periods_per_year: int = 252) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return r.std(ddof=1) * np.sqrt(periods_per_year)


def max_drawdown_from_returns(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return abs(dd.min())


def omega_ratio(r: pd.Series, threshold_daily: float = 0.0) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    gains = np.maximum(r - threshold_daily, 0.0).sum()
    losses = np.maximum(threshold_daily - r, 0.0).sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def beta_alpha_jensen(asset_r: pd.Series, market_r: pd.Series, rf_daily: pd.Series | float = 0.0):
    df = pd.concat([asset_r, market_r], axis=1).dropna()
    df.columns = ["asset", "market"]

    if isinstance(rf_daily, pd.Series):
        df = df.join(rf_daily.rename("rf"), how="inner").dropna()
        asset_excess = df["asset"] - df["rf"]
        market_excess = df["market"] - df["rf"]
    else:
        asset_excess = df["asset"] - rf_daily
        market_excess = df["market"] - rf_daily

    if len(df) < 30:
        return np.nan, np.nan

    X = sm.add_constant(market_excess.values)
    y = asset_excess.values
    model = sm.OLS(y, X).fit()

    alpha_daily = model.params[0]
    beta = model.params[1]
    alpha_annual = (1 + alpha_daily) ** 252 - 1
    return beta, alpha_annual


def modified_var_cf(r: pd.Series, alpha: float = 0.95, periods_per_year: int = 252) -> float:
    r = r.dropna()
    if len(r) < 30:
        return np.nan
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    s = skew(r, bias=False)
    k = kurtosis(r, fisher=True, bias=False)
    z = norm.ppf(1 - alpha)
    z_cf = z + (1 / 6) * (z**2 - 1) * s + (1 / 24) * (z**3 - 3 * z) * k - (1 / 36) * (2 * z**3 - 5 * z) * (s**2)
    var_daily = -(mu + z_cf * sigma)
    var_daily = max(var_daily, 1e-8)
    return var_daily * np.sqrt(periods_per_year)


def modified_sharpe_ratio(r: pd.Series, rf_daily: pd.Series | float = 0.0, alpha: float = 0.95, periods_per_year: int = 252) -> float:
    r = r.dropna()
    if isinstance(rf_daily, pd.Series):
        df = pd.concat([r, rf_daily], axis=1).dropna()
        df.columns = ["r", "rf"]
        excess = df["r"] - df["rf"]
    else:
        excess = r - rf_daily
    if len(excess) < 30:
        return np.nan
    excess_ann = annualize_return(excess, periods_per_year)
    mvar = modified_var_cf(excess, alpha=alpha, periods_per_year=periods_per_year)
    if mvar is None or np.isnan(mvar) or mvar <= 0:
        return np.nan
    return excess_ann / mvar


def robust_zscore(series: pd.Series) -> pd.Series:
    s = series.copy().astype(float)
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))
    if mad == 0 or np.isnan(mad):
        return (s - np.nanmean(s)) / (np.nanstd(s) + 1e-8)
    return 0.6745 * (s - med) / (mad + 1e-8)


def build_asset_metrics(returns_df: pd.DataFrame, market_returns: pd.Series, rf_daily: pd.Series | float = 0.0, liquidity: pd.Series | None = None, min_obs: int = 126) -> pd.DataFrame:
    metrics = []
    for ticker in returns_df.columns:
        r = returns_df[ticker].dropna()
        if len(r) < min_obs:
            continue
        beta, alpha_jensen = beta_alpha_jensen(r, market_returns, rf_daily)
        msr = modified_sharpe_ratio(r, rf_daily)
        omg = omega_ratio(r, threshold_daily=0.0)
        mdd = max_drawdown_from_returns(r)
        ann_ret = annualize_return(r)
        ann_vol = annualize_vol(r)
        metrics.append(
            {
                "ticker": ticker,
                "annual_return": ann_ret,
                "annual_vol": ann_vol,
                "beta": beta,
                "alpha_jensen": alpha_jensen,
                "modified_sharpe": msr,
                "omega": omg,
                "max_drawdown": mdd,
                "liquidity": np.nan if liquidity is None else liquidity.get(ticker, np.nan),
            }
        )
    dfm = pd.DataFrame(metrics)
    if dfm.empty:
        return dfm
    return dfm.replace([np.inf, -np.inf], np.nan)


def create_selection_score(metrics_df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    if weights is None:
        weights = {
            "modified_sharpe": 0.35,
            "omega": 0.25,
            "alpha_jensen": 0.20,
            "max_drawdown": 0.10,
            "liquidity": 0.10,
        }
    df = metrics_df.copy()
    for col in ["modified_sharpe", "omega", "alpha_jensen", "max_drawdown", "liquidity"]:
        if col not in df.columns:
            df[col] = np.nan
    df["z_modified_sharpe"] = robust_zscore(df["modified_sharpe"])
    df["z_omega"] = robust_zscore(df["omega"])
    df["z_alpha_jensen"] = robust_zscore(df["alpha_jensen"])
    df["z_max_drawdown"] = robust_zscore(df["max_drawdown"])
    df["z_liquidity"] = robust_zscore(df["liquidity"])
    df["selection_score"] = (
        weights["modified_sharpe"] * df["z_modified_sharpe"]
        + weights["omega"] * df["z_omega"]
        + weights["alpha_jensen"] * df["z_alpha_jensen"]
        - weights["max_drawdown"] * df["z_max_drawdown"]
        + weights["liquidity"] * df["z_liquidity"]
    )
    return df.sort_values("selection_score", ascending=False).reset_index(drop=True)


def select_top_k_assets_milp(scored_df: pd.DataFrame, k: int = 10, min_liquidity: float = 1_000_000, max_beta: float = 2.0, solver_msg: bool = False):
    df = scored_df.copy().dropna(subset=["selection_score"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("scored_df está vazio após limpeza.")

    model = LpProblem("asset_selection", LpMaximize)
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in df.index}

    model += lpSum(df.loc[i, "selection_score"] * x[i] for i in df.index)
    model += lpSum(x[i] for i in df.index) == k

    if "liquidity" in df.columns and df["liquidity"].notna().any():
        for i in df.index:
            liq = df.loc[i, "liquidity"]
            if pd.notna(liq) and liq < min_liquidity:
                model += x[i] == 0

    if "beta" in df.columns and df["beta"].notna().any():
        for i in df.index:
            beta = df.loc[i, "beta"]
            if pd.notna(beta) and beta > max_beta:
                model += x[i] == 0

    model.solve(PULP_CBC_CMD(msg=solver_msg))
    selected_mask = [i for i in df.index if x[i].value() == 1]
    selected = df.loc[selected_mask].copy().sort_values("selection_score", ascending=False)
    return selected, model


def load_inputs():
    screened_stocks = pd.read_csv(PATHS.screened_stocks_path)
    screened_fiis = pd.read_csv(PATHS.screened_fiis_path)
    historical_prices = pd.read_csv(PATHS.historical_prices_path)
    return screened_stocks, screened_fiis, historical_prices


def main():
    screened_stocks, screened_fiis, historical_prices = load_inputs()

    eligible_tickers = (
        pd.concat(
            [
                screened_stocks[["ticker", "liquidity_proxy"]],
                screened_fiis[["ticker", "liquidity_proxy"]],
            ],
            axis=0,
            ignore_index=True,
        )
        .drop_duplicates(subset="ticker")
        .reset_index(drop=True)
    )

    ticker_list = eligible_tickers["ticker"].tolist()

    historical_prices["date"] = pd.to_datetime(historical_prices["date"])
    historical_prices = historical_prices.sort_values("date").set_index("date")

    available_tickers = [t for t in ticker_list if t in historical_prices.columns]
    if len(available_tickers) == 0:
        raise ValueError("Nenhum ticker elegível foi encontrado em historical_prices.csv")

    prices_df = historical_prices[available_tickers].copy()
    returns_df = prices_df.pct_change().dropna(how="all")

    if "ibov_close" not in historical_prices.columns:
        raise ValueError("A coluna 'ibov_close' não foi encontrada em historical_prices.csv")
    ibov_returns = historical_prices["ibov_close"].pct_change().rename("ibov_returns").dropna()

    if "cdi_return" not in historical_prices.columns:
        raise ValueError("A coluna 'cdi_return' não foi encontrada em historical_prices.csv")
    cdi_daily = historical_prices["cdi_return"].astype(float).rename("cdi_daily").dropna()

    liquidity_series = eligible_tickers.set_index("ticker")["liquidity_proxy"].astype(float).rename("liquidity_proxy")
    liquidity_series = liquidity_series.loc[available_tickers]

    common_index = returns_df.index.intersection(ibov_returns.index).intersection(cdi_daily.index)
    returns_df = returns_df.loc[common_index]
    ibov_returns = ibov_returns.loc[common_index]
    cdi_daily = cdi_daily.loc[common_index]

    metrics_df = build_asset_metrics(
        returns_df=returns_df,
        market_returns=ibov_returns,
        rf_daily=cdi_daily,
        liquidity=liquidity_series,
        min_obs=CONFIG.selection_min_obs,
    )

    scored_df = create_selection_score(
        metrics_df,
        weights={
            "modified_sharpe": 0.40,
            "omega": 0.25,
            "alpha_jensen": 0.15,
            "max_drawdown": 0.10,
            "liquidity": 0.10,
        },
    )

    selected_assets, _ = select_top_k_assets_milp(
        scored_df,
        k=CONFIG.selection_k,
        min_liquidity=CONFIG.selection_min_liquidity,
        max_beta=CONFIG.selection_max_beta,
        solver_msg=False,
    )

    output_cols = ["ticker", "selection_score", "modified_sharpe", "omega", "alpha_jensen", "max_drawdown", "liquidity"]
    print(selected_assets[output_cols])
    selected_assets[output_cols].to_csv(PATHS.selected10_path, index=False)
    print(f"Arquivo gerado: {PATHS.selected10_path}")


if __name__ == "__main__":
    main()
