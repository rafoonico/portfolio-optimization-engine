# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kurtosis, norm, skew

from src.portfolio_config import load_config

CONFIG = load_config()
PATHS = CONFIG.paths


def load_data(selected_path, prices_path):
    selected = pd.read_csv(selected_path)
    prices = pd.read_csv(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").reset_index(drop=True)
    return selected, prices


def classify_asset_type(ticker: str) -> str:
    ticker_clean = ticker.replace(".SA", "")
    return "fii" if ticker_clean.endswith("11") else "stock"


def build_returns_dataframe(selected: pd.DataFrame, prices: pd.DataFrame, lookback_days: int):
    tickers = selected["ticker"].dropna().unique().tolist()
    required_cols = ["date", "cdi_return", "ibov_close"] + tickers
    missing_cols = [c for c in required_cols if c not in prices.columns]
    if missing_cols:
        raise ValueError(f"As seguintes colunas não estão em historical_prices.csv: {missing_cols}")

    df = prices[required_cols].copy().tail(lookback_days + 5)
    price_df = df[["date"] + tickers].copy().sort_values("date").reset_index(drop=True)
    price_df[tickers] = price_df[tickers].ffill()

    returns_df = price_df.set_index("date")[tickers].pct_change()
    ibov_returns = df[["date", "ibov_close"]].set_index("date")["ibov_close"].pct_change().rename("ibov_return")
    cdi_daily = df[["date", "cdi_return"]].set_index("date")["cdi_return"].rename("cdi_return")

    panel = returns_df.join(ibov_returns, how="left").join(cdi_daily, how="left")
    panel = panel.dropna(subset=tickers)
    panel["cdi_return"] = panel["cdi_return"].ffill()

    return returns_df.loc[panel.index], panel["ibov_return"], panel["cdi_return"]


def build_metadata(selected: pd.DataFrame):
    meta = selected.copy()
    meta["asset_type"] = meta["ticker"].apply(classify_asset_type)
    return meta


def cornish_fisher_z(alpha: float, s: float, k_excess: float) -> float:
    z = norm.ppf(alpha)
    return z + (1 / 6) * (z**2 - 1) * s + (1 / 24) * (z**3 - 3 * z) * k_excess - (1 / 36) * (2 * z**3 - 5 * z) * (s**2)


def modified_var(portfolio_returns: np.ndarray, alpha: float = 0.95) -> float:
    mu = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns, ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.inf
    s = skew(portfolio_returns, bias=False)
    k_excess = kurtosis(portfolio_returns, fisher=True, bias=False)
    z_cf = cornish_fisher_z(1 - alpha, s, k_excess)
    var_cf = -(mu + z_cf * sigma)
    return max(var_cf, 1e-12)


def modified_sharpe_ratio(weights: np.ndarray, asset_returns: pd.DataFrame, rf_series: pd.Series, alpha: float = 0.95) -> float:
    port_ret = asset_returns.values @ weights
    excess_ret = port_ret - rf_series.loc[asset_returns.index].values
    mu_excess = np.mean(excess_ret)
    mvar = modified_var(excess_ret, alpha=alpha)
    if mvar <= 0 or np.isnan(mvar):
        return -np.inf
    return mu_excess / mvar


def negative_modified_sharpe(weights: np.ndarray, asset_returns: pd.DataFrame, rf_series: pd.Series, alpha: float = 0.95) -> float:
    msr = modified_sharpe_ratio(weights, asset_returns, rf_series, alpha)
    if np.isnan(msr) or np.isinf(msr):
        return 1e6
    return -msr


def build_constraints(meta: pd.DataFrame, max_fii_allocation: float = 0.40):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    fii_mask = (meta["asset_type"] == "fii").astype(float).values
    if fii_mask.sum() > 0:
        constraints.append({"type": "ineq", "fun": lambda w, fii_mask=fii_mask, cap=max_fii_allocation: cap - np.sum(w * fii_mask)})
    return constraints


def build_bounds(n_assets: int, min_weight: float = 0.0, max_weight: float = 0.20):
    return [(min_weight, max_weight) for _ in range(n_assets)]


def optimize_portfolio_modified_sharpe(selected: pd.DataFrame, returns_df: pd.DataFrame, cdi_daily: pd.Series, alpha: float = 0.95, min_weight: float = 0.0, max_weight: float = 0.20, max_fii_allocation: float = 0.40):
    meta = build_metadata(selected)
    tickers = meta["ticker"].tolist()
    asset_returns = returns_df[tickers].copy()
    n = len(tickers)
    x0 = np.repeat(1 / n, n)
    bounds = build_bounds(n_assets=n, min_weight=min_weight, max_weight=max_weight)
    constraints = build_constraints(meta=meta, max_fii_allocation=max_fii_allocation)

    print("n_assets:", n)
    print("min_weight:", min_weight)
    print("max_weight:", max_weight)
    print("n * min_weight:", n * min_weight)
    print("n * max_weight:", n * max_weight)
    print("tickers:", tickers)

    print("x0:", x0)
    print("objective at x0:", negative_modified_sharpe(x0, asset_returns, cdi_daily, alpha))

    result = minimize(
        fun=negative_modified_sharpe,
        x0=x0,
        args=(asset_returns, cdi_daily, alpha),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"O otimizador falhou: {result.message}")
    weights = pd.Series(result.x, index=tickers, name="weight")
    weights = weights / weights.sum()
    return weights.sort_values(ascending=False), result


def portfolio_statistics(weights: pd.Series, returns_df: pd.DataFrame, cdi_daily: pd.Series):
    tickers = weights.index.tolist()
    asset_returns = returns_df[tickers].copy()
    port_ret = pd.Series(asset_returns.values @ weights.values, index=asset_returns.index, name="portfolio_return")
    excess_ret = port_ret - cdi_daily.loc[port_ret.index]
    ann_factor = 252
    annual_return = (1 + port_ret.mean()) ** ann_factor - 1
    annual_vol = port_ret.std(ddof=1) * np.sqrt(ann_factor)
    sharpe = excess_ret.mean() / port_ret.std(ddof=1) * np.sqrt(ann_factor) if port_ret.std(ddof=1) > 0 else np.nan
    mvar_daily = modified_var(excess_ret.values, alpha=CONFIG.allocation_alpha)
    mod_sharpe = excess_ret.mean() / mvar_daily if mvar_daily > 0 else np.nan
    wealth = (1 + port_ret).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    max_dd = drawdown.min()
    stats = {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_annualized": sharpe,
        "modified_sharpe_daily": mod_sharpe,
        "max_drawdown": max_dd,
    }
    return port_ret, pd.Series(stats)


def main():
    selected, prices = load_data(PATHS.selected10_path, PATHS.historical_prices_path)
    returns_df, ibov_returns, cdi_daily = build_returns_dataframe(selected=selected, prices=prices, lookback_days=CONFIG.allocation_lookback_days)
    print("returns_df shape:", returns_df.shape)
    print("NaN por coluna:")
    print(returns_df.isna().sum())
    print("Tem inf?", np.isinf(returns_df.values).any())
    print("Tem NaN geral?", np.isnan(returns_df.values).any())
    print("CDI NaN?", cdi_daily.isna().sum())
    weights, _ = optimize_portfolio_modified_sharpe(
        selected=selected,
        returns_df=returns_df,
        cdi_daily=cdi_daily,
        alpha=CONFIG.allocation_alpha,
        min_weight=CONFIG.allocation_min_weight,
        max_weight=CONFIG.allocation_max_weight,
        max_fii_allocation=CONFIG.allocation_max_fii_allocation,
    )
    portfolio_returns, portfolio_stats = portfolio_statistics(weights=weights, returns_df=returns_df, cdi_daily=cdi_daily)

    allocation_df = weights.reset_index().rename(columns={"index": "ticker", "weight": "allocation"}).merge(selected, on="ticker", how="left")
    allocation_df["allocation_pct"] = allocation_df["allocation"] * 100

    print("\n========== PESOS ÓTIMOS ==========")
    print(allocation_df[["ticker", "allocation", "allocation_pct"]].to_string(index=False))
    print("\n========== ESTATÍSTICAS DA CARTEIRA ==========")
    print(portfolio_stats)

    allocation_df.to_csv(PATHS.optimal_allocation_path, index=False)
    portfolio_returns.to_csv(PATHS.portfolio_returns_path, index=True)

    print("\nArquivos gerados:")
    print(f"- {PATHS.optimal_allocation_path}")
    print(f"- {PATHS.portfolio_returns_path}")


if __name__ == "__main__":
    main()
