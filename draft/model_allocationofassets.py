# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 00:29:10 2026

@author: USER
"""

# ============================================================
# BLOCO 1 — IMPORTS
# ============================================================
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, norm


# ============================================================
# BLOCO 2 — CONFIGURAÇÕES
# ============================================================
SELECTED_PATH = f"C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/selected10titles.csv"
PRICES_PATH = f"C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/historical_prices.csv"

LOOKBACK_DAYS = 252 * 3   # janela de estimação: ~3 anos úteis
ALPHA = 0.95              # para Modified VaR (95%)
MAX_WEIGHT = 0.20         # peso máximo por ativo
MIN_WEIGHT = 0.00         # long only
MAX_FII_ALLOCATION = 0.40 # no máximo 40% em FIIs
USE_EXP_WEIGHTING = False # se quiser ponderar mais os retornos recentes
LAMBDA_EWMA = 0.97        # usado apenas se USE_EXP_WEIGHTING=True


# ============================================================
# BLOCO 3 — FUNÇÕES AUXILIARES DE DATA PREP
# ============================================================
def load_data(selected_path: str, prices_path: str):
    selected = pd.read_csv(selected_path)
    prices = pd.read_csv(prices_path)

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").reset_index(drop=True)

    return selected, prices


def classify_asset_type(ticker: str) -> str:
    """
    Heurística simples:
    - FIIs na B3 normalmente terminam em 11
    - ações tendem a terminar em 3,4,5,6 etc.
    """
    ticker_clean = ticker.replace(".SA", "")
    if ticker_clean.endswith("11"):
        return "fii"
    return "stock"


def build_returns_dataframe(selected: pd.DataFrame, prices: pd.DataFrame, lookback_days: int):
    tickers = selected["ticker"].dropna().unique().tolist()

    required_cols = ["date", "cdi_return", "ibov_close"] + tickers
    missing_cols = [c for c in required_cols if c not in prices.columns]
    if missing_cols:
        raise ValueError(f"As seguintes colunas não estão em historical_prices.csv: {missing_cols}")

    df = prices[required_cols].copy()

    # recorta janela
    df = df.tail(lookback_days + 5).copy()

    # preços dos ativos
    price_df = df[["date"] + tickers].copy()
    price_df = price_df.sort_values("date").reset_index(drop=True)

    # forward fill apenas dentro da série
    price_df[tickers] = price_df[tickers].ffill()

    # retornos dos ativos
    returns_df = price_df.set_index("date")[tickers].pct_change()

    # ibov -> transformar fechamento em retorno diário
    ibov_returns = (
        df[["date", "ibov_close"]]
        .set_index("date")["ibov_close"]
        .pct_change()
        .rename("ibov_return")
    )

    # cdi já veio como retorno diário
    cdi_daily = (
        df[["date", "cdi_return"]]
        .set_index("date")["cdi_return"]
        .rename("cdi_return")
    )

    # junta tudo e remove linhas inviáveis
    panel = returns_df.join(ibov_returns, how="left").join(cdi_daily, how="left")

    # remove linhas onde algum dos 10 ativos ainda esteja sem retorno
    panel = panel.dropna(subset=tickers)

    # CDI pode ter algum missing residual
    panel["cdi_return"] = panel["cdi_return"].ffill()

    return returns_df.loc[panel.index], panel["ibov_return"], panel["cdi_return"]


def build_metadata(selected: pd.DataFrame):
    meta = selected.copy()
    meta["asset_type"] = meta["ticker"].apply(classify_asset_type)
    return meta


# ============================================================
# BLOCO 4 — MÉTRICAS DE RISCO E OBJETIVO
# ============================================================
def cornish_fisher_z(alpha: float, s: float, k_excess: float) -> float:
    """
    Ajuste de Cornish-Fisher para quantil normal.
    s = skewness
    k_excess = excess kurtosis
    """
    z = norm.ppf(alpha)
    z_cf = (
        z
        + (1/6) * (z**2 - 1) * s
        + (1/24) * (z**3 - 3*z) * k_excess
        - (1/36) * (2*z**3 - 5*z) * (s**2)
    )
    return z_cf


def modified_var(portfolio_returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    Modified VaR via Cornish-Fisher.
    Retorna um número positivo representando perda.
    """
    mu = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns, ddof=1)

    if sigma == 0 or np.isnan(sigma):
        return np.inf

    s = skew(portfolio_returns, bias=False)
    k_excess = kurtosis(portfolio_returns, fisher=True, bias=False)

    z_cf = cornish_fisher_z(1 - alpha, s, k_excess)

    # VaR do retorno -> foco em perda
    var_cf = -(mu + z_cf * sigma)

    # garante positividade
    return max(var_cf, 1e-12)


def modified_sharpe_ratio(
    weights: np.ndarray,
    asset_returns: pd.DataFrame,
    rf_series: pd.Series,
    alpha: float = 0.95
) -> float:
    """
    MSR = média do excesso de retorno / Modified VaR
    """
    port_ret = asset_returns.values @ weights
    excess_ret = port_ret - rf_series.loc[asset_returns.index].values

    mu_excess = np.mean(excess_ret)
    mvar = modified_var(excess_ret, alpha=alpha)

    if mvar <= 0 or np.isnan(mvar):
        return -np.inf

    return mu_excess / mvar


def negative_modified_sharpe(
    weights: np.ndarray,
    asset_returns: pd.DataFrame,
    rf_series: pd.Series,
    alpha: float = 0.95
) -> float:
    msr = modified_sharpe_ratio(weights, asset_returns, rf_series, alpha)
    if np.isnan(msr) or np.isinf(msr):
        return 1e6
    return -msr


# ============================================================
# BLOCO 5 — RESTRIÇÕES
# ============================================================
def build_constraints(meta: pd.DataFrame, max_fii_allocation: float = 0.40):
    n = len(meta)

    constraints = []

    # soma dos pesos = 1
    constraints.append({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    })

    # limite total em FIIs
    fii_mask = (meta["asset_type"] == "fii").astype(float).values

    if fii_mask.sum() > 0:
        constraints.append({
            "type": "ineq",
            "fun": lambda w, fii_mask=fii_mask, cap=max_fii_allocation: cap - np.sum(w * fii_mask)
        })

    return constraints


def build_bounds(n_assets: int, min_weight: float = 0.0, max_weight: float = 0.20):
    return [(min_weight, max_weight) for _ in range(n_assets)]


# ============================================================
# BLOCO 6 — OTIMIZAÇÃO
# ============================================================
def optimize_portfolio_modified_sharpe(
    selected: pd.DataFrame,
    returns_df: pd.DataFrame,
    cdi_daily: pd.Series,
    alpha: float = 0.95,
    min_weight: float = 0.0,
    max_weight: float = 0.20,
    max_fii_allocation: float = 0.40
):
    meta = build_metadata(selected)

    tickers = meta["ticker"].tolist()
    asset_returns = returns_df[tickers].copy()

    n = len(tickers)
    x0 = np.repeat(1 / n, n)

    bounds = build_bounds(
        n_assets=n,
        min_weight=min_weight,
        max_weight=max_weight
    )

    constraints = build_constraints(
        meta=meta,
        max_fii_allocation=max_fii_allocation
    )

    result = minimize(
        fun=negative_modified_sharpe,
        x0=x0,
        args=(asset_returns, cdi_daily, alpha),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "disp": False}
    )

    if not result.success:
        raise RuntimeError(f"O otimizador falhou: {result.message}")

    weights = pd.Series(result.x, index=tickers, name="weight")
    weights = weights / weights.sum()

    return weights.sort_values(ascending=False), result


# ============================================================
# BLOCO 7 — RELATÓRIO DA CARTEIRA
# ============================================================
def portfolio_statistics(weights: pd.Series, returns_df: pd.DataFrame, cdi_daily: pd.Series):
    tickers = weights.index.tolist()
    asset_returns = returns_df[tickers].copy()

    port_ret = pd.Series(
        asset_returns.values @ weights.values,
        index=asset_returns.index,
        name="portfolio_return"
    )

    excess_ret = port_ret - cdi_daily.loc[port_ret.index]

    ann_factor = 252

    annual_return = (1 + port_ret.mean()) ** ann_factor - 1
    annual_vol = port_ret.std(ddof=1) * np.sqrt(ann_factor)
    sharpe = excess_ret.mean() / port_ret.std(ddof=1) * np.sqrt(ann_factor) if port_ret.std(ddof=1) > 0 else np.nan

    mvar_daily = modified_var(excess_ret.values, alpha=ALPHA)
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
        "max_drawdown": max_dd
    }

    return port_ret, pd.Series(stats)


# ============================================================
# BLOCO 8 — PIPELINE PRINCIPAL
# ============================================================
selected, prices = load_data(SELECTED_PATH, PRICES_PATH)

returns_df, ibov_returns, cdi_daily = build_returns_dataframe(
    selected=selected,
    prices=prices,
    lookback_days=LOOKBACK_DAYS
)

weights, opt_result = optimize_portfolio_modified_sharpe(
    selected=selected,
    returns_df=returns_df,
    cdi_daily=cdi_daily,
    alpha=ALPHA,
    min_weight=MIN_WEIGHT,
    max_weight=MAX_WEIGHT,
    max_fii_allocation=MAX_FII_ALLOCATION
)

portfolio_returns, portfolio_stats = portfolio_statistics(
    weights=weights,
    returns_df=returns_df,
    cdi_daily=cdi_daily
)

allocation_df = (
    weights.reset_index()
    .rename(columns={"index": "ticker", "weight": "allocation"})
    .merge(selected, on="ticker", how="left")
)

allocation_df["allocation_pct"] = allocation_df["allocation"] * 100

print("\n========== PESOS ÓTIMOS ==========")
print(allocation_df[["ticker", "allocation", "allocation_pct"]].to_string(index=False))

print("\n========== ESTATÍSTICAS DA CARTEIRA ==========")
print(portfolio_stats)

allocation_df.to_csv(f"C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/optimal_allocation.csv", index=False)
portfolio_returns.to_csv(f"C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/portfolio_returns_selected10.csv", index=True)

print("\nArquivos gerados:")
print("- optimal_allocation.csv")
print("- portfolio_returns_selected10.csv")