# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:59:12 2026

@author: USER
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, norm
from pulp import (
    LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
)

# =========================================================
# 1) MÉTRICAS
# =========================================================

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


def downside_deviation(r: pd.Series, mar_daily: float = 0.0, periods_per_year: int = 252) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    downside = np.minimum(r - mar_daily, 0.0)
    return np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)


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


def modified_var_cf(
    r: pd.Series,
    alpha: float = 0.95,
    periods_per_year: int = 252
) -> float:
    """
    Modified VaR via Cornish-Fisher.
    Retorna VaR positivo em termos anualizados aproximados.
    """
    r = r.dropna()
    if len(r) < 30:
        return np.nan

    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan

    s = skew(r, bias=False)
    k = kurtosis(r, fisher=True, bias=False)  # excess kurtosis
    z = norm.ppf(1 - alpha)

    z_cf = (
        z
        + (1/6) * (z**2 - 1) * s
        + (1/24) * (z**3 - 3*z) * k
        - (1/36) * (2*z**3 - 5*z) * (s**2)
    )

    var_daily = -(mu + z_cf * sigma)
    var_daily = max(var_daily, 1e-8)

    # aproximação anual
    return var_daily * np.sqrt(periods_per_year)


def modified_sharpe_ratio(
    r: pd.Series,
    rf_daily: pd.Series | float = 0.0,
    alpha: float = 0.95,
    periods_per_year: int = 252
) -> float:
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


# =========================================================
# 2) GERA SCORE POR ATIVO
# =========================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    s = series.copy().astype(float)
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))
    if mad == 0 or np.isnan(mad):
        return (s - np.nanmean(s)) / (np.nanstd(s) + 1e-8)
    return 0.6745 * (s - med) / (mad + 1e-8)


def build_asset_metrics(
    returns_df: pd.DataFrame,
    market_returns: pd.Series,
    rf_daily: pd.Series | float = 0.0,
    liquidity: pd.Series | None = None,
    min_obs: int = 126
) -> pd.DataFrame:
    """
    returns_df: index=data, columns=tickers, valores=retornos diários
    market_returns: série de retorno diário do benchmark
    rf_daily: série ou escalar com retorno livre de risco diário
    liquidity: série indexada por ticker com proxy de liquidez
    """
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

        metrics.append({
            "ticker": ticker,
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "beta": beta,
            "alpha_jensen": alpha_jensen,
            "modified_sharpe": msr,
            "omega": omg,
            "max_drawdown": mdd,
            "liquidity": np.nan if liquidity is None else liquidity.get(ticker, np.nan),
        })

    dfm = pd.DataFrame(metrics)

    if dfm.empty:
        return dfm

    # limpa infinitos
    dfm = dfm.replace([np.inf, -np.inf], np.nan)

    return dfm


def create_selection_score(
    metrics_df: pd.DataFrame,
    weights: dict | None = None
) -> pd.DataFrame:
    """
    weights exemplo:
    {
        "modified_sharpe": 0.35,
        "omega": 0.25,
        "alpha_jensen": 0.20,
        "max_drawdown": 0.10,
        "liquidity": 0.10
    }
    """
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
        weights["modified_sharpe"] * df["z_modified_sharpe"] +
        weights["omega"] * df["z_omega"] +
        weights["alpha_jensen"] * df["z_alpha_jensen"] -
        weights["max_drawdown"] * df["z_max_drawdown"] +
        weights["liquidity"] * df["z_liquidity"]
    )

    return df


# =========================================================
# 3) OTIMIZAÇÃO LINEAR BINÁRIA
# =========================================================

def select_top_k_assets_milp(
    scored_df: pd.DataFrame,
    k: int = 10,
    min_liquidity: float | None = None,
    max_beta: float | None = None,
    sector_map: dict | None = None,
    max_per_sector: int | None = None,
    solver_msg: bool = False
):
    """
    scored_df precisa ter:
    - ticker
    - selection_score
    - liquidity (opcional)
    - beta (opcional)
    """

    df = scored_df.copy().reset_index(drop=True)

    if min_liquidity is not None and "liquidity" in df.columns:
        df = df[df["liquidity"] >= min_liquidity].copy()

    if max_beta is not None and "beta" in df.columns:
        df = df[df["beta"] <= max_beta].copy()

    df = df.dropna(subset=["selection_score"]).copy()

    if len(df) < k:
        raise ValueError(f"Após os filtros, sobraram apenas {len(df)} ativos; k={k} é inviável.")

    prob = LpProblem("Asset_Selection", LpMaximize)

    x = {
        i: LpVariable(f"x_{i}", cat=LpBinary)
        for i in df.index
    }

    # Função objetivo
    prob += lpSum(df.loc[i, "selection_score"] * x[i] for i in df.index)

    # Exatamente k ativos
    prob += lpSum(x[i] for i in df.index) == k

    # Restrição opcional por setor
    if sector_map is not None and max_per_sector is not None:
        df["sector"] = df["ticker"].map(sector_map)
        for sector in df["sector"].dropna().unique():
            idx = df.index[df["sector"] == sector].tolist()
            prob += lpSum(x[i] for i in idx) <= max_per_sector

    prob.solve(PULP_CBC_CMD(msg=solver_msg))

    selected = df.loc[[i for i in df.index if x[i].value() == 1]].copy()
    selected = selected.sort_values("selection_score", ascending=False).reset_index(drop=True)

    return selected, prob

# =========================================================
# 4) LEITURA DOS ARQUIVOS
# =========================================================

historical_prices = pd.read_csv(f"C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/historical_prices.csv")
screened_stocks = pd.read_csv(f"C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/screened_stocks.csv")
screened_fiis = pd.read_csv(f"C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/screened_fiis.csv")

# =========================================================
# 5) LISTA DE ATIVOS ELEGÍVEIS
# =========================================================

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

# =========================================================
# 6) AJUSTE DO DATAFRAME HISTÓRICO
# =========================================================

historical_prices["date"] = pd.to_datetime(historical_prices["date"])
historical_prices = historical_prices.sort_values("date").set_index("date")

# garante que só pegaremos tickers que estão de fato no histórico
available_tickers = [t for t in ticker_list if t in historical_prices.columns]

if len(available_tickers) == 0:
    raise ValueError("Nenhum ticker elegível foi encontrado em historical_prices.csv")

# =========================================================
# 7) RETURNS_DF: retornos diários dos ativos elegíveis
# =========================================================
# no seu arquivo histórico, as colunas dos ativos estão em nível de preço
# então transformamos em retorno simples diário

prices_df = historical_prices[available_tickers].copy()

returns_df = prices_df.pct_change()

# opcional: remove linhas em que todos os ativos estão NaN
returns_df = returns_df.dropna(how="all")

# =========================================================
# 8) IBOV_RETURNS: retorno diário do Ibovespa
# =========================================================
# no arquivo, ibov_close está em nível de índice, então convertemos para retorno

if "ibov_close" not in historical_prices.columns:
    raise ValueError("A coluna 'ibov_close' não foi encontrada em historical_prices.csv")

ibov_returns = historical_prices["ibov_close"].pct_change().rename("ibov_returns")
ibov_returns = ibov_returns.dropna()

# =========================================================
# 9) CDI_DAILY: retorno diário do CDI
# =========================================================
# pelo arquivo, cdi_return já parece vir pronto em formato de retorno diário

if "cdi_return" not in historical_prices.columns:
    raise ValueError("A coluna 'cdi_return' não foi encontrada em historical_prices.csv")

cdi_daily = historical_prices["cdi_return"].astype(float).rename("cdi_daily")
cdi_daily = cdi_daily.dropna()

# =========================================================
# 10) LIQUIDITY_SERIES: proxy de liquidez por ticker
# =========================================================
# vamos usar diretamente a coluna liquidity_proxy que já veio do screener

liquidity_series = (
    eligible_tickers
    .set_index("ticker")["liquidity_proxy"]
    .astype(float)
    .rename("liquidity_proxy")
)

# opcional: manter apenas tickers disponíveis no histórico
liquidity_series = liquidity_series.loc[available_tickers]

# =========================================================
# 11) ALINHAMENTO FINAL
# =========================================================
# aqui criamos uma interseção de datas para garantir que tudo esteja alinhado

common_index = (
    returns_df.index
    .intersection(ibov_returns.index)
    .intersection(cdi_daily.index)
)

returns_df = returns_df.loc[common_index]
ibov_returns = ibov_returns.loc[common_index]
cdi_daily = cdi_daily.loc[common_index]

# =========================================================
# 12) SANITY CHECKS
# =========================================================

print("Quantidade de tickers elegíveis:", len(ticker_list))
print("Quantidade de tickers disponíveis no histórico:", len(available_tickers))
print("Shape de returns_df:", returns_df.shape)
print("Shape de ibov_returns:", ibov_returns.shape)
print("Shape de cdi_daily:", cdi_daily.shape)
print("Shape de liquidity_series:", liquidity_series.shape)

print("\nExemplo returns_df:")
print(returns_df.tail())

print("\nExemplo ibov_returns:")
print(ibov_returns.head())

print("\nExemplo cdi_daily:")
print(cdi_daily.head())

print("\nExemplo liquidity_series:")
print(liquidity_series.head())


# =========================================================
# 13) APLICAÇÂO
# =========================================================

metrics_df = build_asset_metrics(
    returns_df=returns_df,
    market_returns=ibov_returns,
    rf_daily=cdi_daily,
    liquidity=liquidity_series,
    min_obs=126
)

scored_df = create_selection_score(
    metrics_df,
    weights={
        "modified_sharpe": 0.40,
        "omega": 0.25,
        "alpha_jensen": 0.15,
        "max_drawdown": 0.10,
        "liquidity": 0.10,
    }
)

selected_assets, model = select_top_k_assets_milp(
    scored_df,
    k=10,
    min_liquidity=1_000_000,
    max_beta=2.0,
    solver_msg=False
)

print(selected_assets[[
    "ticker",
    "selection_score",
    "modified_sharpe",
    "omega",
    "alpha_jensen",
    "max_drawdown",
    "liquidity"
]])

selected_assets[[
    "ticker",
    "selection_score",
    "modified_sharpe",
    "omega",
    "alpha_jensen",
    "max_drawdown",
    "liquidity"
]].to_csv(f"C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/selected10titles.csv", index=False)