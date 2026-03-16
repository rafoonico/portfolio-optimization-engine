# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 11:35:26 2026

@author: USER
"""

import pandas as pd
import numpy as np
from yahooquery import Ticker

fii = pd.read_excel("C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/list_titles_FII_stocks.xlsx",sheet_name="fii",header=0)

stock = pd.read_excel("C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/list_titles_FII_stocks.xlsx",sheet_name="stock",header=0)

# -------------------------
# 1) FIIs: garante sufixo 11
# -------------------------
fii_codes = (
    fii["code"]
    .dropna()
    .str.strip()
    .str.upper()
)

def check_fii_tickers(codes, suffixes=("11", "12", "13", "14")):
    """
    Recebe lista/Series de códigos (ex.: 'HGLG', 'HGLG11', 'HGLG11.SA')
    e devolve:
      - valid: lista de tickers no formato 'HGLG11' (sem .SA)
      - invalid: lista de códigos que não foram encontrados
      - details: df com status por código (opcional, útil pra debug)
    """
    s = pd.Series(list(codes), dtype="string").dropna().str.strip().str.upper()

    # remove .SA se existir
    s = s.str.replace(r"\.SA$", "", regex=True)

    # separa root e já garante que os que já têm número fiquem como estão
    has_num = s.str.match(r".*\d+$")
    roots = s[~has_num]
    already = s[has_num]

    # cria candidatos: pra quem não tem número, tenta root+11, root+12...
    candidates = list(already) + [f"{r}{suf}" for r in roots for suf in suffixes]
    candidates_sa = [c + ".SA" for c in candidates]

    t = Ticker(candidates_sa)
    price = t.price

    # normaliza price se vier list
    if isinstance(price, list):
        price = {d.get("symbol"): d for d in price if isinstance(d, dict) and d.get("symbol")}

    valid_syms = set()
    bad_syms = set()

    if isinstance(price, dict):
        for sym, payload in price.items():
            if isinstance(payload, dict) and payload.get("regularMarketPrice") is not None:
                valid_syms.add(sym)
            else:
                bad_syms.add(sym)

    # escolhe 1 ticker por root (prioriza 11, depois 12, etc.)
    def pick_best_for_root(root):
        for suf in suffixes:
            sym = f"{root}{suf}.SA"
            if sym in valid_syms:
                return sym
        return None

    chosen = []
    invalid_roots = []

    # para os que já tinham número (ex.: HGLG11), valida direto
    for c in already:
        sym = f"{c}.SA"
        if sym in valid_syms:
            chosen.append(c)
        else:
            invalid_roots.append(c)

    # para os roots (ex.: HGLG), escolhe o melhor sufixo disponível
    for r in roots:
        best = pick_best_for_root(r)
        if best:
            chosen.append(best)
        else:
            invalid_roots.append(r)

    details = pd.DataFrame({
        "input_code": list(s),
    })
    return sorted(set(chosen)), sorted(set(invalid_roots)), details

valid_fiis, invalid_fiis, _ = check_fii_tickers(fii_codes)

print("Válidos:", len(valid_fiis))
print("Inválidos:", len(invalid_fiis))
print("Exemplos inválidos:", invalid_fiis[:20])

# -------------------------
# 2) Stocks: expandir roots -> tentar achar tickers válidos
# -------------------------
stock_roots = (
    stock["code"]
    .dropna()
    .str.strip()
    .str.upper()
)

def expand_stock_roots_to_tickers(roots, suffixes=("3", "4", "5", "6")):
    """
    Expande roots tipo 'PETR' -> ['PETR4', ...]
    Valida no Yahoo e escolhe 1 ticker por root pelo maior volume (se disponível).

    Obs:
    - removi '11' dos sufixos por padrão (11 é mais típico de FII).
    - trata casos em que yahooquery retorna string/erro em vez de dict.
    """
    roots = [str(r).strip().upper() for r in roots if pd.notna(r) and str(r).strip()]
    candidates = [f"{r}{s}.SA" for r in roots for s in suffixes]

    t = Ticker(candidates)

    price = t.price

    # yahooquery pode retornar:
    # 1) dict: { "PETR4.SA": {...}, ... }
    # 2) list de dicts (menos comum)
    # 3) dict com erros em string por símbolo

    if isinstance(price, list):
        # converte list -> dict
        price = {d.get("symbol"): d for d in price if isinstance(d, dict) and d.get("symbol")}

    if not isinstance(price, dict):
        # se vier algo totalmente inesperado
        return []

    valid_rows = []
    for sym, payload in price.items():
        # payload pode ser dict OU string (erro)
        if not isinstance(payload, dict):
            continue

        p = payload.get("regularMarketPrice")
        if p is None:
            continue

        vol = payload.get("regularMarketVolume")
        valid_rows.append(
            {
                "symbol": sym,
                "price": p,
                "vol": vol if vol is not None else 0
            }
        )

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

valid_stocks = expand_stock_roots_to_tickers(stock_roots.tolist())

# -------------------------
# 3) Rodando o Screener
# -------------------------


def run_screener(ticker_list, tipo="acao", debug=True):

    def log(msg):
        if debug:
            print(msg)

    log("=" * 70)
    log(f"🔎 INICIANDO SCREENER | Tipo: {tipo} | tickers: {len(ticker_list)}")

    # 1) Consulta
    log("📡 Consultando YahooQuery...")
    t = Ticker(ticker_list)

    summary   = pd.DataFrame(t.summary_detail).T
    key_stats = pd.DataFrame(t.key_stats).T
    financial = pd.DataFrame(t.financial_data).T

    log(f"✔ summary: {summary.shape} | key_stats: {key_stats.shape} | financial: {financial.shape}")

    # 2) Consolida
    df = pd.concat([summary, key_stats, financial], axis=1)
    df = df.reset_index().rename(columns={"index": "ticker"})
    log(f"📊 DF consolidado: {df.shape}")

    # 3) Coerção numérica (somente o que vamos usar)
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
        else:
            log(f"   ⚠ Coluna ausente: {c}")

    # 4) Preço: usa currentPrice; fallback para média OHLC+prevClose
    log("💰 Construindo preço...")
    if "currentPrice" in df.columns and df["currentPrice"].notna().any():
        df["price_used"] = df["currentPrice"]
        log("   ✔ price_used = currentPrice")
    else:
        price_cols = ["regularMarketPreviousClose", "regularMarketOpen", "regularMarketDayLow", "regularMarketDayHigh"]
        have_price = [c for c in price_cols if c in df.columns]
        df["price_used"] = df[have_price].mean(axis=1) if have_price else np.nan
        log(f"   ✔ price_used = mean({have_price})" if have_price else "   ❌ Sem preço disponível")

    # 5) Liquidez proxy
    log("💧 Calculando liquidez_proxy = volume10d * preço...")
    if "averageDailyVolume10Day" in df.columns:
        df["liquidity_proxy"] = df["averageDailyVolume10Day"] * df["price_used"]
        log("   ✔ liquidity_proxy calculada")
    else:
        df["liquidity_proxy"] = np.nan
        log("   ❌ Sem averageDailyVolume10Day")

    # 6) Valuation: prioriza enterpriseToEbitda; fallback EV/EBITDA calculado
    log("🏷️ Definindo EV/EBITDA...")
    if "enterpriseToEbitda" in df.columns and df["enterpriseToEbitda"].notna().any():
        df["ev_ebitda_used"] = df["enterpriseToEbitda"]
        log("   ✔ ev_ebitda_used = enterpriseToEbitda (Yahoo)")
    else:
        # fallback: EV/EBITDA = enterpriseValue / ebitda
        if ("enterpriseValue" in df.columns) and ("ebitda" in df.columns):
            df["ev_ebitda_used"] = df["enterpriseValue"] / df["ebitda"]
            log("   ✔ ev_ebitda_used = enterpriseValue / ebitda (fallback)")
        else:
            df["ev_ebitda_used"] = np.nan
            log("   ❌ Não foi possível construir EV/EBITDA")

    # 7) Diagnóstico rápido de NaNs
    log("🧪 Cobertura (não-nulos) das métricas-chave:")
    for c in ["dividendYield", "liquidity_proxy", "ev_ebitda_used", "marketCap", "priceToBook"]:
        if c in df.columns:
            log(f"   - {c}: {df[c].notna().mean():.1%} não-nulos")

    log(f"📈 Linhas antes do filtro: {len(df)}")

    # 8) Filtros
    if tipo == "acao":
        log("🏭 Aplicando filtros AÇÕES (DY + Liquidez + EV/EBITDA + MarketCap)...")

        # thresholds (ajuste fácil aqui)
        dy_min = 0.06
        liq_min = 5_000_000
        ev_ebitda_max = 10  # EV/EBITDA < 10 é um substituto razoável do EV/EBIT < 8
        mcap_min = 1_000_000_000

        mask = (
            (df["dividendYield"] > dy_min) &
            (df["liquidity_proxy"] > liq_min) &
            (df["marketCap"] > mcap_min) &
            (df["ev_ebitda_used"] > 0) &
            (df["ev_ebitda_used"] < ev_ebitda_max)
        )
        df_filtered = df[mask]

    elif tipo == "fii":
        log("🏢 Aplicando filtros FIIs (DY + Liquidez + P/VP)...")

        dy_min = 0.08
        liq_min = 1_000_000
        pvp_max = 1.05

        mask = (
            (df["dividendYield"] > dy_min) &
            (df["liquidity_proxy"] > liq_min)
        )

        if "priceToBook" in df.columns and df["priceToBook"].notna().any():
            mask &= (df["priceToBook"] < pvp_max)
        else:
            log("   ⚠ Sem priceToBook suficiente — rodando sem filtro de P/VP")

        df_filtered = df[mask]

    else:
        raise ValueError("tipo deve ser 'acao' ou 'fii'")

    log(f"✅ Linhas após filtro: {len(df_filtered)}")

    if len(df_filtered):
        cols_show = ["ticker", "dividendYield", "liquidity_proxy", "ev_ebitda_used", "marketCap", "priceToBook"]
        cols_show = [c for c in cols_show if c in df_filtered.columns]
        log("🏆 Top 10:")
        log(df_filtered[cols_show].sort_values("dividendYield", ascending=False).head(10).to_string(index=False))

    log("=" * 70)

    return df_filtered.sort_values("dividendYield", ascending=False)


screened_stocks = run_screener(valid_stocks, tipo='acao')

screened_fiis = run_screener(valid_fiis, tipo='fii')

# Salvando em "data"
screened_stocks.to_csv("C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/screened_stocks.csv", index=False)
screened_fiis.to_csv("C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/screened_fiis.csv", index=False)

############################################################################################################################################################

