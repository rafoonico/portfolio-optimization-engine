# -*- coding: utf-8 -*-
"""
dataprep_historical_prices.py

Objetivo
--------
1) Ler múltiplos arquivos (CSV/XLSX) com tickers (ex.: screened_stocks.csv, screened_fiis.csv)
2) Consolidar tickers em uma lista única
3) Baixar histórico diário de OHLC via yfinance (todo o histórico disponível)
4) Adicionar:
   - CDI: retorno diário via SGS/BCB usando python-bcb (série 12, CDI diário % ao dia)
   - Ibovespa: pontuação (Close) do ^BVSP via yfinance
5) Montar um dataframe final:
   - index = data (ou período, se reamostrado)
   - colunas = 1 coluna por ticker, contendo a média entre (Open, High, Low, Close) do ticker no dia/período
              + cdi_return (decimal)
              + ibov_close (pontos)

Regra para freq != D
--------------------
Para cada ticker, primeiro agregamos OHLC no período:
  open=first, high=max, low=min, close=last
Depois calculamos a média do período: (open+high+low+close)/4

CDI:
  diário: retorno decimal (ex.: 0.00042)
  período: acumulado (1+r).prod() - 1

IBOV:
  diário: close do dia
  período: último close do período

Exemplo
-------
python dataprep_historical_prices.py \
  --inputs "screened_stocks.csv" "screened_fiis.csv" \
  --ticker-col ticker \
  --freq D \
  --out "historical_dataset.parquet"
"""
# ===============================
# BLOCO 1 — IMPORTS
# ===============================

import pandas as pd
import numpy as np
import time
import yfinance as yf

from bcb import sgs
from datetime import datetime, timedelta

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("Imports carregados.")


# ===============================
# BLOCO 2 — LER OS ARQUIVOS DE TICKERS
# ===============================

stocks = pd.read_csv(
    "C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/screened_stocks.csv"
)

fiis = pd.read_csv(
    "C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/screened_fiis.csv"
)

print("Stocks:", stocks.shape)
print("FIIs:", fiis.shape)

stocks.head()
fiis.head()

# ===============================
# BLOCO 3 — LISTA DE TICKERS
# ===============================

tickers = (
    pd.concat([stocks["ticker"], fiis["ticker"]])
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print("Total tickers:", len(tickers))
print(tickers[:20])

# ===============================
# BLOCO 4 — DOWNLOAD OHLC (Yahoo Finance)
# ===============================

def split_batches(lst, batch_size=8):
    """Divide lista em batches menores"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


all_prices = []

for batch in split_batches(tickers, batch_size=5):

    print("Baixando batch:", batch)

    try:

        data = yf.download(
            batch,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=False,
            progress=False
        )

        all_prices.append(data)

    except Exception as e:

        print("Erro no batch:", batch)
        print(e)

    # pausa para evitar rate limit
    time.sleep(4)


# junta todos os batches
prices = pd.concat(all_prices, axis=1)

print("Download finalizado")
print(prices.shape)
prices.head()

# ===============================
# BLOCO 5 — TRANSFORMAR EM LONG
# ===============================

dfs = []

for t in tickers:

    if t not in prices.columns.get_level_values(0):
        continue

    tmp = prices[t][["Open","High","Low","Close"]].copy()

    tmp.columns = ["open","high","low","close"]
    tmp["ticker"] = t
    tmp["date"] = tmp.index

    dfs.append(tmp)

prices_long = pd.concat(dfs).reset_index(drop=True)

print(prices_long.shape)
prices_long.tail()


# ===============================
# BLOCO 6 — MÉDIA OHLC
# ===============================

prices_long["price_mean"] = prices_long[
    ["open","high","low","close"]
].mean(axis=1)

prices_long.tail()

# ===============================
# BLOCO 7 — PIVOT
# ===============================

df_prices = prices_long.pivot_table(
    index="date",
    columns="ticker",
    values="price_mean"
)

df_prices = df_prices.sort_index()

print(df_prices.shape)
df_prices.tail()

# ===============================
# BLOCO 8 — IBOVESPA
# ===============================

ibov = yf.download(
    "^BVSP",
    interval="1d",
    progress=True
)

ibov = ibov[('Close', '^BVSP')].rename("ibov_close")

print(ibov.shape)
ibov.tail()

# ===============================
# BLOCO 9 — CDI (COM JANELA DE 10 ANOS)
# ===============================

end_date = datetime.today()
start_date = end_date - timedelta(days=365*10)

print("Baixando CDI:")
print("start:", start_date.date())
print("end:", end_date.date())

cdi = sgs.get(
    12,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d")
)

cdi.columns = ["cdi_percent"]

cdi["cdi_return"] = cdi["cdi_percent"] / 100

cdi = cdi[["cdi_return"]]

print(cdi.shape)
cdi.tail()

# ===============================
# BLOCO 10 — MERGE FINAL
# ===============================

df = (
    df_prices
    .join(cdi, how="left")
    .join(ibov, how="left")
)

df = df.sort_index()

# ffill na coluna cdi_return 
df[["cdi_return","ibov_close"]] = df[["cdi_return","ibov_close"]].ffill()

print(df.shape)

df.tail()


# ===============================
# BLOCO 11 — RESAMPLE
# ===============================

# Usa o resample do pandas para pegar informações mensais, anuais e etc

freq = "M"   # D, M, A, etc

df_resampled = df.resample(freq).last() # dá pra usar .mean() também, caso queria uma média do período

df_resampled.head()


# ===============================
# BLOCO 12 — SALVAR
# ===============================

df = df.reset_index()

df.to_csv(
"C:/Users/USER/Documents/Pessoal/ProjetosPessoais/portfolio-optimization-engine/data/historical_prices.csv", index=False
)

print("Dataset salvo.")