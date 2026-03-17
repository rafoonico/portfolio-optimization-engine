from __future__ import annotations

import json
import sys
import os

user_profile = {
    "profession": os.getenv("USER_PROFESSION", ""),
    "knowledge": os.getenv("USER_INVESTMENT_KNOWLEDGE", ""),
    "objective": os.getenv("USER_INVESTMENT_OBJECTIVE", ""),
    "risk": os.getenv("USER_RISK_TOLERANCE", ""),
    "info_style": os.getenv("USER_INFO_STYLE", ""),
    "tone": os.getenv("USER_LANGUAGE_TONE", ""),
    "hobbies": os.getenv("USER_HOBBIES", ""),
}

from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from src.portfolio_config import load_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# CONFIG
# ============================================================
CONFIG = load_config()
PATHS = CONFIG.paths

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
OUTPUT_SUBDIR = os.getenv("REPORT_OUTPUT_SUBDIR", "report_output")
PDF_NAME = os.getenv("REPORT_PDF_FILENAME", "relatorio_carteira_recomendada.pdf")
JSON_NAME = os.getenv("REPORT_CACHE_FILENAME", "web_research_cache.json")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


USE_CACHE = _env_bool("REPORT_USE_CACHE", True)
FORCE_REFRESH = _env_bool("REPORT_FORCE_REFRESH", False)

OUTPUT_DIR = PATHS.data_dir / OUTPUT_SUBDIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPTIMAL_ALLOCATION_PATH = PATHS.optimal_allocation_path
SELECTED_TITLES_PATH = PATHS.selected10_path
SCREENED_STOCKS_PATH = PATHS.screened_stocks_path
SCREENED_FIIS_PATH = PATHS.screened_fiis_path
HISTORICAL_PRICES_PATH = PATHS.historical_prices_path
PORTFOLIO_RETURNS_PATH = PATHS.portfolio_returns_path


# ============================================================
# JSON SCHEMAS
# ============================================================
TICKER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ticker": {"type": "string"},
        "company_name": {"type": "string"},
        "asset_type": {"type": "string", "enum": ["stock", "fii"]},
        "sector": {"type": "string"},
        "business_summary": {"type": "string"},
        "why_it_entered_portfolio": {"type": "string"},
        "recent_news_summary": {"type": "string"},
        "positives": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "publisher": {"type": "string"},
                    "url": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["title", "publisher", "url", "date"],
            },
            "minItems": 3,
            "maxItems": 6,
        },
    },
    "required": [
        "ticker",
        "company_name",
        "asset_type",
        "sector",
        "business_summary",
        "why_it_entered_portfolio",
        "recent_news_summary",
        "positives",
        "risks",
        "sources",
    ],
}

PORTFOLIO_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "executive_summary": {"type": "string"},
        "allocation_rationale": {"type": "string"},
        "main_strengths": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 5,
        },
        "main_risks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 5,
        },
        "methodology_note": {"type": "string"},
    },
    "required": [
        "executive_summary",
        "allocation_rationale",
        "main_strengths",
        "main_risks",
        "methodology_note",
    ],
}


# ============================================================
# HELPERS
# ============================================================
def br_percent(x: float, decimals: int = 2) -> str:
    return f"{100*x:.{decimals}f}%".replace(".", ",")


def br_number(x: float, decimals: int = 2) -> str:
    return f"{x:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def classify_asset_type(ticker: str) -> str:
    t = str(ticker).replace(".SA", "")
    return "fii" if t.endswith("11") else "stock"


def safe_get(series: pd.Series, keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in series.index and pd.notna(series[k]):
            return series[k]
    return None


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def cleaned_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else parts[0]
    return json.loads(text)


def ensure_required_files() -> None:
    required = [
        OPTIMAL_ALLOCATION_PATH,
        SELECTED_TITLES_PATH,
        SCREENED_STOCKS_PATH,
        SCREENED_FIIS_PATH,
        HISTORICAL_PRICES_PATH,
        PORTFOLIO_RETURNS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\n - ".join(missing)
        raise FileNotFoundError(
            "Arquivos obrigatórios não encontrados para gerar o relatório:\n - " + joined
        )


# ============================================================
# DATA LOADING
# ============================================================
def load_local_data() -> Dict[str, pd.DataFrame]:
    ensure_required_files()

    allocation = pd.read_csv(OPTIMAL_ALLOCATION_PATH)
    selected = pd.read_csv(SELECTED_TITLES_PATH)
    stocks = pd.read_csv(SCREENED_STOCKS_PATH)
    fiis = pd.read_csv(SCREENED_FIIS_PATH)
    hist = pd.read_csv(HISTORICAL_PRICES_PATH)
    p_returns = pd.read_csv(PORTFOLIO_RETURNS_PATH)

    allocation["ticker"] = allocation["ticker"].map(normalize_ticker)
    selected["ticker"] = selected["ticker"].map(normalize_ticker)
    stocks["ticker"] = stocks["ticker"].map(normalize_ticker)
    fiis["ticker"] = fiis["ticker"].map(normalize_ticker)

    hist["date"] = pd.to_datetime(hist["date"])
    p_returns["date"] = pd.to_datetime(p_returns["date"])

    return {
        "allocation": allocation,
        "selected": selected,
        "stocks": stocks,
        "fiis": fiis,
        "hist": hist,
        "portfolio_returns": p_returns,
    }


# ============================================================
# LOCAL ENRICHMENT
# ============================================================
def build_asset_snapshot(ticker: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    allocation = data["allocation"]
    stocks = data["stocks"]
    fiis = data["fiis"]

    row_alloc = allocation.loc[allocation["ticker"] == ticker].iloc[0]
    asset_type = classify_asset_type(ticker)
    local = stocks if asset_type == "stock" else fiis
    row_local = local.loc[local["ticker"] == ticker]
    row_local = row_local.iloc[0] if len(row_local) else pd.Series(dtype=object)

    snapshot = {
        "ticker": ticker,
        "asset_type": asset_type,
        "allocation": float(row_alloc["allocation"]),
        "allocation_pct": float(row_alloc["allocation_pct"]),
        "selection_score": float(row_alloc.get("selection_score", np.nan)),
        "modified_sharpe": float(row_alloc.get("modified_sharpe", np.nan)),
        "omega": float(row_alloc.get("omega", np.nan)),
        "alpha_jensen": float(row_alloc.get("alpha_jensen", np.nan)),
        "max_drawdown": float(row_alloc.get("max_drawdown", np.nan)),
        "liquidity": float(row_alloc.get("liquidity", np.nan)),
        "current_price": safe_get(row_local, ["currentPrice", "price_used", "previousClose"]),
        "market_cap": safe_get(row_local, ["marketCap", "nonDilutedMarketCap"]),
        "dividend_yield": safe_get(row_local, ["dividendYield", "trailingAnnualDividendYield"]),
        "trailing_pe": safe_get(row_local, ["trailingPE", "forwardPE"]),
        "price_to_book": safe_get(row_local, ["priceToBook"]),
        "profit_margins": safe_get(row_local, ["profitMargins", "profitMargins.1"]),
        "revenue_growth": safe_get(row_local, ["revenueGrowth"]),
        "earnings_growth": safe_get(row_local, ["earningsGrowth", "earningsQuarterlyGrowth"]),
        "return_on_equity": safe_get(row_local, ["returnOnEquity"]),
        "debt_to_equity": safe_get(row_local, ["debtToEquity"]),
        "ev_ebitda": safe_get(row_local, ["ev_ebitda_used", "enterpriseToEbitda"]),
        "recommendation_key": safe_get(row_local, ["recommendationKey"]),
    }
    return snapshot


def create_metrics_table_df(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    allocation = data["allocation"].copy()
    rows: List[Dict[str, Any]] = []
    for ticker in allocation["ticker"].tolist():
        rows.append(build_asset_snapshot(ticker, data))
    return pd.DataFrame(rows).sort_values("allocation", ascending=False).reset_index(drop=True)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================
def build_performance_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    hist = data["hist"].copy()
    port = data["portfolio_returns"].copy()
    port = port.sort_values("date")

    hist = hist[["date", "ibov_close", "cdi_return"]].copy()
    merged = port.merge(hist, on="date", how="left")
    merged["ibov_return"] = merged["ibov_close"].pct_change()
    merged["cdi_return"] = merged["cdi_return"].ffill()

    merged["portfolio_cum"] = (1 + merged["portfolio_return"]).cumprod()
    merged["ibov_cum"] = (1 + merged["ibov_return"].fillna(0)).cumprod()
    merged["cdi_cum"] = (1 + merged["cdi_return"].fillna(0)).cumprod()

    ann_factor = 252
    mean_daily = merged["portfolio_return"].mean()
    vol_daily = merged["portfolio_return"].std(ddof=1)
    rf_daily = merged["cdi_return"].mean()
    excess_daily = mean_daily - rf_daily
    sharpe = np.nan if vol_daily == 0 else (excess_daily / vol_daily) * np.sqrt(ann_factor)

    wealth = merged["portfolio_cum"]
    dd = wealth / wealth.cummax() - 1

    summary = {
        "start_date": merged["date"].min().strftime("%Y-%m-%d"),
        "end_date": merged["date"].max().strftime("%Y-%m-%d"),
        "annual_return_est": (1 + mean_daily) ** ann_factor - 1,
        "annual_vol_est": vol_daily * np.sqrt(ann_factor),
        "annual_sharpe_est": sharpe,
        "max_drawdown_hist": dd.min(),
        "portfolio_cum_total": merged["portfolio_cum"].iloc[-1] - 1,
        "ibov_cum_total": merged["ibov_cum"].iloc[-1] - 1,
        "cdi_cum_total": merged["cdi_cum"].iloc[-1] - 1,
        "timeseries": merged,
    }
    return summary


# ============================================================
# OPENAI WEB RESEARCH
# ============================================================
def make_client() -> OpenAI:
    return OpenAI()

def build_user_context(profile):

    return f"""
Leitor do relatório:

Profissão: {profile['profession']}
Conhecimento sobre investimentos: {profile['knowledge']}
Objetivo com investimentos: {profile['objective']}
Tolerância a risco: {profile['risk']}

Preferências de comunicação:
- Forma de explicação: {profile['info_style']}
- Tom de linguagem: {profile['tone']}

Interesses pessoais: {profile['hobbies']}

Adapte a linguagem e exemplos do relatório para esse perfil.
"""

def ticker_prompt(snapshot: Dict[str, Any]) -> str:
    return f"""
{build_user_context(user_profile)}

Você é um analista buy side cobrindo ações brasileiras e FIIs.
Pesquise somente em fontes abertas e gratuitas na web.
Escreva em português do Brasil.

Objetivo: produzir um resumo robusto, factual e conciso sobre o ativo {snapshot['ticker']}.

Dados locais do pipeline para contextualizar a análise:
- ticker: {snapshot['ticker']}
- tipo de ativo: {snapshot['asset_type']}
- peso recomendado na carteira: {snapshot['allocation_pct']:.4f}%
- modified_sharpe do pipeline: {snapshot['modified_sharpe']}
- omega do pipeline: {snapshot['omega']}
- alpha_jensen do pipeline: {snapshot['alpha_jensen']}
- max_drawdown do pipeline: {snapshot['max_drawdown']}
- liquidez proxy do pipeline: {snapshot['liquidity']}
- preço atual local: {snapshot['current_price']}
- market cap local: {snapshot['market_cap']}
- dividend yield local: {snapshot['dividend_yield']}
- P/L local: {snapshot['trailing_pe']}
- P/VP local: {snapshot['price_to_book']}
- ROE local: {snapshot['return_on_equity']}
- dívida/patrimônio local: {snapshot['debt_to_equity']}
- EV/EBITDA local: {snapshot['ev_ebitda']}
- recommendationKey local: {snapshot['recommendation_key']}

Entregue:
1) o nome da empresa/fundo e setor;
2) uma explicação breve do negócio e de como ele ganha dinheiro;
3) um resumo do noticiário e dos fatos recentes mais relevantes para tese de investimento;
4) por que esse ativo faz sentido dentro da carteira recomendada;
5) principais pontos positivos e principais riscos;
6) de 3 a 6 fontes abertas e recentes.

Regras:
- Não invente fatos.
- Prefira RI da companhia, B3, fatos relevantes, release de resultados e veículos econômicos conhecidos.
- Se houver divergências, priorize fontes primárias.
- Seja específico, mas sem jargão excessivo.
- Você não vai fazer comentário individual de ativo cujo peso recomendado na carteira seja menor que 1%, mas ainda assim deve entregar as informações acima de forma concisa e objetiva.
""".strip()


def portfolio_prompt(metrics_df: pd.DataFrame, perf: Dict[str, Any]) -> str:
    compact = metrics_df[
        [
            "ticker",
            "asset_type",
            "allocation_pct",
            "modified_sharpe",
            "omega",
            "alpha_jensen",
            "max_drawdown",
            "liquidity",
        ]
    ].to_dict(orient="records")

    return f"""
{build_user_context(user_profile)}

Você é um analista de investimentos escrevendo o resumo executivo de uma carteira recomendada brasileira.
Escreva em português do Brasil.

Dados da carteira:
{json.dumps(compact, ensure_ascii=False)}

Estatísticas agregadas da carteira (janela histórica do backtest/avaliação):
- período: {perf['start_date']} até {perf['end_date']}
- retorno acumulado da carteira: {perf['portfolio_cum_total']}
- retorno acumulado do Ibovespa: {perf['ibov_cum_total']}
- retorno acumulado do CDI: {perf['cdi_cum_total']}
- retorno anualizado estimado: {perf['annual_return_est']}
- volatilidade anualizada estimada: {perf['annual_vol_est']}
- Sharpe anualizado estimado: {perf['annual_sharpe_est']}
- max drawdown histórico da carteira: {perf['max_drawdown_hist']}

Explique:
1) qual é o perfil geral da carteira;
2) por que a distribuição de pesos faz sentido;
3) forças e fragilidades da recomendação;
4) uma nota curta sobre metodologia: seleção prévia dos ativos + otimização de pesos com foco em retorno ajustado ao risco.

Seja analítico, equilibrado e objetivo.
""".strip()


def responses_create_json(client: OpenAI, *, prompt: str, schema: Dict[str, Any], name: str) -> Dict[str, Any]:
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        tools=[{"type": "web_search"}],
        text={
            "format": {
                "type": "json_schema",
                "name": name,
                "schema": schema,
                "strict": True,
            }
        },
    )
    return cleaned_json(response.output_text)


def load_cache() -> Dict[str, Any]:
    cache_path = OUTPUT_DIR / JSON_NAME
    if cache_path.exists() and USE_CACHE:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {"tickers": {}, "portfolio": None}


def save_cache(cache: Dict[str, Any]) -> None:
    (OUTPUT_DIR / JSON_NAME).write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_web_research(metrics_df: pd.DataFrame, perf: Dict[str, Any]) -> Dict[str, Any]:
    client = make_client()
    cache = load_cache()

    ticker_results: Dict[str, Any] = {}
    for _, row in metrics_df.iterrows():
        ticker = row["ticker"]
        if USE_CACHE and not FORCE_REFRESH and ticker in cache.get("tickers", {}):
            ticker_results[ticker] = cache["tickers"][ticker]
            continue

        snapshot = row.to_dict()
        result = responses_create_json(
            client,
            prompt=ticker_prompt(snapshot),
            schema=TICKER_SCHEMA,
            name=f"research_{ticker.replace('.', '_')}",
        )
        ticker_results[ticker] = result
        cache.setdefault("tickers", {})[ticker] = result
        save_cache(cache)

    if USE_CACHE and not FORCE_REFRESH and cache.get("portfolio") is not None:
        portfolio_result = cache["portfolio"]
    else:
        portfolio_result = responses_create_json(
            client,
            prompt=portfolio_prompt(metrics_df, perf),
            schema=PORTFOLIO_SCHEMA,
            name="portfolio_summary",
        )
        cache["portfolio"] = portfolio_result
        save_cache(cache)

    return {"tickers": ticker_results, "portfolio": portfolio_result}


# ============================================================
# CHARTS
# ============================================================
def make_allocation_chart(metrics_df: pd.DataFrame) -> Path:
    out = OUTPUT_DIR / "allocation_chart.png"
    chart_df = metrics_df.sort_values("allocation_pct", ascending=True)

    plt.figure(figsize=(8.5, 5.2))
    plt.barh(chart_df["ticker"], chart_df["allocation_pct"])
    plt.xlabel("Alocação (%)")
    plt.ylabel("Ticker")
    plt.title("Alocação recomendada por ativo")
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def make_performance_chart(perf: Dict[str, Any]) -> Path:
    out = OUTPUT_DIR / "performance_chart.png"
    ts = perf["timeseries"].copy()

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(ts["date"], ts["portfolio_cum"], label="Carteira")
    plt.plot(ts["date"], ts["ibov_cum"], label="Ibovespa")
    plt.plot(ts["date"], ts["cdi_cum"], label="CDI")
    plt.title("Evolução acumulada - carteira vs. benchmarks")
    plt.ylabel("Crescimento de R$ 1,00")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


# ============================================================
# PDF
# ============================================================
def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SubTitleCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#475569"),
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="TickerHeader",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#111827"),
        spaceBefore=8,
        spaceAfter=6,
    ))
    return styles


def paragraph_list(items: List[str], styles, bullet: str = "- ") -> List[Paragraph]:
    out = []
    for item in items:
        out.append(Paragraph(f"{bullet}{item}", styles["BodyCustom"]))
    return out


def metrics_table(metrics_df: pd.DataFrame) -> Table:
    show = metrics_df.copy()
    show = show[[
        "ticker", "asset_type", "allocation_pct", "modified_sharpe",
        "omega", "alpha_jensen", "max_drawdown", "liquidity"
    ]]
    show.columns = [
        "Ticker", "Tipo", "Peso (%)", "Mod. Sharpe",
        "Omega", "Alpha Jensen", "Max DD", "Liquidez"
    ]
    show["Peso (%)"] = show["Peso (%)"].map(lambda x: br_number(x, 2))
    show["Mod. Sharpe"] = show["Mod. Sharpe"].map(lambda x: br_number(x, 3))
    show["Omega"] = show["Omega"].map(lambda x: br_number(x, 3))
    show["Alpha Jensen"] = show["Alpha Jensen"].map(lambda x: br_number(x, 4))
    show["Max DD"] = show["Max DD"].map(lambda x: br_number(x, 3))
    show["Liquidez"] = show["Liquidez"].map(lambda x: br_number(x, 0))

    data = [show.columns.tolist()] + show.values.tolist()
    col_widths = [2.3*cm, 1.8*cm, 2.0*cm, 2.2*cm, 1.9*cm, 2.5*cm, 1.9*cm, 3.0*cm]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.7),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def perf_table(perf: Dict[str, Any]) -> Table:
    rows = [
        ["Período avaliado", f"{perf['start_date']} a {perf['end_date']}"],
        ["Retorno acumulado da carteira", br_percent(perf['portfolio_cum_total'])],
        ["Retorno acumulado do Ibovespa", br_percent(perf['ibov_cum_total'])],
        ["Retorno acumulado do CDI", br_percent(perf['cdi_cum_total'])],
        ["Retorno anualizado estimado", br_percent(perf['annual_return_est'])],
        ["Volatilidade anualizada estimada", br_percent(perf['annual_vol_est'])],
        ["Sharpe anualizado estimado", br_number(perf['annual_sharpe_est'], 2)],
        ["Max drawdown histórico", br_percent(perf['max_drawdown_hist'])],
    ]
    tbl = Table(rows, colWidths=[6.5*cm, 5.0*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def build_pdf(metrics_df: pd.DataFrame, perf: Dict[str, Any], research: Dict[str, Any], chart_paths: Dict[str, Path]) -> Path:
    styles = build_styles()
    pdf_path = OUTPUT_DIR / PDF_NAME
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.7*cm,
        rightMargin=1.7*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
        title="Relatório de Carteira Recomendada",
        author="Pipeline Portfolio + OpenAI Web Research",
    )

    story = []
    portfolio_text = research["portfolio"]
    ticker_texts = research["tickers"]

    story.append(Paragraph("Relatório da Carteira Recomendada", styles["TitleCustom"]))
    story.append(Paragraph(
        "Integração entre pipeline quantitativo local e pesquisa web em fontes abertas para explicar a recomendação, os pesos e os fundamentos de cada papel.",
        styles["SubTitleCustom"],
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Resumo executivo", styles["SectionCustom"]))
    story.append(Paragraph(portfolio_text["executive_summary"], styles["BodyCustom"]))
    story.append(Paragraph(portfolio_text["allocation_rationale"], styles["BodyCustom"]))

    story.append(Paragraph("Pontos fortes", styles["SectionCustom"]))
    story.extend(paragraph_list(portfolio_text["main_strengths"], styles))
    story.append(Paragraph("Principais riscos", styles["SectionCustom"]))
    story.extend(paragraph_list(portfolio_text["main_risks"], styles))

    story.append(Paragraph("Metodologia", styles["SectionCustom"]))
    story.append(Paragraph(portfolio_text["methodology_note"], styles["BodyCustom"]))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Métricas resumidas da carteira", styles["SectionCustom"]))
    story.append(perf_table(perf))
    story.append(Spacer(1, 0.35*cm))
    story.append(Paragraph("Composição e métricas por ativo", styles["SectionCustom"]))
    story.append(metrics_table(metrics_df))
    story.append(Spacer(1, 0.35*cm))

    story.append(Paragraph("Visualizações", styles["SectionCustom"]))
    story.append(Image(str(chart_paths["allocation"]), width=16.5*cm, height=9.3*cm))
    story.append(Spacer(1, 0.2*cm))
    story.append(Image(str(chart_paths["performance"]), width=16.5*cm, height=8.8*cm))
    story.append(PageBreak())

    story.append(Paragraph("Comentário individual dos ativos", styles["SectionCustom"]))

    for _, row in metrics_df.iterrows():
        ticker = row["ticker"]
        t = ticker_texts[ticker]

        story.append(Paragraph(
            f"{ticker} - {t['company_name']} ({'FII' if t['asset_type']=='fii' else 'Ação'}) - peso recomendado: {br_number(row['allocation_pct'], 2)}%",
            styles["TickerHeader"],
        ))

        local_bits = []
        if pd.notna(row.get("current_price")):
            local_bits.append(f"preço local: R$ {br_number(float(row['current_price']), 2)}")
        if pd.notna(row.get("dividend_yield")):
            local_bits.append(f"dividend yield local: {br_percent(float(row['dividend_yield']), 2)}")
        if pd.notna(row.get("trailing_pe")):
            local_bits.append(f"P/L local: {br_number(float(row['trailing_pe']), 2)}")
        if pd.notna(row.get("price_to_book")):
            local_bits.append(f"P/VP local: {br_number(float(row['price_to_book']), 2)}")
        if pd.notna(row.get("return_on_equity")):
            local_bits.append(f"ROE local: {br_percent(float(row['return_on_equity']), 2)}")
        if pd.notna(row.get("debt_to_equity")):
            local_bits.append(f"dívida/patrimônio local: {br_number(float(row['debt_to_equity']), 2)}")

        story.append(Paragraph(f"<b>Setor:</b> {t['sector']}", styles["BodyCustom"]))
        if local_bits:
            story.append(Paragraph(f"<b>Métricas locais usadas como apoio:</b> {'; '.join(local_bits)}.", styles["BodyCustom"]))
        story.append(Paragraph(f"<b>O que é o papel:</b> {t['business_summary']}", styles["BodyCustom"]))
        story.append(Paragraph(f"<b>Por que entrou na carteira:</b> {t['why_it_entered_portfolio']}", styles["BodyCustom"]))
        story.append(Paragraph(f"<b>Noticiário e fatos recentes:</b> {t['recent_news_summary']}", styles["BodyCustom"]))

        story.append(Paragraph("Pontos positivos", styles["SmallCustom"]))
        story.extend(paragraph_list(t["positives"], styles))
        story.append(Paragraph("Riscos", styles["SmallCustom"]))
        story.extend(paragraph_list(t["risks"], styles))

        src_lines = []
        for s in t["sources"]:
            line = f"- {s['publisher']} | {s['title']} | {s['date']} | {s['url']}"
            src_lines.append(line)
        story.append(Paragraph("Fontes abertas consultadas", styles["SmallCustom"]))
        for line in src_lines:
            story.append(Paragraph(line, styles["SmallCustom"]))

        story.append(Spacer(1, 0.2*cm))

    story.append(PageBreak())
    story.append(Paragraph("Ressalvas importantes", styles["SectionCustom"]))
    story.append(Paragraph(
        "Este relatório combina métricas históricas do pipeline com pesquisa automatizada em fontes abertas. Ele não substitui due diligence própria, leitura integral dos releases e avaliação de adequação ao perfil de risco do investidor. Noticiário, fundamentos e preços mudam com frequência; por isso, a recomendação deve ser revalidada antes de qualquer execução.",
        styles["BodyCustom"],
    ))

    doc.build(story)
    return pdf_path


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    data = load_local_data()
    metrics_df = create_metrics_table_df(data)
    perf = build_performance_summary(data)
    research = run_web_research(metrics_df, perf)

    chart_paths = {
        "allocation": make_allocation_chart(metrics_df),
        "performance": make_performance_chart(perf),
    }

    pdf_path = build_pdf(metrics_df, perf, research, chart_paths)

    print("PDF gerado com sucesso:")
    print(pdf_path)
    print("\nCache de pesquisa:")
    print(OUTPUT_DIR / JSON_NAME)


if __name__ == "__main__":
    main()
