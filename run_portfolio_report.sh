#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Portfolio Optimization Engine - main runner
# ============================================================
# Uso:
#   bash run_portfolio_report.sh
#   bash run_portfolio_report.sh --config .env
#
# Regras:
# - centraliza parâmetros de execução aqui
# - carrega segredos do arquivo .env (não versionado)
# - evita hardcode de API key no script
# ============================================================

CONFIG_FILE=".env"
if [[ "${1:-}" == "--config" ]]; then
  CONFIG_FILE="${2:?Informe o caminho do arquivo de configuração após --config}"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERRO] Arquivo de configuração não encontrado: $CONFIG_FILE"
  echo "Crie a partir do .env.example e preencha os valores locais."
  exit 1
fi

# Exporta variáveis definidas no .env
set -a
source "$CONFIG_FILE"
set +a

# ============================================================
# CONFIGURAÇÃO CENTRALIZADA
# ============================================================
: "${ANACONDA_BASE:?Defina ANACONDA_BASE no .env}"
: "${CONDA_ENV_NAME:?Defina CONDA_ENV_NAME no .env}"
: "${PROJECT_ROOT:?Defina PROJECT_ROOT no .env}"
: "${PORTFOLIO_REPORT_DRAFT_DIR:?Defina PORTFOLIO_REPORT_DRAFT_DIR no .env}"
: "${PORTFOLIO_REPORT_DATA_DIR:?Defina PORTFOLIO_REPORT_DATA_DIR no .env}"
: "${OPENAI_API_KEY:?Defina OPENAI_API_KEY no .env}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.4}"
RUN_SCREENING="${RUN_SCREENING:-true}"
RUN_HIST_PRICES="${RUN_HIST_PRICES:-true}"
RUN_SELECT_TITLES="${RUN_SELECT_TITLES:-true}"
RUN_ALLOCATE="${RUN_ALLOCATE:-true}"
RUN_REPORT="${RUN_REPORT:-true}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export OPENAI_API_KEY
export OPENAI_MODEL
export PORTFOLIO_REPORT_DRAFT_DIR
export PORTFOLIO_REPORT_DATA_DIR
export PROJECT_ROOT

# ============================================================
# ATIVAR CONDA
# ============================================================
source "$ANACONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "[INFO] Environment ativado: $CONDA_ENV_NAME"
echo "[INFO] Projeto: $PROJECT_ROOT"
echo "[INFO] Draft dir: $PORTFOLIO_REPORT_DRAFT_DIR"
echo "[INFO] Data dir: $PORTFOLIO_REPORT_DATA_DIR"
echo "[INFO] Modelo OpenAI: $OPENAI_MODEL"

run_step () {
  local enabled="$1"
  local label="$2"
  local script_path="$3"

  if [[ "$enabled" != "true" ]]; then
    echo "[SKIP] $label"
    return 0
  fi

  echo ""
  echo "[RUN ] $label"
  "$PYTHON_BIN" "$script_path"
  echo "[ OK ] $label"
}

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

run_step "$RUN_SCREENING"     "(1/5) Dataprep screening"         "$PROJECT_ROOT/src/dataprep/dataprep_screening.py"
run_step "$RUN_HIST_PRICES"   "(2/5) Dataprep historical prices" "$PROJECT_ROOT/src/dataprep/dataprep_historical_prices.py"
run_step "$RUN_SELECT_TITLES" "(3/5) Seleção dos 10 títulos"     "$PROJECT_ROOT/src/models/model_select10titles.py"
run_step "$RUN_ALLOCATE"      "(4/5) Alocação de ativos"         "$PROJECT_ROOT/src/models/model_allocationofassets.py"
run_step "$RUN_REPORT"        "(5/5) Geração do relatório"       "$PROJECT_ROOT/src/report/generate_portfolio_report.py"

echo ""
echo "[FIM] Pipeline concluído com sucesso."
