# Portfolio Optimization Engine

This repository contains an end-to-end portfolio research pipeline for Brazilian stocks and FIIs (real estate funds). It screens candidate assets, downloads historical data, selects the top assets using risk-adjusted metrics, optimizes portfolio weights, and generates a PDF report enriched with OpenAI-powered web research.

The active implementation lives under `src/`. The shell runner (`run_portfolio_report.sh`) orchestrates the full workflow through environment variables loaded from a local `.env` file.

## What the pipeline does

The solution runs in five sequential stages:

1. `dataprep_screening.py`
   Reads an Excel list of candidate stocks and FIIs, validates tickers, queries Yahoo Finance data, and filters assets using liquidity and valuation rules.
2. `dataprep_historical_prices.py`
   Downloads historical prices for the screened assets, plus Ibovespa and CDI benchmark data.
3. `model_select10titles.py`
   Computes risk-adjusted metrics and selects the top `K` assets with an optimization model.
4. `model_allocationofassets.py`
   Optimizes portfolio weights with a modified Sharpe objective and portfolio constraints.
5. `generate_portfolio_report.py`
   Builds charts, performs OpenAI web research, and exports a PDF report.

## Repository layout

```text
portfolio-optimization-engine/
|-- data/
|   |-- list_titles_FII_stocks.xlsx
|   |-- screened_stocks.csv
|   |-- screened_fiis.csv
|   |-- historical_prices.csv
|   |-- selected10titles.csv
|   |-- optimal_allocation.csv
|   |-- portfolio_returns_selected10.csv
|   `-- report_output/
|       |-- allocation_chart.png
|       |-- performance_chart.png
|       |-- relatorio_carteira_recomendada.pdf
|       `-- web_research_cache.json
|-- draft/
|   `-- legacy or draft artifacts kept outside the active pipeline
|-- src/
|   |-- dataprep/
|   |   |-- dataprep_screening.py
|   |   `-- dataprep_historical_prices.py
|   |-- models/
|   |   |-- model_select10titles.py
|   |   `-- model_allocationofassets.py
|   |-- report/
|   |   `-- generate_portfolio_report.py
|   |-- portfolio_config.py
|   `-- __init__.py
|-- pyproject.toml
|-- requirements.txt
`-- run_portfolio_report.sh
```

### Directory notes

- `src/` is the current codebase used by the runner.
- `data/` contains both the input workbook and all generated artifacts.
- `draft/` exists in the repository and is still required by the shell runner as a configured path, but the current `src/` pipeline does not actively depend on it.
- `requirements.txt` is a Conda environment export for `win-64`, not a standard `pip` requirements file.
- `pyproject.toml` contains the Python dependency list for a normal `pip` installation.

## Prerequisites

- Python `3.10+`
- Internet access
- Bash shell to run `run_portfolio_report.sh`
  - On Linux/macOS: native shell is fine
  - On Windows: use Git Bash, WSL, or another Bash-compatible shell
- Conda if you want to use the provided shell runner as-is

## Installation

You can install the project in two ways.

### Option 1: Recommended for development with `pip`

Use this if you want a clean Python environment and do not need to recreate the exact exported Conda setup.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### Option 2: Recreate the exported Conda environment

Use this if you want to install the environment described by `requirements.txt`.

```bash
conda create --name portfolio-engine --file requirements.txt
conda activate portfolio-engine
```

Important:

- `requirements.txt` must be installed with `conda create --file ...`.
- Do not use `pip install -r requirements.txt`; the file format is not compatible with `pip`.

## External services and API keys

### Mandatory

- `OPENAI_API_KEY`
  Required to generate the final report with OpenAI web research in `src/report/generate_portfolio_report.py`.

### No API key required

The pipeline also depends on public market data sources:

- Yahoo Finance via `yfinance`
- Yahoo Finance via `yahooquery`
- Banco Central do Brasil via `python-bcb`

These services do not require user-managed API keys in the current implementation, but they do require network connectivity and may fail if the upstream services are unavailable or rate-limited.

## Input data expectations

The screening stage expects the file below:

- `data/list_titles_FII_stocks.xlsx`

Expected worksheet structure:

- Sheet `fii`
  Must contain a `code` column
- Sheet `stock`
  Must contain a `code` column
- Sheet `source`
  Present in the sample workbook, but not consumed by the active pipeline

Observed sample columns:

- `fii`: `corporate_name`, `fund`, `code`
- `stock`: `sector`, `subsector`, `segment`, `issuer_trading_name`, `code`, `trading_segment`
- `source`: `title`, `link`

## Configuration and execution parameters

The shell runner accepts only one command-line option:

```bash
bash run_portfolio_report.sh
bash run_portfolio_report.sh --config .env
```

All other runtime controls are environment variables loaded from the configuration file.

### 1. Core runner variables

These variables are validated directly by `run_portfolio_report.sh`.

| Variable | Required | Default | Description |
|---|---|---:|---|
| `ANACONDA_BASE` | Yes | None | Base Conda installation path. The script loads `conda.sh` from here before activating the environment. |
| `CONDA_ENV_NAME` | Yes | None | Name of the Conda environment to activate. |
| `PROJECT_ROOT` | Yes | None | Absolute path to the repository root. |
| `PORTFOLIO_REPORT_DRAFT_DIR` | Yes | None | Path to the `draft/` directory. Required by the runner for configuration completeness, even though the active `src/` pipeline does not currently use it directly. |
| `PORTFOLIO_REPORT_DATA_DIR` | Yes | None | Path to the directory where input and output data files live. |
| `OPENAI_API_KEY` | Yes | None | OpenAI API key used by the report generation stage. |
| `OPENAI_MODEL` | No | `gpt-5.4` | Model passed to the OpenAI Responses API. |
| `RUN_SCREENING` | No | `true` | Enables stage 1: screening. |
| `RUN_HIST_PRICES` | No | `true` | Enables stage 2: historical price download. |
| `RUN_SELECT_TITLES` | No | `true` | Enables stage 3: top-asset selection. |
| `RUN_ALLOCATE` | No | `true` | Enables stage 4: allocation optimization. |
| `RUN_REPORT` | No | `true` | Enables stage 5: PDF report generation. |
| `PYTHON_BIN` | No | `python` | Python executable used by the runner. |

Notes:

- The shell runner validates `OPENAI_API_KEY` before executing any stage. In practice, only the report stage needs it, but the current script requires it up front.
- Boolean flags are expected as strings such as `true` or `false`.

### 2. Path and filename variables

These are loaded by `src/portfolio_config.py` and control where the pipeline reads and writes files.

| Variable | Required | Default | Description |
|---|---|---:|---|
| `INPUT_TITLES_FILENAME` | No | `list_titles_FII_stocks.xlsx` | Input Excel workbook with the candidate universe. |
| `SCREENED_STOCKS_FILENAME` | No | `screened_stocks.csv` | Output CSV for screened stocks. |
| `SCREENED_FIIS_FILENAME` | No | `screened_fiis.csv` | Output CSV for screened FIIs. |
| `HISTORICAL_PRICES_FILENAME` | No | `historical_prices.csv` | Output CSV for historical prices plus CDI and Ibovespa. |
| `SELECTED10_FILENAME` | No | `selected10titles.csv` | Output CSV for selected assets. |
| `OPTIMAL_ALLOCATION_FILENAME` | No | `optimal_allocation.csv` | Output CSV for optimized weights. |
| `PORTFOLIO_RETURNS_FILENAME` | No | `portfolio_returns_selected10.csv` | Output CSV for the portfolio return time series. |

### 3. Screening parameters

| Variable | Required | Default | Description |
|---|---|---:|---|
| `SCREENING_DEBUG` | No | `true` | Enables verbose logs for the screening stage. |

The hard-coded screening rules currently implemented in `src/dataprep/dataprep_screening.py` are:

- Stocks:
  - `dividendYield > 0.06`
  - `liquidity_proxy > 5,000,000`
  - `marketCap > 1,000,000,000`
  - `0 < EV/EBITDA < 10`
- FIIs:
  - `dividendYield > 0.08`
  - `liquidity_proxy > 1,000,000`
  - `priceToBook < 1.05` when the field is available

### 4. Historical data parameters

| Variable | Required | Default | Description |
|---|---|---:|---|
| `HISTORICAL_FREQ` | No | `D` | Resampling frequency applied after download. `D` keeps daily data; other Pandas-compatible frequencies are passed to `resample(...).last()`. |
| `HISTORICAL_BATCH_SIZE` | No | `5` | Number of tickers downloaded per Yahoo Finance batch. |
| `HISTORICAL_PAUSE_SECONDS` | No | `4` | Delay between Yahoo Finance download batches. |
| `HISTORICAL_CDI_YEARS` | No | `10` | Number of years of CDI history requested from BCB. |
| `HISTORICAL_CDI_MAX_RETRIES` | No | `5` | Maximum retry attempts for CDI download. |
| `HISTORICAL_CDI_RETRY_SLEEP_SECONDS` | No | `6` | Pause between CDI retry attempts. |

### 5. Asset selection parameters

| Variable | Required | Default | Description |
|---|---|---:|---|
| `SELECTION_K` | No | `10` | Number of assets to select in the MILP selection model. |
| `SELECTION_MIN_LIQUIDITY` | No | `1000000` | Minimum liquidity proxy allowed in the optimization stage. |
| `SELECTION_MAX_BETA` | No | `2.0` | Maximum beta allowed for selected assets. |
| `SELECTION_MIN_OBS` | No | `126` | Minimum number of return observations required for an asset to be scored. |

The selection model uses:

- annualized return and volatility
- beta and Jensen's alpha
- modified Sharpe ratio
- Omega ratio
- max drawdown
- liquidity proxy

### 6. Allocation parameters

| Variable | Required | Default | Description |
|---|---|---:|---|
| `ALLOCATION_LOOKBACK_DAYS` | No | `756` | Number of recent rows used to build the return panel for optimization. |
| `ALLOCATION_ALPHA` | No | `0.95` | Confidence level used in the modified VaR / modified Sharpe calculations. |
| `ALLOCATION_MAX_WEIGHT` | No | `0.20` | Per-asset maximum weight bound. |
| `ALLOCATION_MIN_WEIGHT` | No | `0.00` | Per-asset minimum weight bound. |
| `ALLOCATION_MAX_FII_ALLOCATION` | No | `0.40` | Maximum combined weight allowed for FII positions. |

Important feasibility note:

- The allocation optimizer enforces weights that sum to `1.0`.
- Make sure `number_of_assets * ALLOCATION_MAX_WEIGHT >= 1.0`.
- Make sure `number_of_assets * ALLOCATION_MIN_WEIGHT <= 1.0`.
- With the default setup (`10` assets, max weight `0.20`), the problem is feasible.

### 7. Report generation parameters

| Variable | Required | Default | Description |
|---|---|---:|---|
| `REPORT_OUTPUT_SUBDIR` | No | `report_output` | Subdirectory under `PORTFOLIO_REPORT_DATA_DIR` where charts, cache, and PDF are written. |
| `REPORT_PDF_FILENAME` | No | `relatorio_carteira_recomendada.pdf` | Final PDF filename. |
| `REPORT_CACHE_FILENAME` | No | `web_research_cache.json` | JSON cache for OpenAI web research results. |
| `REPORT_USE_CACHE` | No | `true` | Reuses previous web research from the cache file when available. |
| `REPORT_FORCE_REFRESH` | No | `false` | Forces fresh web research even if cache exists. |

### 8. Optional user profile parameters for report tone

These variables personalize the natural-language explanation in the generated report:

| Variable | Required | Default | Description |
|---|---|---:|---|
| `USER_PROFESSION` | No | empty | Reader profession. |
| `USER_INVESTMENT_KNOWLEDGE` | No | empty | Reader investment knowledge level. |
| `USER_INVESTMENT_OBJECTIVE` | No | empty | Main investment objective. |
| `USER_RISK_TOLERANCE` | No | empty | Risk tolerance. |
| `USER_INFO_STYLE` | No | empty | Preferred explanation style. |
| `USER_LANGUAGE_TONE` | No | empty | Preferred tone. |
| `USER_HOBBIES` | No | empty | Personal interests that can help shape examples and wording. |

## Example `.env`

The repository does not currently include a versioned `.env.example`, so you should create a local `.env` file yourself. Example:

```dotenv
ANACONDA_BASE=/opt/anaconda3
CONDA_ENV_NAME=portfolio-engine
PROJECT_ROOT=/absolute/path/to/portfolio-optimization-engine
PORTFOLIO_REPORT_DRAFT_DIR=/absolute/path/to/portfolio-optimization-engine/draft
PORTFOLIO_REPORT_DATA_DIR=/absolute/path/to/portfolio-optimization-engine/data
OPENAI_API_KEY=your_openai_api_key

OPENAI_MODEL=gpt-5.4
RUN_SCREENING=true
RUN_HIST_PRICES=true
RUN_SELECT_TITLES=true
RUN_ALLOCATE=true
RUN_REPORT=true
PYTHON_BIN=python

INPUT_TITLES_FILENAME=list_titles_FII_stocks.xlsx
SCREENED_STOCKS_FILENAME=screened_stocks.csv
SCREENED_FIIS_FILENAME=screened_fiis.csv
HISTORICAL_PRICES_FILENAME=historical_prices.csv
SELECTED10_FILENAME=selected10titles.csv
OPTIMAL_ALLOCATION_FILENAME=optimal_allocation.csv
PORTFOLIO_RETURNS_FILENAME=portfolio_returns_selected10.csv

SCREENING_DEBUG=true
HISTORICAL_FREQ=D
HISTORICAL_BATCH_SIZE=5
HISTORICAL_PAUSE_SECONDS=4
HISTORICAL_CDI_YEARS=10
HISTORICAL_CDI_MAX_RETRIES=5
HISTORICAL_CDI_RETRY_SLEEP_SECONDS=6

SELECTION_K=10
SELECTION_MIN_LIQUIDITY=1000000
SELECTION_MAX_BETA=2.0
SELECTION_MIN_OBS=126

ALLOCATION_LOOKBACK_DAYS=756
ALLOCATION_ALPHA=0.95
ALLOCATION_MAX_WEIGHT=0.20
ALLOCATION_MIN_WEIGHT=0.00
ALLOCATION_MAX_FII_ALLOCATION=0.40

REPORT_OUTPUT_SUBDIR=report_output
REPORT_PDF_FILENAME=relatorio_carteira_recomendada.pdf
REPORT_CACHE_FILENAME=web_research_cache.json
REPORT_USE_CACHE=true
REPORT_FORCE_REFRESH=false

USER_PROFESSION=Engineer
USER_INVESTMENT_KNOWLEDGE=Intermediate
USER_INVESTMENT_OBJECTIVE=Income and long-term capital appreciation
USER_RISK_TOLERANCE=Moderate
USER_INFO_STYLE=Clear and practical
USER_LANGUAGE_TONE=Professional
USER_HOBBIES=Technology, travel, books

FORCE_REFRESH=false
```

Windows example values:

```dotenv
ANACONDA_BASE=C:/Users/USER/anaconda3
PROJECT_ROOT=C:/Users/USER/Documents/git-hub-personal/portfolio-optimization-engine
PORTFOLIO_REPORT_DRAFT_DIR=C:/Users/USER/Documents/git-hub-personal/portfolio-optimization-engine/draft
PORTFOLIO_REPORT_DATA_DIR=C:/Users/USER/Documents/git-hub-personal/portfolio-optimization-engine/data
```

## How to run the solution

### Run the full pipeline

```bash
bash run_portfolio_report.sh --config .env
```

If you omit `--config`, the script looks for `.env` in the repository root:

```bash
bash run_portfolio_report.sh
```

### Run only selected stages

Disable stages through the `.env` file:

```dotenv
RUN_SCREENING=false
RUN_HIST_PRICES=true
RUN_SELECT_TITLES=true
RUN_ALLOCATE=true
RUN_REPORT=false
```

This is useful when you already have intermediate CSV files and only want to rerun part of the workflow.

### Run the Python scripts manually

If you prefer not to use the Bash runner, you can execute the stages directly after exporting the same environment variables and setting `PYTHONPATH`.

Linux/macOS:

```bash
export PROJECT_ROOT=/absolute/path/to/portfolio-optimization-engine
export PORTFOLIO_REPORT_DATA_DIR="$PROJECT_ROOT/data"
export PORTFOLIO_REPORT_DRAFT_DIR="$PROJECT_ROOT/draft"
export OPENAI_API_KEY=your_openai_api_key
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

python src/dataprep/dataprep_screening.py
python src/dataprep/dataprep_historical_prices.py
python src/models/model_select10titles.py
python src/models/model_allocationofassets.py
python src/report/generate_portfolio_report.py
```

Windows PowerShell:

```powershell
$env:PROJECT_ROOT = "C:\Users\USER\Documents\git-hub-personal\portfolio-optimization-engine"
$env:PORTFOLIO_REPORT_DATA_DIR = "$env:PROJECT_ROOT\data"
$env:PORTFOLIO_REPORT_DRAFT_DIR = "$env:PROJECT_ROOT\draft"
$env:OPENAI_API_KEY = "your_openai_api_key"
$env:PYTHONPATH = "$env:PROJECT_ROOT\src;$env:PROJECT_ROOT"

python src\dataprep\dataprep_screening.py
python src\dataprep\dataprep_historical_prices.py
python src\models\model_select10titles.py
python src\models\model_allocationofassets.py
python src\report\generate_portfolio_report.py
```

## Generated outputs

After a successful full run, the main outputs are:

- `data/screened_stocks.csv`
- `data/screened_fiis.csv`
- `data/historical_prices.csv`
- `data/selected10titles.csv`
- `data/optimal_allocation.csv`
- `data/portfolio_returns_selected10.csv`
- `data/report_output/allocation_chart.png`
- `data/report_output/performance_chart.png`
- `data/report_output/web_research_cache.json`
- `data/report_output/relatorio_carteira_recomendada.pdf`

## Practical caveats

- The final report content is generated in Brazilian Portuguese, even though this README is in English.
- The report stage depends on the OpenAI Responses API with the `web_search` tool enabled for the selected model.
- The shell runner is Conda-oriented; if you use only `venv`, prefer manual script execution or adapt the runner.
- Upstream data quality depends on Yahoo Finance and BCB availability.
