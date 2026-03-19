from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class PortfolioPaths:
    project_root: Path
    draft_dir: Path
    data_dir: Path
    input_titles_path: Path
    screened_stocks_path: Path
    screened_fiis_path: Path
    historical_prices_path: Path
    selected10_path: Path
    optimal_allocation_path: Path
    portfolio_returns_path: Path


@dataclass(frozen=True)
class PortfolioConfig:
    paths: PortfolioPaths
    screening_debug: bool
    historical_freq: str
    historical_batch_size: int
    historical_pause_seconds: float
    historical_cdi_years: int
    selection_k: int
    selection_min_liquidity: float
    selection_max_beta: float
    selection_min_obs: int
    allocation_lookback_days: int
    allocation_alpha: float
    allocation_max_weight: float
    allocation_min_weight: float
    allocation_max_fii_allocation: float
    historical_cdi_max_retries: int
    historical_cdi_retry_sleep_seconds: int



def load_config() -> PortfolioConfig:
    default_root = Path.cwd()
    project_root = Path(os.getenv("PROJECT_ROOT", str(default_root))).expanduser()
    draft_dir = Path(os.getenv("PORTFOLIO_REPORT_DRAFT_DIR", str(project_root / "draft"))).expanduser()
    data_dir = Path(os.getenv("PORTFOLIO_REPORT_DATA_DIR", str(project_root / "data"))).expanduser()

    data_dir.mkdir(parents=True, exist_ok=True)

    paths = PortfolioPaths(
        project_root=project_root,
        draft_dir=draft_dir,
        data_dir=data_dir,
        input_titles_path=data_dir / os.getenv("INPUT_TITLES_FILENAME", "list_titles_FII_stocks.xlsx"),
        screened_stocks_path=data_dir / os.getenv("SCREENED_STOCKS_FILENAME", "screened_stocks.csv"),
        screened_fiis_path=data_dir / os.getenv("SCREENED_FIIS_FILENAME", "screened_fiis.csv"),
        historical_prices_path=data_dir / os.getenv("HISTORICAL_PRICES_FILENAME", "historical_prices.csv"),
        selected10_path=data_dir / os.getenv("SELECTED10_FILENAME", "selected10titles.csv"),
        optimal_allocation_path=data_dir / os.getenv("OPTIMAL_ALLOCATION_FILENAME", "optimal_allocation.csv"),
        portfolio_returns_path=data_dir / os.getenv("PORTFOLIO_RETURNS_FILENAME", "portfolio_returns_selected10.csv"),
    )

    return PortfolioConfig(
        paths=paths,
        screening_debug=_env_bool("SCREENING_DEBUG", True),
        historical_freq=os.getenv("HISTORICAL_FREQ", "D"),
        historical_batch_size=int(os.getenv("HISTORICAL_BATCH_SIZE", "5")),
        historical_pause_seconds=float(os.getenv("HISTORICAL_PAUSE_SECONDS", "4")),
        historical_cdi_years=int(os.getenv("HISTORICAL_CDI_YEARS", "10")),
        selection_k=int(os.getenv("SELECTION_K", "10")),
        selection_min_liquidity=float(os.getenv("SELECTION_MIN_LIQUIDITY", "1000000")),
        selection_max_beta=float(os.getenv("SELECTION_MAX_BETA", "2.0")),
        selection_min_obs=int(os.getenv("SELECTION_MIN_OBS", "126")),
        allocation_lookback_days=int(os.getenv("ALLOCATION_LOOKBACK_DAYS", str(252 * 3))),
        allocation_alpha=float(os.getenv("ALLOCATION_ALPHA", "0.95")),
        allocation_max_weight=float(os.getenv("ALLOCATION_MAX_WEIGHT", "0.20")),
        allocation_min_weight=float(os.getenv("ALLOCATION_MIN_WEIGHT", "0.00")),
        allocation_max_fii_allocation=float(os.getenv("ALLOCATION_MAX_FII_ALLOCATION", "0.40")),
        historical_cdi_max_retries=int(os.getenv("HISTORICAL_CDI_MAX_RETRIES", "5")),
        historical_cdi_retry_sleep_seconds=int(os.getenv("HISTORICAL_CDI_RETRY_SLEEP_SECONDS", "6")),

    )
