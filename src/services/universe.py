import csv
import os
from typing import List

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

UNIVERSE_PATH = os.path.join(settings.DATA_DIR, "sp500_universe.csv")


def load_universe() -> List[str]:
    """Load ticker symbols from the universe CSV. Returns empty list if file is missing."""
    if not os.path.exists(UNIVERSE_PATH):
        log.warning("Universe file not found", path=UNIVERSE_PATH)
        return []

    tickers: List[str] = []
    with open(UNIVERSE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip().upper()
            if ticker:
                tickers.append(ticker)

    log.info("Loaded universe", count=len(tickers))
    return tickers


def save_universe(rows: List[dict]) -> int:
    """Write universe rows to CSV. Each dict should have ticker, company_name, sector, sub_industry."""
    os.makedirs(os.path.dirname(UNIVERSE_PATH), exist_ok=True)
    fieldnames = ["ticker", "company_name", "sector", "sub_industry"]
    with open(UNIVERSE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    log.info("Saved universe", count=len(rows), path=UNIVERSE_PATH)
    return len(rows)
