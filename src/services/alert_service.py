from typing import List, Optional

from src.core.config import settings
from src.core.logger import get_logger
from src.models.insider_schema import InsiderSignal, InsiderSentiment
from src.services.insider_store import InsiderStore

log = get_logger(__name__)


class AlertService:
    """Evaluates insider signals and generates actionable alerts."""

    def __init__(self, store: InsiderStore) -> None:
        self.store = store
        self.threshold = settings.INSIDER_ANOMALY_THRESHOLD

    async def evaluate(self, signals: List[InsiderSignal]) -> List[InsiderSignal]:
        """Filter signals to actionable alerts and persist them."""
        actionable: List[InsiderSignal] = []
        for sig in signals:
            if sig.anomaly_score < self.threshold:
                continue

            await self.store.save_alert(
                ticker=sig.ticker,
                anomaly_score=sig.anomaly_score,
                insider_sentiment=sig.insider_sentiment.value,
                recommendation=sig.recommendation,
                composite_alpha_score=sig.composite_alpha_score,
            )
            actionable.append(sig)

        log.info(
            "Alert evaluation complete",
            total=len(signals),
            actionable=len(actionable),
            threshold=self.threshold,
        )
        return actionable

    async def get_active_alerts(self, limit: int = 50) -> list:
        """Retrieve recent undelivered alerts."""
        return await self.store.get_alerts(delivered=False, limit=limit)
