from typing import List, Optional
from datetime import date, timedelta
import math

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

from src.core.config import settings
from src.core.logger import get_logger
from src.models.insider_schema import (
    AnomalyType,
    InsiderAnomaly,
    InsiderProfile,
    InsiderSignal,
    InsiderSentiment,
    InsiderTransaction,
    TransactionCode,
)
from src.services.insider_store import InsiderStore

log = get_logger(__name__)

# Tier 1 thresholds
VOLUME_Z_THRESHOLD = 2.0
FREQUENCY_RATIO_THRESHOLD = 0.25
CLUSTER_SELLER_THRESHOLD = 3
HOLDINGS_PCT_THRESHOLD = 0.20

# Role weights for signal scoring
ROLE_WEIGHTS = {
    "ceo": 1.5,
    "cfo": 1.5,
    "officer": 1.2,
    "director": 1.0,
}

PLANNED_TRADE_DISCOUNT = 0.5


class InsiderAnalyzer:
    """
    Two-tier anomaly detection engine for insider trading patterns.

    Tier 1: Statistical rules (z-scores, cluster counting, thresholds).
    Tier 2: Isolation Forest ML model on feature vectors.
    """

    def __init__(self, store: InsiderStore) -> None:
        self.store = store

    async def analyze_ticker(self, ticker: str) -> InsiderSignal:
        """Run full anomaly analysis for a single ticker."""
        txns = await self.store.get_transactions(ticker, days_back=settings.INSIDER_LOOKBACK_DAYS)
        if not txns:
            return InsiderSignal(ticker=ticker)

        insiders = {tx.insider_name for tx in txns}
        all_anomalies: List[InsiderAnomaly] = []

        for name in insiders:
            profile = await self.store.get_profile(ticker, name)
            if profile is None:
                continue
            person_txns = [t for t in txns if t.insider_name == name]
            anomalies = self._tier1_detect(person_txns, profile, ticker)
            all_anomalies.extend(anomalies)

        # Cluster selling (cross-insider)
        cluster_anomaly = await self._detect_cluster_selling(ticker)
        if cluster_anomaly:
            all_anomalies.append(cluster_anomaly)

        # Tier 2: ML enrichment
        ml_score = self._tier2_score(txns)

        # Composite signal scoring
        anomaly_score = self._compute_anomaly_score(all_anomalies, ml_score, txns)
        sentiment = self._derive_sentiment(anomaly_score, txns)

        signal = InsiderSignal(
            ticker=ticker,
            anomaly_score=anomaly_score,
            anomalies=all_anomalies,
            insider_sentiment=sentiment,
        )

        for a in all_anomalies:
            await self.store.save_anomaly(a)

        return signal

    # ------------------------------------------------------------------
    # Tier 1: Statistical anomaly detection
    # ------------------------------------------------------------------

    def _tier1_detect(
        self,
        txns: List[InsiderTransaction],
        profile: InsiderProfile,
        ticker: str,
    ) -> List[InsiderAnomaly]:
        anomalies: List[InsiderAnomaly] = []
        if not txns:
            return anomalies

        latest = txns[0]
        name = latest.insider_name

        # Volume anomaly
        sizes = [t.shares * (t.price_per_share or 0) for t in txns if t.price_per_share]
        if len(sizes) >= 3:
            latest_size = latest.shares * (latest.price_per_share or 0)
            mean = np.mean(sizes)
            std = np.std(sizes, ddof=1)
            if std > 0:
                z = (latest_size - mean) / std
                if abs(z) > VOLUME_Z_THRESHOLD:
                    anomalies.append(
                        InsiderAnomaly(
                            ticker=ticker,
                            insider_name=name,
                            anomaly_type=AnomalyType.VOLUME,
                            severity_score=min(1.0, abs(z) / 5.0),
                            z_score=float(z),
                            description=f"Transaction size z-score={z:.2f} vs historical mean",
                            transactions=[latest],
                        )
                    )

        # Frequency anomaly
        if profile.avg_frequency_days > 0 and len(txns) >= 2:
            days_since = (date.today() - txns[0].transaction_date).days
            if days_since > 0:
                ratio = days_since / profile.avg_frequency_days
                if ratio < FREQUENCY_RATIO_THRESHOLD:
                    anomalies.append(
                        InsiderAnomaly(
                            ticker=ticker,
                            insider_name=name,
                            anomaly_type=AnomalyType.FREQUENCY,
                            severity_score=min(1.0, 1.0 - ratio),
                            z_score=0.0,
                            description=(
                                f"Traded {days_since}d after previous vs avg {profile.avg_frequency_days:.0f}d"
                            ),
                            transactions=[txns[0], txns[1]],
                        )
                    )

        # Holdings percentage anomaly
        if latest.transaction_code == TransactionCode.SALE and latest.shares_owned_after:
            total_before = latest.shares + latest.shares_owned_after
            if total_before > 0:
                pct_sold = latest.shares / total_before
                if pct_sold > HOLDINGS_PCT_THRESHOLD:
                    anomalies.append(
                        InsiderAnomaly(
                            ticker=ticker,
                            insider_name=name,
                            anomaly_type=AnomalyType.HOLDINGS_PERCENTAGE,
                            severity_score=min(1.0, pct_sold),
                            z_score=0.0,
                            description=f"Sold {pct_sold:.1%} of holdings in single transaction",
                            transactions=[latest],
                        )
                    )

        return anomalies

    async def _detect_cluster_selling(self, ticker: str) -> Optional[InsiderAnomaly]:
        sellers = await self.store.get_recent_sellers(
            ticker, window_days=settings.INSIDER_CLUSTER_WINDOW_DAYS
        )
        if len(sellers) >= CLUSTER_SELLER_THRESHOLD:
            return InsiderAnomaly(
                ticker=ticker,
                insider_name="MULTIPLE",
                anomaly_type=AnomalyType.CLUSTER,
                severity_score=min(1.0, len(sellers) / 6.0),
                z_score=0.0,
                description=f"{len(sellers)} insiders sold within {settings.INSIDER_CLUSTER_WINDOW_DAYS}d window",
            )
        return None

    # ------------------------------------------------------------------
    # Tier 2: Isolation Forest ML
    # ------------------------------------------------------------------

    def _tier2_score(self, txns: List[InsiderTransaction]) -> float:
        """Train an Isolation Forest on historical features; return anomaly score for the latest."""
        if len(txns) < 10:
            return 0.0

        features = self._build_feature_matrix(txns)
        if features.shape[0] < 5:
            return 0.0

        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
        )
        model.fit(features)

        latest_features = features[-1:].reshape(1, -1) if features.ndim > 1 else features[-1:].reshape(1, -1)
        raw = model.decision_function(latest_features)[0]
        # decision_function returns negative for anomalies; map to 0..1
        return float(np.clip(1.0 - (raw + 0.5), 0.0, 1.0))

    def _build_feature_matrix(self, txns: List[InsiderTransaction]) -> np.ndarray:
        rows = []
        sorted_txns = sorted(txns, key=lambda t: t.transaction_date)
        for i, tx in enumerate(sorted_txns):
            size = tx.shares * (tx.price_per_share or 0)
            days_since = (
                (tx.transaction_date - sorted_txns[i - 1].transaction_date).days
                if i > 0
                else 0
            )
            pct_sold = 0.0
            if tx.shares_owned_after and tx.transaction_code == TransactionCode.SALE:
                total = tx.shares + tx.shares_owned_after
                pct_sold = tx.shares / total if total > 0 else 0.0
            is_csuite = 1.0 if tx.is_officer else 0.0
            rows.append([size, days_since, pct_sold, is_csuite])
        return np.array(rows, dtype=np.float64)

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def _compute_anomaly_score(
        self,
        anomalies: List[InsiderAnomaly],
        ml_score: float,
        txns: List[InsiderTransaction],
    ) -> float:
        if not anomalies and ml_score == 0.0:
            return 0.0

        tier1_max = max((a.severity_score for a in anomalies), default=0.0)
        type_count = len({a.anomaly_type for a in anomalies})
        co_occurrence_boost = min(0.2, type_count * 0.05) if type_count > 1 else 0.0

        base = 0.6 * tier1_max + 0.4 * ml_score + co_occurrence_boost

        # Role weighting (use highest-role transaction)
        role_weight = 1.0
        for tx in txns[:5]:
            title_lower = tx.insider_title.lower()
            if "ceo" in title_lower or "chief executive" in title_lower:
                role_weight = max(role_weight, ROLE_WEIGHTS["ceo"])
            elif "cfo" in title_lower or "chief financial" in title_lower:
                role_weight = max(role_weight, ROLE_WEIGHTS["cfo"])
            elif tx.is_officer:
                role_weight = max(role_weight, ROLE_WEIGHTS["officer"])

        # 10b5-1 discount
        planned_ratio = sum(1 for tx in txns[:10] if tx.is_10b5_1) / max(len(txns[:10]), 1)
        planned_discount = 1.0 - (planned_ratio * (1.0 - PLANNED_TRADE_DISCOUNT))

        return float(np.clip(base * role_weight * planned_discount, 0.0, 1.0))

    @staticmethod
    def _derive_sentiment(
        anomaly_score: float, txns: List[InsiderTransaction]
    ) -> InsiderSentiment:
        sells = sum(1 for t in txns if t.transaction_code == TransactionCode.SALE)
        buys = sum(1 for t in txns if t.transaction_code == TransactionCode.PURCHASE)
        if anomaly_score > 0.6 and sells > buys:
            return InsiderSentiment.BEARISH
        if buys > sells:
            return InsiderSentiment.BULLISH
        return InsiderSentiment.NEUTRAL
