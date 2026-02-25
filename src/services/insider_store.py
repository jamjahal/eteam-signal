from typing import List, Optional
from datetime import date, datetime, timedelta

import asyncpg

from src.core.config import settings
from src.core.logger import get_logger
from src.models.insider_schema import (
    InsiderTransaction,
    InsiderProfile,
    InsiderAnomaly,
    AnomalyType,
)

log = get_logger(__name__)


class InsiderStore:
    """Async PostgreSQL/TimescaleDB store for insider transaction data."""

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            min_size=settings.POSTGRES_POOL_MIN,
            max_size=settings.POSTGRES_POOL_MAX,
        )
        log.info("InsiderStore connected", host=settings.POSTGRES_HOST)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            log.info("InsiderStore connection pool closed")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("InsiderStore is not connected. Call connect() first.")
        return self._pool

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    async def upsert_transaction(self, tx: InsiderTransaction) -> bool:
        """Insert a transaction, returning True if a new row was created."""
        query = """
            INSERT INTO insider_transactions (
                ticker, insider_name, insider_title, is_officer, is_director,
                transaction_date, transaction_code, shares, price_per_share,
                total_value, shares_owned_after, is_10b5_1, filing_date
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (ticker, insider_name, transaction_date, shares, transaction_code)
            DO NOTHING
            RETURNING id
        """
        row = await self.pool.fetchrow(
            query,
            tx.ticker,
            tx.insider_name,
            tx.insider_title,
            tx.is_officer,
            tx.is_director,
            tx.transaction_date,
            tx.transaction_code.value,
            tx.shares,
            tx.price_per_share,
            tx.total_value,
            tx.shares_owned_after,
            tx.is_10b5_1,
            tx.filing_date,
        )
        return row is not None

    async def upsert_transactions(self, txns: List[InsiderTransaction]) -> int:
        """Bulk-insert transactions. Returns count of newly inserted rows."""
        inserted = 0
        for tx in txns:
            if await self.upsert_transaction(tx):
                inserted += 1
        return inserted

    async def get_transactions(
        self,
        ticker: str,
        days_back: int = 730,
        insider_name: Optional[str] = None,
    ) -> List[InsiderTransaction]:
        """Fetch recent transactions for a ticker, optionally filtered by insider."""
        cutoff = date.today() - timedelta(days=days_back)
        if insider_name:
            query = """
                SELECT * FROM insider_transactions
                WHERE ticker = $1 AND insider_name = $2 AND transaction_date >= $3
                ORDER BY transaction_date DESC
            """
            rows = await self.pool.fetch(query, ticker, insider_name, cutoff)
        else:
            query = """
                SELECT * FROM insider_transactions
                WHERE ticker = $1 AND transaction_date >= $2
                ORDER BY transaction_date DESC
            """
            rows = await self.pool.fetch(query, ticker, cutoff)
        return [self._row_to_transaction(r) for r in rows]

    async def get_recent_sellers(
        self, ticker: str, window_days: int = 14
    ) -> List[str]:
        """Return distinct insider names who sold within the given window."""
        cutoff = date.today() - timedelta(days=window_days)
        query = """
            SELECT DISTINCT insider_name FROM insider_transactions
            WHERE ticker = $1 AND transaction_code = 'S' AND transaction_date >= $2
        """
        rows = await self.pool.fetch(query, ticker, cutoff)
        return [r["insider_name"] for r in rows]

    # ------------------------------------------------------------------
    # Profiles (from continuous aggregate)
    # ------------------------------------------------------------------

    async def get_profile(self, ticker: str, insider_name: str) -> Optional[InsiderProfile]:
        """Build a profile from the continuous aggregate."""
        query = """
            SELECT
                insider_name,
                ticker,
                SUM(trade_count)         AS total_transactions,
                AVG(avg_transaction_size) AS avg_transaction_size,
                AVG(avg_shares)           AS avg_shares,
                AVG(pct_holdings_sold)    AS typical_sell_percentage
            FROM insider_profiles_daily
            WHERE ticker = $1 AND insider_name = $2
            GROUP BY ticker, insider_name
        """
        row = await self.pool.fetchrow(query, ticker, insider_name)
        if not row:
            return None

        last_tx = await self.pool.fetchval(
            "SELECT MAX(transaction_date) FROM insider_transactions WHERE ticker=$1 AND insider_name=$2",
            ticker,
            insider_name,
        )

        first_tx = await self.pool.fetchval(
            "SELECT MIN(transaction_date) FROM insider_transactions WHERE ticker=$1 AND insider_name=$2",
            ticker,
            insider_name,
        )

        total = int(row["total_transactions"])
        span_days = (last_tx - first_tx).days if first_tx and last_tx and total > 1 else 0
        avg_freq = span_days / (total - 1) if total > 1 else 0.0

        return InsiderProfile(
            insider_name=row["insider_name"],
            ticker=row["ticker"],
            avg_transaction_size=float(row["avg_transaction_size"] or 0),
            avg_frequency_days=avg_freq,
            total_transactions=total,
            typical_sell_percentage=float(row["typical_sell_percentage"] or 0),
            last_transaction_date=last_tx,
        )

    # ------------------------------------------------------------------
    # Anomalies
    # ------------------------------------------------------------------

    async def save_anomaly(self, anomaly: InsiderAnomaly) -> int:
        query = """
            INSERT INTO insider_anomalies (ticker, insider_name, anomaly_type, severity_score, z_score, description)
            VALUES ($1,$2,$3,$4,$5,$6)
            RETURNING id
        """
        return await self.pool.fetchval(
            query,
            anomaly.ticker,
            anomaly.insider_name,
            anomaly.anomaly_type.value,
            anomaly.severity_score,
            anomaly.z_score,
            anomaly.description,
        )

    async def get_anomalies(
        self, ticker: Optional[str] = None, min_score: float = 0.0, limit: int = 100
    ) -> List[InsiderAnomaly]:
        if ticker:
            query = """
                SELECT * FROM insider_anomalies
                WHERE ticker = $1 AND severity_score >= $2
                ORDER BY detected_at DESC LIMIT $3
            """
            rows = await self.pool.fetch(query, ticker, min_score, limit)
        else:
            query = """
                SELECT * FROM insider_anomalies
                WHERE severity_score >= $1
                ORDER BY detected_at DESC LIMIT $2
            """
            rows = await self.pool.fetch(query, min_score, limit)
        return [
            InsiderAnomaly(
                ticker=r["ticker"],
                insider_name=r["insider_name"],
                anomaly_type=AnomalyType(r["anomaly_type"]),
                severity_score=r["severity_score"],
                z_score=r["z_score"],
                description=r["description"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    async def save_alert(
        self,
        ticker: str,
        anomaly_score: float,
        insider_sentiment: str,
        recommendation: str,
        composite_alpha_score: Optional[float] = None,
    ) -> int:
        query = """
            INSERT INTO insider_alerts
                (ticker, anomaly_score, insider_sentiment, recommendation, composite_alpha_score)
            VALUES ($1,$2,$3,$4,$5)
            RETURNING id
        """
        return await self.pool.fetchval(
            query, ticker, anomaly_score, insider_sentiment, recommendation, composite_alpha_score
        )

    async def get_alerts(self, delivered: Optional[bool] = None, limit: int = 50) -> list:
        if delivered is not None:
            query = "SELECT * FROM insider_alerts WHERE delivered=$1 ORDER BY created_at DESC LIMIT $2"
            return await self.pool.fetch(query, delivered, limit)
        query = "SELECT * FROM insider_alerts ORDER BY created_at DESC LIMIT $1"
        return await self.pool.fetch(query, limit)

    # ------------------------------------------------------------------
    # Watermarks (for ATOM feed poller)
    # ------------------------------------------------------------------

    async def get_watermark(self, feed_name: str) -> Optional[str]:
        row = await self.pool.fetchrow(
            "SELECT last_seen_accession FROM monitor_watermarks WHERE feed_name=$1",
            feed_name,
        )
        return row["last_seen_accession"] if row else None

    async def set_watermark(self, feed_name: str, accession: str) -> None:
        await self.pool.execute(
            """
            INSERT INTO monitor_watermarks (feed_name, last_seen_accession, last_poll_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (feed_name) DO UPDATE
                SET last_seen_accession = $2, last_poll_at = NOW()
            """,
            feed_name,
            accession,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_transaction(row: asyncpg.Record) -> InsiderTransaction:
        from src.models.insider_schema import TransactionCode

        code_map = {v.value: v for v in TransactionCode}
        raw_code = row["transaction_code"].strip()
        tx_code = code_map.get(raw_code, TransactionCode.OTHER)

        return InsiderTransaction(
            ticker=row["ticker"],
            insider_name=row["insider_name"],
            insider_title=row["insider_title"],
            is_officer=row["is_officer"],
            is_director=row["is_director"],
            transaction_date=row["transaction_date"],
            transaction_code=tx_code,
            shares=row["shares"],
            price_per_share=row["price_per_share"],
            total_value=row["total_value"],
            shares_owned_after=row["shares_owned_after"],
            is_10b5_1=row["is_10b5_1"],
            filing_date=row["filing_date"],
        )
