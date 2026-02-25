-- 001_insider_schema.sql
-- Idempotent migration for insider trading anomaly detection tables.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- insider_transactions  (hypertable, partitioned by transaction_date)
-- ============================================================
CREATE TABLE IF NOT EXISTS insider_transactions (
    id              BIGSERIAL,
    ticker          TEXT        NOT NULL,
    insider_name    TEXT        NOT NULL,
    insider_title   TEXT        NOT NULL DEFAULT '',
    is_officer      BOOLEAN     NOT NULL DEFAULT FALSE,
    is_director     BOOLEAN     NOT NULL DEFAULT FALSE,
    transaction_date DATE       NOT NULL,
    transaction_code CHAR(1)    NOT NULL,
    shares          DOUBLE PRECISION NOT NULL,
    price_per_share DOUBLE PRECISION,
    total_value     DOUBLE PRECISION,
    shares_owned_after DOUBLE PRECISION,
    is_10b5_1       BOOLEAN     NOT NULL DEFAULT FALSE,
    filing_date     DATE        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (ticker, insider_name, transaction_date, shares, transaction_code)
);

SELECT create_hypertable(
    'insider_transactions',
    'transaction_date',
    if_not_exists => TRUE,
    migrate_data  => TRUE
);

CREATE INDEX IF NOT EXISTS idx_insider_tx_ticker
    ON insider_transactions (ticker, transaction_date DESC);

CREATE INDEX IF NOT EXISTS idx_insider_tx_name
    ON insider_transactions (insider_name, transaction_date DESC);

CREATE INDEX IF NOT EXISTS idx_insider_tx_filing
    ON insider_transactions (filing_date DESC);

-- ============================================================
-- insider_profiles_daily  (continuous aggregate)
-- ============================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.continuous_aggregates
        WHERE view_name = 'insider_profiles_daily'
    ) THEN
        EXECUTE $agg$
            CREATE MATERIALIZED VIEW insider_profiles_daily
            WITH (timescaledb.continuous) AS
            SELECT
                ticker,
                insider_name,
                time_bucket('1 day', transaction_date) AS bucket,
                COUNT(*)                                AS trade_count,
                AVG(shares * COALESCE(price_per_share, 0))  AS avg_transaction_size,
                AVG(shares)                             AS avg_shares,
                SUM(CASE WHEN transaction_code = 'S' THEN shares ELSE 0 END)
                    / NULLIF(MAX(shares_owned_after), 0) AS pct_holdings_sold
            FROM insider_transactions
            GROUP BY ticker, insider_name, bucket
        $agg$;
    END IF;
END $$;

-- ============================================================
-- insider_anomalies
-- ============================================================
CREATE TABLE IF NOT EXISTS insider_anomalies (
    id              BIGSERIAL   PRIMARY KEY,
    ticker          TEXT        NOT NULL,
    insider_name    TEXT        NOT NULL,
    anomaly_type    TEXT        NOT NULL,
    severity_score  DOUBLE PRECISION NOT NULL,
    z_score         DOUBLE PRECISION NOT NULL DEFAULT 0,
    description     TEXT        NOT NULL DEFAULT '',
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_anomalies_ticker
    ON insider_anomalies (ticker, detected_at DESC);

-- ============================================================
-- insider_alerts
-- ============================================================
CREATE TABLE IF NOT EXISTS insider_alerts (
    id                  BIGSERIAL   PRIMARY KEY,
    ticker              TEXT        NOT NULL,
    anomaly_score       DOUBLE PRECISION NOT NULL,
    insider_sentiment   TEXT        NOT NULL DEFAULT 'NEUTRAL',
    recommendation      TEXT        NOT NULL DEFAULT '',
    composite_alpha_score DOUBLE PRECISION,
    delivered           BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_ticker
    ON insider_alerts (ticker, created_at DESC);

-- ============================================================
-- monitor_watermarks  (tracks ATOM feed polling position)
-- ============================================================
CREATE TABLE IF NOT EXISTS monitor_watermarks (
    feed_name               TEXT PRIMARY KEY,
    last_seen_accession     TEXT NOT NULL DEFAULT '',
    last_poll_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
