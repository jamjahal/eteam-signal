import json
from typing import Optional

from src.core.logger import get_logger
from src.models.schema import AlphaSignal
from src.models.insider_schema import InsiderSignal, InsiderSentiment
from src.services.llm_client import LLMClient

log = get_logger(__name__)


class CompositeSignalEngine:
    """
    Merges the existing AlphaSignal (filing sentiment drift) with the
    InsiderSignal (anomalous executive selling) into a unified recommendation.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm = llm_client or LLMClient()

    async def compose(
        self,
        ticker: str,
        filing_signal: Optional[AlphaSignal],
        insider_signal: Optional[InsiderSignal],
    ) -> InsiderSignal:
        """
        Combine filing analysis and insider trading signals.
        Returns an enriched InsiderSignal with composite_alpha_score and recommendation.
        """
        base = insider_signal or InsiderSignal(ticker=ticker)
        filing_score = filing_signal.signal_score if filing_signal else 0.0
        insider_score = base.anomaly_score

        composite = self._blend_scores(filing_score, insider_score)
        base.composite_alpha_score = composite

        recommendation = await self._generate_recommendation(
            ticker, filing_signal, base, composite
        )
        base.recommendation = recommendation
        return base

    def _blend_scores(self, filing_score: float, insider_score: float) -> float:
        """Weighted blend with conviction boost for convergent signals."""
        blended = 0.5 * filing_score + 0.5 * insider_score

        # Both signals agree and are strong -> conviction boost
        if filing_score > 0.5 and insider_score > 0.5:
            blended = min(1.0, blended * 1.2)

        return round(blended, 4)

    async def _generate_recommendation(
        self,
        ticker: str,
        filing_signal: Optional[AlphaSignal],
        insider_signal: InsiderSignal,
        composite: float,
    ) -> str:
        """Use Claude to generate a natural-language recommendation."""
        filing_summary = "No filing analysis available."
        if filing_signal:
            filing_summary = (
                f"Filing drift score: {filing_signal.signal_score:.2f} "
                f"(confidence {filing_signal.confidence:.2f}). "
                f"Summary: {filing_signal.summary}"
            )

        anomaly_descriptions = "\n".join(
            f"- [{a.anomaly_type.value}] {a.description} (severity {a.severity_score:.2f})"
            for a in insider_signal.anomalies
        ) or "No anomalies detected."

        system_prompt = (
            "You are a senior quantitative analyst. Produce a concise recommendation "
            "combining SEC filing sentiment analysis with insider trading anomaly data. "
            "Include: (1) what the insiders did, (2) what the filings say, "
            "(3) suggested position, (4) confidence and time horizon, (5) key risk caveats. "
            "Output plain text, 3-5 sentences."
        )
        user_prompt = (
            f"Ticker: {ticker}\n"
            f"Composite Alpha Score: {composite:.2f}\n"
            f"Insider Sentiment: {insider_signal.insider_sentiment.value}\n"
            f"Insider Anomaly Score: {insider_signal.anomaly_score:.2f}\n"
            f"Anomalies:\n{anomaly_descriptions}\n\n"
            f"Filing Analysis:\n{filing_summary}"
        )

        try:
            return await self.llm.generate(system_prompt, user_prompt, temperature=0.1)
        except Exception as e:
            log.error("Failed to generate recommendation", ticker=ticker, error=str(e))
            return self._fallback_recommendation(ticker, insider_signal, composite)

    @staticmethod
    def _fallback_recommendation(
        ticker: str, signal: InsiderSignal, composite: float
    ) -> str:
        if composite > 0.7:
            action = "Strong sell signal"
        elif composite > 0.4:
            action = "Elevated caution"
        else:
            action = "No immediate action"
        return (
            f"{action} for {ticker}. "
            f"Composite score: {composite:.2f}, "
            f"insider sentiment: {signal.insider_sentiment.value}, "
            f"anomalies detected: {len(signal.anomalies)}."
        )
