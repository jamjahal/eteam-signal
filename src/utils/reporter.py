from src.models.schema import AlphaSignal

def generate_markdown_report(signal: AlphaSignal) -> str:
    """
    Convert an AlphaSignal into a readable Markdown report.
    """
    md = f"""
# Alpha Sentinel Report: {signal.ticker}

**Date:** {signal.analysis_date.strftime('%Y-%m-%d')}
**Signal Score:** {signal.signal_score:.2f} / 1.0
**Confidence:** {signal.confidence:.2f} / 1.0

## Executive Summary
{signal.summary}

## Identified Risk Factors
"""
    for risk in signal.risk_factors:
        md += f"- {risk}\n"
        
    md += "\n## Key Evidence (Quotes)\n"
    for quote in signal.key_quotes:
        md += f"> {quote}\n\n"
        
    if signal.critic_notes:
        md += f"\n## Critic's Notes\n_{signal.critic_notes}_"
        
    return md
