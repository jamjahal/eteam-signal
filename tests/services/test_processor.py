import pytest
from datetime import date
from src.services.processor import FilingProcessor

class TestFilingProcessor:
    
    def test_clean_html(self):
        processor = FilingProcessor()
        html = """
        <html>
            <style>body { color: red; }</style>
            <body>
                <p>Hello</p>
                <script>alert('xss');</script>
                <p>World</p>
            </body>
        </html>
        """
        text = processor._clean_html(html)
        assert "Hello" in text
        assert "World" in text
        assert "color: red" not in text
        assert "alert" not in text

    def test_split_by_sections_10k(self):
        processor = FilingProcessor()
        text = """
        Intro text here.
        Item 1. Business
        Business content...
        Item 1A. Risk Factors
        Risk content...
        Item 2. Properties
        Properties content...
        """
        
        sections = processor._split_by_sections(text, "10-K")
        
        assert "Item 1 Business" in sections or "Item 1. Business" in sections or "Item 1" in sections # Depending on normalization
        # Based on implementation: norm_header = re.sub(r'\s+', ' ', header)
        # "Item 1. Business" -> "Item 1. Business"
        
        assert "Item 1A. Risk Factors" in sections
        assert sections["Item 1A. Risk Factors"].strip() == "Risk content..."
        
    def test_process_html_integration(self):
        processor = FilingProcessor()
        html = "<html><body>Item 1A. Risk Factors<br>Dangerous things.</body></html>"
        
        chunks = processor.process_html(
            html_content=html,
            ticker="TEST",
            cik="000",
            form_type="10-K",
            period_end_date=date.today(),
            filing_date=date.today()
        )
        
        assert len(chunks) > 0
        risk_chunk = next((c for c in chunks if "Risk Factors" in c.section_name), None)
        assert risk_chunk is not None
        assert "Dangerous things" in risk_chunk.content
