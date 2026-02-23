import re
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
from src.models.schema import FilingChunk
from src.core.logger import get_logger

log = get_logger(__name__)

class FilingProcessor:
    """
    Parses raw SEC HTML filings into structured chunks.
    Focuses on identifying major sections (Item 1, 1A, 7, etc.).
    """
    
    # Regex patterns for 10-K sections. 
    # This is a simplified heuristic; SEC filings are notoriously messy.
    # We look for bold tags or distinct lines starting with "Item X."
    SECTION_PATTERNS = {
        "10-K": [
            r"Item\s+1\.\s+Business",
            r"Item\s+1A\.\s+Risk\s+Factors",
            r"Item\s+1B\.\s+Unresolved\s+Staff\s+Comments",
            r"Item\s+2\.\s+Properties",
            r"Item\s+3\.\s+Legal\s+Proceedings",
            r"Item\s+4\.\s+Mine\s+Safety\s+Disclosures",
            r"Item\s+5\.\s+Market",
            r"Item\s+6\.\s+Selected\s+Financial\s+Data",
            r"Item\s+7\.\s+Management", # Shortened to catch MD&A
            r"Item\s+7A\.\s+Quantitative",
            r"Item\s+8\.\s+Financial\s+Statements",
            r"Item\s+9\.\s+Changes",
            r"Item\s+9A\.\s+Controls",
            r"Item\s+9B\.\s+Other",
        ]
    }

    def process_html(self, html_content: str, ticker: str, cik: str, 
                     form_type: str, period_end_date, filing_date) -> List[FilingChunk]:
        """
        Main entry point: Cleans HTML -> Splits by Section -> Creates Chunks.
        """
        text_content = self._clean_html(html_content)
        sections = self._split_by_sections(text_content, form_type)
        
        chunks = []
        for section_name, content in sections.items():
            # Further chunk if content is too large? 
            # For now, we keep sections intact or do simple sub-chunking if needed.
            # Ideally, we want semantic chunks. Let's do a simple recursive split if > 2000 tokens later.
            # Here we just return the full section.
            
            if len(content.strip()) < 5: # Skip empty/noise sections
                continue
                
            chunk = FilingChunk(
                ticker=ticker,
                cik=cik,
                form_type=form_type,
                period_end_date=period_end_date,
                filing_date=filing_date,
                section_name=section_name,
                content=content.strip(),
                tokens=len(content.split()), # Rough estimation
                metadata={"source": "sec_edgar"}
            )
            chunks.append(chunk)
            
        log.info("Processed filing", ticker=ticker, chunks_generated=len(chunks))
        return chunks

    def _clean_html(self, html: str) -> str:
        """
        Converts HTML to clean text, removing tables/styling but preserving structure.
        """
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n\n")
        
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _split_by_sections(self, text: str, form_type: str) -> Dict[str, str]:
        """
        Splits text into a dictionary of {Section Name: Content}.
        """
        patterns = self.SECTION_PATTERNS.get(form_type, [])
        if not patterns:
            return {"FULL_TEXT": text}

        # Build a giant regex: (Item 1...)|(Item 1A...)|...
        # We capture the delimiter to know which section starts.
        combined_pattern = f"({'|'.join(patterns)})"
        
        # Split by the pattern, keeping the delimiters
        # re.split with groups returns [prefix, delimiter, content, delimiter, content...]
        parts = re.split(combined_pattern, text, flags=re.IGNORECASE)
        
        sections = {}
        current_section = "PREAMBLE"
        
        # The first part is everything before the first match
        if len(parts) > 0:
            sections[current_section] = parts[0]
            
        # Iterate over the rest: (delimiter, content) pairs
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                header = parts[i].strip()
                content = parts[i+1]
                
                # Normalize header
                # e.g. "Item 1. Business" -> "Item 1"
                # Simplified normalization
                norm_header = re.sub(r'\s+', ' ', header) 
                
                sections[norm_header] = content
                
        return sections
