import pytest
from unittest.mock import Mock, patch
from src.services.retriever import Retriever
from src.models.schema import FilingChunk
from datetime import date

class TestRetriever:
    
    @patch("src.services.retriever.VectorStore")
    @patch("src.services.retriever.SentenceTransformer")
    def test_index_filing(self, mock_encoder_cls, mock_store_cls):
        mock_encoder = Mock()
        mock_encoder_cls.return_value = mock_encoder
        mock_store = Mock()
        mock_store_cls.return_value = mock_store
        
        mock_encoder.encode.return_value.tolist.return_value = [[0.1, 0.2]]
        
        retriever = Retriever()
        chunk = FilingChunk(
            ticker="A", cik="1", form_type="10-K", 
            period_end_date=date.today(), filing_date=date.today(), 
            content="abc"
        )
        
        retriever.index_filing([chunk])
        
        mock_encoder.encode.assert_called()
        mock_store.upsert_chunks.assert_called_once()

    @patch("src.services.retriever.VectorStore")
    @patch("src.services.retriever.SentenceTransformer")
    def test_search(self, mock_encoder_cls, mock_store_cls):
        mock_encoder = Mock()
        mock_encoder_cls.return_value = mock_encoder
        mock_store = Mock()
        mock_store_cls.return_value = mock_store
        
        mock_encoder.encode.return_value.tolist.return_value = [[0.1, 0.2]]
        
        # Mock Qdrant point return
        mock_point = Mock()
        mock_point.score = 0.9
        # Payload must match FilingChunk structure
        mock_point.payload = {
            "id": "123",
            "ticker": "A", 
            "cik": "1", 
            "form_type": "10-K", 
            "period_end_date": "2023-01-01", 
            "filing_date": "2023-01-01", 
            "content": "abc",
            "tokens": 3,
            "metadata": {}
        }
        mock_store.search.return_value = [mock_point]
        
        retriever = Retriever()
        results = retriever.search("query")
        
        assert len(results) == 1
        assert results[0].score == 0.9
        assert results[0].chunk.ticker == "A"
