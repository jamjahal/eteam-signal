import pytest
from unittest.mock import Mock, patch, ANY
from src.services.vector_store import VectorStore
from src.models.schema import FilingChunk
from datetime import date

@patch("src.services.vector_store.QdrantClient")
class TestVectorStore:
    
    def test_init_creates_collection(self, mock_client_cls):
        mock_client = Mock()
        mock_client_cls.return_value = mock_client
        
        # Simulate collection missing, then creating
        mock_client.get_collection.side_effect = Exception("Not found")
        
        store = VectorStore()
        
        mock_client.create_collection.assert_called_once_with(
            collection_name=ANY,
            vectors_config=ANY
        )

    def test_upsert_chunks(self, mock_client_cls):
        mock_client = Mock()
        mock_client_cls.return_value = mock_client
        store = VectorStore()
        
        chunk = FilingChunk(
            ticker="A", cik="1", form_type="10-K", 
            period_end_date=date.today(), filing_date=date.today(), 
            content="abc"
        )
        embeddings = [[0.1, 0.2, 0.3]]
        
        store.upsert_chunks([chunk], embeddings)
        
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args.kwargs['collection_name'] == store.collection_name
        assert len(call_args.kwargs['points']) == 1

    def test_search(self, mock_client_cls):
        mock_client = Mock()
        mock_client_cls.return_value = mock_client
        store = VectorStore()

        mock_point = Mock()
        mock_point.score = 0.9
        mock_point.payload = {"ticker": "A"}
        mock_response = Mock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response

        results = store.search(query_vector=[0.1, 0.1])

        mock_client.query_points.assert_called_once_with(
            collection_name=store.collection_name,
            query=[0.1, 0.1],
            limit=10,
            with_payload=True,
        )
        assert results == [mock_point]
