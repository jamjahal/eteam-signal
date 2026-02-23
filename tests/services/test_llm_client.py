import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.services.llm_client import LLMClient
from src.core.config import settings

@pytest.mark.asyncio
async def test_generate():
    # Mock settings
    settings.ANTHROPIC_API_KEY = "test-key"
    
    with patch("src.services.llm_client.anthropic.AsyncAnthropic") as mock_anthropic:
        mock_client_instance = Mock()
        mock_anthropic.return_value = mock_client_instance
        
        mock_message = Mock()
        mock_message.content = [Mock(text="Generated text")]
        
        # Async mock for messages.create
        mock_client_instance.messages.create = AsyncMock(return_value=mock_message)
        
        client = LLMClient()
        response = await client.generate("system", "user")
        
        assert response == "Generated text"
        mock_client_instance.messages.create.assert_called_once()
