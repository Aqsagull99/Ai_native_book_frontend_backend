import pytest
from unittest.mock import Mock, patch, AsyncMock
from backend.src.content_embedding.qdrant_service import create_collection, save_chunks_to_qdrant, generate_embeddings


@pytest.mark.asyncio
async def test_create_collection_success():
    """Test create_collection function successfully creates a collection."""
    with patch('backend.src.content_embedding.qdrant_service.get_qdrant_client') as mock_get_client:
        mock_client = Mock()
        mock_client.get_collection = Mock(side_effect=Exception("Collection not found"))  # Simulate collection doesn't exist
        mock_client.recreate_collection = Mock(return_value=True)
        mock_get_client.return_value = mock_client

        with patch('backend.src.content_embedding.qdrant_service.get_cohere_client') as mock_cohere:
            mock_cohere_response = Mock()
            mock_cohere_response.embeddings = [[0.1, 0.2, 0.3]]  # Mock embedding with 3 dimensions
            mock_cohere.return_value.embed.return_value = mock_cohere_response

            result = await create_collection("test_collection")

            assert result is True
            mock_client.recreate_collection.assert_called_once()
            # Verify the collection was created with the right parameters
            args, kwargs = mock_client.recreate_collection.call_args
            assert kwargs['collection_name'] == "test_collection"


@pytest.mark.asyncio
async def test_create_collection_already_exists():
    """Test create_collection function when collection already exists."""
    with patch('backend.src.content_embedding.qdrant_service.get_qdrant_client') as mock_get_client:
        mock_client = Mock()
        mock_client.get_collection = Mock(return_value=True)  # Simulate collection exists
        mock_get_client.return_value = mock_client

        result = await create_collection("existing_collection")

        assert result is True
        mock_client.recreate_collection.assert_not_called()


@pytest.mark.asyncio
async def test_save_chunks_to_qdrant_success():
    """Test save_chunks_to_qdrant function successfully saves chunks."""
    chunks = [
        {
            'content': 'Sample content 1',
            'chunk_index': 0,
            'metadata': {'url': 'https://example.com/page1', 'title': 'Page 1'}
        },
        {
            'content': 'Sample content 2',
            'chunk_index': 1,
            'metadata': {'url': 'https://example.com/page2', 'title': 'Page 2'}
        }
    ]

    with patch('backend.src.content_embedding.qdrant_service.get_cohere_client') as mock_cohere:
        mock_embedding_response = Mock()
        mock_embedding_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_cohere.return_value.embed.return_value = mock_embedding_response

        with patch('backend.src.content_embedding.qdrant_service.get_qdrant_client') as mock_get_client:
            mock_client = Mock()
            mock_client.upsert = Mock(return_value=True)
            mock_get_client.return_value = mock_client

            result = await save_chunks_to_qdrant(chunks, "test_collection")

            assert result is True
            # Should have called upsert at least once for the batch
            assert mock_client.upsert.call_count >= 1


@pytest.mark.asyncio
async def test_generate_embeddings_success():
    """Test generate_embeddings function successfully generates embeddings."""
    texts = ["Text 1", "Text 2", "Text 3"]

    with patch('backend.src.content_embedding.qdrant_service.get_cohere_client') as mock_cohere:
        mock_embedding_response = Mock()
        mock_embedding_response.embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_cohere.return_value.embed.return_value = mock_embedding_response

        result = await generate_embeddings(texts)

        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
        assert result[2] == [0.5, 0.6]


def test_generate_embeddings_function_signature():
    """Test that generate_embeddings has the correct function signature."""
    import inspect
    from backend.src.content_embedding.qdrant_service import generate_embeddings

    sig = inspect.signature(generate_embeddings)
    params = list(sig.parameters.keys())

    assert 'texts' in params


@pytest.mark.asyncio
async def test_save_chunks_to_qdrant_empty_chunks():
    """Test save_chunks_to_qdrant function with empty chunks list."""
    result = await save_chunks_to_qdrant([], "test_collection")
    # Should return True for empty list (no error)
    assert result is True


@pytest.mark.asyncio
async def test_create_collection_error_handling():
    """Test create_collection function handles errors properly."""
    from backend.src.content_embedding.utils import QdrantError

    with patch('backend.src.content_embedding.qdrant_service.get_qdrant_client') as mock_get_client:
        mock_client = Mock()
        mock_client.recreate_collection.side_effect = Exception("Connection failed")
        mock_get_client.return_value = mock_client

        with patch('backend.src.content_embedding.qdrant_service.get_cohere_client') as mock_cohere:
            mock_cohere_response = Mock()
            mock_cohere_response.embeddings = [[0.1, 0.2, 0.3]]
            mock_cohere.return_value.embed.return_value = mock_cohere_response

            with pytest.raises(QdrantError):
                await create_collection("failing_collection")


@pytest.mark.asyncio
async def test_generate_embeddings_error_handling():
    """Test generate_embeddings function handles errors properly."""
    from backend.src.content_embedding.utils import EmbeddingError

    with patch('backend.src.content_embedding.qdrant_service.get_cohere_client') as mock_cohere:
        mock_cohere.return_value.embed.side_effect = Exception("API error")

        with pytest.raises(EmbeddingError):
            await generate_embeddings(["test text"])