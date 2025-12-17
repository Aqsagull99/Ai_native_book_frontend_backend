import pytest
from unittest.mock import Mock, patch, AsyncMock
from backend.src.content_embedding.qdrant_service import create_collection, save_chunks_to_qdrant


@pytest.mark.asyncio
async def test_qdrant_storage_integration():
    """Integration test for Qdrant storage functionality."""
    collection_name = "test_integration_collection"

    # Mock the Qdrant client creation and operations
    with patch('backend.src.content_embedding.qdrant_service.get_qdrant_client') as mock_get_client:
        mock_client = Mock()
        # Simulate that the collection doesn't exist initially
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.recreate_collection.return_value = True
        mock_client.upsert.return_value = True
        mock_get_client.return_value = mock_client

        # Mock the Cohere client for embedding generation
        with patch('backend.src.content_embedding.qdrant_service.get_cohere_client') as mock_cohere:
            mock_embedding_response = Mock()
            # Return a simple embedding for testing
            mock_embedding_response.embeddings = [[0.1, 0.2, 0.3, 0.4]]
            mock_cohere.return_value.embed.return_value = mock_embedding_response

            # Test collection creation
            result = await create_collection(collection_name)
            assert result is True
            mock_client.recreate_collection.assert_called_once()

            # Test saving chunks
            test_chunks = [
                {
                    'content': 'Test content for integration',
                    'chunk_index': 0,
                    'metadata': {
                        'url': 'https://example.com/test',
                        'title': 'Test Page',
                        'section_hierarchy': [
                            {'level': 1, 'text': 'Main Section'}
                        ]
                    }
                }
            ]

            result = await save_chunks_to_qdrant(test_chunks, collection_name)
            assert result is True
            mock_client.upsert.assert_called_once()

            # Verify the upsert was called with the right parameters
            args, kwargs = mock_client.upsert.call_args
            assert 'collection_name' in kwargs
            assert kwargs['collection_name'] == collection_name
            assert 'points' in kwargs
            assert len(kwargs['points']) == 1


@pytest.mark.asyncio
async def test_qdrant_storage_with_multiple_chunks():
    """Test Qdrant storage with multiple chunks to verify batch processing."""
    collection_name = "test_batch_collection"

    # Mock the Qdrant client
    with patch('backend.src.content_embedding.qdrant_service.get_qdrant_client') as mock_get_client:
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.recreate_collection.return_value = True
        mock_client.upsert.return_value = True
        mock_get_client.return_value = mock_client

        # Mock the Cohere client
        with patch('backend.src.content_embedding.qdrant_service.get_cohere_client') as mock_cohere:
            # Create embeddings for multiple chunks
            mock_embedding_response = Mock()
            # Return embeddings for 3 chunks
            mock_embedding_response.embeddings = [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2]
            ]
            mock_cohere.return_value.embed.return_value = mock_embedding_response

            # Create multiple test chunks
            test_chunks = [
                {
                    'content': f'Test content chunk {i}',
                    'chunk_index': i,
                    'metadata': {
                        'url': f'https://example.com/test{i}',
                        'title': f'Test Page {i}',
                        'section_hierarchy': [
                            {'level': 1, 'text': f'Section {i}'}
                        ]
                    }
                }
                for i in range(3)
            ]

            result = await save_chunks_to_qdrant(test_chunks, collection_name)
            assert result is True

            # Verify upsert was called (for batch processing)
            assert mock_client.upsert.call_count >= 1  # May be called multiple times for batches