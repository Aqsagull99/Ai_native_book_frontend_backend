import pytest
import asyncio
from unittest.mock import AsyncMock, patch, Mock
from backend.src.content_embedding.embedding_service import main, run_pipeline_with_progress


@pytest.mark.asyncio
async def test_main_function_success():
    """Test main function executes the full pipeline successfully."""
    # Mock all the dependencies
    with patch('backend.src.content_embedding.embedding_service.setup_logging') as mock_logging:
        with patch('backend.src.content_embedding.embedding_service.get_all_url', new_callable=AsyncMock) as mock_get_urls:
            with patch('backend.src.content_embedding.embedding_service.extract_text_from_url', new_callable=AsyncMock) as mock_extract:
                with patch('backend.src.content_embedding.embedding_service.chunk_text_with_metadata') as mock_chunk:
                    with patch('backend.src.content_embedding.embedding_service.batch_save_chunks_to_qdrant', new_callable=AsyncMock) as mock_save:
                        with patch('backend.src.content_embedding.embedding_service.create_collection', new_callable=AsyncMock) as mock_create_collection:
                            # Setup mocks
                            mock_get_urls.return_value = ['https://example.com/page1', 'https://example.com/page2']
                            mock_extract.return_value = {
                                'content': 'Sample content',
                                'title': 'Test Page',
                                'metadata': {'url': 'https://example.com/page1'}
                            }
                            mock_chunk.return_value = [
                                {
                                    'content': 'Sample content',
                                    'chunk_index': 0,
                                    'metadata': {'url': 'https://example.com/page1'}
                                }
                            ]
                            mock_save.return_value = True
                            mock_create_collection.return_value = True

                            # Run the main function
                            await main()

                            # Verify all steps were called
                            mock_create_collection.assert_called_once()
                            mock_get_urls.assert_called_once()
                            assert mock_extract.call_count == 2  # Called for each URL
                            assert mock_chunk.call_count == 2   # Called for each extracted content
                            assert mock_save.call_count >= 1    # Called for saving chunks


@pytest.mark.asyncio
async def test_run_pipeline_with_progress():
    """Test run_pipeline_with_progress function executes successfully."""
    # Mock all the dependencies
    with patch('backend.src.content_embedding.embedding_service.setup_logging') as mock_logging:
        with patch('backend.src.content_embedding.embedding_service.get_all_url', new_callable=AsyncMock) as mock_get_urls:
            with patch('backend.src.content_embedding.embedding_service.extract_text_from_url', new_callable=AsyncMock) as mock_extract:
                with patch('backend.src.content_embedding.embedding_service.chunk_text_with_metadata') as mock_chunk:
                    with patch('backend.src.content_embedding.embedding_service.batch_save_chunks_to_qdrant', new_callable=AsyncMock) as mock_save:
                        with patch('backend.src.content_embedding.embedding_service.create_collection', new_callable=AsyncMock) as mock_create_collection:
                            # Setup mocks
                            mock_get_urls.return_value = ['https://example.com/page1']
                            mock_extract.return_value = {
                                'content': 'Sample content',
                                'title': 'Test Page',
                                'metadata': {'url': 'https://example.com/page1'}
                            }
                            mock_chunk.return_value = [
                                {
                                    'content': 'Sample content',
                                    'chunk_index': 0,
                                    'metadata': {'url': 'https://example.com/page1'}
                                }
                            ]
                            mock_save.return_value = True
                            mock_create_collection.return_value = True

                            # Run the function
                            await run_pipeline_with_progress()

                            # Verify all steps were called
                            mock_create_collection.assert_called_once()
                            mock_get_urls.assert_called_once()
                            mock_extract.assert_called_once()
                            mock_chunk.assert_called_once()
                            mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_main_function_with_url_error():
    """Test main function handles URL processing errors gracefully."""
    # Mock all the dependencies
    with patch('backend.src.content_embedding.embedding_service.setup_logging') as mock_logging:
        with patch('backend.src.content_embedding.embedding_service.get_all_url', new_callable=AsyncMock) as mock_get_urls:
            with patch('backend.src.content_embedding.embedding_service.extract_text_from_url', new_callable=AsyncMock) as mock_extract:
                with patch('backend.src.content_embedding.embedding_service.chunk_text_with_metadata') as mock_chunk:
                    with patch('backend.src.content_embedding.embedding_service.batch_save_chunks_to_qdrant', new_callable=AsyncMock) as mock_save:
                        with patch('backend.src.content_embedding.embedding_service.create_collection', new_callable=AsyncMock) as mock_create_collection:
                            # Setup mocks
                            mock_get_urls.return_value = ['https://example.com/page1', 'https://example.com/page2']

                            # Make the first URL extraction succeed and the second fail
                            def extract_side_effect(url):
                                if url == 'https://example.com/page1':
                                    return {
                                        'content': 'Sample content',
                                        'title': 'Test Page',
                                        'metadata': {'url': url}
                                    }
                                else:
                                    raise Exception("Failed to extract")

                            mock_extract.side_effect = extract_side_effect
                            mock_chunk.return_value = [
                                {
                                    'content': 'Sample content',
                                    'chunk_index': 0,
                                    'metadata': {'url': 'https://example.com/page1'}
                                }
                            ]
                            mock_save.return_value = True
                            mock_create_collection.return_value = True

                            # Run the main function - should not raise an exception
                            await main()

                            # Verify all steps were called appropriately
                            mock_create_collection.assert_called_once()
                            mock_get_urls.assert_called_once()
                            assert mock_extract.call_count == 2  # Called for each URL
                            mock_chunk.assert_called_once()     # Only called for successful extraction
                            mock_save.assert_called_once()      # Only called for successful chunking


def test_main_function_signature():
    """Test that main has the correct function signature."""
    import inspect
    from backend.src.content_embedding.embedding_service import main

    sig = inspect.signature(main)
    params = list(sig.parameters.keys())

    # main should be an async function with no parameters
    assert len(params) == 0


def test_run_pipeline_with_progress_signature():
    """Test that run_pipeline_with_progress has the correct function signature."""
    import inspect
    from backend.src.content_embedding.embedding_service import run_pipeline_with_progress

    sig = inspect.signature(run_pipeline_with_progress)
    params = list(sig.parameters.keys())

    # run_pipeline_with_progress should be an async function with no parameters
    assert len(params) == 0