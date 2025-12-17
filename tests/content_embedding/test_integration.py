import pytest
import asyncio
from unittest.mock import AsyncMock, patch, Mock
from backend.src.content_embedding.crawler import get_all_url
from backend.src.content_embedding.text_extractor import extract_text_from_url
from backend.src.content_embedding.chunker import chunk_text_with_metadata
from backend.src.content_embedding.qdrant_service import create_collection, save_chunks_to_qdrant


@pytest.mark.asyncio
async def test_crawling_extraction_integration():
    """Integration test for crawling and extraction pipeline."""
    # Mock the HTTP requests to avoid making real network calls
    with patch('httpx.AsyncClient.get') as mock_get:
        # Mock response for the base URL
        mock_response = AsyncMock()
        mock_response.text = '''
        <html>
            <head><title>Test Site</title></head>
            <body>
                <a href="/page1">Page 1</a>
                <a href="/page2">Page 2</a>
            </body>
        </html>
        '''
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test crawling
        urls = await get_all_url('https://example.com', max_depth=1)
        assert len(urls) >= 1  # At least the base URL

        # Test extraction (with a separate mock for the specific page)
        with patch('httpx.AsyncClient.get') as mock_page_get:
            mock_page_response = AsyncMock()
            mock_page_response.text = '''
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <h1>Main Content</h1>
                    <p>This is the main content of the page.</p>
                    <p>Additional content here.</p>
                </body>
            </html>
            '''
            mock_page_response.raise_for_status.return_value = None
            mock_page_get.return_value = mock_page_response

            content_data = await extract_text_from_url('https://example.com/page1')
            assert content_data['title'] == 'Test Page'
            assert 'main content of the page' in content_data['content']

            # Test chunking
            chunks = chunk_text_with_metadata(
                text=content_data['content'],
                metadata=content_data['metadata'],
                chunk_size=50,
                overlap=10
            )
            assert len(chunks) > 0
            assert 'content' in chunks[0]
            assert 'metadata' in chunks[0]


@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """End-to-end test for the crawling -> extraction -> chunking pipeline."""
    # Mock the crawling step
    with patch('backend.src.content_embedding.crawler._get_urls_from_sitemap',
               new_callable=AsyncMock) as mock_sitemap:
        mock_sitemap.return_value = ['https://example.com/test-page']

        # Mock the HTTP request for content extraction
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.text = '''
            <html>
                <head><title>Integration Test Page</title></head>
                <body>
                    <main>
                        <h1>Integration Test Content</h1>
                        <p>This is a test paragraph for the integration test.</p>
                        <p>Additional content to ensure chunking works properly.</p>
                    </main>
                </body>
            </html>
            '''
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Execute the pipeline steps
            urls = await get_all_url('https://example.com', max_depth=1)
            assert len(urls) == 1

            content_data = await extract_text_from_url(urls[0])
            assert content_data['title'] == 'Integration Test Page'
            assert 'integration test' in content_data['content'].lower()

            chunks = chunk_text_with_metadata(
                text=content_data['content'],
                metadata=content_data['metadata'],
                chunk_size=60,
                overlap=12
            )

            assert len(chunks) > 0
            for chunk in chunks:
                assert 'content' in chunk
                assert 'metadata' in chunk
                assert 'url' in chunk['metadata']


def test_pipeline_data_flow():
    """Test that data flows correctly between pipeline components."""
    # Test data
    content = "This is a sample content for testing the pipeline data flow. " * 10
    metadata = {
        'url': 'https://example.com/test',
        'title': 'Test Page',
        'description': 'Test description'
    }

    # Test chunking with metadata
    chunks = chunk_text_with_metadata(
        text=content,
        metadata=metadata,
        chunk_size=50,
        overlap=10
    )

    # Verify that metadata is preserved in each chunk
    assert len(chunks) > 0
    for chunk in chunks:
        chunk_metadata = chunk['metadata']
        assert chunk_metadata['url'] == metadata['url']
        assert chunk_metadata['title'] == metadata['title']
        assert chunk_metadata['description'] == metadata['description']
        assert 'chunk_size' in chunk_metadata
        assert 'overlap' in chunk_metadata
        assert 'chunk_position' in chunk_metadata


@pytest.mark.asyncio
async def test_end_to_end_full_pipeline():
    """Comprehensive end-to-end test covering the full functionality."""
    # This test simulates the full pipeline execution
    # Mock all external dependencies to avoid actual network calls

    # Mock sitemap parsing
    with patch('backend.src.content_embedding.crawler._get_urls_from_sitemap',
               new_callable=AsyncMock) as mock_sitemap:
        mock_sitemap.return_value = [
            'https://example.com/page1',
            'https://example.com/page2'
        ]

        # Mock HTTP requests for content extraction
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.text = '''
            <html>
                <head><title>Integration Test Page</title></head>
                <body>
                    <main class="main-wrapper">
                        <h1>Integration Test Content</h1>
                        <p>This is the main content for integration testing.</p>
                        <p>With multiple paragraphs to ensure chunking works properly.</p>
                        <p>Additional content to make the text longer for better testing.</p>
                    </main>
                </body>
            </html>
            '''
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Mock Cohere client for embedding generation
            with patch('backend.src.content_embedding.qdrant_service.get_cohere_client') as mock_cohere:
                mock_embedding_response = Mock()
                # Return embeddings for the chunks (simplified for testing)
                mock_embedding_response.embeddings = [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2]
                ]
                mock_cohere.return_value.embed.return_value = mock_embedding_response

                # Mock Qdrant client for storage
                with patch('backend.src.content_embedding.qdrant_service.get_qdrant_client') as mock_qdrant:
                    mock_qdrant_instance = Mock()
                    mock_qdrant_instance.get_collection.side_effect = Exception("Collection not found")
                    mock_qdrant_instance.recreate_collection.return_value = True
                    mock_qdrant_instance.upsert.return_value = True
                    mock_qdrant.return_value = mock_qdrant_instance

                    # Now test the full pipeline components working together
                    # 1. Get URLs
                    urls = await get_all_url('https://example.com', max_depth=1)
                    assert len(urls) == 2

                    # 2. Extract content from first URL
                    content_data = await extract_text_from_url(urls[0])
                    assert content_data['title'] == 'Integration Test Page'
                    assert 'main content for integration testing' in content_data['content'].lower()

                    # 3. Chunk the content
                    chunks = chunk_text_with_metadata(
                        text=content_data['content'],
                        metadata=content_data['metadata'],
                        chunk_size=60,
                        overlap=12
                    )
                    assert len(chunks) > 0

                    # 4. Create Qdrant collection
                    collection_result = await create_collection("test_collection")
                    assert collection_result is True

                    # 5. Save chunks to Qdrant
                    save_result = await save_chunks_to_qdrant(chunks, "test_collection")
                    assert save_result is True

                    print("All pipeline components work together successfully!")