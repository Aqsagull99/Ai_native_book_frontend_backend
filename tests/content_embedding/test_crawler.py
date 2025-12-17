import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from backend.src.content_embedding.crawler import get_all_url, _get_urls_from_sitemap


@pytest.mark.asyncio
async def test_get_all_url_with_sitemap():
    """Test get_all_url function with sitemap parsing."""
    # Mock the sitemap parsing to return test URLs
    with patch('backend.src.content_embedding.crawler._get_urls_from_sitemap',
               new_callable=AsyncMock) as mock_sitemap:
        mock_sitemap.return_value = [
            'https://example.com/page1',
            'https://example.com/page2',
            'https://example.com/page3'
        ]

        result = await get_all_url('https://example.com', max_depth=2)

        assert len(result) == 3
        assert 'https://example.com/page1' in result
        assert 'https://example.com/page2' in result
        assert 'https://example.com/page3' in result


@pytest.mark.asyncio
async def test_get_all_url_fallback_to_crawling():
    """Test get_all_url function falls back to crawling when sitemap is empty."""
    with patch('backend.src.content_embedding.crawler._get_urls_from_sitemap',
               new_callable=AsyncMock) as mock_sitemap:
        mock_sitemap.return_value = []  # Empty sitemap

        with patch('backend.src.content_embedding.crawler._crawl_recursively',
                   new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = [
                'https://example.com/page1',
                'https://example.com/page2'
            ]

            result = await get_all_url('https://example.com', max_depth=1)

            assert len(result) == 2
            assert 'https://example.com/page1' in result
            assert 'https://example.com/page2' in result


def test_get_all_url_function_signature():
    """Test that get_all_url has the correct function signature."""
    import inspect
    from backend.src.content_embedding.crawler import get_all_url

    sig = inspect.signature(get_all_url)
    params = list(sig.parameters.keys())

    assert 'base_url' in params
    assert 'max_depth' in params

    # Check default value for max_depth
    assert sig.parameters['max_depth'].default == 2


@pytest.mark.asyncio
async def test_get_urls_from_sitemap_success():
    """Test _get_urls_from_sitemap function with valid sitemap."""
    sitemap_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url>
            <loc>https://example.com/page1</loc>
        </url>
        <url>
            <loc>https://example.com/page2</loc>
        </url>
    </urlset>'''

    with patch('httpx.AsyncClient.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.text = sitemap_content
        mock_response.content = sitemap_content.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        from backend.src.content_embedding.crawler import _get_urls_from_sitemap
        result = await _get_urls_from_sitemap('https://example.com/sitemap.xml')

        assert len(result) == 2
        assert 'https://example.com/page1' in result
        assert 'https://example.com/page2' in result


@pytest.mark.asyncio
async def test_get_urls_from_sitemap_failure():
    """Test _get_urls_from_sitemap function handles failure gracefully."""
    with patch('httpx.AsyncClient.get') as mock_get:
        mock_get.side_effect = Exception("Network error")

        from backend.src.content_embedding.crawler import _get_urls_from_sitemap
        result = await _get_urls_from_sitemap('https://example.com/sitemap.xml')

        assert result == []  # Should return empty list on failure