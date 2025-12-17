import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from backend.src.content_embedding.text_extractor import extract_text_from_url


@pytest.mark.asyncio
async def test_extract_text_from_url_success():
    """Test extract_text_from_url function with valid HTML content."""
    html_content = '''
    <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="This is a test page">
        </head>
        <body>
            <h1>Main Title</h1>
            <p>This is the main content of the page.</p>
            <p>Additional content here.</p>
        </body>
    </html>
    '''

    with patch('httpx.AsyncClient.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.text = html_content
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = await extract_text_from_url('https://example.com/test')

        assert result['url'] == 'https://example.com/test'
        assert result['title'] == 'Test Page'
        assert 'main content of the page' in result['content']
        assert 'Additional content here' in result['content']
        assert result['metadata']['description'] == 'This is a test page'


@pytest.mark.asyncio
async def test_extract_text_from_url_with_docusaurus_selectors():
    """Test extract_text_from_url function with Docusaurus-specific selectors."""
    html_content = '''
    <html>
        <head>
            <title>Docusaurus Page</title>
        </head>
        <body>
            <nav>Navigation content</nav>
            <header>Header content</header>
            <main class="main-wrapper">
                <article class="markdown">
                    <h1>Docusaurus Content</h1>
                    <p>This is the main content that should be extracted.</p>
                    <p>More content here.</p>
                </article>
            </main>
            <footer>Footer content</footer>
        </body>
    </html>
    '''

    with patch('httpx.AsyncClient.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.text = html_content
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = await extract_text_from_url('https://example.com/docs')

        assert result['title'] == 'Docusaurus Page'
        # Should extract content from the main/article sections, not navigation or footer
        assert 'main content that should be extracted' in result['content']
        assert 'Navigation content' not in result['content']
        assert 'Header content' not in result['content']
        assert 'Footer content' not in result['content']


@pytest.mark.asyncio
async def test_extract_text_from_url_missing_title():
    """Test extract_text_from_url function when title is missing."""
    html_content = '''
    <html>
        <head></head>
        <body>
            <p>Content without a title.</p>
        </body>
    </html>
    '''

    with patch('httpx.AsyncClient.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.text = html_content
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = await extract_text_from_url('https://example.com/no-title')

        assert result['title'] == 'No Title'
        assert 'Content without a title' in result['content']


@pytest.mark.asyncio
async def test_extract_text_from_url_network_error():
    """Test extract_text_from_url function handles network errors."""
    with patch('httpx.AsyncClient.get') as mock_get:
        mock_get.side_effect = Exception("Network error")

        from backend.src.content_embedding.utils import TextExtractionError

        with pytest.raises(TextExtractionError):
            await extract_text_from_url('https://example.com/error')


def test_extract_text_from_url_function_signature():
    """Test that extract_text_from_url has the correct function signature."""
    import inspect
    from backend.src.content_embedding.text_extractor import extract_text_from_url

    sig = inspect.signature(extract_text_from_url)
    params = list(sig.parameters.keys())

    assert 'url' in params