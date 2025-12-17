import httpx
from bs4 import BeautifulSoup
from typing import Dict, Any
import logging
from urllib.parse import urljoin
from .utils import TextExtractionError, Config

logger = logging.getLogger(__name__)


def _validate_url_input(url: str) -> None:
    """
    Validate URL input for text extraction.

    Args:
        url: The URL to validate

    Raises:
        TextExtractionError: If validation fails
    """
    if not url or not isinstance(url, str):
        raise TextExtractionError("URL must be a non-empty string")

    if not url.startswith(('http://', 'https://')):
        raise TextExtractionError("URL must be a valid URL starting with http:// or https://")

    if len(url) > 2048:  # Standard URL length limit
        raise TextExtractionError("URL exceeds maximum allowed length of 2048 characters")


async def extract_text_from_url(url: str) -> Dict[str, Any]:
    """
    Extract clean text content from a single web page.

    Args:
        url: The URL of the web page to extract content from

    Returns:
        Dictionary containing the extracted content and metadata
    """
    # Validate input
    _validate_url_input(url)

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Content-Extractor-Bot/1.0"}
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else "No Title"

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Try to find main content using Docusaurus-specific selectors
        main_content = None

        # Look for main content areas in Docusaurus sites
        selectors_to_try = [
            'main',  # Most common main content area
            '.main-wrapper',  # Docusaurus main wrapper
            '.container',  # Container divs
            '.theme-doc-page',  # Docusaurus doc page
            '.markdown',  # Markdown content
            '.doc-content',  # Documentation content
            '.article',  # Article content
            'article',  # HTML5 article tag
            '.content',  # General content class
            '.docs-content',  # Documentation specific
        ]

        for selector in selectors_to_try:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # If no main content found with specific selectors, use body
        if not main_content:
            main_content = soup.find('body')

        if main_content:
            # Extract text from the main content
            text_content = main_content.get_text(separator=' ', strip=True)

            # Clean up extra whitespace
            import re
            text_content = re.sub(r'\s+', ' ', text_content).strip()
        else:
            # Fallback: extract all text from body
            body = soup.find('body')
            if body:
                text_content = body.get_text(separator=' ', strip=True)
                text_content = re.sub(r'\s+', ' ', text_content).strip()
            else:
                text_content = soup.get_text(separator=' ', strip=True)
                text_content = re.sub(r'\s+', ' ', text_content).strip()

        # Extract metadata
        metadata = _extract_metadata(soup, url)

        result = {
            'url': url,
            'title': title,
            'content': text_content,
            'html_content': str(main_content) if main_content else response.text,
            'metadata': metadata
        }

        logger.info(f"Successfully extracted content from {url} (title: {title})")
        return result

    except httpx.RequestError as e:
        logger.error(f"Failed to fetch URL {url}: {str(e)}")
        raise TextExtractionError(f"Failed to fetch URL {url}: {str(e)}")
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {str(e)}")
        raise TextExtractionError(f"Error extracting text from {url}: {str(e)}")


def _extract_metadata(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """
    Extract metadata from the HTML soup object.

    Args:
        soup: BeautifulSoup object of the HTML content
        url: The URL of the page

    Returns:
        Dictionary containing metadata
    """
    metadata = {
        'url': url,
        'title': '',
        'description': '',
        'keywords': '',
        'author': '',
        'language': '',
        'section_hierarchy': [],
        'content_type': 'webpage'
    }

    # Extract title
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text().strip()

    # Extract meta description
    desc_tag = soup.find('meta', attrs={'name': 'description'})
    if desc_tag and desc_tag.get('content'):
        metadata['description'] = desc_tag.get('content').strip()

    # Extract meta keywords
    keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
    if keywords_tag and keywords_tag.get('content'):
        metadata['keywords'] = keywords_tag.get('content').strip()

    # Extract author
    author_tag = soup.find('meta', attrs={'name': 'author'})
    if author_tag and author_tag.get('content'):
        metadata['author'] = author_tag.get('content').strip()

    # Extract language
    lang_tag = soup.find('html')
    if lang_tag and lang_tag.get('lang'):
        metadata['language'] = lang_tag.get('lang').strip()

    # Try to extract section hierarchy from headings
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for heading in headings:
        heading_text = heading.get_text().strip()
        if heading_text:
            level = int(heading.name[1])  # Extract number from h1, h2, etc.
            metadata['section_hierarchy'].append({
                'level': level,
                'text': heading_text
            })

    return metadata