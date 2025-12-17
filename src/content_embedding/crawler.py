import asyncio
import httpx
from typing import List, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import time
import logging
from .utils import Config, CrawlerError, get_environment_variable

logger = logging.getLogger(__name__)


def _validate_input_params(base_url: str, max_depth: int) -> None:
    """
    Validate input parameters for URL crawling.

    Args:
        base_url: The base URL to validate
        max_depth: The max depth to validate

    Raises:
        CrawlerError: If validation fails
    """
    if not base_url or not isinstance(base_url, str):
        raise CrawlerError("base_url must be a non-empty string")

    if not base_url.startswith(('http://', 'https://')):
        raise CrawlerError("base_url must be a valid URL starting with http:// or https://")

    if not isinstance(max_depth, int) or max_depth < 0 or max_depth > 10:
        raise CrawlerError("max_depth must be an integer between 0 and 10")


async def get_all_url(base_url: str, max_depth: int = 2) -> List[str]:
    """
    Crawl the deployed Docusaurus book and extract all accessible URLs.

    Args:
        base_url: The base URL of the website to crawl
        max_depth: Maximum depth for recursive crawling (default 2)

    Returns:
        List of URLs found on the website
    """
    # Validate input parameters
    _validate_input_params(base_url, max_depth)

    # Parse the base URL to get the domain
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

    # Get URLs from sitemap first
    sitemap_urls = await _get_urls_from_sitemap(Config.SITEMAP_URL)

    # If sitemap parsing fails or returns no URLs, fall back to crawling
    if not sitemap_urls:
        logger.info("No URLs found in sitemap, starting recursive crawl...")
        sitemap_urls = await _crawl_recursively(base_url, max_depth, base_domain)

    # Remove duplicates while preserving order
    unique_urls = list(dict.fromkeys(sitemap_urls))

    logger.info(f"Found {len(unique_urls)} unique URLs")
    return unique_urls


async def _get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse sitemap to extract URLs.

    Args:
        sitemap_url: URL of the sitemap.xml file

    Returns:
        List of URLs extracted from the sitemap
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(sitemap_url, timeout=30.0)
            response.raise_for_status()

            # Parse the XML content
            root = ET.fromstring(response.content)

            # Find all <url><loc> elements in the sitemap
            urls = []
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None and loc_elem.text:
                    url = loc_elem.text.strip()
                    # Fix incorrect domain in sitemap (replace placeholder with actual domain)
                    if url.startswith('https://your-docusaurus-site.example.com'):
                        url = url.replace('https://your-docusaurus-site.example.com',
                                         Config.BOOK_BASE_URL)
                    elif url.startswith('https://your-book.vercel.app'):
                        url = url.replace('https://your-book.vercel.app',
                                         Config.BOOK_BASE_URL)
                    urls.append(url)

            logger.info(f"Extracted {len(urls)} URLs from sitemap")
            return urls

    except Exception as e:
        logger.warning(f"Failed to parse sitemap {sitemap_url}: {str(e)}")
        return []


async def _crawl_recursively(base_url: str, max_depth: int, base_domain: str) -> List[str]:
    """
    Recursively crawl the website to discover URLs.

    Args:
        base_url: The base URL to start crawling from
        max_depth: Maximum depth for recursive crawling
        base_domain: The base domain to limit crawling to

    Returns:
        List of URLs found during crawling
    """
    visited_urls: Set[str] = set()
    urls_to_visit = [(base_url, 0)]  # (url, depth)

    # Create httpx client with custom settings
    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "Content-Embedding-Bot/1.0"}
    ) as client:

        while urls_to_visit:
            current_url, depth = urls_to_visit.pop(0)

            # Skip if already visited or depth exceeds max
            if current_url in visited_urls or depth > max_depth:
                continue

            visited_urls.add(current_url)
            logger.info(f"Crawling (depth {depth}): {current_url}")

            try:
                # Add delay to respect website policies
                await asyncio.sleep(Config.REQUEST_DELAY)

                response = await client.get(current_url)
                response.raise_for_status()

                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find all links on the page
                for link in soup.find_all('a', href=True):
                    href = link['href']

                    # Convert relative URLs to absolute URLs
                    absolute_url = urljoin(current_url, href)
                    parsed_url = urlparse(absolute_url)

                    # Only add URLs from the same domain and with proper structure
                    if (parsed_url.netloc == urlparse(base_domain).netloc and
                        absolute_url not in visited_urls and
                        not absolute_url.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', '.docx'))):

                        # Ensure it's a proper page URL (not an anchor, email, etc.)
                        if parsed_url.scheme in ['http', 'https'] and not parsed_url.fragment:
                            urls_to_visit.append((absolute_url, depth + 1))

            except httpx.RequestError as e:
                logger.warning(f"Failed to crawl {current_url}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error while crawling {current_url}: {str(e)}")
                continue

    return list(visited_urls)


async def _is_valid_url(url: str, base_domain: str) -> bool:
    """
    Check if a URL is valid for crawling.

    Args:
        url: The URL to validate
        base_domain: The base domain to limit crawling to

    Returns:
        True if the URL is valid, False otherwise
    """
    try:
        parsed = urlparse(url)
        # Only allow URLs from the same domain
        return parsed.netloc == urlparse(base_domain).netloc
    except Exception:
        return False