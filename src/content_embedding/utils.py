import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient


# Load environment variables
load_dotenv()


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('content_embedding.log')
        ]
    )


def get_environment_variable(var_name: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is required but not set")
    return value


def get_cohere_client() -> cohere.Client:
    """Initialize and return Cohere client with proper error handling."""
    try:
        api_key = get_environment_variable('COHERE_API_KEY')
        return cohere.Client(api_key)
    except Exception as e:
        raise EmbeddingError(f"Failed to initialize Cohere client: {str(e)}")


def get_qdrant_client() -> QdrantClient:
    """Initialize and return Qdrant client with proper error handling."""
    try:
        url = get_environment_variable('QDRANT_URL')
        api_key = get_environment_variable('QDRANT_API_KEY')
        return QdrantClient(url=url, api_key=api_key)
    except Exception as e:
        raise QdrantError(f"Failed to initialize Qdrant client: {str(e)}")


class Config:
    """Configuration class to manage application settings."""

    # Content embedding settings
    BOOK_BASE_URL: str = os.getenv('BOOK_BASE_URL', 'https://ai-native-book-frontend.vercel.app')
    TARGET_SITE: str = os.getenv('TARGET_SITE', 'https://ai-native-book-frontend.vercel.app')
    SITEMAP_URL: str = os.getenv('SITEMAP_URL', f'{BOOK_BASE_URL}/sitemap.xml')

    # Chunking settings
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '512'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '102'))  # 20% of chunk size

    # Crawling settings
    MAX_DEPTH: int = int(os.getenv('MAX_DEPTH', '2'))
    REQUEST_DELAY: float = float(os.getenv('REQUEST_DELAY', '1.0'))  # seconds between requests

    # Qdrant settings
    COLLECTION_NAME: str = os.getenv('COLLECTION_NAME', 'ai_native_book')

    # Cohere settings
    COHERE_MODEL: str = os.getenv('COHERE_MODEL', 'embed-english-v3.0')


class ContentEmbeddingError(Exception):
    """Base exception for content embedding pipeline."""
    pass


class CrawlerError(ContentEmbeddingError):
    """Exception raised when crawling operations fail."""
    pass


class TextExtractionError(ContentEmbeddingError):
    """Exception raised when text extraction operations fail."""
    pass


class ChunkingError(ContentEmbeddingError):
    """Exception raised when text chunking operations fail."""
    pass


class QdrantError(ContentEmbeddingError):
    """Exception raised when Qdrant operations fail."""
    pass


class EmbeddingError(ContentEmbeddingError):
    """Exception raised when embedding generation operations fail."""
    pass


def validate_config() -> bool:
    """
    Validate the application configuration and ensure required environment variables are set.

    Returns:
        True if configuration is valid, raises exception if invalid
    """
    required_vars = [
        'COHERE_API_KEY',
        'QDRANT_URL',
        'QDRANT_API_KEY'
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        raise ContentEmbeddingError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Validate that chunk size and overlap are reasonable
    if Config.CHUNK_SIZE <= 0 or Config.CHUNK_SIZE > 10000:
        raise ContentEmbeddingError(f"CHUNK_SIZE must be between 1 and 10000, got {Config.CHUNK_SIZE}")

    if Config.CHUNK_OVERLAP < 0 or Config.CHUNK_OVERLAP >= Config.CHUNK_SIZE:
        raise ContentEmbeddingError(f"CHUNK_OVERLAP must be between 0 and CHUNK_SIZE-1, got {Config.CHUNK_OVERLAP}")

    if Config.MAX_DEPTH < 0 or Config.MAX_DEPTH > 10:
        raise ContentEmbeddingError(f"MAX_DEPTH must be between 0 and 10, got {Config.MAX_DEPTH}")

    # Validate URLs
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if not url_pattern.match(Config.BOOK_BASE_URL):
        raise ContentEmbeddingError(f"BOOK_BASE_URL is not a valid URL: {Config.BOOK_BASE_URL}")

    return True