import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .utils import Config, ChunkingError

logger = logging.getLogger(__name__)


def _validate_chunking_params(text: str, chunk_size: int, overlap: int) -> None:
    """
    Validate parameters for text chunking.

    Args:
        text: The text to validate
        chunk_size: The chunk size to validate
        overlap: The overlap to validate

    Raises:
        ChunkingError: If validation fails
    """
    if not text or not isinstance(text, str):
        raise ChunkingError("Text must be a non-empty string")

    if not isinstance(chunk_size, int) or chunk_size <= 0 or chunk_size > 10000:
        raise ChunkingError("chunk_size must be a positive integer not exceeding 10000")

    if not isinstance(overlap, int) or overlap < 0:
        raise ChunkingError("overlap must be a non-negative integer")

    if overlap >= chunk_size:
        raise ChunkingError("overlap must be less than chunk_size")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 102) -> List[Dict[str, Any]]:
    """
    Split extracted text into manageable chunks for embedding.

    Args:
        text: The text to be chunked
        chunk_size: Size of each chunk (default 512 tokens)
        overlap: Overlap between chunks (default 102 tokens, 20% of chunk_size)

    Returns:
        List of dictionaries containing chunked text with metadata
    """
    # Validate input parameters
    _validate_chunking_params(text, chunk_size, overlap)

    try:
        logger.info(f"Starting to chunk text of length {len(text)} with chunk_size={chunk_size}, overlap={overlap}")

        # Use RecursiveCharacterTextSplitter from langchain for intelligent chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Split the text into chunks
        chunks = text_splitter.split_text(text)

        logger.info(f"Text split into {len(chunks)} chunks")

        # Create chunked results with metadata
        chunked_results = []
        for i, chunk in enumerate(chunks):
            chunked_results.append({
                'id': f'chunk_{i}',
                'content': chunk,
                'chunk_index': i,
                'token_count': len(chunk),  # Approximate token count
                'metadata': {
                    'chunk_size': chunk_size,
                    'overlap': overlap,
                    'chunk_position': i,
                    'total_chunks': len(chunks)
                }
            })

        logger.info(f"Successfully created {len(chunked_results)} text chunks")
        return chunked_results

    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise ChunkingError(f"Error chunking text: {str(e)}")


def validate_metadata_hierarchy(metadata: Dict[str, Any]) -> bool:
    """
    Validate that document hierarchy is preserved in metadata.

    Args:
        metadata: Metadata to validate

    Returns:
        True if metadata hierarchy is valid, False otherwise
    """
    try:
        # Check if section_hierarchy exists and is properly structured
        section_hierarchy = metadata.get('section_hierarchy', [])

        if not isinstance(section_hierarchy, list):
            return False

        # Validate each section in the hierarchy
        for section in section_hierarchy:
            if not isinstance(section, dict):
                return False
            if 'level' not in section or 'text' not in section:
                return False
            if not isinstance(section['level'], int) or not isinstance(section['text'], str):
                return False

        return True
    except Exception:
        return False


def chunk_text_with_metadata(text: str, metadata: Dict[str, Any],
                           chunk_size: int = Config.CHUNK_SIZE,
                           overlap: int = Config.CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split text into chunks while preserving document metadata.

    Args:
        text: The text to be chunked
        metadata: Metadata to be preserved with each chunk
        chunk_size: Size of each chunk (default from Config)
        overlap: Overlap between chunks (default from Config)

    Returns:
        List of dictionaries containing chunked text with preserved metadata
    """
    # Validate input parameters
    _validate_chunking_params(text, chunk_size, overlap)

    if not isinstance(metadata, dict):
        raise ChunkingError("metadata must be a dictionary")

    try:
        logger.info(f"Starting to chunk text with metadata, length {len(text)}, chunk_size={chunk_size}, overlap={overlap}")

        # Use RecursiveCharacterTextSplitter from langchain for intelligent chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Split the text into chunks
        chunks = text_splitter.split_text(text)

        logger.info(f"Text split into {len(chunks)} chunks with metadata")

        # Create chunked results with metadata
        chunked_results = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_size': chunk_size,
                'overlap': overlap,
                'chunk_position': i,
                'total_chunks': len(chunks)
            })

            chunked_results.append({
                'id': f'chunk_{i}',
                'content': chunk,
                'chunk_index': i,
                'token_count': len(chunk),  # Approximate token count
                'metadata': chunk_metadata
            })

        logger.info(f"Successfully created {len(chunked_results)} text chunks with metadata")
        return chunked_results

    except Exception as e:
        logger.error(f"Error chunking text with metadata: {str(e)}")
        raise ChunkingError(f"Error chunking text with metadata: {str(e)}")


def test_metadata_preservation_with_hierarchical_content():
    """
    Function to test metadata structure with hierarchical content.
    This is more of a utility for testing purposes.
    """
    # This function would typically be used in testing scenarios
    # to validate that hierarchical content is preserved correctly
    pass