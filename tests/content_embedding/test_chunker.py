import pytest
from backend.src.content_embedding.chunker import chunk_text, chunk_text_with_metadata


def test_chunk_text_basic():
    """Test chunk_text function with basic text."""
    text = "This is a sample text. " * 50  # Create a longer text
    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert len(chunks) > 0
    assert all('content' in chunk for chunk in chunks)
    assert all('chunk_index' in chunk for chunk in chunks)
    assert all('metadata' in chunk for chunk in chunks)


def test_chunk_text_with_small_text():
    """Test chunk_text function with text smaller than chunk size."""
    text = "Short text."
    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0]['content'] == text


def test_chunk_text_with_metadata_preservation():
    """Test chunk_text function preserves metadata."""
    text = "This is a sample text. " * 20
    chunks = chunk_text(text, chunk_size=40, overlap=8)

    assert len(chunks) > 0
    for i, chunk in enumerate(chunks):
        assert 'id' in chunk
        assert 'content' in chunk
        assert 'chunk_index' in chunk
        assert chunk['chunk_index'] == i
        assert 'metadata' in chunk
        assert 'chunk_size' in chunk['metadata']
        assert 'overlap' in chunk['metadata']


def test_chunk_text_with_configurable_params():
    """Test chunk_text function with different chunk sizes and overlaps."""
    text = "This is a sample text. " * 30

    # Test with different parameters
    chunks1 = chunk_text(text, chunk_size=30, overlap=5)
    chunks2 = chunk_text(text, chunk_size=60, overlap=12)

    # Larger chunks should result in fewer total chunks
    assert len(chunks1) >= len(chunks2)


def test_chunk_text_with_metadata_function():
    """Test chunk_text_with_metadata function."""
    text = "This is a sample text. " * 20
    metadata = {'url': 'https://example.com', 'title': 'Test Page', 'author': 'Test Author'}

    chunks = chunk_text_with_metadata(text, metadata, chunk_size=50, overlap=10)

    assert len(chunks) > 0
    for chunk in chunks:
        assert 'content' in chunk
        assert 'metadata' in chunk
        chunk_metadata = chunk['metadata']
        assert chunk_metadata['url'] == 'https://example.com'
        assert chunk_metadata['title'] == 'Test Page'
        assert chunk_metadata['author'] == 'Test Author'
        assert chunk_metadata['chunk_size'] == 50
        assert chunk_metadata['overlap'] == 10


def test_chunk_text_edge_cases():
    """Test chunk_text function with edge cases."""
    # Empty text
    chunks = chunk_text("", chunk_size=50, overlap=10)
    assert len(chunks) == 0

    # Single character
    chunks = chunk_text("A", chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]['content'] == "A"

    # Text exactly the size of chunk
    text = "A" * 50
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]['content'] == text


def test_chunk_text_function_signature():
    """Test that chunk_text has the correct function signature."""
    import inspect
    from backend.src.content_embedding.chunker import chunk_text

    sig = inspect.signature(chunk_text)
    params = list(sig.parameters.keys())

    assert 'text' in params
    assert 'chunk_size' in params
    assert 'overlap' in params

    # Check default values
    assert sig.parameters['chunk_size'].default == 512
    assert sig.parameters['overlap'].default == 102


def test_chunk_text_with_metadata_function_signature():
    """Test that chunk_text_with_metadata has the correct function signature."""
    import inspect
    from backend.src.content_embedding.chunker import chunk_text_with_metadata

    sig = inspect.signature(chunk_text_with_metadata)
    params = list(sig.parameters.keys())

    assert 'text' in params
    assert 'metadata' in params
    assert 'chunk_size' in params
    assert 'overlap' in params


def test_validate_metadata_hierarchy_valid():
    """Test validate_metadata_hierarchy with valid metadata."""
    from backend.src.content_embedding.chunker import validate_metadata_hierarchy

    valid_metadata = {
        'section_hierarchy': [
            {'level': 1, 'text': 'Main Section'},
            {'level': 2, 'text': 'Subsection'},
            {'level': 3, 'text': 'Sub-subsection'}
        ]
    }

    result = validate_metadata_hierarchy(valid_metadata)
    assert result is True


def test_validate_metadata_hierarchy_invalid():
    """Test validate_metadata_hierarchy with invalid metadata."""
    from backend.src.content_embedding.chunker import validate_metadata_hierarchy

    # Test with invalid structure
    invalid_metadata1 = {
        'section_hierarchy': 'not a list'
    }

    result1 = validate_metadata_hierarchy(invalid_metadata1)
    assert result1 is False

    # Test with missing keys
    invalid_metadata2 = {
        'section_hierarchy': [
            {'level': 1}  # Missing 'text'
        ]
    }

    result2 = validate_metadata_hierarchy(invalid_metadata2)
    assert result2 is False

    # Test with wrong data types
    invalid_metadata3 = {
        'section_hierarchy': [
            {'level': 'not an int', 'text': 'valid text'}
        ]
    }

    result3 = validate_metadata_hierarchy(invalid_metadata3)
    assert result3 is False