# Content Embedding Pipeline

This module provides functionality to crawl websites, extract text content, chunk it, and store embeddings in Qdrant for RAG applications.

## Overview

The content embedding pipeline performs the following steps:
1. Crawls a website to discover all accessible URLs
2. Extracts clean text content from each page
3. Chunks the text into manageable pieces
4. Generates embeddings using Cohere
5. Stores embeddings in Qdrant Cloud for semantic search

## Components

### 1. Crawler (`crawler.py`)
- Discovers URLs using sitemap parsing and recursive crawling
- Respects website policies with rate limiting
- Handles errors gracefully

### 2. Text Extractor (`text_extractor.py`)
- Extracts clean text from HTML pages
- Preserves document structure and metadata
- Uses Docusaurus-specific selectors

### 3. Chunker (`chunker.py`)
- Splits text into configurable chunks
- Preserves metadata for each chunk
- Uses intelligent splitting to maintain context

### 4. Qdrant Service (`qdrant_service.py`)
- Creates Qdrant collections
- Stores embeddings with metadata
- Implements batch processing

### 5. Embedding Service (`embedding_service.py`)
- Orchestrates the entire pipeline
- Provides main execution function
- Includes progress tracking

## Configuration

The pipeline uses the following environment variables (defined in `.env`):

```bash
# Cohere Configuration
COHERE_API_KEY=your_cohere_api_key

# Qdrant Configuration
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key

# Website Configuration
BOOK_BASE_URL=https://your-book.vercel.app
TARGET_SITE=https://ai-native-book-frontend.vercel.app
SITEMAP_URL=https://ai-native-book-frontend.vercel.app/sitemap.xml

# Processing Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=102
MAX_DEPTH=2
REQUEST_DELAY=1.0
COLLECTION_NAME=ai_native_book
COHERE_MODEL=embed-english-v3.0
```

## Usage

### Running the Pipeline

```bash
cd backend
python run_content_embedding.py
```

### Using Individual Components

```python
from content_embedding.crawler import get_all_url
from content_embedding.text_extractor import extract_text_from_url
from content_embedding.chunker import chunk_text_with_metadata
from content_embedding.qdrant_service import save_chunks_to_qdrant

# Example usage
urls = await get_all_url("https://example.com")
content_data = await extract_text_from_url(urls[0])
chunks = chunk_text_with_metadata(content_data['content'], content_data['metadata'])
await save_chunks_to_qdrant(chunks)
```

## Error Handling

The pipeline includes comprehensive error handling with custom exception types:
- `CrawlerError`: For crawling-related errors
- `TextExtractionError`: For text extraction errors
- `ChunkingError`: For chunking errors
- `QdrantError`: For Qdrant operations errors
- `EmbeddingError`: For embedding generation errors

## Testing

Run the tests using pytest:

```bash
cd backend
python -m pytest tests/content_embedding/
```

## Architecture

```
[Website URLs] → [Crawler] → [Text Extractor] → [Chunker] → [Qdrant Service]
     ↓              ↓             ↓            ↓            ↓
  get_all_url  extract_text   chunk_text   generate     save_to_qdrant
                _from_url     _with_meta   embeddings
```

The pipeline is designed to be modular, allowing each component to be used independently while providing a complete solution when orchestrated together.