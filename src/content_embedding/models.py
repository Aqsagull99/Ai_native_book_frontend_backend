"""
Content chunk data model for vector retrieval validation
Based on data-model.md specification
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class ContentChunk:
    """Represents a segment of book content that has been embedded and stored in the vector database."""

    content: str
    url: str
    title: str
    chunk_index: int
    source_metadata: Dict[str, Any]
    created_at: datetime
    vector: Optional[list] = None  # Optional since during retrieval we might not always include the vector


@dataclass
class SearchResult:
    """Represents a single result from a vector similarity search."""

    content_chunk: str
    similarity_score: float
    metadata: Dict[str, Any]
    rank: int  # Position in the results (1-indexed)


@dataclass
class QueryRequest:
    """Represents a search query request."""

    query_text: str
    top_k: int = 5  # Number of results to retrieve (default: 5)
    metadata_filter: Optional[Dict[str, Any]] = None  # Additional filters for search (optional)


@dataclass
class QueryResponse:
    """Represents the response to a search query."""

    query_text: str
    results: list
    execution_time: float  # Time taken to execute the query in seconds
    timestamp: datetime


@dataclass
class RetrievalMetrics:
    """Metrics for tracking retrieval performance and accuracy."""

    total_queries: int
    successful_queries: int
    avg_response_time: float  # Average query response time in seconds
    avg_similarity_score: float  # Average similarity score of results
    metadata_accuracy: float  # Percentage of results with complete metadata (0-100)
    relevance_accuracy: float  # Percentage of results that are semantically relevant (0-100)
    consistency_rate: float  # Percentage of consistent results across repeated queries (0-100)


@dataclass
class ValidationResult:
    """Represents the validation result for a query."""

    query_request: QueryRequest
    query_response: QueryResponse
    metadata_preserved: bool  # Whether all metadata fields are present and valid
    relevance_score: float  # Human-validated relevance score (0-1)
    consistency_score: float  # For repeated queries, similarity of results (0-1)
    validation_timestamp: datetime