"""
Data models for the RAG agent with OpenAI Agents SDK
Based on data-model.md specification
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ContentChunk:
    """Represents a segment of book content that has been embedded and stored in the vector database."""

    content: str
    similarity_score: float  # Cosine similarity score (0-1)
    metadata: Dict[str, Any]  # Full metadata from the stored chunk
    rank: int  # Position in the results (1-indexed)


@dataclass
class AgentRequest:
    """Represents a request to the RAG agent."""

    query_text: str
    top_k: int = 5  # Number of results to retrieve (default: 5)
    temperature: float = 0.7  # Temperature parameter for LLM generation (default: 0.7)
    max_tokens: int = 1000  # Maximum tokens in response (default: 1000)
    conversation_context: Optional[Dict[str, Any]] = None  # Previous conversation turns for context
    include_citations: bool = True  # Whether to include source citations (default: true)


@dataclass
class AgentResponse:
    """Represents the response from the RAG agent."""

    query_text: str
    answer: str  # The generated answer based on retrieved content
    retrieved_chunks: List['ContentChunk']  # Content chunks used in answer generation
    confidence_score: float  # Overall confidence in the answer (0-1)
    execution_time: float  # Query execution time in seconds
    timestamp: datetime  # When the query was processed
    conversation_context: Optional[Dict[str, Any]] = None  # Updated conversation context


@dataclass
class AgentState:
    """Represents the state of an agent conversation session."""

    session_id: str  # Unique identifier for the conversation session
    conversation_history: List[Dict[str, str]]  # Complete conversation history (query-response pairs)
    current_context: str  # Current context for the agent
    last_accessed: datetime  # When this state was last accessed


@dataclass
class AgentMetrics:
    """Metrics for tracking agent performance and accuracy."""

    total_queries: int  # Number of queries processed
    successful_queries: int  # Number of queries that returned results
    avg_response_time: float  # Average query response time in seconds
    avg_confidence_score: float  # Average confidence in responses (0-1)
    metadata_accuracy: float  # Percentage of results with complete metadata (0-100)
    relevance_accuracy: float  # Percentage of results that are semantically relevant (0-100)
    consistency_rate: float  # Percentage of consistent results across repeated queries (0-100)
    timestamp: datetime  # When metrics were collected


@dataclass
class AgentResult:
    """Represents the result of an agent interaction for validation purposes."""

    agent_request: AgentRequest
    agent_response: AgentResponse
    metadata_preserved: bool  # Whether all metadata fields are present and valid
    relevance_score: float  # Human-validated relevance score (0-1)
    consistency_score: float  # For repeated queries, similarity of results (0-1)
    validation_timestamp: datetime  # When validation was performed


@dataclass
class RetrievalTool:
    """Represents the vector retrieval tool for the OpenAI Agents SDK."""

    name: str = "vector_retrieval"
    description: str = "Retrieve relevant content chunks from the vector database based on semantic similarity"
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize the parameters for the retrieval tool."""
        if self.parameters is None:
            self.parameters = {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant content"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve (default: 5)",
                        "default": 5
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Whether to include metadata in results (default: true)",
                        "default": True
                    }
                },
                "required": ["query"]
            }