"""
API Service for RAG Agent Integration

This module implements the FastAPI endpoints for the RAG agent integration,
allowing the frontend to send queries and receive grounded responses with
source citations.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator

from src.rag_agent.agent import create_rag_agent
from src.rag_agent.models import AgentRequest, AgentResponse, ContentChunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router for RAG agent endpoints
router = APIRouter(prefix="", tags=["rag-agent"])

# Request models
class UserQueryRequest(BaseModel):
    """Request model for user queries to the RAG agent."""
    query_text: str = Field(..., min_length=1, max_length=1000, description="The user's query text (500-1000 character limit)")
    selected_text: Optional[str] = Field(None, max_length=5000, description="Optional selected text from the book content (included as context in the query)")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve (default 5)")
    include_citations: bool = Field(True, description="Whether to include source citations and links in the response")

    @validator('query_text')
    def validate_query_length(cls, v):
        if len(v) > 1000:
            raise ValueError('Query exceeds 1000 character limit. Please shorten your question.')
        if len(v) < 1:
            raise ValueError('Query cannot be empty.')
        return v

# Response models
class ContentChunkResponse(BaseModel):
    """Response model for content chunks."""
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    rank: int

class RAGResponse(BaseModel):
    """Response model for RAG agent queries."""
    query_text: str
    answer: str
    retrieved_chunks: list[ContentChunkResponse]
    confidence_score: float
    execution_time: float
    timestamp: str

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None

# Initialize the RAG agent
rag_agent = create_rag_agent()

@router.post("/query",
             summary="Process a user query through the RAG agent",
             description="Sends a user query to the RAG agent and returns a grounded response based on book content with source citations and links",
             response_model=RAGResponse,
             responses={
                 200: {"description": "Successful response from the RAG agent with source citations"},
                 400: {"description": "Invalid request parameters (e.g., query exceeds character limit)"},
                 500: {"description": "Internal server error"}
             })
async def process_query(request: UserQueryRequest):
    """
    Process a user query through the RAG agent.

    This endpoint receives a query from the frontend, processes it using the RAG agent,
    and returns a response with source citations and links to original content.
    """
    try:
        logger.info(f"Processing query: {request.query_text[:50]}...")

        # Combine the query with selected text if provided
        final_query = request.query_text
        if request.selected_text:
            final_query = f"Context: {request.selected_text}\n\nQuestion: {request.query_text}"

        # Process the query using the RAG agent
        response = rag_agent.process_query_with_agents_sdk(
            query=final_query,
            top_k=request.top_k
        )

        logger.info(f"Query processed successfully, retrieved {len(response.retrieved_chunks)} chunks")

        # Return the response with proper structure
        return RAGResponse(
            query_text=request.query_text,
            answer=response.answer,
            retrieved_chunks=[
                ContentChunkResponse(
                    content=chunk.content,
                    similarity_score=chunk.similarity_score,
                    metadata=chunk.metadata,
                    rank=chunk.rank
                ) for chunk in response.retrieved_chunks
            ],
            confidence_score=response.confidence_score,
            execution_time=response.execution_time,
            timestamp=response.timestamp.isoformat()
        )

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "VALIDATION_ERROR",
                "message": str(ve)
            }
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "QUERY_PROCESSING_ERROR",
                "message": "Failed to process query due to internal error. Please try again."
            }
        )

@router.get("/health",
            summary="Check the health status of the RAG agent",
            description="Returns the health status of the RAG agent service",
            response_model=HealthResponse,
            responses={
                200: {"description": "RAG agent is healthy"}
            })
async def health_check():
    """
    Health check endpoint to verify the RAG agent service is running.
    """
    try:
        # Simple check - we could add more sophisticated checks here
        health_status = HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            details={
                "rag_agent_initialized": rag_agent is not None
            }
        )
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unavailable",
            timestamp=datetime.now().isoformat(),
            details={
                "error": str(e)
            }
        )

# Additional endpoints can be added here as needed
