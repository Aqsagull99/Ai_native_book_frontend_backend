"""
API Service for RAG Agent Integration

This module implements the FastAPI endpoints for the RAG agent integration,
allowing the frontend to send queries and receive grounded responses with
source citations.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import logging
import uuid
import asyncio

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator
import asyncio
from src.rag_agent.models import AgentRequest, AgentResponse, ContentChunk
from src.rag_agent.rate_limiter import get_rate_limiter
from src.rag_agent.config import Config

# Import agent functions but handle initialization errors gracefully
# Global variable to cache the agent or error state
_cached_agent = None
_agent_init_error = None

def create_rag_agent():
    global _cached_agent, _agent_init_error

    # Return cached agent if already created successfully
    if _cached_agent is not None:
        return _cached_agent

    # Return cached error agent if initialization already failed
    if _agent_init_error is not None:
        return _agent_init_error

    try:
        from src.rag_agent.agent import RAGAgent
        agent = RAGAgent()
        _cached_agent = agent
        return agent
    except Exception as e:
        logger.error(f"Failed to create RAG agent: {str(e)}")
        error_msg = str(e)  # Store the error message in a variable that can be accessed in the nested function
        _agent_init_error = e
        # Return a mock agent that handles errors gracefully
        class MockRAGAgent:
            def process_query_with_agents_sdk(self, query: str, top_k: int = 5):
                from datetime import datetime
                from src.rag_agent.models import AgentResponse, ContentChunk
                return AgentResponse(
                    query_text=query,
                    answer=f"Service unavailable: {error_msg}",
                    retrieved_chunks=[],
                    confidence_score=0.0,
                    execution_time=0.0,
                    timestamp=datetime.now()
                )
        _agent_init_error = MockRAGAgent()
        return _agent_init_error

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

# Initialize the RAG agent (will be created per request to avoid event loop issues)
# rag_agent = create_rag_agent()

# In-memory job store for async query processing (ephemeral; use Redis or DB for production)
_job_store: dict[str, dict] = {}

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
        # Check rate limit before processing the query
        if Config.RATE_LIMIT_ENABLED:
            rate_limiter = get_rate_limiter()
            if not rate_limiter.increment_request():
                # Rate limit exceeded
                usage_info = rate_limiter.get_usage_info()
                logger.warning(f"Rate limit exceeded: {usage_info['total_requests']}/{usage_info['daily_limit']} requests")

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": f"Daily request limit exceeded. {usage_info['daily_limit']} requests per day allowed.",
                        "usage_info": usage_info
                    }
                )

        logger.info(f"Processing query: {request.query_text[:50]}...")

        # Create a new agent instance for this request
        rag_agent = create_rag_agent()

        # Combine the query with selected text if provided
        final_query = request.query_text
        if request.selected_text:
            final_query = f"Context: {request.selected_text}\n\nQuestion: {request.query_text}"

        # Process the query using asyncio.to_thread to avoid event loop issues
        # This ensures the synchronous agent processing runs in a separate thread
        import asyncio
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    rag_agent.process_query_with_agents_sdk,
                    final_query,
                    request.top_k
                ),
                timeout=60.0  # 60 second timeout
            )
        except asyncio.TimeoutError:
            logger.error("Query processing timed out after 60 seconds")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail={
                    "error": "QUERY_TIMEOUT",
                    "message": "Query processing timed out. Please try again with a simpler query."
                }
            )
        except RuntimeError as e:
            if "There is no current event loop in thread" in str(e):
                logger.error(f"Event loop issue in thread: {str(e)}")
                # This shouldn't happen with our agent fixes, but adding extra handling
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "EVENT_LOOP_ERROR",
                        "message": "Internal event loop error occurred"
                    }
                )
            else:
                raise

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
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Ensure we always return a valid JSON response
        error_detail = {
            "error": "QUERY_PROCESSING_ERROR",
            "message": f"Failed to process query due to internal error: {str(e)}"
        }
        # Log the actual error for debugging
        logger.error(f"Full error details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


@router.post("/query-async",
             summary="Enqueue a user query for async processing",
             description="Returns a job id which can be polled for results",
             responses={
                 202: {"description": "Job accepted"},
                 400: {"description": "Invalid request"}
             })
async def process_query_async(request: UserQueryRequest):
    # Check rate limit before processing the query
    if Config.RATE_LIMIT_ENABLED:
        rate_limiter = get_rate_limiter()
        if not rate_limiter.increment_request():
            # Rate limit exceeded
            usage_info = rate_limiter.get_usage_info()
            logger.warning(f"Rate limit exceeded for async query: {usage_info['total_requests']}/{usage_info['daily_limit']} requests")

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Daily request limit exceeded. {usage_info['daily_limit']} requests per day allowed.",
                    "usage_info": usage_info
                }
            )

    # Create job id and store pending state
    job_id = str(uuid.uuid4())
    _job_store[job_id] = {"status": "pending", "created_at": datetime.utcnow().isoformat()}

    # Prepare final query string
    final_query = request.query_text
    if request.selected_text:
        final_query = f"Context: {request.selected_text}\n\nQuestion: {request.query_text}"

    async def _run_job(jid: str, query_text: str, top_k: int):
        try:
            rag_agent = create_rag_agent()
            # Run the possibly-blocking processing in a thread to avoid blocking the event loop
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        rag_agent.process_query_with_agents_sdk,
                        query_text,
                        top_k
                    ),
                    timeout=60.0  # 60 second timeout
                )
            except RuntimeError as e:
                if "There is no current event loop in thread" in str(e):
                    logger.error(f"Event loop issue in async job thread: {str(e)}")
                    raise Exception(f"Event loop error in async processing: {str(e)}")
                else:
                    raise

            # Serialize response into a simple dict
            result = {
                "query_text": request.query_text,
                "answer": response.answer,
                "retrieved_chunks": [
                    {
                        "content": c.content,
                        "similarity_score": c.similarity_score,
                        "metadata": c.metadata,
                        "rank": c.rank
                    } for c in response.retrieved_chunks
                ],
                "confidence_score": response.confidence_score,
                "execution_time": response.execution_time,
                "timestamp": response.timestamp.isoformat() if hasattr(response, 'timestamp') else datetime.utcnow().isoformat()
            }

            _job_store[jid]["status"] = "completed"
            _job_store[jid]["result"] = result
        except Exception as e:
            logger.error(f"Async job {jid} failed: {e}")
            _job_store[jid]["status"] = "failed"
            _job_store[jid]["error"] = str(e)

    # Schedule background job
    asyncio.create_task(_run_job(job_id, final_query, request.top_k))

    # Return 202 Accepted with job id
    return {"job_id": job_id, "status": "pending", "poll_url": f"/api/rag/query/result/{job_id}"}


@router.get("/query/result/{job_id}", summary="Get async query result")
async def get_query_result(job_id: str):
    job = _job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "JOB_NOT_FOUND", "message": "Job id not found"})

    # Return status and result/error if present
    resp = {"job_id": job_id, "status": job.get("status")}
    if job.get("status") == "completed":
        resp["result"] = job.get("result")
    elif job.get("status") == "failed":
        resp["error"] = job.get("error")

    return resp

@router.get("/rate-limit-status",
            summary="Check the current rate limit status",
            description="Returns the current usage and remaining requests for the daily rate limit",
            responses={
                200: {"description": "Rate limit status information"}
            })
async def rate_limit_status():
    """
    Rate limit status endpoint to check current usage and remaining requests.
    """
    try:
        if Config.RATE_LIMIT_ENABLED:
            rate_limiter = get_rate_limiter()
            usage_info = rate_limiter.get_usage_info()

            return {
                "status": "rate_limiting_enabled",
                "usage_info": usage_info,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "rate_limiting_disabled",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Rate limit status check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "RATE_LIMIT_STATUS_ERROR",
                "message": "Rate limit status check failed"
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
                "rag_agent_initialized": True,
                "rate_limiting_enabled": Config.RATE_LIMIT_ENABLED
            }
        )
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        # Ensure we always return a valid JSON response even on error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "HEALTH_CHECK_ERROR",
                "message": "Health check failed"
            }
        )

# Additional endpoints can be added here as needed
