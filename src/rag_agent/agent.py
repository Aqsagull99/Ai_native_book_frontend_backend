"""
RAG Agent implementation using OpenAI Agents Python SDK
Implements the core agent logic for answering book-related questions using vector retrieval
Following the documentation at https://openai.github.io/openai-agents-python
Uses OpenRouter by default (with fallback to Google Gemini) via OpenAI-compatible endpoint
"""

import time
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunConfig, function_tool
import os
from dotenv import load_dotenv

from src.rag_agent.models import AgentRequest, AgentResponse, ContentChunk
from src.rag_agent.qdrant_service import search_vectors
from src.content_embedding.retrieval_service import create_query_embedding
from src.rag_agent.config import Config
from src.rag_agent.performance_monitor import (
    track_performance, record_retrieval_start, record_retrieval_end,
    record_generation_start, record_generation_end, record_response
)

# Disable tracing to avoid requiring OPENAI_API_KEY for telemetry
set_tracing_disabled(disabled=True)

# Load environment variables
load_dotenv()

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also set up a separate logger for metrics
metrics_logger = logging.getLogger('agent.metrics')


def vector_retrieval_function(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant content chunks from the vector database based on semantic similarity.

    Args:
        query: The search query to find relevant content
        top_k: Number of results to retrieve (default: 5)

    Returns:
        List of retrieved content chunks with their metadata
    """
    try:
        logger.info(f"Executing vector retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")

        # Create embedding for the query text
        query_vector = create_query_embedding(query)

        # Perform semantic search using Qdrant
        search_results = search_vectors(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=None,  # No additional filters for now
            collection_name=Config.QDRANT_COLLECTION_NAME
        )

        # Format results
        formatted_results = []
        for i, result in enumerate(search_results):
            content_chunk = {
                "content": result["content"],
                "similarity_score": result["similarity_score"],
                "metadata": result["metadata"],
                "rank": i + 1
            }
            formatted_results.append(content_chunk)

        logger.info(f"Retrieved {len(formatted_results)} results for query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
        return formatted_results

    except Exception as e:
        logger.error(f"Error in vector retrieval function: {str(e)}")
        raise

@function_tool
def vector_retrieval_tool(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant content chunks from the vector database based on semantic similarity.
    This is the tool version for the agent.

    Args:
        query: The search query to find relevant content
        top_k: Number of results to retrieve (default: 5)

    Returns:
        List of retrieved content chunks with their metadata
    """
    return vector_retrieval_function(query, top_k)


class RAGAgent:
    """
    RAG Agent class that implements the core functionality for answering book-related questions
    using vector retrieval and the OpenAI Agents Python SDK with OpenRouter (fallback to Google Gemini) via OpenAI-compatible endpoint.
    """

    def __init__(self):
        """Initialize the RAG agent with configuration and clients."""
        # Validate configuration
        is_valid, error_msg = Config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error_msg}")

        # Prioritize OpenRouter configuration (as per user's request to use OpenRouter)
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

        if openrouter_api_key:
            # Use OpenRouter API
            logger.info("Using OpenRouter API for LLM service")
            os.environ["OPENAI_API_KEY"] = openrouter_api_key

            # Create AsyncOpenAI client with OpenRouter API endpoint
            external_client = AsyncOpenAI(
                api_key=openrouter_api_key,
                base_url=Config.OPENROUTER_BASE_URL,
            )

            # Create the model using OpenAIChatCompletionsModel with the external client
            model = OpenAIChatCompletionsModel(
                model=Config.OPENROUTER_MODEL,  # Use the configured OpenRouter model
                openai_client=external_client
            )

            logger.info(f"✅ RAG Agent initialized with OpenRouter using model: {Config.OPENROUTER_MODEL}")
        else:
            # Fallback to Google Gemini if OpenRouter is not configured
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("Neither OPENROUTER_API_KEY nor GEMINI_API_KEY environment variable is available")

            logger.info("Falling back to Google Gemini API for LLM service")
            os.environ["OPENAI_API_KEY"] = gemini_api_key

            # Create AsyncOpenAI client with Google Gemini API endpoint
            external_client = AsyncOpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )

            # Create the model using OpenAIChatCompletionsModel with the external client
            model = OpenAIChatCompletionsModel(
                model=Config.GEMINI_MODEL_NAME,  # Use the configured Gemini model
                openai_client=external_client
            )

            logger.info(f"✅ RAG Agent initialized with Google Gemini using model: {Config.GEMINI_MODEL_NAME}")

        # Create the main RAG agent using OpenAI Agents SDK
        self.config = RunConfig(
            model=model,
            model_provider=external_client,
            tracing_disabled=True  # Disable tracing to avoid requiring OPENAI_API_KEY for telemetry
        )

        self.agent = Agent(
            name="RAG Book Assistant",
            instructions="You are a helpful RAG agent that answers questions about books. Use the vector_retrieval tool to find relevant information before answering. Focus on providing accurate, well-cited answers based on the book content.",
            tools=[vector_retrieval_tool]  # Pass the tool function directly
        )

    def process_query_with_agents_sdk(self, query: str, top_k: int = 5) -> AgentResponse:
        """
        Process a query using the OpenAI Agents Python SDK with tool calling through OpenRouter (fallback to Google Gemini) endpoint.

        Args:
            query: The user's query
            top_k: Number of results to retrieve

        Returns:
            AgentResponse with the answer and metadata
        """
        start_time = time.time()
        logger.info(f"Starting query processing with OpenAI Agents Python SDK (OpenRouter/Gemini) for: '{query[:50]}{'...' if len(query) > 50 else ''}' (top_k={top_k})")

        try:
            # Run the agent with the user query using the Google Gemini configuration
            result = Runner.run_sync(
                starting_agent=self.agent,
                input=f"Query: {query}\nPlease retrieve relevant information and provide a comprehensive answer based on the book content. Retrieve {top_k} relevant chunks before answering.",
                run_config=self.config
            )

            # Extract the final output from the agent result
            answer = result.final_output if result.final_output else "I couldn't find sufficient information to answer your question."

            # Get the content for response formatting by calling the retrieval function directly
            tool_results = vector_retrieval_function(query, top_k)

            # Format the retrieved chunks
            retrieved_chunks = []
            for result in tool_results:
                chunk = ContentChunk(
                    content=result["content"],
                    similarity_score=result["similarity_score"],
                    metadata=result["metadata"],
                    rank=result["rank"]
                )
                retrieved_chunks.append(chunk)

            # Calculate confidence score as average similarity of retrieved chunks
            confidence_score = sum(c.similarity_score for c in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0.0

            total_time = time.time() - start_time

            # Create the response
            response = AgentResponse(
                query_text=query,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                confidence_score=confidence_score,
                execution_time=total_time,
                timestamp=datetime.now()
            )

            logger.info(f"Query processed successfully: '{query[:30]}{'...' if len(query) > 30 else ''}' "
                       f"in {total_time:.4f}s with {len(retrieved_chunks)} retrieved chunks, confidence: {confidence_score:.3f}")

            # Log metrics
            metrics_logger.info(f"query_event - query_length={len(query)}, top_k={top_k}, "
                              f"retrieved_chunks={len(retrieved_chunks)}, confidence_score={confidence_score:.3f}, "
                              f"total_time={total_time:.4f}s")

            return response

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Error processing query with OpenAI Agents Python SDK (OpenRouter/Gemini): {str(e)}")
            metrics_logger.error(f"query_error - query_length={len(query)}, total_time={total_time:.4f}s, error={str(e)}")

            # Return an error response
            response = AgentResponse(
                query_text=query,
                answer="Sorry, I encountered an error while processing your query.",
                retrieved_chunks=[],
                confidence_score=0.0,
                execution_time=total_time,
                timestamp=datetime.now()
            )

            return response

    def process_query(self, query: str, top_k: int = 5, include_citations: bool = True) -> AgentResponse:
        """
        Process a user query through the RAG pipeline using the OpenAI Agents SDK with OpenRouter (fallback to Google Gemini).

        Args:
            query: The user's query
            top_k: Number of results to retrieve
            include_citations: Whether to include source citations in the response

        Returns:
            AgentResponse with the answer and metadata
        """
        return self.process_query_with_agents_sdk(query, top_k)


def create_rag_agent() -> RAGAgent:
    """
    Create and return a configured RAG agent instance.

    Returns:
        Configured RAGAgent instance
    """
    return RAGAgent()


def process_agent_request(agent_request: AgentRequest) -> AgentResponse:
    """
    Process an agent request and return the response using the OpenAI Agents SDK with OpenRouter (fallback to Google Gemini).

    Args:
        agent_request: The agent request with query and parameters

    Returns:
        AgentResponse with the answer and metadata
    """
    agent = create_rag_agent()

    return agent.process_query_with_agents_sdk(
        query=agent_request.query_text,
        top_k=agent_request.top_k
    )


# Example usage function
def run_sample_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Run a sample query to demonstrate the OpenAI Agents SDK RAG agent functionality with OpenRouter (fallback to Google Gemini).

    Args:
        query: The query to process
        top_k: Number of results to retrieve

    Returns:
        Dictionary with query results for demonstration
    """
    agent = create_rag_agent()
    response = agent.process_query_with_agents_sdk(query, top_k)

    # Format the response for display
    result = {
        "query": response.query_text,
        "answer": response.answer,
        "retrieved_chunks_count": len(response.retrieved_chunks),
        "confidence_score": response.confidence_score,
        "execution_time": response.execution_time,
        "timestamp": response.timestamp.isoformat(),
        "retrieved_sources": [
            {
                "url": chunk.metadata.get("url", "N/A"),
                "title": chunk.metadata.get("title", "N/A"),
                "similarity_score": chunk.similarity_score,
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            }
            for chunk in response.retrieved_chunks
        ]
    }

    return result







