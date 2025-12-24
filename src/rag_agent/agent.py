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


def is_book_related_query(query: str) -> bool:
    """
    Determine if a query is related to book content by checking for keywords.

    Args:
        query: The search query to analyze

    Returns:
        Boolean indicating if the query is likely book-related
    """
    book_related_keywords = [
        'ai', 'machine learning', 'robotics', 'neural network', 'computer vision',
        'nlp', 'natural language processing', 'deep learning', 'reinforcement learning',
        'algorithm', 'programming', 'code', 'python', 'javascript', 'typescript',
        'html', 'css', 'web development', 'next.js', 'robot', 'automation',
        'artificial intelligence', 'ml', 'data', 'model', 'training', 'learning',
        'book', 'chapter', 'content', 'topic', 'subject', 'course', 'tutorial'
    ]

    query_lower = query.lower()
    # Check if any book-related keywords appear in the query
    for keyword in book_related_keywords:
        if keyword in query_lower:
            return True

    # If no clear book-related keywords, assume it might be author/general query
    return False


def author_info_function(query: str) -> List[Dict[str, Any]]:
    """
    Provide information about Aqsa Gull when queried about the author.

    Args:
        query: The search query to analyze

    Returns:
        List with author information if relevant, empty list otherwise
    """
    author_related_keywords = [
        'aqsa gull', 'author', 'you', 'yourself', 'who are you', 'biography',
        'background', 'experience', 'teaching', 'web developer', 'developer',
        'creator', 'made this', 'wrote this', 'teacher', 'instructor'
    ]

    query_lower = query.lower()

    # Check if the query is about the author
    for keyword in author_related_keywords:
        if keyword in query_lower:
            return [{
                "content": "Aqsa Gull is a passionate web developer, teacher, and author who specializes in simplifying technical concepts. Aqsa has expertise in HTML, CSS, JavaScript, TypeScript, and Next.js, and believes in step-by-step learning approaches that help beginners understand complex topics.",
                "similarity_score": 1.0,
                "metadata": {
                    "url": "N/A",
                    "title": "About the Author",
                    "chunk_index": -1,
                    "source_metadata": {},
                    "created_at": ""
                },
                "rank": 1
            }]

    return []


def general_response_function(query: str) -> List[Dict[str, Any]]:
    """
    Provide general responses for non-book, non-author related queries.

    Args:
        query: The search query to analyze

    Returns:
        List with general information if needed
    """
    # For general queries, return an empty list to indicate no specific content from book
    return []


def smart_retrieval_function(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Smart retrieval function that can classify query type and respond appropriately.

    Args:
        query: The search query to find relevant content
        top_k: Number of results to retrieve (default: 5)

    Returns:
        List of retrieved content chunks with their metadata
    """
    try:
        logger.info(f"Executing smart retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

        # Check if it's an author-related query first
        author_results = author_info_function(query)
        if author_results:
            logger.info(f"Author-related query detected: '{query[:30]}{'...' if len(query) > 30 else ''}'")
            return author_results

        # Check if it's a book-related query
        if is_book_related_query(query):
            logger.info(f"Book-related query detected: '{query[:30]}{'...' if len(query) > 30 else ''}'")

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

            logger.info(f"Retrieved {len(formatted_results)} results for book-related query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
            return formatted_results
        else:
            # For general queries, provide general response
            logger.info(f"General query detected: '{query[:30]}{'...' if len(query) > 30 else ''}'")
            return general_response_function(query)

    except Exception as e:
        logger.error(f"Error in smart retrieval function: {str(e)}")
        # Fallback to vector retrieval in case of error
        return vector_retrieval_function(query, top_k)


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
    return smart_retrieval_function(query, top_k)


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
            name="Aqsa Gull AI Assistant",
            instructions="""You are an intelligent, friendly, and professional AI assistant representing Aqsa Gull — a talented author, web developer, and teacher. You are not just a book assistant. You are Aqsa Gull's digital reflection.

Your responsibilities:
1. Answer questions related to the book content accurately and clearly.
2. Answer general user queries even if they are outside the book.
3. Represent Aqsa Gull's professional background: web developer (HTML, CSS, JavaScript, TypeScript, Next.js), teacher and mentor for beginners, author of this book.
4. Maintain a friendly, motivating, and beginner-friendly tone.

PERSONALITY & TONE:
- Be friendly and polite
- Use light humor when appropriate
- Be confident but humble
- Be motivational for students
- Use simple language (avoid over-complex explanations)

Example tone: "Don't worry 😊 programming sab ko mushkil lagti hai pehle."

BOOK CONTENT HANDLING:
- If a query is related to the book → answer strictly based on book content.
- If the book does not cover the question: Say so politely, then provide a general helpful explanation
Example: "This topic is not directly covered in the book, but here's a simple explanation to help you…"

ABOUT AQSA GULL (AUTHOR CONTEXT):
- Aqsa Gull is a passionate web developer, a teacher who simplifies technical concepts, and an author who believes in step-by-step learning. Stay authentic and do not exaggerate.

FUN & CASUAL QUERIES:
Handle casual or funny questions with light, respectful humor.
Examples:
- "Are you human?" → "I'm AI 😄 but trained with human logic."
- "Are you smarter than Aqsa?" → "No one beats the creator 😉"
Avoid offensive or inappropriate humor.

MOTIVATION & STUDENT SUPPORT:
If a user feels confused or demotivated, encourage them, normalize struggle, and emphasize practice and patience.
Example: "Mistakes are part of learning 💙 even professional developers face bugs daily."

RESPONSE RULES:
- Keep answers short and clear
- Use bullet points when helpful
- Avoid unnecessary technical jargon
- Be respectful at all times

SAFETY & BOUNDARIES:
- Do not provide harmful, illegal, or misleading information.
- Do not claim to be human.
- If unsure, respond honestly and guide politely.

FINAL GOAL: Make users feel supported, motivated, confident in learning, and connected to Aqsa Gull's teaching style. You are not just answering questions — you are guiding, inspiring, and simplifying.""",
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
        import asyncio

        start_time = time.time()
        logger.info(f"Starting query processing with OpenAI Agents Python SDK (OpenRouter/Gemini) for: '{query[:50]}{'...' if len(query) > 50 else ''}' (top_k={top_k})")

        try:
            # Check if we're in a thread without an event loop and handle appropriately
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # If we get here, we're in an async context
                result = Runner.run_sync(
                    starting_agent=self.agent,
                    input=f"Query: {query}\nPlease retrieve relevant information and provide a comprehensive answer based on the book content. Retrieve {top_k} relevant chunks before answering.",
                    run_config=self.config
                )
            except RuntimeError:
                # No event loop running, we're in a thread - create a new event loop for this thread
                # This handles the case when running in a thread created by asyncio.to_thread()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = Runner.run_sync(
                        starting_agent=self.agent,
                        input=f"Query: {query}\nPlease retrieve relevant information and provide a comprehensive answer based on the book content. Retrieve {top_k} relevant chunks before answering.",
                        run_config=self.config
                    )
                finally:
                    loop.close()
                    # Reset the event loop for this thread
                    asyncio.set_event_loop(None)

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
            logger.error(f"Full error details: {repr(e)}")
            metrics_logger.error(f"query_error - query_length={len(query)}, total_time={total_time:.4f}s, error={str(e)}")

            # Return an error response
            # For debugging in deployment, we'll return a more descriptive error
            error_msg = f"Sorry, I encountered an error while processing your query: {type(e).__name__}"
            if "429" in str(e) or "Too Many Requests" in str(e):
                error_msg = "Service temporarily unavailable due to rate limits. Please try again later."
            elif "API" in str(e) or "Key" in str(e) or "Authentication" in str(e):
                error_msg = "Service temporarily unavailable due to authentication issues."

            response = AgentResponse(
                query_text=query,
                answer=error_msg,
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







