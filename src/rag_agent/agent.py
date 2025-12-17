"""
RAG Agent implementation using OpenAI Agents SDK
Implements the core agent logic for answering book-related questions using vector retrieval
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import OpenAI

from src.rag_agent.models import AgentRequest, AgentResponse, ContentChunk
from src.rag_agent.retrieval_tool import VectorRetrievalTool
from src.rag_agent.llm_service import GeminiClientService
from src.rag_agent.config import Config
from src.rag_agent.openai_agents_sdk import OpenAIAgentsSDK
from src.rag_agent.performance_monitor import (
    track_performance, record_retrieval_start, record_retrieval_end,
    record_generation_start, record_generation_end, record_response
)

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also set up a separate logger for metrics
metrics_logger = logging.getLogger('agent.metrics')


class RAGAgent:
    """
    RAG Agent class that implements the core functionality for answering book-related questions
    using vector retrieval and the OpenAI Agents SDK with Google Gemini.
    """

    def __init__(self):
        """Initialize the RAG agent with configuration and clients."""
        # Validate configuration
        is_valid, error_msg = Config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error_msg}")

        # Initialize OpenAI Agents SDK with MCP context 7
        self.openai_agents_sdk = OpenAIAgentsSDK()

        # Initialize the vector retrieval tool
        self.retrieval_tool = VectorRetrievalTool()

        # Initialize the Gemini client service
        self.gemini_client = GeminiClientService()

        # Initialize OpenAI client for assistant functionality (lazily)
        self._openai_client = None
        self._assistant = None
        self._thread_id = None

        logger.info(f"✅ RAG Agent initialized with OpenAI Agents SDK and MCP Context {Config.MCP_CONTEXT_ID}")

    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables. OpenAI functionality will be limited.")
                return None
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    @property
    def assistant(self):
        """Lazy initialization of assistant."""
        return self._assistant

    @assistant.setter
    def assistant(self, value):
        self._assistant = value

    @property
    def thread_id(self):
        """Lazy initialization of thread_id."""
        return self._thread_id

    @thread_id.setter
    def thread_id(self, value):
        self._thread_id = value

    def initialize_assistant(self):
        """Initialize the OpenAI assistant with the retrieval tool."""
        try:
            # Check if OpenAI API key is available
            if not self.openai_client:
                logger.warning("OpenAI API key not available. Skipping assistant initialization.")
                return

            # Create assistant with the retrieval tool
            self.assistant = self.openai_client.beta.assistants.create(
                name="RAG Book Assistant",
                instructions="You are a helpful RAG agent that answers questions about books. Use the vector_retrieval tool to find relevant information before answering.",
                model="gpt-4-turbo",  # Using GPT as the underlying model for the assistant
                tools=[self.retrieval_tool.get_tool_spec()]
            )

            # Create a thread for the conversation
            thread = self.openai_client.beta.threads.create()
            self.thread_id = thread.id

            logger.info(f"✅ Assistant initialized with ID: {self.assistant.id}")
            logger.info(f"✅ Thread created with ID: {self.thread_id}")

        except Exception as e:
            logger.error(f"Error initializing assistant: {str(e)}")
            raise

    def retrieve_content(self, query: str, top_k: int = 5) -> List[ContentChunk]:
        """
        Retrieve relevant content chunks using the vector retrieval tool.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of ContentChunk objects with relevant content
        """
        start_time = time.time()
        logger.info(f"Starting content retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}' (top_k={top_k})")

        # Create a dummy request for tracking (since we might not have the full AgentRequest object)
        dummy_request = AgentRequest(query_text=query, top_k=top_k)
        perf_session_id = track_performance(dummy_request)
        record_retrieval_start(perf_session_id)

        try:
            # Use the vector retrieval tool to get results
            raw_results = self.retrieval_tool.call(query, top_k=top_k)

            # Convert raw results to ContentChunk objects
            content_chunks = []
            for result in raw_results:
                chunk = ContentChunk(
                    content=result["content"],
                    similarity_score=result["similarity_score"],
                    metadata=result["metadata"],
                    rank=result["rank"]
                )
                content_chunks.append(chunk)

            retrieval_time = time.time() - start_time
            record_retrieval_end(perf_session_id)

            logger.info(f"Retrieved {len(content_chunks)} content chunks for query: '{query[:30]}{'...' if len(query) > 30 else ''}' in {retrieval_time:.4f}s")

            # Log metrics
            metrics_logger.info(f"retrieval_event - query_length={len(query)}, top_k={top_k}, results_count={len(content_chunks)}, retrieval_time={retrieval_time:.4f}s")

            return content_chunks

        except Exception as e:
            retrieval_time = time.time() - start_time
            record_retrieval_end(perf_session_id)
            logger.error(f"Error during content retrieval for query '{query[:30]}{'...' if len(query) > 30 else ''}': {str(e)}")
            metrics_logger.error(f"retrieval_error - query_length={len(query)}, retrieval_time={retrieval_time:.4f}s, error={str(e)}")
            return []

    def generate_answer(self, query: str, context_chunks: List[ContentChunk]) -> str:
        """
        Generate an answer based on the query and retrieved context.

        Args:
            query: The original user query
            context_chunks: List of relevant content chunks

        Returns:
            Generated answer as a string
        """
        start_time = time.time()
        logger.info(f"Starting answer generation for query: '{query[:50]}{'...' if len(query) > 50 else ''}' with {len(context_chunks)} context chunks")

        # Create a dummy request for tracking
        dummy_request = AgentRequest(query_text=query, top_k=len(context_chunks))
        perf_session_id = track_performance(dummy_request)
        record_generation_start(perf_session_id)

        try:
            # Use the Gemini client service to generate the answer
            answer = self.gemini_client.generate_response(query, context_chunks)

            generation_time = time.time() - start_time
            record_generation_end(perf_session_id)

            logger.info(f"Answer generated successfully for query: '{query[:30]}{'...' if len(query) > 30 else ''}' in {generation_time:.4f}s")
            logger.debug(f"Generated answer preview: '{answer[:100]}{'...' if len(answer) > 100 else ''}'")

            # Log metrics
            metrics_logger.info(f"generation_event - query_length={len(query)}, context_chunks={len(context_chunks)}, answer_length={len(answer)}, generation_time={generation_time:.4f}s")

            return answer

        except Exception as e:
            generation_time = time.time() - start_time
            record_generation_end(perf_session_id)
            logger.error(f"Error during answer generation for query '{query[:30]}{'...' if len(query) > 30 else ''}': {str(e)}")
            metrics_logger.error(f"generation_error - query_length={len(query)}, context_chunks={len(context_chunks)}, generation_time={generation_time:.4f}s, error={str(e)}")
            return "Sorry, I encountered an error while generating the answer."

    def process_query_with_agents_sdk(self, query: str, top_k: int = 5) -> AgentResponse:
        """
        Process a query using the OpenAI Agents SDK with tool calling.

        Args:
            query: The user's query
            top_k: Number of results to retrieve

        Returns:
            AgentResponse with the answer and metadata
        """
        start_time = time.time()
        logger.info(f"Starting query processing with OpenAI Agents SDK for: '{query[:50]}{'...' if len(query) > 50 else ''}' (top_k={top_k})")

        # Create agent request for tracking
        agent_request = AgentRequest(query_text=query, top_k=top_k)
        perf_session_id = track_performance(agent_request)

        # Check if OpenAI API key is available
        if not self.openai_client:
            logger.warning("OpenAI API key not available. Using direct RAG processing instead of OpenAI Agents SDK.")
            # Fallback to direct retrieval and generation
            return self.process_query(query, top_k)

        # Initialize assistant if not already done
        if not self.assistant:
            logger.info("Initializing assistant for the first time")
            self.initialize_assistant()

        # If assistant still isn't initialized (due to missing API key), use direct processing
        if not self.assistant:
            logger.warning("Assistant not initialized. Using direct RAG processing.")
            return self.process_query(query, top_k)

        try:
            # Add the user message to the thread
            logger.debug(f"Adding message to thread {self.thread_id}")
            message = self.openai_client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=query
            )

            # Run the assistant
            logger.debug("Starting assistant run with vector retrieval tool")
            run = self.openai_client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant.id,
                # Force tool use for vector retrieval
                tools=[self.retrieval_tool.get_tool_spec()]
            )

            # Wait for the run to complete
            import time as time_module
            assistant_start_time = time_module.time()
            while run.status in ["queued", "in_progress"]:
                time_module.sleep(1)
                run = self.openai_client.beta.threads.runs.retrieve(
                    thread_id=self.thread_id,
                    run_id=run.id
                )

            assistant_time = time_module.time() - assistant_start_time
            logger.debug(f"Assistant completed with status: {run.status} in {assistant_time:.4f}s")

            # Get the messages from the thread
            messages = self.openai_client.beta.threads.messages.list(
                thread_id=self.thread_id,
                order="desc"
            )

            # Extract the assistant's response
            answer = ""
            if messages.data:
                answer = messages.data[0].content[0].text.value

            # For now, retrieve content separately to populate the response
            context_chunks = self.retrieve_content(query, top_k)

            # Calculate confidence score as average similarity of retrieved chunks
            confidence_score = sum(c.similarity_score for c in context_chunks) / len(context_chunks) if context_chunks else 0.0

            total_time = time.time() - start_time

            # Create the response
            response = AgentResponse(
                query_text=query,
                answer=answer or "I couldn't find sufficient information to answer your question.",
                retrieved_chunks=context_chunks,
                confidence_score=confidence_score,
                execution_time=total_time,
                timestamp=datetime.now()
            )

            # Record the response with performance tracking
            record_response(perf_session_id, response)

            logger.info(f"Query processed successfully: '{query[:30]}{'...' if len(query) > 30 else ''}' "
                       f"in {total_time:.4f}s with {len(context_chunks)} retrieved chunks, confidence: {confidence_score:.3f}")

            # Log metrics
            metrics_logger.info(f"query_event - query_length={len(query)}, top_k={top_k}, "
                              f"retrieved_chunks={len(context_chunks)}, confidence_score={confidence_score:.3f}, "
                              f"total_time={total_time:.4f}s, assistant_time={assistant_time:.4f}s")

            return response

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Error processing query with OpenAI Agents SDK: {str(e)}")
            metrics_logger.error(f"query_error - query_length={len(query)}, total_time={total_time:.4f}s, error={str(e)}")

            # Fallback to direct retrieval and generation
            logger.info("Falling back to direct retrieval and generation")
            return self.process_query(query, top_k)

    def process_query(self, query: str, top_k: int = 5, include_citations: bool = True) -> AgentResponse:
        """
        Process a user query through the RAG pipeline.

        Args:
            query: The user's query
            top_k: Number of results to retrieve
            include_citations: Whether to include source citations in the response

        Returns:
            AgentResponse with the answer and metadata
        """
        start_time = time.time()

        # Retrieve relevant content
        context_chunks = self.retrieve_content(query, top_k)

        # Generate answer based on retrieved content
        answer = self.generate_answer(query, context_chunks)

        # Calculate confidence score as average similarity of retrieved chunks
        confidence_score = sum(c.similarity_score for c in context_chunks) / len(context_chunks) if context_chunks else 0.0

        # Create the response
        response = AgentResponse(
            query_text=query,
            answer=answer,
            retrieved_chunks=context_chunks,
            confidence_score=confidence_score,
            execution_time=time.time() - start_time,
            timestamp=datetime.now()
        )

        logger.info(f"Processed query: '{query[:50]}{'...' if len(query) > 50 else ''}' "
                   f"in {response.execution_time:.4f}s with {len(context_chunks)} retrieved chunks")

        return response


def create_rag_agent() -> RAGAgent:
    """
    Create and return a configured RAG agent instance.

    Returns:
        Configured RAGAgent instance
    """
    return RAGAgent()


def process_agent_request(agent_request: AgentRequest) -> AgentResponse:
    """
    Process an agent request and return the response.

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
    Run a sample query to demonstrate the RAG agent functionality.

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