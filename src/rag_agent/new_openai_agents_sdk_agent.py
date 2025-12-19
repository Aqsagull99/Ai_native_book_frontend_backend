# """
# New RAG Agent Implementation using OpenAI Agents Python SDK
# Following the documentation at https://openai.github.io/openai-agents-python
# """

# import os
# import logging
# from typing import Dict, Any, List, Optional
# from datetime import datetime

# from agents import Agent, Runner, handoff, function_tool

# from src.rag_agent.models import AgentRequest, AgentResponse, ContentChunk
# from src.rag_agent.qdrant_service import search_vectors
# from src.content_embedding.retrieval_service import create_query_embedding
# from src.rag_agent.config import Config
# from src.rag_agent.llm_service import GeminiClientService

# # Set up comprehensive logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Also set up a separate logger for metrics
# metrics_logger = logging.getLogger('agent.metrics')


# @function_tool
# def vector_retrieval_tool(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#     """
#     Retrieve relevant content chunks from the vector database based on semantic similarity.

#     Args:
#         query: The search query to find relevant content
#         top_k: Number of results to retrieve (default: 5)

#     Returns:
#         List of retrieved content chunks with their metadata
#     """
#     try:
#         logger.info(f"Executing vector retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

#         # Validate inputs
#         if not query or not query.strip():
#             raise ValueError("Query cannot be empty")

#         if top_k <= 0 or top_k > 100:
#             raise ValueError("top_k must be between 1 and 100")

#         # Create embedding for the query text
#         query_vector = create_query_embedding(query)

#         # Perform semantic search using Qdrant
#         search_results = search_vectors(
#             query_vector=query_vector,
#             top_k=top_k,
#             metadata_filter=None,  # No additional filters for now
#             collection_name=Config.QDRANT_COLLECTION_NAME
#         )

#         # Format results
#         formatted_results = []
#         for i, result in enumerate(search_results):
#             content_chunk = {
#                 "content": result["content"],
#                 "similarity_score": result["similarity_score"],
#                 "metadata": result["metadata"],
#                 "rank": i + 1
#             }
#             formatted_results.append(content_chunk)

#         logger.info(f"Retrieved {len(formatted_results)} results for query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
#         return formatted_results

#     except Exception as e:
#         logger.error(f"Error in vector retrieval tool: {str(e)}")
#         raise


# class OpenAIAgentsSDKRAGAgent:
#     """
#     RAG Agent class that implements the core functionality for answering book-related questions
#     using vector retrieval and the OpenAI Agents Python SDK with Google Gemini.
#     """

#     def __init__(self):
#         """Initialize the RAG agent with configuration and clients."""
#         # Validate configuration
#         is_valid, error_msg = Config.validate()
#         if not is_valid:
#             raise ValueError(f"Invalid configuration: {error_msg}")

#         # Initialize the Gemini client service
#         self.gemini_client = GeminiClientService()

#         # Create the main RAG agent
#         self.rag_agent = Agent(
#             name="RAG Book Assistant",
#             instructions="You are a helpful RAG agent that answers questions about books. Use the vector_retrieval tool to find relevant information before answering.",
#             tools=[vector_retrieval_tool]
#         )

#         logger.info("✅ OpenAI Agents SDK RAG Agent initialized successfully")

#     def process_query(self, query: str, top_k: int = 5) -> AgentResponse:
#         """
#         Process a user query through the RAG pipeline using the OpenAI Agents SDK.

#         Args:
#             query: The user's query
#             top_k: Number of results to retrieve

#         Returns:
#             AgentResponse with the answer and metadata
#         """
#         start_time = datetime.now()
#         logger.info(f"Starting query processing with OpenAI Agents SDK for: '{query[:50]}{'...' if len(query) > 50 else ''}' (top_k={top_k})")

#         try:
#             # Run the agent with the user query
#             result = Runner.run_sync(
#                 self.rag_agent,
#                 f"Query: {query}\nPlease retrieve relevant information and provide a comprehensive answer based on the book content. Retrieve {top_k} relevant chunks before answering."
#             )

#             # Extract the final output from the agent result
#             answer = result.final_output if result.final_output else "I couldn't find sufficient information to answer your question."

#             # The vector retrieval tool should have been called during the agent run
#             # Extract tool call results if available
#             retrieved_chunks = []
#             confidence_score = 0.0

#             # Since the tool was called, we need to get the results differently
#             # For now, let's call the tool separately to get the content for response formatting
#             tool_results = vector_retrieval_tool(query, top_k)

#             for result in tool_results:
#                 chunk = ContentChunk(
#                     content=result["content"],
#                     similarity_score=result["similarity_score"],
#                     metadata=result["metadata"],
#                     rank=result["rank"]
#                 )
#                 retrieved_chunks.append(chunk)

#             # Calculate confidence score as average similarity of retrieved chunks
#             if retrieved_chunks:
#                 confidence_score = sum(c.similarity_score for c in retrieved_chunks) / len(retrieved_chunks)

#             total_time = (datetime.now() - start_time).total_seconds()

#             # Create the response
#             response = AgentResponse(
#                 query_text=query,
#                 answer=answer,
#                 retrieved_chunks=retrieved_chunks,
#                 confidence_score=confidence_score,
#                 execution_time=total_time,
#                 timestamp=datetime.now()
#             )

#             logger.info(f"Query processed successfully: '{query[:30]}{'...' if len(query) > 30 else ''}' "
#                        f"in {total_time:.4f}s with {len(retrieved_chunks)} retrieved chunks, confidence: {confidence_score:.3f}")

#             # Log metrics
#             metrics_logger.info(f"query_event - query_length={len(query)}, top_k={top_k}, "
#                               f"retrieved_chunks={len(retrieved_chunks)}, confidence_score={confidence_score:.3f}, "
#                               f"total_time={total_time:.4f}s")

#             return response

#         except Exception as e:
#             total_time = (datetime.now() - start_time).total_seconds()
#             logger.error(f"Error processing query with OpenAI Agents SDK: {str(e)}")
#             metrics_logger.error(f"query_error - query_length={len(query)}, total_time={total_time:.4f}s, error={str(e)}")

#             # Return an error response
#             response = AgentResponse(
#                 query_text=query,
#                 answer="Sorry, I encountered an error while processing your query.",
#                 retrieved_chunks=[],
#                 confidence_score=0.0,
#                 execution_time=total_time,
#                 timestamp=datetime.now()
#             )

#             return response


# def create_openai_agents_sdk_rag_agent() -> OpenAIAgentsSDKRAGAgent:
#     """
#     Create and return a configured OpenAI Agents SDK RAG agent instance.

#     Returns:
#         Configured OpenAIAgentsSDKRAGAgent instance
#     """
#     return OpenAIAgentsSDKRAGAgent()


# def process_agent_request_with_agents_sdk(agent_request: AgentRequest) -> AgentResponse:
#     """
#     Process an agent request using the OpenAI Agents SDK and return the response.

#     Args:
#         agent_request: The agent request with query and parameters

#     Returns:
#         AgentResponse with the answer and metadata
#     """
#     agent = create_openai_agents_sdk_rag_agent()

#     return agent.process_query(
#         query=agent_request.query_text,
#         top_k=agent_request.top_k
#     )


# # Example usage function
# def run_sample_query(query: str, top_k: int = 5) -> Dict[str, Any]:
#     """
#     Run a sample query to demonstrate the OpenAI Agents SDK RAG agent functionality.

#     Args:
#         query: The query to process
#         top_k: Number of results to retrieve

#     Returns:
#         Dictionary with query results for demonstration
#     """
#     agent = create_openai_agents_sdk_rag_agent()
#     response = agent.process_query(query, top_k)

#     # Format the response for display
#     result = {
#         "query": response.query_text,
#         "answer": response.answer,
#         "retrieved_chunks_count": len(response.retrieved_chunks),
#         "confidence_score": response.confidence_score,
#         "execution_time": response.execution_time,
#         "timestamp": response.timestamp.isoformat(),
#         "retrieved_sources": [
#             {
#                 "url": chunk.metadata.get("url", "N/A"),
#                 "title": chunk.metadata.get("title", "N/A"),
#                 "similarity_score": chunk.similarity_score,
#                 "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
#             }
#             for chunk in response.retrieved_chunks
#         ]
#     }

#     return result