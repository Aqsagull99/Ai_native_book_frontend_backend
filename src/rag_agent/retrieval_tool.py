"""
Vector Retrieval Tool for OpenAI Agents SDK
Implements the retrieval tool that can be used by the OpenAI Assistant
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from openai import OpenAI

from src.rag_agent.models import ContentChunk, RetrievalTool
from src.rag_agent.qdrant_service import search_vectors
from src.content_embedding.retrieval_service import create_query_embedding
from src.content_embedding.utils import Config as EmbeddingConfig
from src.rag_agent.config import Config


class VectorRetrievalTool:
    """
    Vector retrieval tool implementation for OpenAI Agents SDK.
    This tool allows the agent to search the Qdrant vector database
    and retrieve relevant content chunks based on semantic similarity.
    """

    def __init__(self):
        """Initialize the vector retrieval tool with required services."""
        # Validate configuration
        is_valid, error_msg = Config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error_msg}")

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("✅ Vector retrieval tool initialized")

    def call(self, query: str, top_k: int = 5, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Execute the vector retrieval tool with the given parameters.

        Args:
            query: The search query to find relevant content
            top_k: Number of results to retrieve (default: 5)
            include_metadata: Whether to include metadata in results (default: True)

        Returns:
            List of retrieved content chunks with their metadata
        """
        try:
            self.logger.info(f"Executing vector retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

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

            # Format results as ContentChunk objects
            formatted_results = []
            for i, result in enumerate(search_results):
                content_chunk = {
                    "content": result["content"],
                    "similarity_score": result["similarity_score"],
                    "metadata": result["metadata"] if include_metadata else {},
                    "rank": i + 1
                }
                formatted_results.append(content_chunk)

            self.logger.info(f"Retrieved {len(formatted_results)} results for query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error in vector retrieval tool: {str(e)}")
            raise

    def get_tool_spec(self) -> Dict[str, Any]:
        """
        Get the tool specification for OpenAI Agents SDK.

        Returns:
            Dictionary containing the tool specification
        """
        tool = RetrievalTool()
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        }

    def validate_tool_connection(self) -> bool:
        """
        Validate that the retrieval tool can connect to the vector database.

        Returns:
            True if the connection is working, False otherwise
        """
        try:
            # Test with a simple query
            test_results = self.call("test", top_k=1)

            if len(test_results) > 0:
                self.logger.info("✅ Vector retrieval tool connection validated successfully")
                return True
            else:
                self.logger.warning("⚠️ Vector retrieval tool returned no results for test query")
                return False

        except Exception as e:
            self.logger.error(f"❌ Vector retrieval tool connection validation failed: {str(e)}")
            return False


def create_vector_retrieval_tool() -> VectorRetrievalTool:
    """
    Create and return a configured vector retrieval tool instance.

    Returns:
        Configured VectorRetrievalTool instance
    """
    return VectorRetrievalTool()


# Example usage
if __name__ == "__main__":
    # This would be used for testing the tool directly
    tool = create_vector_retrieval_tool()

    # Validate connection
    if tool.validate_tool_connection():
        print("Vector retrieval tool is working correctly")

        # Test a sample query
        results = tool.call("What is artificial intelligence?", top_k=3)
        print(f"Retrieved {len(results)} results")
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result['content'][:100]}...")
    else:
        print("Vector retrieval tool connection failed")