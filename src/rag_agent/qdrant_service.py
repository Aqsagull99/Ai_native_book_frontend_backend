"""
Qdrant service for vector retrieval in the RAG agent
Integrates with the existing content embedding infrastructure
"""

import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_qdrant_client():
    """Get Qdrant client instance with configuration from environment variables."""
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")

    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )


def validate_qdrant_connection(collection_name: str = "ai_native_book") -> bool:
    """
    Validate connection to Qdrant Cloud and check if the collection exists.

    Args:
        collection_name: Name of the collection to validate (default "ai_native_book")

    Returns:
        bool: True if connection is valid and collection exists, False otherwise
    """
    try:
        client = get_qdrant_client()

        # Check if the collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if collection_name not in collection_names:
            logger.error(f"Collection '{collection_name}' not found in Qdrant")
            logger.info(f"Available collections: {collection_names}")
            return False

        logger.info("✅ Qdrant connection successful")
        logger.info(f"✅ Collection '{collection_name}' exists")

        # Get collection info
        collection_info = client.get_collection(collection_name)
        logger.info(f"📊 Collection points count: {collection_info.points_count}")

        if collection_info.points_count == 0:
            logger.error("❌ Collection is empty - no embeddings stored")
            return False

        logger.info(f"✅ Collection contains {collection_info.points_count} embeddings")
        return True

    except Exception as e:
        logger.error(f"❌ Error validating Qdrant connection: {str(e)}")
        return False


def search_vectors(query_vector: List[float], top_k: int = 5,
                   metadata_filter: Optional[Dict[str, Any]] = None,
                   collection_name: str = "ai_native_book") -> List[Dict[str, Any]]:
    """
    Perform similarity search against the vector database.

    Args:
        query_vector: The vector to search for similar vectors
        top_k: Number of results to return
        metadata_filter: Optional filter to apply to the search
        collection_name: Name of the collection to search in

    Returns:
        List of search results with content, similarity scores, and metadata
    """
    try:
        client = get_qdrant_client()

        # Prepare filter if provided
        qdrant_filter = None
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )

            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Perform the search using the query_points method (newer Qdrant API)
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,  # Include payload (metadata) in results
            with_vectors=False  # Don't include vectors in response for efficiency
        )

        # Format results
        formatted_results = []
        for idx, result in enumerate(search_results.points):  # Access the points attribute
            formatted_result = {
                "content": result.payload.get('content', ''),
                "similarity_score": result.score,
                "metadata": {
                    "url": result.payload.get('url', ''),
                    "title": result.payload.get('title', ''),
                    "chunk_index": result.payload.get('chunk_index', -1),
                    "source_metadata": result.payload.get('source_metadata', {}),
                    "created_at": result.payload.get('created_at', ''),
                },
                "rank": idx + 1  # 1-indexed
            }
            formatted_results.append(formatted_result)

        return formatted_results

    except Exception as e:
        logger.error(f"Error during vector search: {str(e)}")
        return []


def get_collection_info(collection_name: str = "ai_native_book") -> Dict[str, Any]:
    """
    Get information about the collection.

    Args:
        collection_name: Name of the collection to get info for

    Returns:
        Dict with collection information
    """
    try:
        client = get_qdrant_client()
        collection_info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "total_points": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "status": collection_info.status
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        return {}


def get_sample_point(collection_name: str = "ai_native_book") -> Optional[Dict[str, Any]]:
    """
    Get a sample point from the collection to understand the data structure.

    Args:
        collection_name: Name of the collection to get sample from

    Returns:
        Sample point data or None if not available
    """
    try:
        client = get_qdrant_client()

        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=1
        )
        points, next_page = scroll_result

        if points:
            sample_point = points[0]  # Get the first point
            return {
                "id": sample_point.id,
                "payload": sample_point.payload,
                "vector": sample_point.vector
            }
        return None
    except Exception as e:
        logger.error(f"Error getting sample point: {str(e)}")
        return None