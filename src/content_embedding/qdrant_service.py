import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter
import cohere
from .utils import Config, QdrantError, EmbeddingError, get_qdrant_client, get_cohere_client
import logging

logger = logging.getLogger(__name__)


async def create_collection(collection_name: str = "ai_native_book") -> bool:
    """
    Create Qdrant collection with appropriate vector dimensions for Cohere embeddings.

    Args:
        collection_name: Name of the collection to create (default "ai_native_book")

    Returns:
        True if collection was created successfully, False otherwise
    """
    try:
        client = get_qdrant_client()

        # Get Cohere client to determine embedding dimensions
        cohere_client = get_cohere_client()

        # Test embedding to get dimensions (using a simple text)
        test_embedding = cohere_client.embed(
            texts=["test"],
            model=Config.COHERE_MODEL,
            input_type="search_document"
        )

        vector_size = len(test_embedding.embeddings[0])

        # Check if collection already exists
        try:
            existing_collection = client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
            return True
        except:
            # Collection doesn't exist, proceed with creation
            pass

        # Create the collection
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        logger.info(f"Successfully created Qdrant collection '{collection_name}' with {vector_size} dimensions")
        return True

    except Exception as e:
        logger.error(f"Failed to create Qdrant collection '{collection_name}': {str(e)}")
        raise QdrantError(f"Failed to create Qdrant collection: {str(e)}")


async def save_chunks_to_qdrant(chunks: List[Dict], collection_name: str = "ai_native_book") -> bool:
    """
    Save text chunks with metadata to Qdrant Cloud.

    Args:
        chunks: List of chunks with content and metadata
        collection_name: Name of the Qdrant collection (default "ai_native_book")

    Returns:
        True if chunks were saved successfully, False otherwise
    """
    try:
        # Get Cohere client for embedding generation
        cohere_client = get_cohere_client()

        # Prepare points for Qdrant
        points = []

        for i, chunk in enumerate(chunks):
            # Generate embedding for the chunk content
            embedding_response = cohere_client.embed(
                texts=[chunk['content']],
                model=Config.COHERE_MODEL,
                input_type="search_document"
            )

            embedding = embedding_response.embeddings[0]

            # Create a Qdrant point
            point = models.PointStruct(
                id=i,  # Using index as ID, in production you might want UUIDs
                vector=embedding,
                payload={
                    'content': chunk['content'],
                    'url': chunk.get('metadata', {}).get('url', ''),
                    'title': chunk.get('metadata', {}).get('title', ''),
                    'chunk_index': chunk.get('chunk_index', i),
                    'source_metadata': chunk.get('metadata', {}),
                    'created_at': asyncio.get_event_loop().time()  # timestamp
                }
            )

            points.append(point)

            # Log progress every 10 chunks
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks for embedding")

        # Get Qdrant client and upload points in batches
        qdrant_client = get_qdrant_client()

        # Upload in batches for efficiency
        batch_size = 10
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )

            logger.info(f"Uploaded batch {i//batch_size + 1} of {(len(points) - 1)//batch_size + 1}")

        logger.info(f"Successfully saved {len(chunks)} chunks to Qdrant collection '{collection_name}'")
        return True

    except Exception as e:
        logger.error(f"Failed to save chunks to Qdrant: {str(e)}")
        raise QdrantError(f"Failed to save chunks to Qdrant: {str(e)}")


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Cohere API.

    Args:
        texts: List of texts to generate embeddings for

    Returns:
        List of embedding vectors
    """
    try:
        cohere_client = get_cohere_client()

        response = cohere_client.embed(
            texts=texts,
            model=Config.COHERE_MODEL,
            input_type="search_document"
        )

        embeddings = [embedding for embedding in response.embeddings]
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")


async def batch_save_chunks_to_qdrant(chunks: List[Dict], collection_name: str = "ai_native_book",
                                    batch_size: int = 10) -> bool:
    """
    Save chunks to Qdrant in batches for efficiency.

    Args:
        chunks: List of chunks with content and metadata
        collection_name: Name of the Qdrant collection
        batch_size: Number of chunks to process in each batch

    Returns:
        True if all chunks were saved successfully, False otherwise
    """
    try:
        qdrant_client = get_qdrant_client()
        cohere_client = get_cohere_client()

        total_chunks = len(chunks)

        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]

            # Prepare texts for embedding
            texts = [chunk['content'] for chunk in batch_chunks]

            # Generate embeddings for the batch
            embedding_response = cohere_client.embed(
                texts=texts,
                model=Config.COHERE_MODEL,
                input_type="search_document"
            )

            embeddings = embedding_response.embeddings

            # Prepare points for this batch
            points = []
            for i, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                point = models.PointStruct(
                    id=batch_start + i,  # Global ID
                    vector=embedding,
                    payload={
                        'content': chunk['content'],
                        'url': chunk.get('metadata', {}).get('url', ''),
                        'title': chunk.get('metadata', {}).get('title', ''),
                        'chunk_index': chunk.get('chunk_index', batch_start + i),
                        'source_metadata': chunk.get('metadata', {}),
                        'created_at': asyncio.get_event_loop().time()
                    }
                )
                points.append(point)

            # Upload the batch to Qdrant
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.info(f"Uploaded batch {batch_start//batch_size + 1} "
                       f"({batch_start + 1}-{batch_start + len(batch_chunks)}) "
                       f"of {total_chunks} chunks")

        logger.info(f"Successfully batch saved {total_chunks} chunks to Qdrant collection '{collection_name}'")
        return True

    except Exception as e:
        logger.error(f"Failed to batch save chunks to Qdrant: {str(e)}")
        raise QdrantError(f"Failed to batch save chunks to Qdrant: {str(e)}")


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

        # Perform the search using the correct Qdrant API method for vector search
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,  # Include payload (metadata) in results
            with_vectors=False  # Don't include vectors in response for efficiency
        )

        # Format results - query_points returns QueryResponse object with points
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
            sample_point = points[0]
            return {
                "id": sample_point.id,
                "payload": sample_point.payload,
                "vector": sample_point.vector
            }
        return None
    except Exception as e:
        logger.error(f"Error getting sample point: {str(e)}")
        return None


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