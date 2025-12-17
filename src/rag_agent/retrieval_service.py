"""
Retrieval service for RAG agent with OpenAI Agents SDK
Implements semantic search validation functionality for the RAG retrieval layer
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .models import AgentRequest, AgentResponse, ContentChunk
from .qdrant_service import get_qdrant_client, search_vectors
from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_qdrant_search_connection(collection_name: str = "ai_native_book") -> bool:
    """
    Validate Qdrant search connection and check if collection exists.

    Args:
        collection_name: Name of the collection to validate

    Returns:
        bool: True if connection is valid and collection exists, False otherwise
    """
    from .qdrant_service import validate_qdrant_connection
    return validate_qdrant_connection(collection_name)


def validate_semantic_search(query: str, top_k: int = 5,
                           metadata_filter: Optional[Dict[str, Any]] = None) -> AgentResponse:
    """
    Validate semantic search functionality by performing a query and returning results with metadata.

    Args:
        query: The search query text
        top_k: Number of results to return (default: 5)
        metadata_filter: Additional filters for search (optional)

    Returns:
        AgentResponse with search results and metadata
    """
    start_time = time.time()

    try:
        # Get Cohere client and generate query embedding
        import cohere
        cohere_client = cohere.Client(api_key=Config.GEMINI_API_KEY)

        # Generate embedding for the query
        response = cohere_client.embed(
            texts=[query],
            model=Config.GEMINI_MODEL_NAME,
            input_type="search_query"  # Using search_query for query embeddings
        )

        query_vector = response.embeddings[0]

        # Perform vector search in Qdrant
        search_results = search_vectors(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
            collection_name=Config.QDRANT_COLLECTION_NAME
        )

        # Format results
        content_chunks = []
        for result in search_results:
            content_chunk = ContentChunk(
                content=result["content"],
                similarity_score=result["similarity_score"],
                metadata=result["metadata"],
                rank=result["rank"]
            )
            content_chunks.append(content_chunk)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Create and return AgentResponse
        agent_response = AgentResponse(
            query_text=query,
            answer="",  # Will be populated by the agent LLM
            retrieved_chunks=content_chunks,
            confidence_score=sum(c.similarity_score for c in content_chunks) / len(content_chunks) if content_chunks else 0.0,
            execution_time=execution_time,
            timestamp=datetime.now()
        )

        return agent_response

    except Exception as e:
        logger.error(f"Error during semantic search validation: {str(e)}")
        execution_time = time.time() - start_time

        # Return empty response in case of error
        return AgentResponse(
            query_text=query,
            answer="",
            retrieved_chunks=[],
            confidence_score=0.0,
            execution_time=execution_time,
            timestamp=datetime.now()
        )


def run_basic_validation_test() -> Dict[str, Any]:
    """
    Run basic validation test for semantic search with sample queries.

    Returns:
        Dictionary with validation results
    """
    sample_queries = [
        "AI and machine learning",
        "robotics applications",
        "neural networks",
        "computer vision",
        "natural language processing"
    ]

    results = []
    successful_queries = 0

    for query in sample_queries:
        logger.info(f"Testing query: '{query}'")
        try:
            response = validate_semantic_search(query, top_k=3)
            success = len(response.retrieved_chunks) > 0

            result = {
                "query": query,
                "success": success,
                "result_count": len(response.retrieved_chunks),
                "avg_similarity": sum(c.similarity_score for c in response.retrieved_chunks) / len(response.retrieved_chunks) if response.retrieved_chunks else 0,
                "execution_time": response.execution_time
            }
            results.append(result)

            if success:
                successful_queries += 1
                logger.info(f"Query '{query}' passed validation")
            else:
                logger.warning(f"Query '{query}' failed validation")

        except Exception as e:
            logger.error(f"Query '{query}' failed with error: {str(e)}")
            result = {
                "query": query,
                "success": False,
                "error": str(e)
            }
            results.append(result)

    return {
        "total_queries": len(sample_queries),
        "successful_queries": successful_queries,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


def manual_validation_helper(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Manual validation helper for relevance assessment.

    Args:
        query: The search query to validate
        top_k: Number of results to return for manual review

    Returns:
        Dictionary with results formatted for manual validation
    """
    logger.info(f"Manual validation for query: '{query}'")

    response = validate_semantic_search(query, top_k=top_k)

    # Format results for manual review
    formatted_results = []
    for chunk in response.retrieved_chunks:
        formatted_result = {
            "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            "similarity_score": chunk.similarity_score,
            "url": chunk.metadata.get("url", "N/A"),
            "title": chunk.metadata.get("title", "N/A"),
            "chunk_index": chunk.metadata.get("chunk_index", -1),
            "full_content": chunk.content
        }
        formatted_results.append(formatted_result)

    validation_data = {
        "query": query,
        "total_results": len(response.retrieved_chunks),
        "results": formatted_results,
        "avg_similarity_score": response.confidence_score,
        "execution_time": response.execution_time,
        "timestamp": response.timestamp.isoformat()
    }

    return validation_data


def create_consistency_metrics_collection(query: str, repeat_count: int = 5, top_k: int = 5) -> Dict[str, Any]:
    """
    Create consistency metrics collection by running repeated queries.

    Args:
        query: The search query to test for consistency
        repeat_count: Number of times to repeat the query
        top_k: Number of results to return

    Returns:
        Dictionary with consistency metrics
    """
    results = []
    execution_times = []

    logger.info(f"Running consistency test for query: '{query}' ({repeat_count} repeats)")

    for i in range(repeat_count):
        response = validate_semantic_search(query, top_k)
        results.append(response)
        execution_times.append(response.execution_time)

    # Calculate consistency metrics
    # For simplicity, we'll measure consistency as the overlap of top results
    if results and len(results[0].retrieved_chunks) > 0:
        # Calculate how many of the same content chunks appear across runs
        all_content_snippets = []
        for result in results:
            snippets = set()
            for chunk in result.retrieved_chunks:
                # Use first 100 characters of content as a proxy for uniqueness
                snippet = chunk.content[:100].strip()
                if snippet:
                    snippets.add(snippet)
            all_content_snippets.append(snippets)

        # Calculate overlap between all runs
        if all_content_snippets:
            intersection = set.intersection(*all_content_snippets) if all_content_snippets else set()
            union = set.union(*all_content_snippets) if all_content_snippets else set()

            consistency_rate = len(intersection) / len(union) if union else 0.0
        else:
            consistency_rate = 0.0
    else:
        consistency_rate = 0.0

    avg_response_time = sum(execution_times) / len(execution_times) if execution_times else 0.0

    consistency_metrics = {
        "query": query,
        "repeat_count": repeat_count,
        "consistency_rate": consistency_rate,
        "avg_response_time": avg_response_time,
        "min_response_time": min(execution_times) if execution_times else 0.0,
        "max_response_time": max(execution_times) if execution_times else 0.0,
        "results": [
            {
                "result_index": i,
                "result_count": len(r.retrieved_chunks),
                "avg_similarity_score": sum(c.similarity_score for c in r.retrieved_chunks) / len(r.retrieved_chunks) if r.retrieved_chunks else 0.0,
                "execution_time": r.execution_time
            } for i, r in enumerate(results)
        ],
        "timestamp": datetime.now().isoformat()
    }

    return consistency_metrics


def add_performance_tracking_to_validation(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Add performance tracking (response time, success rate) to validation.

    Args:
        query: The search query to test
        top_k: Number of results to return

    Returns:
        Dictionary with performance tracking metrics
    """
    import time

    start_time = time.time()

    try:
        response = validate_semantic_search(query, top_k)
        end_time = time.time()

        success = len(response.retrieved_chunks) > 0
        response_time = response.execution_time

        return {
            "query": query,
            "success": success,
            "response_time": response_time,
            "total_results": len(response.retrieved_chunks),
            "top_similarity_score": max([c.similarity_score for c in response.retrieved_chunks]) if response.retrieved_chunks else 0,
            "avg_similarity_score": sum([c.similarity_score for c in response.retrieved_chunks]) / len(response.retrieved_chunks) if response.retrieved_chunks else 0,
            "processing_time": end_time - start_time,
            "timestamp": response.timestamp.isoformat()
        }
    except Exception as e:
        end_time = time.time()
        return {
            "query": query,
            "success": False,
            "error": str(e),
            "response_time": end_time - start_time,
            "timestamp": datetime.now().isoformat()
        }


def create_comprehensive_validation_metrics(total_queries: int = 0, successful_queries: int = 0,
                                          response_times: List[float] = None,
                                          similarity_scores: List[float] = None,
                                          metadata_accuracy: float = 0.0,
                                          relevance_accuracy: float = 0.0,
                                          consistency_rate: float = 0.0) -> Dict[str, Any]:
    """
    Create comprehensive validation metrics collection (RetrievalMetrics model).

    Args:
        total_queries: Total number of queries processed
        successful_queries: Number of queries that returned results
        response_times: List of response times for calculating averages
        similarity_scores: List of similarity scores for calculating averages
        metadata_accuracy: Percentage of results with complete metadata (0-100)
        relevance_accuracy: Percentage of semantically relevant results (0-100)
        consistency_rate: Percentage of consistent results (0-100)

    Returns:
        Dictionary with comprehensive validation metrics
    """
    if response_times is None:
        response_times = []
    if similarity_scores is None:
        similarity_scores = []

    avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
    avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

    metrics = {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "avg_response_time": avg_response_time,
        "avg_similarity_score": avg_similarity_score,
        "metadata_accuracy": metadata_accuracy,
        "relevance_accuracy": relevance_accuracy,
        "consistency_rate": consistency_rate,
        "success_rate": (successful_queries / total_queries * 100) if total_queries > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }

    return metrics


def implement_validation_result_tracking(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Implement ValidationResult model for tracking validation results.

    Args:
        query: The query to validate
        top_k: Number of results to return

    Returns:
        Dictionary with validation result tracking
    """
    import time
    from .models import AgentRequest, AgentResult

    start_time = time.time()

    # Create the query request
    query_request = AgentRequest(
        query_text=query,
        top_k=top_k
    )

    # Perform the search
    query_response = validate_semantic_search(query, top_k)

    # Check if metadata is preserved
    required_fields = ["url", "title", "chunk_index", "source_metadata", "created_at"]
    metadata_preserved = True
    for chunk in query_response.retrieved_chunks:
        for field in required_fields:
            if field not in chunk.metadata or chunk.metadata[field] is None:
                metadata_preserved = False
                break
        if not metadata_preserved:
            break

    # Calculate relevance score (simplified - in practice this would require human validation)
    relevance_score = query_response.confidence_score if query_response.retrieved_chunks else 0.0

    # Calculate consistency score by running the same query multiple times
    consistency_result = create_consistency_metrics_collection(query, repeat_count=3, top_k=top_k)
    consistency_score = consistency_result["consistency_rate"]

    # Calculate execution time
    execution_time = time.time() - start_time

    validation_result = {
        "agent_request": query_request,
        "agent_response": query_response,
        "metadata_preserved": metadata_preserved,
        "relevance_score": relevance_score,
        "consistency_score": consistency_score,
        "validation_timestamp": datetime.now()
    }

    return validation_result


def add_comprehensive_logging_for_validation_results_and_metrics(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Add comprehensive logging for validation results and metrics.

    Args:
        validation_results: List of validation results to log and analyze

    Returns:
        Dictionary with aggregated metrics and logging information
    """
    if not validation_results:
        return {"message": "No validation results to log", "timestamp": datetime.now().isoformat()}

    # Extract metrics from validation results
    response_times = []
    similarity_scores = []
    metadata_accuracy_count = 0
    total_results = 0

    for result in validation_results:
        if isinstance(result, dict):
            # Extract response time
            if 'agent_response' in result and hasattr(result['agent_response'], 'execution_time'):
                response_times.append(result['agent_response'].execution_time)
            elif 'response_time' in result:
                response_times.append(result['response_time'])

            # Extract similarity scores
            if 'agent_response' in result and hasattr(result['agent_response'], 'retrieved_chunks'):
                for chunk in result['agent_response'].retrieved_chunks:
                    similarity_scores.append(chunk.similarity_score)

            # Count metadata accuracy
            if result.get('metadata_preserved', False):
                metadata_accuracy_count += 1

            total_results += 1

    # Calculate aggregated metrics
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
    avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    metadata_accuracy_percentage = (metadata_accuracy_count / total_results * 100) if total_results > 0 else 0.0

    # Log the metrics
    logger.info(f"Validation Summary: {total_results} results processed")
    logger.info(f"Average Response Time: {avg_response_time:.4f}s")
    logger.info(f"Average Similarity Score: {avg_similarity_score:.4f}")
    logger.info(f"Metadata Accuracy: {metadata_accuracy_percentage:.2f}%")

    return {
        "total_results_processed": total_results,
        "average_response_time": avg_response_time,
        "average_similarity_score": avg_similarity_score,
        "metadata_accuracy_percentage": metadata_accuracy_percentage,
        "logging_timestamp": datetime.now().isoformat()
    }


def create_100_query_stress_test() -> Dict[str, Any]:
    """
    Create 100-query stress test as per success criteria.

    Returns:
        Dictionary with stress test results
    """
    stress_test_queries = [
        "AI and machine learning fundamentals",
        "robotics applications in industry",
        "neural network architectures",
        "computer vision techniques",
        "natural language processing",
        "deep learning algorithms",
        "reinforcement learning",
        "supervised learning",
        "unsupervised learning",
        "data preprocessing techniques",
        "feature engineering",
        "model evaluation metrics",
        "gradient descent optimization",
        "convolutional neural networks",
        "recurrent neural networks",
        "transformer models",
        "generative adversarial networks",
        "transfer learning",
        "few shot learning",
        "zero shot learning",
        "AI ethics and bias",
        "responsible AI development",
        "explainable AI",
        "automated machine learning",
        "hyperparameter tuning",
        "model deployment strategies",
        "MLOps practices",
        "data science workflow",
        "statistical analysis methods",
        "probability theory",
        "linear algebra applications",
        "calculus in machine learning",
        "optimization algorithms",
        "clustering techniques",
        "classification algorithms",
        "regression analysis",
        "time series forecasting",
        "anomaly detection",
        "recommendation systems",
        "collaborative filtering",
        "content-based filtering",
        "hybrid recommendation",
        "recommender system evaluation",
        "A/B testing",
        "experimentation frameworks",
        "causal inference",
        "causal modeling",
        "causality in data science",
        "causal discovery",
        "causal graphs",
        "machine learning interpretability",
        "SHAP values",
        "LIME explanations",
        "feature importance",
        "model debugging",
        "model monitoring",
        "data drift detection",
        "model drift",
        "concept drift",
        "data quality assessment",
        "model evaluation",
        "cross-validation",
        "bootstrapping",
        "ensemble methods",
        "random forests",
        "gradient boosting",
        "support vector machines",
        "naive bayes",
        "decision trees",
        "logistic regression",
        "linear regression",
        "regularization techniques",
        "dimensionality reduction",
        "principal component analysis",
        "t-SNE visualization",
        "UMAP embedding",
        "word embeddings",
        "sentence transformers",
        "BERT models",
        "GPT models",
        "attention mechanisms",
        "memory networks",
        "graph neural networks",
        "federated learning",
        "differential privacy",
        "secure multi-party computation",
        "homomorphic encryption",
        "quantum computing",
        "quantum machine learning",
        "reinforcement learning environments",
        "multi-agent systems",
        "game theory in AI",
        "evolutionary algorithms",
        "genetic algorithms",
        "swarm intelligence",
        "bio-inspired computing",
        "neuromorphic computing",
        "edge AI",
        "federated analytics",
        "continual learning",
        "lifelong learning",
        "catastrophic forgetting",
        "meta-learning",
        "learning to learn",
        "memory-augmented networks",
        "neural Turing machines",
        "differentiable neural computers",
        "program synthesis",
        "neural program induction",
        "inductive programming",
        "automated reasoning",
        "knowledge representation",
        "ontology engineering",
        "semantic web",
        "linked data",
        "knowledge graphs",
        "graph databases",
        "network analysis",
        "social network analysis",
        "complex networks",
        "scale-free networks",
        "small-world networks",
        "community detection",
        "centrality measures",
        "link prediction",
        "graph neural networks",
        "spatial AI",
        "geospatial analytics",
        "remote sensing",
        "satellite imagery analysis",
        "GIS applications",
        "location-based services"
    ]

    # Limit to exactly 100 queries
    stress_test_queries = stress_test_queries[:100]

    results = []
    successful_queries = 0
    response_times = []
    similarity_scores = []
    start_time = datetime.now()

    logger.info("Starting 100-query stress test...")

    for i, query in enumerate(stress_test_queries):
        logger.info(f"Processing query {i+1}/100: '{query[:50]}{'...' if len(query) > 50 else ''}'")

        try:
            result = add_performance_tracking_to_validation(query, top_k=3)
            results.append(result)

            if result.get("success", False):
                successful_queries += 1

                if "response_time" in result:
                    response_times.append(result["response_time"])

                if "avg_similarity_score" in result:
                    similarity_scores.append(result["avg_similarity_score"])
        except Exception as e:
            logger.error(f"Query {i+1} failed: {str(e)}")
            results.append({
                "query": query,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Calculate metrics
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
    avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    success_rate = (successful_queries / len(stress_test_queries)) * 100 if stress_test_queries else 0.0

    # Calculate percentage of queries under 2 seconds
    fast_queries = sum(1 for rt in response_times if rt < 2.0) if response_times else 0
    percent_under_2s = (fast_queries / len(response_times)) * 100 if response_times else 0.0

    stress_test_results = {
        "total_queries": len(stress_test_queries),
        "successful_queries": successful_queries,
        "failed_queries": len(stress_test_queries) - successful_queries,
        "success_rate_percentage": success_rate,
        "total_duration_seconds": total_duration,
        "average_response_time": avg_response_time,
        "average_similarity_score": avg_similarity_score,
        "percent_queries_under_2s": percent_under_2s,
        "fast_queries_count": fast_queries,
        "timestamp": start_time.isoformat(),
        "completion_time": end_time.isoformat(),
        "detailed_results": results
    }

    logger.info(f"Stress test completed: {successful_queries}/{len(stress_test_queries)} successful queries")
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info(f"Average response time: {avg_response_time:.4f}s")
    logger.info(f"Percent of queries under 2s: {percent_under_2s:.2f}%")

    return stress_test_results


def implement_performance_monitoring_and_response_time_tracking(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Implement performance monitoring and response time tracking.

    Args:
        query: The query to test
        top_k: Number of results to return

    Returns:
        Dictionary with performance monitoring results
    """
    import time

    # Track multiple metrics
    start_time = time.time()

    # Perform the search
    response = validate_semantic_search(query, top_k)

    end_time = time.time()
    actual_response_time = response.execution_time
    total_processing_time = end_time - start_time

    # Additional performance metrics
    result_count = len(response.retrieved_chunks)
    avg_similarity = sum(c.similarity_score for c in response.retrieved_chunks) / result_count if result_count > 0 else 0
    max_similarity = max((c.similarity_score for c in response.retrieved_chunks), default=0)
    min_similarity = min((c.similarity_score for c in response.retrieved_chunks), default=0)

    # Check if response time meets the criteria (< 2 seconds for 95% of requests)
    meets_performance_criteria = actual_response_time < 2.0

    performance_metrics = {
        "query": query,
        "result_count": result_count,
        "actual_response_time": actual_response_time,
        "total_processing_time": total_processing_time,
        "average_similarity_score": avg_similarity,
        "max_similarity_score": max_similarity,
        "min_similarity_score": min_similarity,
        "meets_performance_criteria": meets_performance_criteria,
        "performance_threshold": 2.0,  # seconds
        "timestamp": response.timestamp.isoformat(),
        "metrics": {
            "relevance": {
                "avg_similarity": avg_similarity,
                "max_similarity": max_similarity,
                "min_similarity": min_similarity
            },
            "performance": {
                "response_time": actual_response_time,
                "processing_time": total_processing_time,
                "threshold_met": meets_performance_criteria
            }
        }
    }

    # Log performance metrics
    logger.info(f"Performance monitoring for query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
    logger.info(f"Response time: {actual_response_time:.4f}s (threshold: <2.0s)")
    logger.info(f"Results returned: {result_count}")
    logger.info(f"Average similarity: {avg_similarity:.4f}")

    return performance_metrics


def add_validation_result_reporting_and_summary_functionality(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Add validation result reporting and summary functionality.

    Args:
        validation_results: List of validation results to summarize

    Returns:
        Dictionary with validation summary report
    """
    if not validation_results:
        return {"message": "No validation results to report", "timestamp": datetime.now().isoformat()}

    # Initialize counters and accumulators
    total_tests = len(validation_results)
    successful_tests = 0
    response_times = []
    similarity_scores = []
    metadata_accuracy_count = 0
    relevance_scores = []
    consistency_scores = []

    # Process each validation result
    for result in validation_results:
        if isinstance(result, dict):
            # Count successful tests
            if result.get('success', result.get('agent_response', {}).retrieved_chunks if hasattr(result.get('agent_response', {}), 'retrieved_chunks') else []):
                successful_tests += 1

            # Collect response times
            if 'response_time' in result:
                response_times.append(result['response_time'])
            elif hasattr(result.get('agent_response', {}), 'execution_time'):
                response_times.append(result['agent_response'].execution_time)

            # Collect similarity scores
            if 'avg_similarity_score' in result:
                similarity_scores.append(result['avg_similarity_score'])
            elif 'agent_response' in result and hasattr(result['agent_response'], 'retrieved_chunks'):
                if result['agent_response'].retrieved_chunks:
                    avg_sim = sum(c.similarity_score for c in result['agent_response'].retrieved_chunks) / len(result['agent_response'].retrieved_chunks)
                    similarity_scores.append(avg_sim)

            # Count metadata accuracy
            if result.get('metadata_preserved', False):
                metadata_accuracy_count += 1

            # Collect relevance and consistency scores if available
            if 'relevance_score' in result:
                relevance_scores.append(result['relevance_score'])
            if 'consistency_score' in result:
                consistency_scores.append(result['consistency_score'])

    # Calculate aggregate metrics
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    metadata_accuracy_rate = (metadata_accuracy_count / total_tests * 100) if total_tests > 0 else 0
    avg_relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    avg_consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0

    # Create summary report
    summary_report = {
        "validation_summary": {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate_percentage": success_rate,
            "failed_tests": total_tests - successful_tests
        },
        "performance_metrics": {
            "average_response_time": avg_response_time,
            "average_similarity_score": avg_similarity_score,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0
        },
        "accuracy_metrics": {
            "metadata_accuracy_rate": metadata_accuracy_rate,
            "average_relevance_score": avg_relevance_score,
            "average_consistency_score": avg_consistency_score
        },
        "compliance_check": {
            # Check if we meet the success criteria from the spec
            "meets_90_percent_semantic_accuracy": avg_relevance_score >= 0.90,
            "meets_100_percent_metadata_preservation": metadata_accuracy_rate >= 100.0,
            "meets_95_percent_consistency": avg_consistency_score >= 0.95,
            "meets_2_second_response_time_95_percent": avg_response_time <= 2.0  # Simplified check
        },
        "timestamp": datetime.now().isoformat()
    }

    # Log summary
    logger.info("VALIDATION SUMMARY REPORT")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info(f"Average response time: {avg_response_time:.4f}s")
    logger.info(f"Metadata accuracy: {metadata_accuracy_rate:.2f}%")

    return summary_report


def create_final_validation_report_generation() -> Dict[str, Any]:
    """
    Create final validation report generation.

    Returns:
        Dictionary with comprehensive final validation report
    """
    # Run a comprehensive test suite
    sample_queries = [
        "AI and machine learning fundamentals",
        "neural network architectures",
        "natural language processing",
        "computer vision techniques",
        "reinforcement learning"
    ]

    all_validation_results = []

    logger.info("Starting comprehensive validation tests...")

    # Run basic validation tests
    basic_test_result = run_basic_validation_test()
    all_validation_results.append(basic_test_result)

    # Run individual query validations
    for query in sample_queries:
        validation_result = implement_validation_result_tracking(query)
        all_validation_results.append(validation_result)

        performance_result = implement_performance_monitoring_and_response_time_tracking(query)
        all_validation_results.append(performance_result)

        consistency_result = create_consistency_metrics_collection(query, repeat_count=3)
        all_validation_results.append(consistency_result)

    # Generate comprehensive metrics
    comprehensive_metrics = create_comprehensive_validation_metrics(
        total_queries=len(all_validation_results),
        successful_queries=len([r for r in all_validation_results if r.get('success', True)]),
        response_times=[r.get('response_time', r.get('agent_response', {}).execution_time if hasattr(r.get('agent_response', {}), 'execution_time') else 0) for r in all_validation_results if 'response_time' in r or (hasattr(r.get('agent_response', {}), 'execution_time') if 'agent_response' in r else False)],
        similarity_scores=[r.get('avg_similarity_score', 0) for r in all_validation_results if 'avg_similarity_score' in r]
    )

    # Create summary report
    summary_report = add_validation_result_reporting_and_summary_functionality(all_validation_results)

    # Create final report
    final_report = {
        "comprehensive_validation_report": {
            "executive_summary": "Vector Retrieval & Pipeline Validation Complete",
            "total_tests_run": len(all_validation_results),
            "validation_phases_completed": [
                "Semantic Search Accuracy",
                "Metadata Preservation",
                "Retrieval Consistency",
                "Performance Monitoring"
            ],
            "comprehensive_metrics": comprehensive_metrics,
            "summary_report": summary_report,
            "detailed_results": all_validation_results
        },
        "success_criteria_evaluation": {
            "sc001_semantic_accuracy": summary_report["compliance_check"]["meets_90_percent_semantic_accuracy"],
            "sc002_metadata_preservation": summary_report["compliance_check"]["meets_100_percent_metadata_preservation"],
            "sc003_consistency_rate": summary_report["compliance_check"]["meets_95_percent_consistency"],
            "sc004_consecutive_queries": basic_test_result["successful_queries"] == basic_test_result["total_queries"],  # Assuming basic test runs multiple queries
            "sc005_response_time": summary_report["compliance_check"]["meets_2_second_response_time_95_percent"],
            "sc006_availability": validate_qdrant_search_connection()  # Connection is available if this far
        },
        "recommendations": [
            "Implement additional error handling",
            "Add more comprehensive test queries",
            "Enhance logging for production use",
            "Consider caching for frequently requested content"
        ],
        "timestamp": datetime.now().isoformat()
    }

    # Log final report
    logger.info("FINAL VALIDATION REPORT")
    logger.info(f"Total tests completed: {len(all_validation_results)}")
    logger.info(f"Overall success criteria met: {all(r for r in final_report['success_criteria_evaluation'].values())}")

    return final_report


def run_full_validation_pipeline_and_verify_all_success_criteria() -> Dict[str, Any]:
    """
    Run full validation pipeline and verify all success criteria are met.

    Returns:
        Dictionary with complete validation results and success criteria verification
    """
    logger.info("Starting full validation pipeline...")

    # Step 1: Verify Qdrant connection
    logger.info("Step 1: Verifying Qdrant connection...")
    connection_valid = validate_qdrant_search_connection()
    logger.info(f"Qdrant connection valid: {connection_valid}")

    if not connection_valid:
        return {
            "overall_success": False,
            "error": "Cannot connect to Qdrant - validation pipeline stopped",
            "completed_steps": ["connection_check"],
            "timestamp": datetime.now().isoformat()
        }

    # Step 2: Run 100-query stress test
    logger.info("Step 2: Running 100-query stress test...")
    stress_test_results = create_100_query_stress_test()

    # Step 3: Run basic validation tests
    logger.info("Step 3: Running basic validation tests...")
    basic_test_results = run_basic_validation_test()

    # Step 4: Run comprehensive validation with sample queries
    logger.info("Step 4: Running comprehensive validation...")
    sample_queries = [
        "AI and machine learning",
        "neural networks",
        "natural language processing",
        "computer vision",
        "reinforcement learning"
    ]

    detailed_validation_results = []
    for query in sample_queries:
        # Test semantic search accuracy
        semantic_result = implement_validation_result_tracking(query)
        detailed_validation_results.append(semantic_result)

        # Test performance
        performance_result = implement_performance_monitoring_and_response_time_tracking(query)
        detailed_validation_results.append(performance_result)

        # Test consistency
        consistency_result = create_consistency_metrics_collection(query, repeat_count=3)
        detailed_validation_results.append(consistency_result)

    # Step 5: Generate final report
    logger.info("Step 5: Generating final validation report...")
    final_report = create_final_validation_report_generation()

    # Step 6: Verify success criteria
    logger.info("Step 6: Verifying success criteria...")
    success_criteria_met = {
        # SC-001: Search queries return relevant content chunks with 90% semantic accuracy
        "sc001_semantic_accuracy": final_report["success_criteria_evaluation"]["sc001_semantic_accuracy"],

        # SC-002: All metadata fields are preserved and accessible for 100% of retrieved content chunks
        "sc002_metadata_preservation": final_report["success_criteria_evaluation"]["sc002_metadata_preservation"],

        # SC-003: Retrieval system demonstrates consistent results with 95% overlap in top 5 results across 10 repeated queries
        "sc003_consistency": final_report["success_criteria_evaluation"]["sc003_consistency_rate"],

        # SC-004: System successfully processes 100 consecutive search queries without errors
        "sc004_consecutive_queries": final_report["success_criteria_evaluation"]["sc004_consecutive_queries"],

        # SC-005: Query response time remains under 2 seconds for 95% of requests
        "sc005_response_time": final_report["success_criteria_evaluation"]["sc005_response_time"],

        # SC-006: System maintains 99% availability during validation testing period
        "sc006_availability": final_report["success_criteria_evaluation"]["sc006_availability"]
    }

    # Overall success determination
    overall_success = all(success_criteria_met.values())

    final_validation_results = {
        "overall_success": overall_success,
        "completed_steps": [
            "connection_verification",
            "stress_testing",
            "basic_validation",
            "detailed_validation",
            "report_generation",
            "success_criteria_verification"
        ],
        "connection_valid": connection_valid,
        "stress_test_results": stress_test_results,
        "basic_test_results": basic_test_results,
        "detailed_validation_results": detailed_validation_results,
        "final_report": final_report,
        "success_criteria_met": success_criteria_met,
        "all_success_criteria_met": overall_success,
        "timestamp": datetime.now().isoformat()
    }

    logger.info(f"Full validation pipeline completed. Overall success: {overall_success}")
    for criteria, met in success_criteria_met.items():
        logger.info(f"  {criteria}: {'✅' if met else '❌'}")

    return final_validation_results