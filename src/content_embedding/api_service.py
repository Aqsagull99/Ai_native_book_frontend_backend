"""
API service for vector retrieval validation
Implements the validation API endpoints as defined in contracts/validation-api.yaml
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .retrieval_service import (
    validate_semantic_search,
    run_basic_validation_test,
    add_metadata_preservation_validation_to_retrieval,
    create_consistency_metrics_collection,
    add_performance_tracking_to_validation
)
from .qdrant_service import get_collection_info, validate_qdrant_connection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_search_endpoint(query: str, top_k: int = 5, validate_metadata: bool = True) -> Dict[str, Any]:
    """
    Implement POST /validate-search endpoint per contracts/validation-api.yaml

    Args:
        query: The search query text
        top_k: Number of results to return (optional, default: 5)
        validate_metadata: Whether to validate metadata preservation (optional, default: true)

    Returns:
        Dictionary matching the API contract response format
    """
    # Perform the semantic search
    query_response = validate_semantic_search(query, top_k)

    # Format results according to API contract
    formatted_results = []
    for result in query_response.results:
        formatted_result = {
            "content": result.content_chunk,
            "similarity_score": result.similarity_score,
            "metadata": result.metadata,
            "rank": result.rank
        }
        formatted_results.append(formatted_result)

    # Get collection info for metrics
    collection_info = get_collection_info()

    # Prepare validation results
    metadata_validation_passed = True
    if validate_metadata:
        metadata_check = add_metadata_preservation_validation_to_retrieval(query, top_k)
        metadata_validation_passed = metadata_check["all_metadata_valid"]

    # Calculate validation metrics
    semantic_relevance = len(formatted_results) > 0  # Basic check: did we get results?
    metadata_accuracy = 100.0  # Placeholder - would need more sophisticated calculation
    consistency = 100.0  # Placeholder - would need to run consistency tests

    response = {
        "query": query,
        "results": formatted_results,
        "metrics": {
            "execution_time": query_response.execution_time,
            "total_points_in_collection": collection_info.get("total_points", 0),
            "metadata_validation_passed": metadata_validation_passed
        },
        "validation": {
            "semantic_relevance": semantic_relevance,
            "metadata_accuracy": metadata_accuracy,
            "consistency": consistency
        }
    }

    return response


def validation_status_endpoint() -> Dict[str, Any]:
    """
    Implement GET /validation-status endpoint per contracts/validation-api.yaml

    Returns:
        Dictionary matching the API contract response format
    """
    # Check if Qdrant connection is valid
    qdrant_valid = validate_qdrant_connection()
    status = "completed" if qdrant_valid else "failed"  # Simplified status

    # Get collection info
    collection_info = get_collection_info()

    # For now, return basic metrics - in a real implementation, we'd track more detailed metrics
    response = {
        "status": status,
        "metrics": {
            "total_queries_executed": 0,  # Would track actual count in a real implementation
            "success_rate": 100.0 if qdrant_valid else 0.0,  # Placeholder
            "avg_response_time": 0.0,  # Would track actual average in a real implementation
            "relevance_accuracy": 0.0,  # Would calculate from actual tests
            "metadata_preservation_rate": 0.0  # Would calculate from actual tests
        },
        "last_validation_timestamp": datetime.now().isoformat(),
        "collection_info": collection_info
    }

    return response


def add_request_response_validation(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add request/response validation based on contract specifications.

    Args:
        request_data: The incoming request data to validate

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "sanitized_request": {}
    }

    # Validate required fields for search endpoint
    if "query" not in request_data or not isinstance(request_data["query"], str) or not request_data["query"].strip():
        validation_results["valid"] = False
        validation_results["errors"].append("Query is required and must be a non-empty string")

    # Validate top_k if provided
    if "top_k" in request_data:
        try:
            top_k = int(request_data["top_k"])
            if top_k <= 0 or top_k > 100:  # Reasonable upper limit
                validation_results["valid"] = False
                validation_results["errors"].append("top_k must be between 1 and 100")
            else:
                validation_results["sanitized_request"]["top_k"] = top_k
        except (ValueError, TypeError):
            validation_results["valid"] = False
            validation_results["errors"].append("top_k must be a valid integer")
    else:
        validation_results["sanitized_request"]["top_k"] = 5  # Default

    # Validate validate_metadata if provided
    if "validate_metadata" in request_data:
        if not isinstance(request_data["validate_metadata"], bool):
            validation_results["valid"] = False
            validation_results["errors"].append("validate_metadata must be a boolean")
        else:
            validation_results["sanitized_request"]["validate_metadata"] = request_data["validate_metadata"]
    else:
        validation_results["sanitized_request"]["validate_metadata"] = True  # Default

    # Add the query if valid
    if "query" in request_data and isinstance(request_data["query"], str) and request_data["query"].strip():
        validation_results["sanitized_request"]["query"] = request_data["query"].strip()

    return validation_results


def format_api_error(error_message: str, status_code: int = 400) -> Dict[str, Any]:
    """
    Implement API error handling and response formatting.

    Args:
        error_message: The error message to include in the response
        status_code: The HTTP status code (default 400)

    Returns:
        Dictionary with formatted error response
    """
    return {
        "error": error_message,
        "status_code": status_code,
        "timestamp": datetime.now().isoformat()
    }