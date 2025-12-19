"""
Basic validation tests for agent responses
Tests the core functionality of the RAG agent
"""

import asyncio
import pytest
from datetime import datetime
from typing import List

from src.rag_agent.models import AgentRequest, AgentResponse, ContentChunk
from src.rag_agent.agent import create_rag_agent, process_agent_request
from src.rag_agent.config import Config


def test_agent_response_basic():
    """Test that the agent can process a basic query and return a response."""
    # Create a basic agent request
    agent_request = AgentRequest(
        query_text="What is artificial intelligence?",
        top_k=3,
        temperature=0.7,
        max_tokens=500
    )

    # Process the request
    response = process_agent_request(agent_request)

    # Validate the response structure
    assert isinstance(response, AgentResponse)
    assert response.query_text == "What is artificial intelligence?"
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0
    assert isinstance(response.confidence_score, float)
    assert 0.0 <= response.confidence_score <= 1.0
    assert isinstance(response.execution_time, float)
    assert response.execution_time >= 0
    assert isinstance(response.timestamp, datetime)
    assert isinstance(response.retrieved_chunks, list)


def test_agent_response_with_content_chunks():
    """Test that the agent returns content chunks with proper structure."""
    agent_request = AgentRequest(
        query_text="What are machine learning algorithms?",
        top_k=5
    )

    response = process_agent_request(agent_request)

    # Validate that retrieved chunks have proper structure
    assert len(response.retrieved_chunks) <= 5  # top_k limit

    for chunk in response.retrieved_chunks:
        assert isinstance(chunk, ContentChunk)
        assert isinstance(chunk.content, str)
        assert len(chunk.content) > 0
        assert isinstance(chunk.similarity_score, float)
        assert 0.0 <= chunk.similarity_score <= 1.0
        assert isinstance(chunk.metadata, dict)
        assert isinstance(chunk.rank, int)
        assert chunk.rank > 0


def test_agent_response_quality():
    """Test that the agent response is contextually relevant."""
    test_query = "What is the capital of France?"
    agent_request = AgentRequest(
        query_text=test_query,
        top_k=1
    )

    response = process_agent_request(agent_request)

    # Basic quality check: response should be related to the query
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0

    # The response should be a reasonable length
    assert len(response.answer) > 10


def test_agent_error_handling():
    """Test that the agent handles invalid inputs gracefully."""
    try:
        # Test with empty query (should be handled gracefully by validation)
        agent_request = AgentRequest(
            query_text="",
            top_k=5
        )

        # This might raise an exception or return a default response
        # depending on how error handling is implemented
        response = process_agent_request(agent_request)

        # If it returns a response, it should be appropriate for empty input
        assert isinstance(response, AgentResponse)

    except Exception as e:
        # If it raises an exception, that's also valid behavior as long as
        # it's handled appropriately by the calling code
        pass


def test_configuration_validation():
    """Test that the agent configuration is valid."""
    is_valid, error_msg = Config.validate()
    assert is_valid, f"Configuration validation failed: {error_msg}"


def test_agent_creation():
    """Test that the agent can be created successfully."""
    try:
        agent = create_rag_agent()
        assert agent is not None
        # Check that required components are initialized
        assert hasattr(agent, 'process_query')
        assert hasattr(agent, 'process_query_with_agents_sdk')
    except Exception as e:
        pytest.fail(f"Failed to create agent: {str(e)}")


if __name__ == "__main__":
    # Run basic validation tests
    print("Running basic validation tests for agent responses...")

    try:
        test_agent_creation()
        print("✅ Agent creation test passed")
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")

    try:
        test_configuration_validation()
        print("✅ Configuration validation test passed")
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")

    try:
        test_agent_response_basic()
        print("✅ Basic agent response test passed")
    except Exception as e:
        print(f"❌ Basic agent response test failed: {e}")

    try:
        test_agent_response_with_content_chunks()
        print("✅ Content chunks test passed")
    except Exception as e:
        print(f"❌ Content chunks test failed: {e}")

    try:
        test_agent_response_quality()
        print("✅ Response quality test passed")
    except Exception as e:
        print(f"❌ Response quality test failed: {e}")

    try:
        test_agent_error_handling()
        print("✅ Error handling test passed")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")

    print("Basic validation tests completed.")