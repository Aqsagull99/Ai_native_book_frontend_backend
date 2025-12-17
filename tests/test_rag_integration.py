"""
Integration tests for RAG agent API endpoints.

These tests verify the end-to-end functionality of the RAG chatbot integration,
including query processing, response generation, and source citation.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime

from main import app  # Import the main FastAPI app
from src.rag_agent.models import AgentResponse, ContentChunk

# Create test client
client = TestClient(app)

def test_rag_query_endpoint():
    """Test the POST /api/rag/query endpoint with a basic query."""

    # Mock the RAG agent response
    mock_response = AgentResponse(
        query_text="test query",
        answer="This is a test answer",
        retrieved_chunks=[
            ContentChunk(
                content="Test content chunk",
                similarity_score=0.95,
                metadata={"title": "Test Title", "url": "http://example.com", "section": "Test Section"},
                rank=1
            )
        ],
        confidence_score=0.85,
        execution_time=0.5,
        timestamp=datetime.now()
    )

    # Mock the rag_agent.process_query_with_agents_sdk method
    with patch('src.rag_agent.api_service.rag_agent') as mock_agent:
        mock_agent.process_query_with_agents_sdk.return_value = mock_response

        response = client.post(
            "/api/rag/query",
            json={
                "query_text": "What is the main concept of this book?",
                "selected_text": "Some selected text for context",
                "top_k": 5,
                "include_citations": True
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "query_text" in data
        assert "answer" in data
        assert "retrieved_chunks" in data
        assert "confidence_score" in data
        assert "execution_time" in data
        assert "timestamp" in data

        assert data["query_text"] == "What is the main concept of this book?"
        assert len(data["retrieved_chunks"]) > 0
        assert data["confidence_score"] >= 0 and data["confidence_score"] <= 1

def test_rag_query_endpoint_with_validation_error():
    """Test the POST /api/rag/query endpoint with invalid query (too long)."""

    response = client.post(
        "/api/rag/query",
        json={
            "query_text": "a" * 1001,  # Exceeds 1000 character limit
            "selected_text": "Some selected text for context",
            "top_k": 5,
            "include_citations": True
        }
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data["detail"]
    assert data["detail"]["error"] == "VALIDATION_ERROR"

def test_rag_query_endpoint_with_empty_query():
    """Test the POST /api/rag/query endpoint with empty query."""

    response = client.post(
        "/api/rag/query",
        json={
            "query_text": "",
            "selected_text": "Some selected text for context",
            "top_k": 5,
            "include_citations": True
        }
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data["detail"]
    assert data["detail"]["error"] == "VALIDATION_ERROR"

def test_rag_health_endpoint():
    """Test the GET /api/rag/health endpoint."""

    response = client.get("/api/rag/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "timestamp" in data
    assert "details" in data
    assert data["status"] in ["healthy", "unavailable"]

def test_rag_query_with_selected_text_context():
    """Test the POST /api/rag/query endpoint with selected text context."""

    # Mock the RAG agent response
    mock_response = AgentResponse(
        query_text="test query",
        answer="This is a test answer with context",
        retrieved_chunks=[
            ContentChunk(
                content="Test content chunk with context",
                similarity_score=0.90,
                metadata={"title": "Test Title with Context", "url": "http://example.com/context", "section": "Test Section"},
                rank=1
            )
        ],
        confidence_score=0.80,
        execution_time=0.6,
        timestamp=datetime.now()
    )

    # Mock the rag_agent.process_query_with_agents_sdk method
    with patch('src.rag_agent.api_service.rag_agent') as mock_agent:
        mock_agent.process_query_with_agents_sdk.return_value = mock_response

        response = client.post(
            "/api/rag/query",
            json={
                "query_text": "How does this concept work?",
                "selected_text": "The concept of AI agents is fundamental to understanding this book.",
                "top_k": 3,
                "include_citations": True
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["query_text"] == "How does this concept work?"
        assert len(data["retrieved_chunks"]) > 0
        assert data["confidence_score"] >= 0 and data["confidence_score"] <= 1

if __name__ == "__main__":
    pytest.main([__file__])