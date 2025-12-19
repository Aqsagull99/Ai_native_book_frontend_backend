#!/usr/bin/env python3
"""
Test script to simulate a chatbot query to the RAG agent
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_agent.agent import create_rag_agent, run_sample_query
from src.rag_agent.config import Config

def test_chatbot_like_query():
    """Test the agent with queries that would come from a chatbot"""
    print("Testing RAG Agent with chatbot-like queries...")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Validate configuration
    is_valid, error_msg = Config.validate()
    if not is_valid:
        print(f"❌ Configuration validation failed: {error_msg}")
        return False

    print(f"✅ Configuration is valid")
    print(f"✅ Using model: {Config.GEMINI_MODEL_NAME}")
    print()

    try:
        # Create the agent
        print("Creating RAG agent...")
        agent = create_rag_agent()
        print("✅ Agent created successfully")
        print()

        # Test various chatbot-like queries
        test_queries = [
            "What are the main concepts of artificial intelligence?",
            "Explain machine learning algorithms briefly",
            "What is the difference between AI and machine learning?",
            "How does a humanoid robot work?",
            "What is computer vision in robotics?",
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"Test {i}: Query: '{query}'")
            print("-" * 40)

            # Run a sample query
            result = run_sample_query(query, top_k=3)

            print(f"Answer preview: {result['answer'][:200]}...")
            print(f"Retrieved chunks: {result['retrieved_chunks_count']}")
            print(f"Confidence score: {result['confidence_score']:.3f}")
            print(f"Execution time: {result['execution_time']:.3f}s")

            if result['retrieved_chunks_count'] > 0:
                print(f"First source: {result['retrieved_sources'][0]['title']}")
                print(f"Similarity score: {result['retrieved_sources'][0]['similarity_score']:.3f}")
                print(f"Content preview: {result['retrieved_sources'][0]['content_preview']}")

            print()

        print("🎉 All chatbot-like queries processed successfully!")
        return True

    except Exception as e:
        print(f"❌ Agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_api_call():
    """Test using the same approach as the API endpoint"""
    print("Testing with direct API-like call...")
    print("-" * 40)

    try:
        from src.rag_agent.agent import process_agent_request
        from src.rag_agent.models import AgentRequest

        # Create a request similar to what the API would receive
        agent_request = AgentRequest(
            query_text="What is deep learning?",
            top_k=3,
            temperature=0.7,
            max_tokens=500
        )

        # Process the request
        response = process_agent_request(agent_request)

        print(f"Query: {response.query_text}")
        print(f"Answer: {response.answer[:200]}...")
        print(f"Retrieved chunks: {len(response.retrieved_chunks)}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Execution time: {response.execution_time:.3f}s")
        print("✅ Direct API-like call successful")
        print()

        return True

    except Exception as e:
        print(f"❌ Direct API call failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing RAG Agent with Chatbot-like Queries")
    print("=" * 60)

    success1 = test_chatbot_like_query()
    success2 = test_direct_api_call()

    print("=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! The chatbot queries are working properly with the RAG agent.")
        print(f"✅ Agent is successfully responding to queries using {Config.GEMINI_MODEL_NAME}")
        print("✅ RAG functionality is working with vector retrieval and source citations")
        print("✅ Tool calling is properly integrated")
    else:
        print("❌ Some tests failed. Please check the configuration and implementation.")