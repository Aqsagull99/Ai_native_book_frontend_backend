#!/usr/bin/env python3
"""
Test script to verify the agent works properly with gemini-1.5-flash model
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_agent.agent import create_rag_agent, run_sample_query
from src.rag_agent.config import Config

def test_agent_with_gemini_15_flash():
    """Test the agent with the configured gemini-1.5-flash model"""
    print("Testing agent with gemini-1.5-flash model...")

    # Load environment variables
    load_dotenv()

    # Verify configuration
    print(f"GEMINI_MODEL_NAME from config: {Config.GEMINI_MODEL_NAME}")
    print(f"QDRANT_COLLECTION_NAME from config: {Config.QDRANT_COLLECTION_NAME}")

    # Validate configuration
    is_valid, error_msg = Config.validate()
    if not is_valid:
        print(f"❌ Configuration validation failed: {error_msg}")
        return False

    print("✅ Configuration is valid")

    try:
        # Create the agent
        print("Creating RAG agent...")
        agent = create_rag_agent()
        print("✅ Agent created successfully")

        # Test a simple query
        test_query = "What is artificial intelligence?"
        print(f"Testing query: '{test_query}'")

        # Run a sample query
        result = run_sample_query(test_query, top_k=3)

        print("\n--- Query Result ---")
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer'][:200]}...")  # First 200 chars
        print(f"Retrieved chunks: {result['retrieved_chunks_count']}")
        print(f"Confidence score: {result['confidence_score']:.3f}")
        print(f"Execution time: {result['execution_time']:.3f}s")

        if result['retrieved_chunks_count'] > 0:
            print(f"First source: {result['retrieved_sources'][0]['title']}")
            print(f"Similarity score: {result['retrieved_sources'][0]['similarity_score']:.3f}")

        print("✅ Agent test completed successfully")
        return True

    except Exception as e:
        print(f"❌ Agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_agent_call():
    """Test calling the agent directly"""
    print("\nTesting direct agent call...")

    try:
        # Create the agent
        agent = create_rag_agent()

        # Call the agent directly with a simple query
        response = agent.process_query_with_agents_sdk(
            query="What are the main topics covered in machine learning?",
            top_k=2
        )

        print(f"Direct agent response: {response.answer[:150]}...")
        print(f"Retrieved {len(response.retrieved_chunks)} chunks")
        print("✅ Direct agent call successful")
        return True

    except Exception as e:
        print(f"❌ Direct agent call failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing RAG Agent with gemini-1.5-flash model")
    print("=" * 50)

    success1 = test_agent_with_gemini_15_flash()
    success2 = test_direct_agent_call()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! The agent is working properly with gemini-1.5-flash.")
    else:
        print("❌ Some tests failed. Please check the configuration and environment variables.")

    print(f"\nRemember to set these environment variables:")
    print(f"- GEMINI_API_KEY (your Google AI API key)")
    print(f"- GEMINI_MODEL_NAME=gemini-1.5-flash (already set in .env)")
    print(f"- QDRANT_URL (your Qdrant Cloud URL)")
    print(f"- QDRANT_API_KEY (your Qdrant API key)")