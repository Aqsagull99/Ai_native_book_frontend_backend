"""
Test script for the new OpenAI Agents Python SDK RAG agent implementation
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.rag_agent.agent import create_rag_agent, run_sample_query
from src.rag_agent.models import AgentRequest

def test_new_agent():
    """Test the new OpenAI Agents SDK implementation"""
    print("Testing the new OpenAI Agents Python SDK RAG agent...")

    try:
        # Create the agent
        agent = create_rag_agent()
        print("✅ Agent created successfully")

        # Test a simple query
        query = "What is artificial intelligence?"
        print(f"\nTesting query: '{query}'")

        # Run the sample query
        result = run_sample_query(query, top_k=3)

        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Retrieved chunks: {result['retrieved_chunks_count']}")
        print(f"Confidence score: {result['confidence_score']:.3f}")
        print(f"Execution time: {result['execution_time']:.4f}s")

        if result['retrieved_sources']:
            print(f"First source: {result['retrieved_sources'][0]['url']}")

        print("\n✅ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_directly():
    """Test the agent by calling it directly"""
    print("\nTesting agent directly...")

    try:
        agent = create_rag_agent()

        # Create a sample request
        request = AgentRequest(
            query_text="What are the key concepts in machine learning?",
            top_k=2
        )

        # Process the request
        response = agent.process_query_with_agents_sdk(
            query=request.query_text,
            top_k=request.top_k
        )

        print(f"Direct agent response:")
        print(f"Query: {response.query_text}")
        print(f"Answer: {response.answer[:200]}...")
        print(f"Retrieved chunks: {len(response.retrieved_chunks)}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Time: {response.execution_time:.4f}s")

        print("\n✅ Direct test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Direct test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting tests for OpenAI Agents Python SDK RAG agent...")

    # Check if required environment variables are set
    required_vars = ['GEMINI_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"\n⚠️  Warning: Missing environment variables: {missing_vars}")
        print("Please set these variables in your .env file to run the full test.")
        print("The agent will still initialize but may not function fully without them.")

    # Run tests
    success1 = test_new_agent()
    success2 = test_agent_directly()

    if success1 and success2:
        print("\n🎉 All tests passed! The OpenAI Agents Python SDK implementation is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")