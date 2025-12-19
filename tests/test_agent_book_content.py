"""
Test script to verify the agent retrieves and uses actual book content from Qdrant
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.rag_agent.agent import create_rag_agent
from src.rag_agent.qdrant_service import search_vectors
from src.content_embedding.retrieval_service import create_query_embedding

def test_qdrant_connection():
    """Test if we can connect to Qdrant and retrieve data"""
    print("Testing Qdrant connection and data retrieval...")

    try:
        # Create a test query embedding
        test_query = "artificial intelligence"
        query_vector = create_query_embedding(test_query)

        # Search in Qdrant
        results = search_vectors(
            query_vector=query_vector,
            top_k=3,
            collection_name=os.getenv('QDRANT_COLLECTION_NAME', 'ai_native_book')
        )

        print(f"✅ Successfully retrieved {len(results)} results from Qdrant")
        if results:
            print(f"First result preview: {results[0]['content'][:100]}...")
            print(f"Metadata: {results[0]['metadata']}")

        return True
    except Exception as e:
        print(f"❌ Qdrant connection failed: {str(e)}")
        return False

def test_agent_with_book_content():
    """Test the agent with queries related to book content"""
    print("\nTesting agent with book-related queries...")

    try:
        agent = create_rag_agent()
        print("✅ Agent created successfully")

        # Test queries related to book content
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning concepts",
            "What are neural networks?",
            "Describe the history of AI",
            "What is deep learning?"
        ]

        for i, query in enumerate(test_queries):
            print(f"\nTest {i+1}: Querying '{query}'")

            try:
                response = agent.process_query_with_agents_sdk(query, top_k=3)

                print(f"Answer: {response.answer[:200]}...")
                print(f"Retrieved {len(response.retrieved_chunks)} chunks")
                print(f"Confidence: {response.confidence_score:.3f}")

                if response.retrieved_chunks:
                    first_chunk = response.retrieved_chunks[0]
                    print(f"First retrieved chunk preview: {first_chunk.content[:100]}...")
                    print(f"Source: {first_chunk.metadata.get('url', 'N/A')}")

            except Exception as e:
                print(f"Error processing query '{query}': {str(e)}")

        return True

    except Exception as e:
        print(f"❌ Agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_retrieval():
    """Test the retrieval function directly"""
    print("\nTesting direct retrieval from Qdrant...")

    try:
        # Test the vector retrieval tool directly
        from src.rag_agent.agent import vector_retrieval_tool

        results = vector_retrieval_tool("artificial intelligence", top_k=2)

        print(f"✅ Retrieved {len(results)} results using vector_retrieval_tool")
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result['content'][:100]}...")
            print(f"Similarity: {result['similarity_score']:.3f}")
            print(f"Metadata: {result['metadata']}")

        return True
    except Exception as e:
        print(f"❌ Direct retrieval test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting validation tests for agent and book content retrieval...\n")

    # Check if required environment variables are set
    required_vars = ['GEMINI_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {missing_vars}")
        print("Please set these variables in your .env file.")
    else:
        print("✅ All required environment variables are set")

    # Run tests
    qdrant_ok = test_qdrant_connection()
    retrieval_ok = test_direct_retrieval() if qdrant_ok else False
    agent_ok = test_agent_with_book_content() if qdrant_ok else False

    print(f"\n" + "="*50)
    print("TEST RESULTS SUMMARY:")
    print(f"Qdrant Connection: {'✅ PASS' if qdrant_ok else '❌ FAIL'}")
    print(f"Direct Retrieval: {'✅ PASS' if retrieval_ok else '❌ FAIL'}")
    print(f"Agent Queries: {'✅ PASS' if agent_ok else '❌ FAIL'}")

    if qdrant_ok and retrieval_ok:
        print("\n🎉 The agent can successfully retrieve book content from Qdrant!")
        if agent_ok:
            print("🎉 The agent is properly using the retrieved book content in responses!")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")