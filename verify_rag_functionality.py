#!/usr/bin/env python3
"""
Script to verify RAG (Retrieval-Augmented Generation) functionality
This will show you that the agent is using Qdrant vector database to retrieve information
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def verify_rag_functionality():
    """Verify that the agent is using RAG (Retrieval-Augmented Generation)."""
    try:
        from src.rag_agent.agent import RAGAgent
        from src.rag_agent.config import Config

        print("🔍 RAG Functionality Verification")
        print("=" * 60)
        print(f"🎯 Using LLM: {'OpenRouter' if Config.OPENROUTER_API_KEY else 'Gemini'}")
        print(f"🔗 Qdrant URL: {Config.QDRANT_URL}")
        print(f"📚 Collection: {Config.QDRANT_COLLECTION_NAME}")
        print("=" * 60)

        agent = RAGAgent()
        print("✅ Agent created successfully!")

        # Test queries that should find specific book content
        test_queries = [
            {
                "query": "What is ROS2?",
                "expected_book_content": True,
                "description": "Should find ROS2 information in the book"
            },
            {
                "query": "Explain humanoid robotics",
                "expected_book_content": True,
                "description": "Should find humanoid robotics content in the book"
            },
            {
                "query": "What is the capstone project about?",
                "expected_book_content": True,
                "description": "Should find capstone project details in the book"
            }
        ]

        print("\n📊 TESTING RAG FUNCTIONALITY")
        print("=" * 60)

        for i, test in enumerate(test_queries, 1):
            print(f"\n📋 Test {i}: {test['query']}")
            print(f"📝 Expected: {test['description']}")
            print("-" * 40)

            response = agent.process_query(test['query'], top_k=3)

            print(f"🤖 Answer Preview: {response.answer[:200]}...")
            print(f"📊 Confidence Score: {response.confidence_score:.3f}")
            print(f"🔗 Retrieved Chunks: {len(response.retrieved_chunks)}")

            # Show retrieved chunks to verify they contain book content
            if response.retrieved_chunks:
                print("📖 RETRIEVED CONTENT FROM BOOK:")
                for j, chunk in enumerate(response.retrieved_chunks, 1):
                    print(f"  Chunk {j}:")
                    print(f"    Title: {chunk.metadata.get('title', 'N/A')}")
                    print(f"    URL: {chunk.metadata.get('url', 'N/A')}")
                    print(f"    Similarity: {chunk.similarity_score:.3f}")
                    print(f"    Content: {chunk.content[:150]}...")
                    print()
            else:
                print("❌ NO CONTENT RETRIEVED - This indicates RAG is not working")

            print("=" * 60)

        # Test a query that should NOT find relevant content
        print("\n📋 CONTROL TEST: Query that likely has no book content")
        print("-" * 40)
        control_response = agent.process_query("What is the weather today?", top_k=3)
        print(f"🤖 Answer: {control_response.answer[:200]}...")
        print(f"📊 Retrieved Chunks: {len(control_response.retrieved_chunks)}")
        print("💡 Note: This should retrieve fewer relevant chunks since it's not in the book")
        print("=" * 60)

        print("\n🎯 VERIFICATION COMPLETE")
        print("✅ If you see 'Retrieved Chunks' > 0 and 'Content' from the book, RAG is working!")
        print("✅ If you see specific URLs and titles from the AI-native book, RAG is working!")
        print("✅ The agent is using vector retrieval from Qdrant database!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def quick_rag_test(query):
    """Quick test to see if RAG is working for a specific query."""
    try:
        from src.rag_agent.agent import RAGAgent
        agent = RAGAgent()

        print(f"🔍 Testing RAG for: '{query}'")
        response = agent.process_query(query, top_k=3)

        print(f"🤖 Answer: {response.answer[:300]}...")
        print(f"📊 Retrieved Chunks: {len(response.retrieved_chunks)}")

        if response.retrieved_chunks:
            print("\n📖 SOURCES FROM BOOK:")
            for i, chunk in enumerate(response.retrieved_chunks, 1):
                print(f"  {i}. {chunk.metadata.get('title', 'N/A')}")
                print(f"     Similarity: {chunk.similarity_score:.3f}")
                print(f"     Content preview: {chunk.content[:100]}...")
        else:
            print("❌ No sources retrieved - RAG may not be working properly")

        return response

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific query
        query = " ".join(sys.argv[1:])
        quick_rag_test(query)
    else:
        # Run full verification
        verify_rag_functionality()