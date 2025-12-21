#!/usr/bin/env python3
"""
Direct agent testing script - Ready to run!
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_agent():
    """Test the RAG agent with various queries."""
    try:
        # Import the agent after setting up the path
        from src.rag_agent.agent import RAGAgent

        print("🚀 Creating RAG Agent...")
        agent = RAGAgent()
        print("✅ Agent created successfully!")

        # Show which LLM provider is being used
        from src.rag_agent.config import Config
        if Config.OPENROUTER_API_KEY:
            print(f"🎯 Using OpenRouter with model: {Config.OPENROUTER_MODEL}")
        else:
            print(f"🎯 Using Gemini with model: {Config.GEMINI_MODEL_NAME}")

        print("\n" + "="*60)
        print("TESTING QUERIES")
        print("="*60)

        # List of test queries
        queries = [
            "What is ROS2?",
            "Explain humanoid robotics",
            "What is the capstone project about?",
            "How does the VLA system work?",
            "What is Qdrant?"
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n📋 Query {i}: {query}")
            print("-" * 40)

            try:
                response = agent.process_query(query, top_k=3)
                print(f"🤖 Answer: {response.answer}")
                print(f"📊 Confidence: {response.confidence_score:.3f}")
                print(f"🔗 Retrieved chunks: {len(response.retrieved_chunks)}")
                print(f"⏱️  Execution time: {response.execution_time:.2f}s")
            except Exception as e:
                print(f"❌ Error processing query: {e}")

            print("="*60)

        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"❌ Error creating or using agent: {e}")
        import traceback
        traceback.print_exc()

def quick_test(query_text):
    """Quick test with a single query."""
    try:
        from src.rag_agent.agent import RAGAgent
        agent = RAGAgent()

        print(f"🔍 Processing query: '{query_text}'")
        response = agent.process_query(query_text, top_k=3)

        print(f"🤖 Answer: {response.answer}")
        print(f"📊 Confidence: {response.confidence_score:.3f}")
        print(f"🔗 Retrieved chunks: {len(response.retrieved_chunks)}")

        return response

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If command line argument provided, test that specific query
        query = " ".join(sys.argv[1:])
        quick_test(query)
    else:
        # Otherwise run full test suite
        test_agent()