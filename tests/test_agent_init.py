#!/usr/bin/env python3
"""
Test script to check the RAG agent initialization.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_agent_initialization():
    """Test the RAG agent initialization."""
    print("Testing RAG agent initialization...")

    try:
        # Import and create the agent (api_service handles errors internally)
        from src.rag_agent.api_service import create_rag_agent
        print("Creating RAG agent...")

        agent = create_rag_agent()
        print("✅ RAG agent created successfully!")

        # Check if it's the mock agent (error state) or real agent
        if hasattr(agent, 'process_query_with_agents_sdk'):
            print("✅ Real RAG agent is available (not in error state)")
        else:
            print("❌ Agent might be in error state")

        # Test a simple query processing to see if there are runtime errors
        print("Testing query processing...")
        response = agent.process_query_with_agents_sdk("test", top_k=1)
        print(f"✅ Query processed, answer: {response.answer[:50]}...")

        return True
    except Exception as e:
        print(f"❌ Error during agent initialization or testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent_initialization()
    if success:
        print("\n🎉 Agent initialization test completed successfully!")
    else:
        print("\n💥 Agent initialization test failed!")
        sys.exit(1)