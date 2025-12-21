#!/usr/bin/env python3
"""
Test script to check the health of the deployed application.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import health_check

async def test_health():
    """Test the health check function."""
    print("Testing health check...")

    try:
        health_info = await health_check()
        print("Health check result:")
        print(f"Status: {health_info['status']}")
        print(f"Environment: {health_info['environment']}")
        print(f"Services: {health_info['services']}")

        # Check if RAG agent is available
        if health_info['services']['rag_agent_available']:
            print("✅ RAG agent is available")
        else:
            print("❌ RAG agent is NOT available")

        # Check environment variables
        env = health_info['environment']
        if not env['qdrant_url_set'] or not env['qdrant_api_key_set']:
            print("❌ Missing Qdrant configuration")
        else:
            print("✅ Qdrant configuration is set")

        if not env['gemini_api_key_set']:
            print("⚠️  Gemini API key is not set")
        else:
            print("✅ Gemini API key is set")

        # Check if OpenRouter is available by checking the config directly
        from src.rag_agent.config import Config
        if Config.OPENROUTER_API_KEY:
            print("✅ OpenRouter API key is set")
        else:
            print("❌ OpenRouter API key is NOT set")

        return True
    except Exception as e:
        print(f"❌ Error in health check: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_health())
    if success:
        print("\n🎉 Health check completed!")
    else:
        print("\n💥 Health check failed!")
        sys.exit(1)