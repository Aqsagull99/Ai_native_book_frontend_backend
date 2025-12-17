"""
OpenAI Agents SDK Configuration for Context 7 MCP Servers
Implements the OpenAI Agents SDK configuration with context 7 MCP architecture
"""

import os
import logging
from typing import Dict, Any, Optional

from openai import OpenAI
import google.generativeai as genai

from src.rag_agent.config import Config
from src.rag_agent.models import ContentChunk


class OpenAIAgentsSDK:
    """
    Wrapper class for OpenAI Agents SDK functionality with MCP context 7.
    This class provides the interface to the OpenAI Assistants API while
    incorporating the MCP (Multi-Context Protocol) architecture.
    """

    def __init__(self):
        """Initialize the OpenAI Agents SDK with MCP context 7 configuration."""
        # Validate configuration
        is_valid, error_msg = Config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error_msg}")

        # Initialize OpenAI client (lazily)
        self._openai_client = None

        # Configure Google Generative AI as well for Gemini integration
        genai.configure(api_key=Config.GEMINI_API_KEY)

        # Set up MCP context (context 7 as specified)
        self.mcp_context_id = Config.MCP_CONTEXT_ID  # Should be 7
        if self.mcp_context_id != 7:
            raise ValueError(f"MCP Context ID must be 7, got {self.mcp_context_id}")

        # Initialize the assistant
        self.assistant = None
        self.thread_id = None

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"✅ OpenAI Agents SDK initialized with MCP Context {self.mcp_context_id}")

    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.warning("OPENAI_API_KEY not found in environment variables. OpenAI functionality will be limited.")
                return None
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def initialize_assistant(self, name: str = "RAG Agent",
                           instructions: str = "You are a helpful RAG agent that answers questions based on provided context."):
        """
        Initialize the OpenAI Assistant with the specified parameters.

        Args:
            name: Name of the assistant
            instructions: System instructions for the assistant
        """
        try:
            # Check if OpenAI API key is available
            if not self.openai_client:
                self.logger.warning("OpenAI API key not available. Skipping assistant initialization.")
                return None

            # Create or retrieve the assistant
            self.assistant = self.openai_client.beta.assistants.create(
                name=name,
                instructions=instructions,
                model=Config.GEMINI_MODEL_NAME.replace("gemini-", "gpt-4"),  # Fallback model for OpenAI
                tools=[{"type": "function", "function": {
                    "name": "vector_retrieval",
                    "description": "Retrieve relevant content chunks from the vector database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"},
                            "top_k": {"type": "integer", "description": "Number of results to return"}
                        },
                        "required": ["query", "top_k"]
                    }
                }}]
            )

            self.logger.info(f"✅ Assistant '{name}' initialized with ID: {self.assistant.id}")
            return self.assistant

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize assistant: {str(e)}")
            raise

    def create_thread(self):
        """Create a new thread for the assistant conversation."""
        try:
            # Check if OpenAI API key is available
            if not self.openai_client:
                self.logger.warning("OpenAI API key not available. Cannot create thread.")
                return None

            thread = self.openai_client.beta.threads.create()
            self.thread_id = thread.id
            self.logger.info(f"✅ Thread created with ID: {self.thread_id}")
            return thread

        except Exception as e:
            self.logger.error(f"❌ Failed to create thread: {str(e)}")
            raise

    def add_message_to_thread(self, message: str):
        """
        Add a user message to the current thread.

        Args:
            message: The user's message to add to the thread
        """
        if not self.thread_id:
            raise ValueError("No thread exists. Call create_thread() first.")

        # Check if OpenAI API key is available
        if not self.openai_client:
            self.logger.warning("OpenAI API key not available. Cannot add message to thread.")
            return None

        try:
            message_object = self.openai_client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=message
            )
            self.logger.info(f"✅ Message added to thread {self.thread_id}")
            return message_object

        except Exception as e:
            self.logger.error(f"❌ Failed to add message to thread: {str(e)}")
            raise

    def run_assistant(self):
        """Run the assistant on the current thread and wait for completion."""
        if not self.assistant:
            raise ValueError("Assistant not initialized. Call initialize_assistant() first.")
        if not self.thread_id:
            raise ValueError("No thread exists. Call create_thread() first.")

        # Check if OpenAI API key is available
        if not self.openai_client:
            self.logger.warning("OpenAI API key not available. Cannot run assistant.")
            return None

        try:
            run = self.openai_client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant.id
            )

            # Wait for the run to complete
            import time
            while run.status in ["queued", "in_progress"]:
                time.sleep(1)
                run = self.openai_client.beta.threads.runs.retrieve(
                    thread_id=self.thread_id,
                    run_id=run.id
                )

            self.logger.info(f"✅ Assistant run completed with status: {run.status}")
            return run

        except Exception as e:
            self.logger.error(f"❌ Failed to run assistant: {str(e)}")
            raise

    def get_thread_messages(self):
        """Get all messages from the current thread."""
        if not self.thread_id:
            raise ValueError("No thread exists. Call create_thread() first.")

        # Check if OpenAI API key is available
        if not self.openai_client:
            self.logger.warning("OpenAI API key not available. Cannot get thread messages.")
            return None

        try:
            messages = self.openai_client.beta.threads.messages.list(
                thread_id=self.thread_id
            )
            return messages

        except Exception as e:
            self.logger.error(f"❌ Failed to get thread messages: {str(e)}")
            raise

    def validate_mcp_context(self) -> bool:
        """
        Validate that the MCP context 7 is properly configured.

        Returns:
            True if the MCP context is properly configured, False otherwise
        """
        try:
            # Check if the context ID is 7 as required
            if self.mcp_context_id != 7:
                self.logger.error(f"❌ MCP Context ID is {self.mcp_context_id}, expected 7")
                return False

            # Additional validation could go here
            self.logger.info("✅ MCP Context 7 is properly configured")
            return True

        except Exception as e:
            self.logger.error(f"❌ MCP Context validation failed: {str(e)}")
            return False


def create_openai_agents_sdk() -> OpenAIAgentsSDK:
    """
    Create and return a configured OpenAI Agents SDK instance.

    Returns:
        Configured OpenAIAgentsSDK instance
    """
    return OpenAIAgentsSDK()


# Example usage
if __name__ == "__main__":
    # This would be used for testing the service directly
    sdk = create_openai_agents_sdk()

    # Validate MCP context
    if sdk.validate_mcp_context():
        print("OpenAI Agents SDK with MCP Context 7 is properly configured")
    else:
        print("OpenAI Agents SDK MCP Context 7 configuration failed")