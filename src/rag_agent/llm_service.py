"""
Google Gemini client service for the RAG agent with OpenAI Agents SDK
Implements the Google AI client as a dedicated service for LLM interactions
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import google.generativeai as genai

from src.rag_agent.config import Config
from src.rag_agent.models import ContentChunk


class GeminiClientService:
    """
    Dedicated service for interacting with Google Gemini LLM.
    This service handles all LLM interactions and provides a clean interface
    for the agent to generate responses based on retrieved context.
    """

    def __init__(self):
        """Initialize the Gemini client service with configuration."""
        # Validate configuration
        is_valid, error_msg = Config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error_msg}")

        # Configure Google Generative AI
        genai.configure(api_key=Config.GEMINI_API_KEY)

        # Initialize the model
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"✅ Gemini client service initialized with model: {Config.GEMINI_MODEL_NAME}")

    def generate_response(self, query: str, context_chunks: List[ContentChunk],
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """
        Generate a response based on the query and provided context chunks.

        Args:
            query: The original user query
            context_chunks: List of relevant content chunks retrieved from vector database
            temperature: Temperature parameter for response generation (optional)
            max_tokens: Maximum tokens in response (optional)

        Returns:
            Generated response as a string
        """
        try:
            # Use provided values or defaults from config
            temp = temperature if temperature is not None else Config.AGENT_TEMPERATURE
            max_tok = max_tokens if max_tokens is not None else Config.AGENT_MAX_TOKENS

            # Construct the context from retrieved chunks
            context_text = "\n\n".join([
                f"Source: {chunk.metadata.get('url', 'Unknown')}\n"
                f"Content: {chunk.content}\n"
                f"Similarity Score: {chunk.similarity_score}"
                for chunk in context_chunks
            ])

            # Create the prompt for the LLM
            prompt = f"""
            You are an AI assistant that answers questions based on provided book content.
            Use only the information provided in the context to answer the question.
            If the context doesn't contain enough information to answer the question, say so.

            Context:\n{context_text}\n\n
            Question: {query}\n\n
            Answer:
            """

            # Generate content using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": max_tok
                }
            )

            result = response.text if response and hasattr(response, 'text') and response.text else "I couldn't find sufficient information in the provided context to answer your question."
            self.logger.info(f"Generated response for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

            return result

        except Exception as e:
            self.logger.error(f"Error during response generation: {str(e)}")
            return "Sorry, I encountered an error while generating the answer."

    def validate_client_connection(self) -> bool:
        """
        Validate that the Gemini client is properly configured and can make requests.

        Returns:
            True if the client is working, False otherwise
        """
        try:
            # Try a simple test generation
            test_response = self.model.generate_content(
                "Say 'connection test successful' if you can respond to this message.",
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 50
                }
            )

            if test_response.text and "connection" in test_response.text.lower():
                self.logger.info("✅ Gemini client connection validated successfully")
                return True
            else:
                self.logger.warning("⚠️ Gemini client connection validation returned unexpected response")
                return False

        except Exception as e:
            self.logger.error(f"❌ Gemini client connection validation failed: {str(e)}")
            return False


def create_gemini_client_service() -> GeminiClientService:
    """
    Create and return a configured Gemini client service instance.

    Returns:
        Configured GeminiClientService instance
    """
    return GeminiClientService()


# Example usage
if __name__ == "__main__":
    # This would be used for testing the service directly
    service = create_gemini_client_service()

    # Validate connection
    if service.validate_client_connection():
        print("Gemini client service is working correctly")
    else:
        print("Gemini client service connection failed")