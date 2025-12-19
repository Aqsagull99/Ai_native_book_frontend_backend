"""
Configuration for the RAG agent with OpenAI Agents SDK
"""

import os
from typing import Optional


class Config:
    """Configuration class for RAG agent settings."""

    # Google AI/Gemini settings
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL_NAME: str = os.getenv('GEMINI_MODEL_NAME', 'gemini-flash-lite-latest')

    # Qdrant settings
    QDRANT_URL: str = os.getenv('QDRANT_URL', '')
    QDRANT_API_KEY: str = os.getenv('QDRANT_API_KEY', '')
    QDRANT_COLLECTION_NAME: str = os.getenv('QDRANT_COLLECTION_NAME', 'ai_native_book')

    # Agent settings
    AGENT_TEMPERATURE: float = float(os.getenv('AGENT_TEMPERATURE', '0.7'))
    AGENT_MAX_TOKENS: int = int(os.getenv('AGENT_MAX_TOKENS', '1000'))
    DEFAULT_TOP_K: int = int(os.getenv('DEFAULT_TOP_K', '5'))

    # MCP Context settings (context 7 as specified)
    MCP_CONTEXT_ID: int = int(os.getenv('MCP_CONTEXT_ID', '7'))

    # Validation settings
    REQUIRED_METADATA_FIELDS: list = ['url', 'title', 'chunk_index', 'source_metadata', 'created_at']

    @classmethod
    def validate(cls) -> tuple[bool, str]:
        """
        Validate that all required configuration values are present.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not cls.GEMINI_API_KEY:
            return False, "GEMINI_API_KEY environment variable is required"

        if not cls.QDRANT_URL:
            return False, "QDRANT_URL environment variable is required"

        if not cls.QDRANT_API_KEY:
            return False, "QDRANT_API_KEY environment variable is required"

        return True, ""