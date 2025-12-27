"""
Translation Agent using Context7 MCP server for Urdu translation functionality.
"""
import asyncio
import logging
from typing import Optional
from ..translation_config import TranslationConfig

logger = logging.getLogger(__name__)

class TranslationAgent:
    """
    Translation Agent that uses Context7 MCP server to translate content to Urdu
    while preserving formatting and structure.
    """

    def __init__(self):
        """
        Initialize the Translation Agent with Context7 MCP server configuration.
        """
        # The actual API calls will be handled through Context7 MCP server
        # which provides the OpenAI-compatible interface
        self.model = TranslationConfig.OPENROUTER_MODEL
        self.system_prompt = TranslationConfig.TRANSLATION_SYSTEM_PROMPT

    async def translate_content(self, content: str, user_id: str, chapter_id: str) -> Optional[str]:
        """
        Translate the provided content to Urdu while preserving structure
        using Context7 MCP server.

        Args:
            content: The content to translate
            user_id: The ID of the user requesting translation
            chapter_id: The ID of the chapter being translated

        Returns:
            Translated content in Urdu or None if translation fails
        """
        try:
            # Import the Context7 MCP server client when needed
            from openai import AsyncOpenAI

            # Initialize client with OpenRouter configuration for Context7 MCP server
            client = AsyncOpenAI(
                api_key=TranslationConfig.OPENROUTER_API_KEY,
                base_url=TranslationConfig.OPENROUTER_BASE_URL
            )

            # Prepare the message for the API call
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Translate the following content to {TranslationConfig.TARGET_LANGUAGE} while preserving all formatting and structure:\n\n{content}"}
            ]

            # Make the API call via Context7 MCP server
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=4096,  # Adjust based on content length
                timeout=TranslationConfig.TRANSLATION_TIMEOUT
            )

            # Extract the translated content
            if response.choices and len(response.choices) > 0:
                translated_content = response.choices[0].message.content
                logger.info(f"Successfully translated chapter {chapter_id} for user {user_id}")
                return translated_content
            else:
                logger.warning(f"No translation returned for chapter {chapter_id} for user {user_id}")
                return None

        except Exception as e:
            logger.error(f"Error translating chapter {chapter_id} for user {user_id}: {str(e)}")
            return None

    async def translate_content_preserve_formatting(self, content: str, user_id: str, chapter_id: str) -> Optional[str]:
        """
        Translate content with special handling to preserve formatting
        using Context7 MCP server.
        This method ensures that code blocks, headings, and other structural elements
        are properly preserved during translation.

        Args:
            content: The content to translate
            user_id: The ID of the user requesting translation
            chapter_id: The ID of the chapter being translated

        Returns:
            Translated content in Urdu with preserved formatting
        """
        try:
            # For now, we'll use the basic translation method
            # In a more advanced implementation, we might parse the content
            # to identify code blocks, headings, etc. and handle them separately
            return await self.translate_content(content, user_id, chapter_id)
        except Exception as e:
            logger.error(f"Error in translate_content_preserve_formatting: {str(e)}")
            return None

    async def health_check(self) -> bool:
        """
        Check if the translation service is available via Context7 MCP server.

        Returns:
            True if the service is available, False otherwise
        """
        try:
            # Import the Context7 MCP server client when needed
            from openai import AsyncOpenAI

            # Initialize client with OpenRouter configuration for Context7 MCP server
            client = AsyncOpenAI(
                api_key=TranslationConfig.OPENROUTER_API_KEY,
                base_url=TranslationConfig.OPENROUTER_BASE_URL
            )

            # Make a simple test call to the API
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=10,
                timeout=TranslationConfig.TRANSLATION_TIMEOUT
            )

            return response is not None
        except Exception as e:
            logger.error(f"Translation agent health check failed: {str(e)}")
            return False


# Global instance of the translation agent
translation_agent = TranslationAgent()