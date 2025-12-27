"""
Personalization Agent using OpenRouter API for AI-powered content personalization.
"""
import asyncio
import logging
import time
from typing import Optional, Dict, Any
from ..personalization_config import PersonalizationConfig

logger = logging.getLogger(__name__)

class PersonalizationAgent:
    """
    Personalization Agent that uses OpenRouter LLM to personalize chapter content
    based on user's reading level and preferences while preserving structure.
    """

    def __init__(self):
        """
        Initialize the Personalization Agent with OpenRouter configuration.
        """
        self.model = PersonalizationConfig.OPENROUTER_MODEL
        self.is_available = PersonalizationConfig.is_configured()

    async def personalize_content(
        self,
        content: str,
        preferences: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Personalize the provided content based on user preferences using OpenRouter LLM.

        Args:
            content: The HTML content to personalize
            preferences: Dictionary with reading_level, technical_explanations, example_density
            user_profile: Optional user profile with software_experience, hardware_experience

        Returns:
            Dictionary with personalized_content and metadata, or None if personalization fails
        """
        start_time = time.time()

        if not self.is_available:
            logger.error("Personalization agent not configured - OPENROUTER_API_KEY missing")
            return None

        try:
            from openai import AsyncOpenAI

            # Extract preferences with defaults
            reading_level = preferences.get('reading_level', 'intermediate')
            technical_explanations = preferences.get('technical_explanations', False)
            example_density = preferences.get('example_density', 'normal')

            # Build the system prompt based on preferences
            system_prompt = PersonalizationConfig.get_full_prompt(
                reading_level=reading_level,
                technical_explanations=technical_explanations,
                example_density=example_density
            )

            # Initialize client with OpenRouter configuration
            client = AsyncOpenAI(
                api_key=PersonalizationConfig.OPENROUTER_API_KEY,
                base_url=PersonalizationConfig.OPENROUTER_BASE_URL
            )

            # Build user context from profile if available
            user_context = ""
            if user_profile:
                sw_exp = user_profile.get('software_experience', 'unknown')
                hw_exp = user_profile.get('hardware_experience', 'unknown')
                user_context = f"\n\nUser background: Software experience: {sw_exp}, Hardware experience: {hw_exp}"

            # Prepare the message for the API call
            user_message = f"""Personalize the following HTML content for a {reading_level} reader.{user_context}

Content to personalize:

{content}

Remember: Output ONLY the personalized HTML content. No explanations, no markdown code blocks."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            # Make the API call
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent personalization
                max_tokens=8192,  # Allow for longer content
                timeout=PersonalizationConfig.PERSONALIZATION_TIMEOUT
            )

            # Extract the personalized content
            if response.choices and len(response.choices) > 0:
                personalized_content = response.choices[0].message.content

                # Clean up any markdown code block wrapper if present
                if personalized_content.startswith("```html"):
                    personalized_content = personalized_content[7:]
                if personalized_content.startswith("```"):
                    personalized_content = personalized_content[3:]
                if personalized_content.endswith("```"):
                    personalized_content = personalized_content[:-3]
                personalized_content = personalized_content.strip()

                processing_time_ms = int((time.time() - start_time) * 1000)

                logger.info(f"Successfully personalized content in {processing_time_ms}ms")

                return {
                    "personalized_content": personalized_content,
                    "preferences_applied": {
                        "reading_level": reading_level,
                        "technical_explanations": technical_explanations,
                        "example_density": example_density
                    },
                    "processing_time_ms": processing_time_ms
                }
            else:
                logger.warning("No personalized content returned from LLM")
                return None

        except Exception as e:
            logger.error(f"Error personalizing content: {str(e)}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the personalization service is available.

        Returns:
            Dictionary with health status
        """
        result = {
            "status": "unhealthy",
            "service": "personalization",
            "ai_available": False,
            "openrouter_configured": PersonalizationConfig.is_configured()
        }

        if not self.is_available:
            result["error"] = "OpenRouter API key not configured"
            return result

        try:
            from openai import AsyncOpenAI

            # Initialize client with OpenRouter configuration
            client = AsyncOpenAI(
                api_key=PersonalizationConfig.OPENROUTER_API_KEY,
                base_url=PersonalizationConfig.OPENROUTER_BASE_URL
            )

            # Make a simple test call
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=10,
                timeout=10
            )

            if response:
                result["status"] = "healthy"
                result["ai_available"] = True
                return result

        except Exception as e:
            logger.error(f"Personalization agent health check failed: {str(e)}")
            result["error"] = str(e)

        return result


# Global instance of the personalization agent
personalization_agent = PersonalizationAgent()
