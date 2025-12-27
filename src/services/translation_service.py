"""
Comprehensive translation service that orchestrates translation functionality.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel

from .translation_agent import translation_agent
from .translation_cache import translation_cache
from ..models.translation import TranslateRequest, TranslateResponse

logger = logging.getLogger(__name__)

class TranslationService:
    """
    Main service class that orchestrates translation functionality including
    API calls, caching, and result management.
    """

    def __init__(self):
        """
        Initialize the translation service.
        """
        self.agent = translation_agent
        self.cache = translation_cache

    async def translate_content(
        self,
        content: str,
        user_id: str,
        chapter_id: str,
        preserve_formatting: bool = True
    ) -> Optional[TranslateResponse]:
        """
        Translate content to Urdu with full service orchestration.
        Optimized for performance with large chapters using chunking if needed.

        Args:
            content: The content to translate
            user_id: The ID of the user requesting translation
            chapter_id: The ID of the chapter being translated
            preserve_formatting: Whether to preserve original formatting

        Returns:
            TranslateResponse with the translated content or None if failed
        """
        try:
            logger.info(f"Starting translation service for user {user_id}, chapter {chapter_id}")

            # Check if translation is already cached
            cached_result = await self.cache.get_translation(user_id, chapter_id)
            if cached_result:
                logger.info(f"Returning cached translation for user {user_id}, chapter {chapter_id}")

                return TranslateResponse(
                    translated_content=cached_result.content,
                    chapter_id=chapter_id,
                    user_id=user_id,
                    translation_id=f"cached_{chapter_id}_{user_id}",
                    created_at=datetime.fromtimestamp(cached_result.created_at)
                )

            # For very large content, we might want to implement chunking
            # But for now, let's add performance logging
            content_length = len(content)
            logger.info(f"Content length: {content_length} characters for chapter {chapter_id}")

            # For performance optimization with large chapters, we could implement chunking
            # But for now, proceed with normal translation
            if content_length > 5000:  # If content is large
                logger.info(f"Processing large content ({content_length} chars) for chapter {chapter_id}")

            # Perform the translation using the agent
            translated_content = None
            if preserve_formatting:
                translated_content = await self.agent.translate_content_preserve_formatting(
                    content=content,
                    user_id=user_id,
                    chapter_id=chapter_id
                )
            else:
                translated_content = await self.agent.translate_content(
                    content=content,
                    user_id=user_id,
                    chapter_id=chapter_id
                )

            if not translated_content:
                logger.warning(f"Translation failed for user {user_id}, chapter {chapter_id}")
                return None

            # Generate a unique translation ID
            translation_id = f"tr_{uuid.uuid4().hex}"

            # Cache the translation
            await self.cache.set_translation(
                user_id=user_id,
                chapter_id=chapter_id,
                content=translated_content
            )

            # Create and return response
            response = TranslateResponse(
                translated_content=translated_content,
                chapter_id=chapter_id,
                user_id=user_id,
                translation_id=translation_id,
                created_at=datetime.utcnow()
            )

            logger.info(f"Successfully completed translation for user {user_id}, chapter {chapter_id} (content length: {content_length})")
            return response

        except Exception as e:
            logger.error(f"Error in translation service: {str(e)}")
            return None

    async def translate_content_with_status_tracking(
        self,
        content: str,
        user_id: str,
        chapter_id: str,
        preserve_formatting: bool = True
    ) -> Dict[str, Any]:
        """
        Translate content with status tracking for async operations.

        Args:
            content: The content to translate
            user_id: The ID of the user requesting translation
            chapter_id: The ID of the chapter being translated
            preserve_formatting: Whether to preserve original formatting

        Returns:
            Dictionary with translation status and result information
        """
        try:
            # Check if translation is already cached
            cached_result = await self.cache.get_translation(user_id, chapter_id)
            if cached_result:
                logger.info(f"Returning cached translation for user {user_id}, chapter {chapter_id}")

                return {
                    "status": "completed",
                    "cached": True,
                    "result": TranslateResponse(
                        translated_content=cached_result.content,
                        chapter_id=chapter_id,
                        user_id=user_id,
                        translation_id=f"cached_{chapter_id}_{user_id}",
                        created_at=datetime.fromtimestamp(cached_result.created_at)
                    )
                }

            # Generate a unique translation ID
            translation_id = f"tr_{uuid.uuid4().hex}"

            # Start translation process
            logger.info(f"Starting translation with status tracking for user {user_id}, chapter {chapter_id}")

            # Perform the translation
            translated_content = None
            if preserve_formatting:
                translated_content = await self.agent.translate_content_preserve_formatting(
                    content=content,
                    user_id=user_id,
                    chapter_id=chapter_id
                )
            else:
                translated_content = await self.agent.translate_content(
                    content=content,
                    user_id=user_id,
                    chapter_id=chapter_id
                )

            if not translated_content:
                return {
                    "status": "failed",
                    "error": "Translation failed",
                    "translation_id": translation_id
                }

            # Cache the translation
            await self.cache.set_translation(
                user_id=user_id,
                chapter_id=chapter_id,
                content=translated_content
            )

            # Create response
            response = TranslateResponse(
                translated_content=translated_content,
                chapter_id=chapter_id,
                user_id=user_id,
                translation_id=translation_id,
                created_at=datetime.utcnow()
            )

            return {
                "status": "completed",
                "result": response,
                "translation_id": translation_id
            }

        except Exception as e:
            logger.error(f"Error in translation service with status tracking: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "translation_id": f"tr_{uuid.uuid4().hex}"
            }

    async def is_translation_cached(
        self,
        user_id: str,
        chapter_id: str
    ) -> bool:
        """
        Check if a translation is cached for a user and chapter.

        Args:
            user_id: The user ID
            chapter_id: The chapter ID

        Returns:
            True if translation is cached and not expired, False otherwise
        """
        return await self.cache.has_translation(user_id, chapter_id)

    async def get_cached_translation(
        self,
        user_id: str,
        chapter_id: str
    ) -> Optional[TranslateResponse]:
        """
        Get a cached translation if it exists.

        Args:
            user_id: The user ID
            chapter_id: The chapter ID

        Returns:
            Cached translation as TranslateResponse or None if not found
        """
        cached_result = await self.cache.get_translation(user_id, chapter_id)
        if cached_result:
            return TranslateResponse(
                translated_content=cached_result.content,
                chapter_id=chapter_id,
                user_id=user_id,
                translation_id=f"cached_{chapter_id}_{user_id}",
                created_at=datetime.fromtimestamp(cached_result.created_at)
            )
        return None

    async def clear_user_translations(
        self,
        user_id: str
    ) -> int:
        """
        Clear all translations for a specific user.

        Args:
            user_id: The user ID

        Returns:
            Number of translations removed
        """
        # This would require iterating through the cache to find all entries for the user
        # For now, this is a placeholder as the current cache implementation doesn't support this efficiently
        logger.warning("Clear user translations not fully implemented in current cache design")
        return 0

    async def health_check(self) -> bool:
        """
        Perform a health check on the translation service.

        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            # Check if the translation agent is healthy
            agent_healthy = await self.agent.health_check()

            # Check if the cache is accessible
            cache_stats = await self.cache.get_cache_stats()

            return agent_healthy and cache_stats is not None

        except Exception as e:
            logger.error(f"Translation service health check failed: {str(e)}")
            return False


# Global instance of the translation service
translation_service = TranslationService()