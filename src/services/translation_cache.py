"""
In-memory caching mechanism for translated content per user and chapter.
"""
import asyncio
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CachedTranslation:
    """
    Data class to represent a cached translation.
    """
    content: str
    user_id: str
    chapter_id: str
    created_at: float
    expires_at: float
    source_language: str = "English"
    target_language: str = "Urdu"


class TranslationCache:
    """
    In-memory caching mechanism for translated content.
    Caches translations per user and chapter with TTL (time-to-live).
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initialize the cache with TTL and max size settings.

        Args:
            ttl_seconds: Time-to-live for cached translations in seconds (default: 1 hour)
            max_size: Maximum number of cached translations (default: 1000)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[Tuple[str, str], CachedTranslation] = {}  # (user_id, chapter_id) -> CachedTranslation
        self._lock = asyncio.Lock()

    def _generate_cache_key(self, user_id: str, chapter_id: str) -> Tuple[str, str]:
        """
        Generate a cache key from user_id and chapter_id.

        Args:
            user_id: The user ID
            chapter_id: The chapter ID

        Returns:
            A tuple that serves as the cache key
        """
        return (user_id, chapter_id)

    def _is_expired(self, cached_translation: CachedTranslation) -> bool:
        """
        Check if a cached translation has expired.

        Args:
            cached_translation: The cached translation to check

        Returns:
            True if expired, False otherwise
        """
        return time.time() > cached_translation.expires_at

    async def get_translation(self, user_id: str, chapter_id: str) -> Optional[CachedTranslation]:
        """
        Get a cached translation if it exists and hasn't expired.

        Args:
            user_id: The user ID
            chapter_id: The chapter ID

        Returns:
            Cached translation if found and not expired, None otherwise
        """
        async with self._lock:
            key = self._generate_cache_key(user_id, chapter_id)
            cached_translation = self._cache.get(key)

            if cached_translation and not self._is_expired(cached_translation):
                return cached_translation
            elif cached_translation:  # Entry exists but is expired
                del self._cache[key]

            return None

    async def set_translation(self, user_id: str, chapter_id: str, content: str) -> bool:
        """
        Set a translation in the cache.

        Args:
            user_id: The user ID
            chapter_id: The chapter ID
            content: The translated content

        Returns:
            True if successfully cached, False if cache is at max capacity
        """
        async with self._lock:
            # Check if we need to evict entries due to max size
            if len(self._cache) >= self.max_size:
                # Simple LRU-like eviction: remove the oldest entry
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at,
                    default=None
                )
                if oldest_key:
                    del self._cache[oldest_key]

            # Create new cached translation
            now = time.time()
            cached_translation = CachedTranslation(
                content=content,
                user_id=user_id,
                chapter_id=chapter_id,
                created_at=now,
                expires_at=now + self.ttl_seconds
            )

            key = self._generate_cache_key(user_id, chapter_id)
            self._cache[key] = cached_translation
            return True

    async def has_translation(self, user_id: str, chapter_id: str) -> bool:
        """
        Check if a translation exists in cache and hasn't expired.

        Args:
            user_id: The user ID
            chapter_id: The chapter ID

        Returns:
            True if translation exists and is not expired, False otherwise
        """
        translation = await self.get_translation(user_id, chapter_id)
        return translation is not None

    async def delete_translation(self, user_id: str, chapter_id: str) -> bool:
        """
        Delete a translation from the cache.

        Args:
            user_id: The user ID
            chapter_id: The chapter ID

        Returns:
            True if translation was found and deleted, False otherwise
        """
        async with self._lock:
            key = self._generate_cache_key(user_id, chapter_id)
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear_expired(self) -> int:
        """
        Clear all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, cached_translation in self._cache.items()
                if self._is_expired(cached_translation)
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    async def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        async with self._lock:
            total = len(self._cache)
            expired_count = sum(1 for ct in self._cache.values() if self._is_expired(ct))
            valid_count = total - expired_count

            return {
                "total_entries": total,
                "valid_entries": valid_count,
                "expired_entries": expired_count,
                "max_capacity": self.max_size,
                "utilization_percent": (total / self.max_size) * 100 if self.max_size > 0 else 0
            }


# Global instance of the translation cache
translation_cache = TranslationCache()