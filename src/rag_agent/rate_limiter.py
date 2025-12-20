"""
Rate Limiter Service for RAG Agent

Implements daily rate limiting to prevent exceeding API quotas.
Tracks requests per day and enforces limits of 200 requests per day.
"""
import time
import threading
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class DailyRateLimiter:
    """
    Daily rate limiter that tracks requests per day and enforces limits.
    """

    def __init__(self, daily_limit: int = 200):
        """
        Initialize the rate limiter.

        Args:
            daily_limit: Maximum number of requests allowed per day (default: 200)
        """
        self.daily_limit = daily_limit
        self.request_counts: Dict[str, int] = {}
        self.last_reset_date: datetime = self._get_current_date()
        self.lock = threading.Lock()

        logger.info(f"Rate limiter initialized with daily limit: {daily_limit} requests")

    def _get_current_date(self) -> datetime:
        """Get the current date (without time) for tracking daily limits."""
        now = datetime.now()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _reset_if_new_day(self):
        """Reset the request count if it's a new day."""
        current_date = self._get_current_date()

        if current_date > self.last_reset_date:
            with self.lock:
                # Double-check after acquiring lock
                if current_date > self.last_reset_date:
                    logger.info(f"Resetting daily request count (was {self.request_counts.get('total', 0)})")
                    self.request_counts.clear()
                    self.last_reset_date = current_date

    def increment_request(self) -> bool:
        """
        Increment the request count and check if the limit has been exceeded.

        Returns:
            bool: True if request is allowed, False if rate limit exceeded
        """
        with self.lock:
            self._reset_if_new_day()

            # Get current total count
            total_requests = self.request_counts.get('total', 0)

            # Check if we've reached the daily limit
            if total_requests >= self.daily_limit:
                logger.warning(f"Rate limit exceeded: {total_requests}/{self.daily_limit} requests")
                return False

            # Increment the counter
            self.request_counts['total'] = total_requests + 1
            logger.debug(f"Request #{total_requests + 1} allowed (limit: {self.daily_limit})")

            return True

    def get_usage_info(self) -> Dict[str, any]:
        """
        Get current rate limit usage information.

        Returns:
            Dict with usage information including remaining requests
        """
        with self.lock:
            self._reset_if_new_day()

            total_requests = self.request_counts.get('total', 0)
            remaining = max(0, self.daily_limit - total_requests)

            # Calculate reset time (next midnight)
            tomorrow = self.last_reset_date + timedelta(days=1)
            reset_timestamp = int(tomorrow.timestamp())

            return {
                'total_requests': total_requests,
                'daily_limit': self.daily_limit,
                'remaining_requests': remaining,
                'reset_timestamp': reset_timestamp,
                'reset_datetime': tomorrow.isoformat(),
                'limit_exceeded': total_requests >= self.daily_limit
            }

# Global rate limiter instance
rate_limiter: Optional[DailyRateLimiter] = None

def get_rate_limiter() -> DailyRateLimiter:
    """
    Get the global rate limiter instance, creating it if it doesn't exist.

    Returns:
        DailyRateLimiter instance
    """
    global rate_limiter
    if rate_limiter is None:
        from .config import Config
        daily_limit = int(getattr(Config, 'DAILY_REQUEST_LIMIT', 200))
        rate_limiter = DailyRateLimiter(daily_limit=daily_limit)
    return rate_limiter