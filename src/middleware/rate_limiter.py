"""
Rate limiting middleware for translation API endpoints.
"""
import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, Deque
from fastapi import Request, HTTPException, status
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window algorithm.
    """
    def __init__(self, max_requests: int = 10, window_size: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the window
            window_size: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests: Dict[str, Deque[float]] = defaultdict(deque)

    def is_allowed(self, identifier: str) -> tuple[bool, int]:
        """
        Check if a request from the given identifier is allowed.

        Args:
            identifier: Unique identifier for the requester (e.g., user ID or IP)

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        current_time = time.time()

        # Clean up old requests outside the window
        while (self.requests[identifier] and
               current_time - self.requests[identifier][0] > self.window_size):
            self.requests[identifier].popleft()

        # Check if we're under the limit
        current_requests = len(self.requests[identifier])
        is_allowed = current_requests < self.max_requests

        remaining = self.max_requests - current_requests

        if is_allowed:
            # Add current request to the queue
            self.requests[identifier].append(current_time)

        return is_allowed, remaining

    def get_reset_time(self, identifier: str) -> float:
        """
        Get the time when the rate limit will reset for the identifier.

        Args:
            identifier: Unique identifier for the requester

        Returns:
            Unix timestamp when the rate limit will reset
        """
        if not self.requests[identifier]:
            return time.time()

        # The reset time is when the oldest request expires
        oldest_request = self.requests[identifier][0]
        return oldest_request + self.window_size


# Global rate limiter instance
rate_limiter = RateLimiter(
    max_requests=10,  # 10 requests per minute by default
    window_size=60    # 60 second window
)


async def check_rate_limit(request: Request, identifier: str = None) -> None:
    """
    Check if the request is within rate limits.

    Args:
        request: The incoming request
        identifier: Optional identifier (if not provided, will use user ID or IP)

    Raises:
        HTTPException: If rate limit is exceeded
    """
    # Determine identifier - prefer user ID if available, otherwise use IP
    if not identifier:
        # Try to get user ID from session (this is a simplified approach)
        # In a real implementation, you'd extract this from the auth middleware
        user_id = getattr(request.state, 'user_id', None) if hasattr(request.state, 'user_id') else None
        if user_id:
            identifier = f"user:{user_id}"
        else:
            # Use IP address as fallback
            client_host = request.client.host if request.client else "unknown"
            forwarded_for = request.headers.get("x-forwarded-for")
            identifier = f"ip:{forwarded_for or client_host}"

    is_allowed, remaining = rate_limiter.is_allowed(identifier)

    # Add rate limit headers to response
    request.state.rate_limit_remaining = remaining
    request.state.rate_limit_reset = rate_limiter.get_reset_time(identifier)

    if not is_allowed:
        reset_time = rate_limiter.get_reset_time(identifier)
        retry_after = int(reset_time - time.time())

        logger.warning(f"Rate limit exceeded for {identifier}")

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Try again in {retry_after} seconds.",
                "retry_after": retry_after
            }
        )


def get_rate_limit_headers(request: Request) -> dict:
    """
    Get rate limit headers for the response.

    Args:
        request: The incoming request

    Returns:
        Dictionary of rate limit headers
    """
    if hasattr(request.state, 'rate_limit_remaining'):
        remaining = getattr(request.state, 'rate_limit_remaining', 0)
        reset_time = getattr(request.state, 'rate_limit_reset', 0)
        reset_timestamp = int(reset_time)

        return {
            "X-RateLimit-Limit": str(rate_limiter.max_requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_timestamp),
        }

    return {}