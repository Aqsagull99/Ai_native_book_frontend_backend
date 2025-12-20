"""
Test script to verify the rate limiter functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_agent.rate_limiter import DailyRateLimiter

def test_rate_limiter():
    print("Testing DailyRateLimiter...")

    # Create a rate limiter with a small limit for testing
    rate_limiter = DailyRateLimiter(daily_limit=3)

    print(f"Daily limit: {rate_limiter.daily_limit}")

    # Test allowing requests up to the limit
    for i in range(5):
        allowed = rate_limiter.increment_request()
        usage_info = rate_limiter.get_usage_info()

        if allowed:
            print(f"Request {i+1}: ALLOWED - {usage_info['remaining_requests']} requests remaining")
        else:
            print(f"Request {i+1}: REJECTED - Rate limit exceeded")
            print(f"Usage info: {usage_info}")
            break

    print("Rate limiter test completed.")

if __name__ == "__main__":
    test_rate_limiter()