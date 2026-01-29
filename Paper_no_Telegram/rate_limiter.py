"""
Rate Limiter Module

This module provides rate limiting functionality to prevent API throttling.
Uses a sliding window algorithm to track API calls and enforce limits.

Features:
- Thread-safe rate limiting
- Separate limiters for different API types
- File-based audit logging
- Retry logic with exponential backoff
- Daily log rotation

Groww API Rate Limits (estimated):
- Data API: 10 requests/second
- Order API: 5 requests/second
- LTP API: 20 requests/second

Author: Algo Trading Bot
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from collections import deque
import functools
import os

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm.

    Usage:
        limiter = RateLimiter(max_requests=10, window_seconds=1)
        limiter.wait_if_needed()  # Blocks if limit exceeded
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        name: str = "default"
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            name: Name for logging purposes
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.name = name

        self._requests: deque = deque()
        self._lock = threading.Lock()

        # Audit logging
        self._audit_log_path = f"rate_limit_audit_{name}.log"
        self._total_requests = 0
        self._throttled_requests = 0

    def can_proceed(self) -> bool:
        """
        Check if a request can proceed without blocking.

        Returns:
            True if request is allowed
        """
        with self._lock:
            self._cleanup_old_requests()
            return len(self._requests) < self.max_requests

    def wait_if_needed(self, timeout: float = 30.0) -> bool:
        """
        Wait until a request is allowed.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if request is allowed, False if timeout
        """
        start_time = time.time()

        while True:
            with self._lock:
                self._cleanup_old_requests()

                if len(self._requests) < self.max_requests:
                    self._requests.append(time.time())
                    self._total_requests += 1
                    return True

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self._throttled_requests += 1
                logger.warning(f"Rate limiter {self.name}: Timeout waiting for slot")
                return False

            # Calculate wait time
            with self._lock:
                if self._requests:
                    oldest = self._requests[0]
                    wait_time = self.window_seconds - (time.time() - oldest)
                    wait_time = max(0.01, min(wait_time, timeout - elapsed))
                else:
                    wait_time = 0.01

            time.sleep(wait_time)

    def _cleanup_old_requests(self):
        """Remove requests outside the sliding window."""
        cutoff = time.time() - self.window_seconds

        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    def acquire(self) -> bool:
        """
        Acquire a rate limit slot (blocking).

        Returns:
            True if acquired
        """
        return self.wait_if_needed()

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            self._cleanup_old_requests()
            return {
                "name": self.name,
                "current_requests": len(self._requests),
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
                "total_requests": self._total_requests,
                "throttled_requests": self._throttled_requests,
                "available_slots": self.max_requests - len(self._requests)
            }

    def reset(self):
        """Reset the rate limiter."""
        with self._lock:
            self._requests.clear()
            self._total_requests = 0
            self._throttled_requests = 0

    def audit_log(self, message: str):
        """Write to audit log."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open(self._audit_log_path, 'a') as f:
                f.write(f"{timestamp} - {self.name} - {message}\n")
        except Exception as e:
            logger.error(f"Audit log error: {e}")


def rate_limited(
    limiter: RateLimiter,
    max_retries: int = 3,
    backoff_factor: float = 2.0
):
    """
    Decorator for rate-limited functions.

    Args:
        limiter: RateLimiter instance
        max_retries: Maximum retry attempts
        backoff_factor: Exponential backoff multiplier
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                # Wait for rate limit slot
                if not limiter.wait_if_needed():
                    raise Exception(f"Rate limit timeout for {func.__name__}")

                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # Check for rate limit errors
                    if 'rate limit' in error_str or '429' in error_str:
                        wait_time = backoff_factor ** attempt
                        logger.warning(f"Rate limit hit, waiting {wait_time}s "
                                      f"(attempt {attempt + 1}/{max_retries + 1})")
                        time.sleep(wait_time)
                    else:
                        raise

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator


# Pre-configured rate limiters for Groww API
_data_api_limiter: Optional[RateLimiter] = None
_order_api_limiter: Optional[RateLimiter] = None
_ltp_api_limiter: Optional[RateLimiter] = None
_general_api_limiter: Optional[RateLimiter] = None


def get_data_api_limiter() -> RateLimiter:
    """Get or create rate limiter for data API calls."""
    global _data_api_limiter
    if _data_api_limiter is None:
        _data_api_limiter = RateLimiter(
            max_requests=10,
            window_seconds=1,
            name="data_api"
        )
    return _data_api_limiter


def get_order_api_limiter() -> RateLimiter:
    """Get or create rate limiter for order API calls."""
    global _order_api_limiter
    if _order_api_limiter is None:
        _order_api_limiter = RateLimiter(
            max_requests=5,
            window_seconds=1,
            name="order_api"
        )
    return _order_api_limiter


def get_ltp_api_limiter() -> RateLimiter:
    """Get or create rate limiter for LTP API calls."""
    global _ltp_api_limiter
    if _ltp_api_limiter is None:
        _ltp_api_limiter = RateLimiter(
            max_requests=20,
            window_seconds=1,
            name="ltp_api"
        )
    return _ltp_api_limiter


def get_general_api_limiter() -> RateLimiter:
    """Get or create general rate limiter."""
    global _general_api_limiter
    if _general_api_limiter is None:
        _general_api_limiter = RateLimiter(
            max_requests=30,
            window_seconds=1,
            name="general_api"
        )
    return _general_api_limiter


class RateLimitAuditor:
    """
    Audit logger for API rate limiting.

    Provides file-based logging with daily rotation.
    """

    def __init__(self, log_dir: str = ".", prefix: str = "rate_limit"):
        self.log_dir = log_dir
        self.prefix = prefix
        self._lock = threading.Lock()
        self._current_date = None
        self._log_file = None

    def _get_log_path(self) -> str:
        """Get current log file path."""
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"{self.prefix}_{today}.log")

    def _ensure_log_file(self):
        """Ensure log file is open for current date."""
        today = datetime.now().date()
        if today != self._current_date:
            if self._log_file:
                self._log_file.close()
            self._log_file = open(self._get_log_path(), 'a')
            self._current_date = today

    def log(self, limiter_name: str, event: str, details: str = ""):
        """Log a rate limit event."""
        with self._lock:
            try:
                self._ensure_log_file()
                timestamp = datetime.now().strftime("%H:%M:%S.%f")
                self._log_file.write(f"{timestamp}|{limiter_name}|{event}|{details}\n")
                self._log_file.flush()
            except Exception as e:
                logger.error(f"Audit log error: {e}")

    def close(self):
        """Close log file."""
        if self._log_file:
            self._log_file.close()


if __name__ == "__main__":
    print("Rate Limiter Module")
    print("=" * 50)

    # Test rate limiter
    limiter = RateLimiter(max_requests=5, window_seconds=1, name="test")

    print("\nTesting rate limiter (5 req/sec):")
    for i in range(10):
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        print(f"Request {i+1}: waited {elapsed:.3f}s")

    print(f"\nStats: {limiter.get_stats()}")
