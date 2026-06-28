"""Per-IP rate limiting.

A small Redis-backed fixed-window limiter exposed as a FastAPI dependency. It
protects the (paid, rate-limited) LLM backends from abuse on a public demo. The
same Redis instance used as the Celery broker is reused here as the counter store.
"""
import os
import redis
from fastapi import HTTPException, Request

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client = redis.from_url(REDIS_URL)


def rate_limiter(limit: int, window_seconds: int):
    """
    Fixed-window rate limiter, keyed by client IP + route.
    Returns a FastAPI dependency that raises 429 once `limit` requests
    from the same IP hit this route within `window_seconds`.
    """

    def dependency(request: Request):
        client_ip = request.client.host if request.client else "unknown"
        key = f"ratelimit:{request.url.path}:{client_ip}"

        # INCR creates the key at 1 if it doesn't exist yet.
        count = _redis_client.incr(key)
        if count == 1:
            # First request in this window — start the expiry clock.
            _redis_client.expire(key, window_seconds)

        if count > limit:
            ttl = _redis_client.ttl(key)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {ttl}s.",
            )

    return dependency
