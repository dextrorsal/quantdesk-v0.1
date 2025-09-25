"""
Rate Limiting Module
Implements rate limiting to prevent API abuse and DoS attacks
"""
import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, requests_per_minute: int = 100, burst_size: int = 200):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst requests allowed
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str = "default") -> bool:
        """
        Check if request is allowed
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            current_time = time.time()
            time_passed = current_time - self.last_update
            
            # Add tokens based on time passed
            tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
            self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
            self.last_update = current_time
            
            # Check if we have tokens available
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                return False

class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation"""
    
    def __init__(self, requests_per_minute: int = 100, window_size: int = 60):
        """
        Initialize sliding window rate limiter
        
        Args:
            requests_per_minute: Maximum requests per minute
            window_size: Window size in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.window_size = window_size
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str = "default") -> bool:
        """
        Check if request is allowed using sliding window
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            current_time = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests outside the window
            while client_requests and client_requests[0] <= current_time - self.window_size:
                client_requests.popleft()
            
            # Check if we're within the limit
            if len(client_requests) < self.requests_per_minute:
                client_requests.append(current_time)
                return True
            else:
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                return False

class IPRateLimiter:
    """IP-based rate limiter"""
    
    def __init__(self, requests_per_minute: int = 100):
        """
        Initialize IP rate limiter
        
        Args:
            requests_per_minute: Maximum requests per minute per IP
        """
        self.requests_per_minute = requests_per_minute
        self.ip_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, ip_address: str) -> bool:
        """
        Check if request from IP is allowed
        
        Args:
            ip_address: Client IP address
            
        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            current_time = time.time()
            ip_requests = self.ip_requests[ip_address]
            
            # Remove requests older than 1 minute
            while ip_requests and ip_requests[0] <= current_time - 60:
                ip_requests.popleft()
            
            # Check if we're within the limit
            if len(ip_requests) < self.requests_per_minute:
                ip_requests.append(current_time)
                return True
            else:
                logger.warning(f"Rate limit exceeded for IP: {ip_address}")
                return False

class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load"""
    
    def __init__(self, base_requests_per_minute: int = 100):
        """
        Initialize adaptive rate limiter
        
        Args:
            base_requests_per_minute: Base rate limit
        """
        self.base_requests_per_minute = base_requests_per_minute
        self.current_limit = base_requests_per_minute
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.lock = asyncio.Lock()
        
        # Start background task to adjust limits
        asyncio.create_task(self._adjust_limits())
    
    async def is_allowed(self, client_id: str = "default") -> bool:
        """
        Check if request is allowed with adaptive limits
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            if client_id not in self.rate_limiters:
                self.rate_limiters[client_id] = RateLimiter(
                    requests_per_minute=self.current_limit
                )
            
            return await self.rate_limiters[client_id].is_allowed(client_id)
    
    async def _adjust_limits(self):
        """Background task to adjust rate limits based on system load"""
        while True:
            try:
                # Get system load (simplified)
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                # Adjust limits based on system load
                if cpu_percent > 80 or memory_percent > 80:
                    # High load - reduce limits
                    self.current_limit = max(50, self.current_limit * 0.8)
                elif cpu_percent < 50 and memory_percent < 50:
                    # Low load - increase limits
                    self.current_limit = min(200, self.current_limit * 1.1)
                
                # Update all existing rate limiters
                for limiter in self.rate_limiters.values():
                    limiter.requests_per_minute = self.current_limit
                
                logger.info(f"Adjusted rate limit to {self.current_limit} requests/minute")
                
            except Exception as e:
                logger.error(f"Error adjusting rate limits: {e}")
            
            # Wait 30 seconds before next adjustment
            await asyncio.sleep(30)

# Global rate limiter instances
default_rate_limiter = RateLimiter(requests_per_minute=100)
ip_rate_limiter = IPRateLimiter(requests_per_minute=100)
# Note: AdaptiveRateLimiter will be initialized when needed to avoid asyncio issues

# Rate limiting decorator
def rate_limit(requests_per_minute: int = 100, per_ip: bool = False):
    """
    Decorator for rate limiting endpoints
    
    Args:
        requests_per_minute: Maximum requests per minute
        per_ip: Whether to limit per IP address
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get client identifier
            client_id = "default"
            if per_ip:
                # Try to get IP from request
                request = kwargs.get('request')
                if request:
                    client_id = request.client.host
                else:
                    client_id = "unknown"
            
            # Check rate limit
            if per_ip:
                if not await ip_rate_limiter.is_allowed(client_id):
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded. Please try again later."
                    )
            else:
                if not await default_rate_limiter.is_allowed(client_id):
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded. Please try again later."
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Utility functions
async def check_rate_limit(client_id: str, requests_per_minute: int = 100) -> bool:
    """
    Check if client is within rate limit
    
    Args:
        client_id: Client identifier
        requests_per_minute: Rate limit
        
    Returns:
        True if allowed, False if rate limited
    """
    limiter = RateLimiter(requests_per_minute=requests_per_minute)
    return await limiter.is_allowed(client_id)

async def get_rate_limit_status(client_id: str) -> Dict[str, any]:
    """
    Get current rate limit status for client
    
    Args:
        client_id: Client identifier
        
    Returns:
        Dictionary with rate limit status
    """
    return {
        "client_id": client_id,
        "requests_per_minute": default_rate_limiter.requests_per_minute,
        "current_tokens": default_rate_limiter.tokens,
        "last_update": default_rate_limiter.last_update
    }
