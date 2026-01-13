"""
Security module for The Local Network Backend

Features:
- API key validation
- Rate limiting (requests per minute)
- Request validation (max prompt length)
- Daily budget caps (token limits)

Author: mikeat7 network
"""

import os
import time
from typing import Optional, Dict
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib


# API key storage (in production, use environment variables or database)
VALID_API_KEYS = set()

# Load API keys from environment
def load_api_keys():
    """Load API keys from environment variable"""
    global VALID_API_KEYS
    api_key_env = os.getenv('API_KEYS', '')
    if api_key_env:
        VALID_API_KEYS = set(api_key_env.split(','))


# Rate limiting storage
# Format: {api_key: [(timestamp1, tokens1), (timestamp2, tokens2), ...]}
rate_limit_storage: Dict[str, list] = defaultdict(list)


# Configuration
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv('RATE_LIMIT_RPM', '5'))
RATE_LIMIT_TOKENS_PER_DAY = int(os.getenv('RATE_LIMIT_TOKENS_DAY', '100000'))
MAX_PROMPT_LENGTH = int(os.getenv('MAX_PROMPT_LENGTH', '5000'))
MAX_TOKENS_PER_REQUEST = int(os.getenv('MAX_TOKENS_REQUEST', '2048'))


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key

    Args:
        api_key: API key to validate

    Returns:
        True if valid, False otherwise
    """
    # If no API keys configured, allow all requests (development mode)
    if not VALID_API_KEYS:
        return True

    # Check if key is in valid set
    return api_key in VALID_API_KEYS


def check_rate_limit(api_key: str, tokens_requested: int = 512) -> Optional[str]:
    """
    Check if request exceeds rate limits

    Args:
        api_key: API key making the request
        tokens_requested: Number of tokens to generate

    Returns:
        Error message if rate limit exceeded, None otherwise
    """
    now = time.time()

    # Get request history for this API key
    history = rate_limit_storage[api_key]

    # Clean up old entries (older than 24 hours)
    cutoff_24h = now - 86400  # 24 hours in seconds
    history[:] = [(ts, tokens) for ts, tokens in history if ts > cutoff_24h]

    # Check requests per minute
    cutoff_1m = now - 60  # 1 minute in seconds
    recent_requests = [ts for ts, _ in history if ts > cutoff_1m]

    if len(recent_requests) >= RATE_LIMIT_REQUESTS_PER_MINUTE:
        return f"Rate limit exceeded: {RATE_LIMIT_REQUESTS_PER_MINUTE} requests per minute"

    # Check tokens per day
    tokens_today = sum(tokens for ts, tokens in history if ts > cutoff_24h)

    if tokens_today + tokens_requested > RATE_LIMIT_TOKENS_PER_DAY:
        return f"Daily token limit exceeded: {RATE_LIMIT_TOKENS_PER_DAY} tokens per day"

    # Record this request
    history.append((now, tokens_requested))

    return None  # No rate limit exceeded


def validate_request(data: dict) -> Optional[str]:
    """
    Validate request parameters

    Args:
        data: Request JSON data

    Returns:
        Error message if invalid, None otherwise
    """
    # Check required fields
    if 'prompt' not in data:
        return "Missing required field: 'prompt'"

    # Validate prompt length
    prompt = data['prompt']
    if not isinstance(prompt, str):
        return "Prompt must be a string"

    if len(prompt) == 0:
        return "Prompt cannot be empty"

    if len(prompt) > MAX_PROMPT_LENGTH:
        return f"Prompt too long (max {MAX_PROMPT_LENGTH} characters)"

    # Validate max_tokens if provided
    if 'max_tokens' in data:
        max_tokens = data['max_tokens']
        if not isinstance(max_tokens, int):
            return "max_tokens must be an integer"

        if max_tokens < 1:
            return "max_tokens must be >= 1"

        if max_tokens > MAX_TOKENS_PER_REQUEST:
            return f"max_tokens too large (max {MAX_TOKENS_PER_REQUEST})"

    # Validate temperature if provided
    if 'temperature' in data:
        temp = data['temperature']
        if not isinstance(temp, (int, float)):
            return "temperature must be a number"

        if temp < 0 or temp > 2:
            return "temperature must be between 0 and 2"

    return None  # Valid request


def generate_api_key(prefix: str = "sk") -> str:
    """
    Generate a new API key

    Args:
        prefix: Prefix for the key (default: "sk")

    Returns:
        Generated API key (e.g., "sk_abc123...")
    """
    # Generate random bytes
    random_bytes = os.urandom(32)

    # Hash to hex string
    key_hash = hashlib.sha256(random_bytes).hexdigest()[:32]

    # Format as API key
    api_key = f"{prefix}_{key_hash}"

    return api_key


def reset_rate_limits(api_key: Optional[str] = None):
    """
    Reset rate limits (for testing/admin purposes)

    Args:
        api_key: Specific key to reset, or None to reset all
    """
    if api_key:
        rate_limit_storage[api_key] = []
    else:
        rate_limit_storage.clear()


# Initialize API keys on module load
load_api_keys()


if __name__ == "__main__":
    # Test security functions
    print("Security Module Test")
    print("=" * 60)

    # Generate test API key
    test_key = generate_api_key()
    print(f"\nGenerated API key: {test_key}")

    # Test validation
    print("\n1. Testing API key validation:")
    print(f"   Valid key: {validate_api_key(test_key)}")
    print(f"   Invalid key: {validate_api_key('invalid_key')}")

    # Test rate limiting
    print("\n2. Testing rate limiting:")
    for i in range(7):
        error = check_rate_limit(test_key, tokens_requested=100)
        if error:
            print(f"   Request {i + 1}: ❌ {error}")
        else:
            print(f"   Request {i + 1}: ✅ Allowed")

    # Test request validation
    print("\n3. Testing request validation:")

    valid_request = {
        "prompt": "Test prompt",
        "max_tokens": 512,
        "temperature": 0.7
    }
    error = validate_request(valid_request)
    print(f"   Valid request: {'✅ OK' if error is None else f'❌ {error}'}")

    invalid_requests = [
        {"max_tokens": 512},  # Missing prompt
        {"prompt": ""},  # Empty prompt
        {"prompt": "x" * 10000},  # Too long
        {"prompt": "Test", "max_tokens": "invalid"},  # Invalid type
        {"prompt": "Test", "temperature": 5.0},  # Temperature out of range
    ]

    for req in invalid_requests:
        error = validate_request(req)
        print(f"   Invalid request: {error}")

    print("\n" + "=" * 60)
    print("Security module test complete!")
    print("\nConfiguration:")
    print(f"  Rate limit: {RATE_LIMIT_REQUESTS_PER_MINUTE} req/min")
    print(f"  Daily tokens: {RATE_LIMIT_TOKENS_PER_DAY}")
    print(f"  Max prompt: {MAX_PROMPT_LENGTH} chars")
    print(f"  Max tokens/req: {MAX_TOKENS_PER_REQUEST}")
