"""Bucket name resolution utilities for R2 storage."""
import os
import logging
import time
import random
import string
import re
from typing import Optional

logger = logging.getLogger(__name__)


def is_r2_bucket(bucket_name: str) -> bool:
    """Check if bucket name indicates R2 storage."""
    return bucket_name == 'minstraly-storage' or bucket_name.startswith('r2://')


def resolve_bucket_name(bucket_name: Optional[str] = None, country_code: Optional[str] = None) -> str:
    """
    Resolve R2 bucket name. Only R2 storage is supported.
    
    Returns the R2 bucket name (defaults to 'minstraly-storage').
    The country_code parameter is ignored as we only use a single R2 bucket.
    Non-R2 bucket names are ignored and the default R2 bucket is returned.
    
    :param bucket_name: Optional explicit bucket name (will be validated as R2, ignored if not R2)
    :param country_code: Ignored (kept for API compatibility)
    :return: R2 bucket name
    """
    # Default R2 bucket
    default_r2_bucket = os.getenv('R2_BUCKET_NAME', 'minstraly-storage')
    
    if bucket_name:
        # Clean up the bucket name
        bn = str(bucket_name).replace('r2://', '').replace('gs://', '').strip()
        # Strip protocol if present
        if bn.startswith('https://') or bn.startswith('http://'):
            bn = bn.split('://', 1)[1]
        # If URL-like, take host part only
        if '/' in bn:
            bn = bn.split('/')[0]
        # Validate it's an R2 bucket
        if is_r2_bucket(bn):
            return bn
        else:
            # Non-R2 bucket name provided (likely old Firebase bucket) - ignore and use default R2 bucket
            logger.warning(f"⚠️ Non-R2 bucket name '{bn}' provided (likely legacy Firebase bucket). Ignoring and using default R2 bucket '{default_r2_bucket}'.")
    
    # Return default R2 bucket
    return default_r2_bucket


def make_safe_slug(value: str) -> str:
    """Create a filesystem and URL-safe slug from a string."""
    if value is None:
        return ""
    slug = value.strip().lower()
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^a-z0-9_-]", "", slug)
    slug = slug.strip("_-")
    return slug or "voice"


def build_voice_id_with_user(voice_name: str, user_id: str) -> str:
    """Build voice id in the format voice_{name}_{userID} using sanitized parts."""
    name_part = make_safe_slug(voice_name or "voice")
    user_part = make_safe_slug(user_id or "")
    if user_part:
        return f"voice_{name_part}_{user_part}"
    return f"voice_{name_part}"


def generate_unique_voice_id(voice_name: str, length: int = 8, max_attempts: int = 10) -> str:
    """
    Generate a unique voice ID with random alphanumeric characters to prevent naming collisions.
    Uses timestamp-based suffix to ensure uniqueness (R2 storage doesn't require pre-check).
    
    Args:
        voice_name: The base voice name
        length: Length of the random alphanumeric suffix (default: 8)
        max_attempts: Maximum attempts to generate unique ID (default: 10, not used but kept for compatibility)
        
    Returns:
        Unique voice ID in format: voice_{voice_name}_{random_alphanumeric}_{timestamp}
        
    Example:
        generate_unique_voice_id("christestclone") -> "voice_christestclone_A7b2K9x1_123456"
    """
    # Generate random alphanumeric suffix (letters + numbers)
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    # Add timestamp to ensure uniqueness
    timestamp = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
    voice_id = f"voice_{voice_name}_{random_suffix}_{timestamp}"
    
    logger.info(f"✅ Generated unique voice ID: {voice_id}")
    return voice_id

