"""R2 storage operations for Cloudflare R2."""
import os
import logging
import traceback
import base64
from typing import Optional

from .bucket_resolver import resolve_bucket_name, is_r2_bucket

logger = logging.getLogger(__name__)


def _encode_metadata_value(value: str) -> str:
    """
    Encode a metadata value to ensure it's ASCII-compatible for S3/R2 metadata.
    
    S3/R2 metadata can only contain ASCII characters. If the value contains
    non-ASCII characters, it will be base64 encoded with a prefix indicator.
    
    :param value: The metadata value to encode
    :return: ASCII-compatible string (either original or base64 encoded with prefix)
    """
    try:
        # Try to encode as ASCII - if it fails, the string contains non-ASCII
        value.encode('ascii')
        # String is already ASCII, return as-is
        return value
    except UnicodeEncodeError:
        # String contains non-ASCII characters, encode with base64
        encoded = base64.b64encode(value.encode('utf-8')).decode('ascii')
        # Prefix with special marker to indicate it's base64 encoded
        return f"base64:{encoded}"


def upload_to_r2(data: bytes, destination_key: str, content_type: str = "application/octet-stream", metadata: dict = None, bucket_name: Optional[str] = None) -> Optional[str]:
    """
    Upload data to Cloudflare R2 using boto3 S3 client.
    
    :param data: Binary data to upload
    :param destination_key: Destination key/path in R2
    :param content_type: MIME type of the file
    :param metadata: Optional metadata dict (will be stored as R2 metadata)
    :param bucket_name: Optional bucket name (defaults to R2_BUCKET_NAME env var)
    :return: Public URL or None if failed
    """
    try:
        import boto3
        
        # Get R2 credentials from environment
        r2_account_id = os.getenv('R2_ACCOUNT_ID')
        r2_access_key_id = os.getenv('R2_ACCESS_KEY_ID')
        r2_secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
        r2_endpoint = os.getenv('R2_ENDPOINT')
        # Use provided bucket_name or fall back to environment variable
        r2_bucket_name = bucket_name or os.getenv('R2_BUCKET_NAME', 'minstraly-storage')
        r2_public_url = os.getenv('NEXT_PUBLIC_R2_PUBLIC_URL') or os.getenv('R2_PUBLIC_URL')
        
        if not all([r2_account_id, r2_access_key_id, r2_secret_access_key, r2_endpoint]):
            logger.error("❌ R2 credentials not configured")
            return None
        
        # Create S3 client for R2
        s3_client = boto3.client(
            's3',
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            region_name='auto'
        )
        
        # Prepare metadata for R2
        extra_args = {
            'ContentType': content_type,
        }
        if metadata:
            # R2 metadata must be strings and ASCII-compatible
            # Encode non-ASCII values using base64 to ensure compatibility
            encoded_metadata = {}
            for k, v in metadata.items():
                key_str = str(k)
                value_str = str(v)
                # Encode the value if it contains non-ASCII characters
                encoded_metadata[key_str] = _encode_metadata_value(value_str)
            extra_args['Metadata'] = encoded_metadata
        
        # Upload to R2
        s3_client.put_object(
            Bucket=r2_bucket_name,
            Key=destination_key,
            Body=data,
            **extra_args
        )
        
        logger.info(f"✅ Uploaded to R2: {destination_key} ({len(data)} bytes)")
        
        # Return public URL if available
        if r2_public_url:
            public_url = f"{r2_public_url.rstrip('/')}/{destination_key}"
            return public_url
        
        # Fallback: return R2 path
        return destination_key
        
    except Exception as e:
        logger.error(f"❌ R2 upload failed: {e}")
        logger.error(f"❌ R2 upload traceback: {traceback.format_exc()}")
        return None


def download_from_r2(source_key: str) -> Optional[bytes]:
    """
    Download data from Cloudflare R2 using boto3 S3 client.
    
    :param source_key: Source key/path in R2
    :return: Binary data or None if failed
    """
    try:
        import boto3
        
        # Get R2 credentials from environment
        r2_account_id = os.getenv('R2_ACCOUNT_ID')
        r2_access_key_id = os.getenv('R2_ACCESS_KEY_ID')
        r2_secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
        r2_endpoint = os.getenv('R2_ENDPOINT')
        r2_bucket_name = os.getenv('R2_BUCKET_NAME', 'minstraly-storage')
        
        if not all([r2_account_id, r2_access_key_id, r2_secret_access_key, r2_endpoint]):
            logger.error("❌ R2 credentials not configured")
            return None
        
        # Create S3 client for R2
        s3_client = boto3.client(
            's3',
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            region_name='auto'
        )
        
        # Download from R2
        response = s3_client.get_object(
            Bucket=r2_bucket_name,
            Key=source_key
        )
        
        data = response['Body'].read()
        logger.info(f"✅ Downloaded from R2: {source_key} ({len(data)} bytes)")
        return data
        
    except Exception as e:
        logger.error(f"❌ R2 download failed: {e}")
        logger.error(f"❌ R2 download traceback: {traceback.format_exc()}")
        return None


def init_firestore_client():
    """Initialize Firestore using explicit service account if provided.

    Prefers RUNPOD_SECRET_Firebase (JSON) to build credentials explicitly so we
    don't rely on ambient ADC in the RunPod runtime. Falls back to default client.
    """
    try:
        import json
        from google.cloud import firestore
        from google.oauth2 import service_account  # type: ignore

        sa_json_str = os.environ.get("RUNPOD_SECRET_Firebase")
        if sa_json_str:
            try:
                sa_info = json.loads(sa_json_str)
                credentials = service_account.Credentials.from_service_account_info(sa_info)
                project_id = sa_info.get("project_id")
                client = firestore.Client(project=project_id, credentials=credentials)
                return client
            except Exception as inner_e:
                logger.warning(f"⚠️ Failed to init Firestore from RUNPOD_SECRET_Firebase; falling back to default ADC: {inner_e}")

        # Fallback to default ADC
        return firestore.Client()
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firestore client: {e}")
        return None

