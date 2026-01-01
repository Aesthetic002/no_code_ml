"""
Cloud Storage Configuration
Supports AWS S3, Supabase, or local storage fallback
"""

import os
import io
import boto3
from typing import Optional, BinaryIO
from enum import Enum

class StorageBackend(Enum):
    AWS_S3 = "aws_s3"
    SUPABASE = "supabase"
    LOCAL = "local"

class CloudStorageConfig:
    """Cloud Storage Configuration Manager"""
    
    def __init__(self):
        self.backend = os.getenv("STORAGE_BACKEND", "local").lower()
        
        # AWS S3 Configuration
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.s3_bucket = os.getenv("S3_BUCKET_NAME")
        
        # Supabase Configuration
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase_bucket = os.getenv("SUPABASE_BUCKET", "models")
        
        self.s3_client = None
        if self.backend == "aws_s3" and self.aws_access_key:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )

class CloudStorageManager:
    """Manager for uploading/downloading files to cloud storage"""
    
    def __init__(self, config: CloudStorageConfig):
        self.config = config
        self.backend = config.backend
    
    def upload_model(self, model_data: bytes, username: str, model_name: str, model_type: str) -> str:
        """
        Upload model to cloud storage
        Returns: Remote path/URL of the uploaded file
        """
        remote_path = f"{username}/{model_type}/{model_name}.pkl"
        
        if self.backend == "aws_s3":
            return self._upload_to_s3(model_data, remote_path)
        elif self.backend == "supabase":
            return self._upload_to_supabase(model_data, remote_path)
        else:
            return remote_path  # Local storage
    
    def download_model(self, remote_path: str) -> bytes:
        """Download model from cloud storage"""
        if self.backend == "aws_s3":
            return self._download_from_s3(remote_path)
        elif self.backend == "supabase":
            return self._download_from_supabase(remote_path)
        else:
            return None  # Local storage
    
    def _upload_to_s3(self, file_data: bytes, remote_path: str) -> str:
        """Upload to AWS S3"""
        try:
            print(f"📤 Uploading to S3: {self.config.s3_bucket}/{remote_path} ({len(file_data)} bytes)")
            self.config.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=remote_path,
                Body=file_data
            )
            url = f"s3://{self.config.s3_bucket}/{remote_path}"
            print(f"✅ S3 upload successful: {url}")
            return url
        except Exception as e:
            print(f"❌ S3 upload error: {str(e)}")
            raise Exception(f"S3 upload failed: {str(e)}")
    
    def _download_from_s3(self, remote_path: str) -> bytes:
        """Download from AWS S3"""
        try:
            # Extract the key from the S3 URL format (s3://bucket/key)
            if remote_path.startswith("s3://"):
                # Format: s3://bucket-name/username/model_type/model_name.pkl
                key = remote_path.split("/", 3)[3]  # Get everything after s3://bucket/
            else:
                # Already a key path
                key = remote_path
            
            print(f"📥 Downloading from S3: {self.config.s3_bucket}/{key}")
            response = self.config.s3_client.get_object(
                Bucket=self.config.s3_bucket,
                Key=key
            )
            data = response['Body'].read()
            print(f"✅ S3 download successful: {len(data)} bytes")
            return data
        except Exception as e:
            print(f"❌ S3 download error: {str(e)}")
            raise Exception(f"S3 download failed: {str(e)}")
    
    def _upload_to_supabase(self, file_data: bytes, remote_path: str) -> str:
        """Upload to Supabase Storage"""
        try:
            from supabase import create_client, Client
            
            supabase: Client = create_client(
                self.config.supabase_url,
                self.config.supabase_key
            )
            
            supabase.storage.from_(self.config.supabase_bucket).upload(
                remote_path,
                file_data
            )
            
            # Get public URL
            url = supabase.storage.from_(self.config.supabase_bucket).get_public_url(remote_path)
            return url
        except Exception as e:
            raise Exception(f"Supabase upload failed: {str(e)}")
    
    def _download_from_supabase(self, remote_path: str) -> bytes:
        """Download from Supabase Storage"""
        try:
            from supabase import create_client, Client
            
            supabase: Client = create_client(
                self.config.supabase_url,
                self.config.supabase_key
            )
            
            response = supabase.storage.from_(self.config.supabase_bucket).download(remote_path)
            return response
        except Exception as e:
            raise Exception(f"Supabase download failed: {str(e)}")
