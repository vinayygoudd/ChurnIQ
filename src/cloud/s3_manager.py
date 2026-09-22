from __future__ import annotations

import logging
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.cloud.aws_config import aws_config

logger = logging.getLogger(__name__)


class S3Manager:
    """Small, environment-driven S3 client for ChurnIQ artifacts."""

    def __init__(self, bucket: str | None = None, prefix: str | None = None):
        self.bucket = bucket or aws_config.bucket
        self.prefix = (prefix or aws_config.prefix).strip("/")
        if not self.bucket:
            raise ValueError("AWS_S3_BUCKET is not configured.")

        self.client = boto3.client(
            "s3",
            region_name=aws_config.region,
            endpoint_url=aws_config.endpoint_url,
        )

    def _key(self, filename: str) -> str:
        filename = filename.lstrip("/")
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def upload_file(self, local_path: str | Path, remote_name: str | None = None) -> str:
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        key = self._key(remote_name or path.name)
        try:
            self.client.upload_file(str(path), self.bucket, key)
        except (BotoCoreError, ClientError):
            logger.exception("S3 upload failed for %s", path)
            raise

        logger.info("Uploaded %s to s3://%s/%s", path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download_file(self, remote_name: str, local_path: str | Path) -> Path:
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        key = self._key(remote_name)

        try:
            self.client.download_file(self.bucket, key, str(destination))
        except (BotoCoreError, ClientError):
            logger.exception("S3 download failed for %s", key)
            raise

        logger.info("Downloaded s3://%s/%s to %s", self.bucket, key, destination)
        return destination

    def exists(self, remote_name: str) -> bool:
        key = self._key(remote_name)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
