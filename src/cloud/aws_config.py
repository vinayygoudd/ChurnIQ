import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AWSConfig:
    region: str = os.getenv("AWS_REGION", "ap-south-1")
    bucket: str = os.getenv("AWS_S3_BUCKET", "")
    prefix: str = os.getenv("AWS_S3_PREFIX", "churniq")
    endpoint_url: str | None = os.getenv("AWS_ENDPOINT_URL") or None


aws_config = AWSConfig()
