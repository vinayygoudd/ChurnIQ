"""Upload ChurnIQ model/data artifacts to S3.

Usage:
    python -m src.cloud.s3_sync
"""

from pathlib import Path

from src.cloud.s3_manager import S3Manager


ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS = [
    ROOT / "data" / "raw" / "Telco-Customer-Churn.csv",
    ROOT / "models" / "churn_model.joblib",
    ROOT / "models" / "metadata.json",
]


def main() -> None:
    manager = S3Manager()

    for path in ARTIFACTS:
        if path.exists():
            print(manager.upload_file(path))
        else:
            print(f"SKIP: {path} does not exist")


if __name__ == "__main__":
    main()
