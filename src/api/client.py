from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_TIMEOUT = 30


class ChurnAPIError(RuntimeError):
    pass


def api_base_url() -> str:
    """Return the production API URL from CHURN_API_URL."""
    url = os.getenv("CHURN_API_URL", "").strip().rstrip("/")
    if not url:
        raise ChurnAPIError(
            "CHURN_API_URL is not configured. Set it to the AWS EC2 Flask API URL."
        )
    return url


def _request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    url = f"{api_base_url()}/{path.lstrip('/')}"
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)

    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ChurnAPIError(f"API request failed: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise ChurnAPIError("API returned a non-JSON response.") from exc


def health() -> dict[str, Any]:
    return _request("GET", "/health")


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    return _request("POST", "/predict", json=payload)


def explain(payload: dict[str, Any]) -> dict[str, Any]:
    return _request("POST", "/explain", json=payload)


def model_info() -> dict[str, Any]:
    return _request("GET", "/model-info")


def analytics() -> dict[str, Any]:
    return _request("GET", "/analytics")
