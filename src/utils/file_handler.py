"""Utility functions for reading and writing JSON files.

Falls back to direct file I/O when the FastAPI data server is unavailable.
"""

import json
import os
from pathlib import Path
from typing import Any

import requests

API_BASE = os.environ.get("DATA_API_URL", "http://localhost:8000")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _to_relative(file_path: Path) -> str:
    return str(file_path.relative_to(DATA_DIR)).replace("\\", "/")


def load_json(file_path: Path) -> Any:
    try:
        resp = requests.get(f"{API_BASE}/api/data/{_to_relative(file_path)}", timeout=3)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return resp.json()
    except (requests.ConnectionError, requests.Timeout):
        if file_path.exists():
            return json.loads(file_path.read_text(encoding="utf-8"))
        return []


def save_json(file_path: Path, data: Any) -> None:
    try:
        resp = requests.put(
            f"{API_BASE}/api/data/{_to_relative(file_path)}",
            json={"data": data},
            timeout=3,
        )
        resp.raise_for_status()
    except (requests.ConnectionError, requests.Timeout):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
