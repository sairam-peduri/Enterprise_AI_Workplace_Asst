"""FastAPI server that serves all JSON data files for the Enterprise AI Workplace Assistant.

Run with:
    uvicorn fastapi_server:app --reload --port 8000
"""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Enterprise AI Data API")

DATA_DIR = Path(__file__).resolve().parent / "data"


class DataPayload(BaseModel):
    data: Any


@app.get("/api/data/{path:path}")
def read_data(path: str):
    file_path = DATA_DIR / path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")


@app.put("/api/data/{path:path}")
def write_data(path: str, payload: DataPayload):
    file_path = DATA_DIR / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload.data, indent=2, default=str),
        encoding="utf-8",
    )
    return {"ok": True}
