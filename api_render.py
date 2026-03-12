from __future__ import annotations

import os
import secrets
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from projection_core import load_excel_bytes, train_projection_model, validate_required_columns

MAX_FILE_SIZE_MB = 15


def _build_cors_origins() -> list[str]:
    origins_raw = os.getenv("FRONTEND_ORIGINS", "")
    origins = [origin.strip()
               for origin in origins_raw.split(",") if origin.strip()]
    if origins:
        return origins
    # Default local origins for development only.
    return ["http://localhost:3000", "http://localhost:8501"]


app = FastAPI(title="Proyecciones API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_cors_origins(),
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "Proyecciones API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "predict_compat": "/predict",
        },
    }


@app.exception_handler(404)
async def not_found_handler(_: Request, __: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "ok": False,
            "detail": "Route not found.",
            "expected": ["/", "/health", "/api/v1/predict", "/predict"],
        },
    )


def _validate_api_key(x_api_key: str | None) -> None:
    expected = os.getenv("API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="API_KEY is not configured in server environment.",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _read_excel_from_upload(file: UploadFile, raw: bytes):
    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=400, detail="Only .xlsx files are allowed.")

    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_MB}MB.",
        )

    try:
        df = load_excel_bytes(raw)
        validate_required_columns(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return df


@app.post("/api/v1/predict")
async def predict_from_excel(
    file: UploadFile = File(...),
    selected_var: str | None = Form(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _validate_api_key(x_api_key)

    raw = await file.read()
    df = _read_excel_from_upload(file, raw)

    try:
        result = train_projection_model(df, selected_var)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "result": result,
    }


@app.post("/predict")
async def predict_from_excel_compat(
    file: UploadFile = File(...),
    selected_var: str | None = Form(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    return await predict_from_excel(
        file=file,
        selected_var=selected_var,
        x_api_key=x_api_key,
    )


@app.post("/ProyAst.py")
async def predict_from_excel_legacy(
    file: UploadFile = File(...),
    selected_var: str | None = Form(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    return await predict_from_excel(
        file=file,
        selected_var=selected_var,
        x_api_key=x_api_key,
    )


@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def json_fallback(path_name: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "ok": False,
            "detail": "Route not found.",
            "requested_path": f"/{path_name}",
            "expected": ["/", "/health", "/api/v1/predict", "/predict", "/ProyAst.py"],
        },
    )
