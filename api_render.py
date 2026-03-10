from __future__ import annotations

import os
import secrets
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"Bloque&Varid", "Tallos/m2", "Produccion"}
MAX_FILE_SIZE_MB = 15


def _build_cors_origins() -> list[str]:
    origins_raw = os.getenv("FRONTEND_ORIGINS", "")
    origins = [origin.strip() for origin in origins_raw.split(",") if origin.strip()]
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


def _validate_api_key(x_api_key: str | None) -> None:
    expected = os.getenv("API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="API_KEY is not configured in server environment.",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _read_excel_from_upload(file: UploadFile, raw: bytes) -> pd.DataFrame:
    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are allowed.")

    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_MB}MB.",
        )

    try:
        df = pd.read_excel(BytesIO(raw), engine="openpyxl")
    except Exception as exc:  # pragma: no cover - defensive validation
        raise HTTPException(status_code=400, detail=f"Invalid Excel file: {exc}") from exc

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="Excel file is empty.")

    return df


def _train_projection_model(df: pd.DataFrame, selected_var: str | None) -> dict[str, Any]:
    if selected_var:
        subset = df[df["Bloque&Varid"] == selected_var].copy()
    else:
        # Default: use first available variety.
        selected_var = str(df["Bloque&Varid"].dropna().iloc[0])
        subset = df[df["Bloque&Varid"] == selected_var].copy()

    if subset.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No rows found for selected_var='{selected_var}'.",
        )

    subset = subset.dropna(subset=["Tallos/m2", "Produccion"])
    if len(subset) < 8:
        raise HTTPException(
            status_code=400,
            detail="Not enough rows to train the model for selected_var.",
        )

    x = subset[["Tallos/m2"]].astype(float)
    y = subset["Produccion"].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x)
    rmse = float(np.sqrt(np.mean((predictions - y.to_numpy()) ** 2)))

    result_preview = pd.DataFrame(
        {
            "tallos_m2": x["Tallos/m2"].round(4),
            "produccion_real": y.round(2),
            "prediccion_modelo": np.round(predictions, 2),
        }
    )

    return {
        "selected_var": selected_var,
        "rows_used": int(len(subset)),
        "rmse": rmse,
        "preview": result_preview.tail(20).to_dict(orient="records"),
    }


@app.post("/api/v1/predict")
async def predict_from_excel(
    file: UploadFile = File(...),
    selected_var: str | None = Form(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _validate_api_key(x_api_key)

    raw = await file.read()
    df = _read_excel_from_upload(file, raw)

    result = _train_projection_model(df, selected_var)

    return {
        "ok": True,
        "result": result,
    }
