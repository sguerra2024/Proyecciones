from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"Bloque&Varid", "Tallos/m2", "Produccion"}


def load_excel_dataframe(source: Any) -> pd.DataFrame:
    try:
        df = pd.read_excel(source, engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"No fue posible leer el Excel: {exc}") from exc

    if df.empty:
        raise ValueError("El archivo Excel no contiene datos.")

    return df


def load_excel_bytes(raw: bytes) -> pd.DataFrame:
    return load_excel_dataframe(BytesIO(raw))


def validate_required_columns(df: pd.DataFrame, required_columns: set[str] | None = None) -> None:
    expected = required_columns or REQUIRED_COLUMNS
    missing = expected - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Faltan columnas requeridas: {missing_text}")


def get_fincas(df: pd.DataFrame) -> list[str]:
    if "Finca" not in df.columns:
        return []

    fincas = df["Finca"].dropna().astype(str).unique().tolist()
    return sorted(fincas)


def get_varieties(df: pd.DataFrame, finca: str | None = None) -> list[str]:
    working_df = df
    if finca and "Finca" in working_df.columns:
        working_df = working_df[working_df["Finca"].astype(str) == str(finca)]

    varieties = working_df["Bloque&Varid"].dropna().astype(
        str).unique().tolist()
    return sorted(varieties)


def train_projection_model(df: pd.DataFrame, selected_var: str | None) -> dict[str, Any]:
    validate_required_columns(df)

    if selected_var:
        subset = df[df["Bloque&Varid"].astype(str) == str(selected_var)].copy()
    else:
        first_var = df["Bloque&Varid"].dropna().astype(str)
        if first_var.empty:
            raise ValueError("No hay variedades disponibles en el archivo.")
        selected_var = first_var.iloc[0]
        subset = df[df["Bloque&Varid"].astype(str) == str(selected_var)].copy()

    subset = subset.dropna(
        subset=["Tallos/m2", "Produccion"]).reset_index(drop=True)
    if len(subset) < 8:
        raise ValueError(
            "No hay suficientes filas para entrenar el modelo de la variedad seleccionada.")

    features = subset[["Tallos/m2"]].astype(float)
    target = subset["Produccion"].astype(float)

    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(features)
    rmse = float(np.sqrt(np.mean((predictions - target.to_numpy()) ** 2)))

    chart_df = pd.DataFrame(
        {
            "tallos_m2": features["Tallos/m2"].round(4),
            "produccion_real": target.round(2),
            "prediccion_modelo": np.round(predictions, 2),
        }
    )

    return {
        "selected_var": str(selected_var),
        "rows_used": int(len(subset)),
        "rmse": rmse,
        "chart_df": chart_df,
        "preview": chart_df.tail(20).to_dict(orient="records"),
    }


def find_reference_pattern(df: pd.DataFrame, selected_var: str) -> dict[str, Any] | None:
    validate_required_columns(df, {"Bloque&Varid", "Tallos/m2"})

    target_rows = df[df["Bloque&Varid"].astype(
        str) == str(selected_var)].copy()
    target_series = pd.to_numeric(
        target_rows["Tallos/m2"], errors="coerce").dropna().reset_index(drop=True)
    if len(target_series) < 4:
        return None

    candidates: list[dict[str, Any]] = []
    grouped = df.dropna(subset=["Bloque&Varid"]).groupby("Bloque&Varid")
    for candidate_var, candidate_rows in grouped:
        candidate_name = str(candidate_var)
        if candidate_name == str(selected_var):
            continue

        candidate_series = pd.to_numeric(
            candidate_rows["Tallos/m2"], errors="coerce").dropna().reset_index(drop=True)
        common_length = min(len(target_series), len(candidate_series))
        if common_length < 4:
            continue

        mse = float(
            np.mean(
                (
                    target_series.iloc[:common_length].to_numpy()
                    - candidate_series.iloc[:common_length].to_numpy()
                )
                ** 2
            )
        )
        candidates.append(
            {
                "reference_var": candidate_name,
                "mse": mse,
                "series": candidate_series,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["mse"])
    return candidates[0]


def scale_reference_projection(reference_series: pd.Series, actual_series: pd.Series) -> pd.Series:
    reference_values = pd.to_numeric(
        reference_series, errors="coerce").dropna().reset_index(drop=True)
    actual_values = pd.to_numeric(
        actual_series, errors="coerce").dropna().reset_index(drop=True)
    if reference_values.empty or actual_values.empty:
        return pd.Series(dtype=float)

    common_length = min(len(reference_values), len(actual_values))
    reference_values = reference_values.iloc[:common_length]
    actual_values = actual_values.iloc[:common_length]

    reference_mean = float(reference_values.mean())
    scale_factor = 1.0 if reference_mean == 0 else float(
        actual_values.mean()) / reference_mean
    return (reference_values * scale_factor).reset_index(drop=True)
