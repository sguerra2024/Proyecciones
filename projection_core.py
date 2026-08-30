from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REQUIRED_COLUMNS = {"Bloque&Varid", "Tallos/m2", "Produccion"}
PRODUCTION_REQUIRED_COLUMNS = REQUIRED_COLUMNS | {
    "Anio", "Semana", "m2Variedad"
}
MODEL_COLUMNS = [
    "Tallos/m2",
    "Tallos_m2_patron",
    "Produccion_patron",
    "Tallos_m2_patron_ponderado",
    "Produccion_patron_ponderado",
    "Incremento_tallos_patron",
    "Incremento_produccion_patron",
    "sn_alto",
]
MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "max_depth": 16,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": "sqrt",
}


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


def build_production_model() -> RandomForestRegressor:
    """Crea el unico Random Forest usado por Streamlit y la API."""
    return RandomForestRegressor(**MODEL_PARAMS)


def fit_production_model(
    features: pd.DataFrame,
    target: "pd.Series | np.ndarray",
) -> RandomForestRegressor:
    if len(features) < 5:
        raise ValueError("No hay suficientes datos para entrenar el modelo.")

    model = build_production_model()
    model.fit(features, np.asarray(target, dtype=float).reshape(-1))
    return model


def _exclude_latest_four_weeks(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["Anio"] = pd.to_numeric(working["Anio"], errors="coerce")
    working["Semana"] = pd.to_numeric(working["Semana"], errors="coerce")
    working = working.dropna(subset=["Anio", "Semana"])
    working = working.sort_values(
        ["Anio", "Semana"], ascending=[False, False]
    ).iloc[4:]
    return working.sort_values(["Anio", "Semana"]).reset_index(drop=True)


def _build_weekly_pattern(pattern_rows: pd.DataFrame) -> pd.DataFrame:
    pattern = pattern_rows[[
        "Anio", "Semana", "Tallos/m2", "Produccion"
    ]].copy()
    for column in ["Anio", "Semana", "Tallos/m2", "Produccion"]:
        pattern[column] = pd.to_numeric(pattern[column], errors="coerce")
    pattern = pattern.dropna(subset=["Anio", "Semana"])
    pattern[["Anio", "Semana"]] = pattern[["Anio", "Semana"]].astype(int)
    pattern = (
        pattern.groupby(["Anio", "Semana"], as_index=False)
        .agg({"Tallos/m2": "mean", "Produccion": "sum"})
        .rename(columns={
            "Tallos/m2": "Tallos_m2_patron",
            "Produccion": "Produccion_patron",
        })
        .sort_values(["Anio", "Semana"])
        .reset_index(drop=True)
    )
    pattern["Incremento_tallos_patron"] = (
        pattern["Tallos_m2_patron"].diff().fillna(0.0)
    )
    pattern["Incremento_produccion_patron"] = (
        pattern["Produccion_patron"].diff().fillna(0.0)
    )
    return pattern


def _prepare_production_dataset(
    target_rows: pd.DataFrame,
    weekly_pattern: pd.DataFrame,
    pattern_weight: float = 1.0,
) -> pd.DataFrame:
    working = target_rows[[
        "Anio", "Semana", "Tallos/m2", "Produccion"
    ]].copy()
    for column in ["Anio", "Semana", "Tallos/m2", "Produccion"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna().sort_values(["Anio", "Semana"])
    working[["Anio", "Semana"]] = working[["Anio", "Semana"]].astype(int)
    working = working.merge(weekly_pattern, on=["Anio", "Semana"], how="left")
    working["Tallos_m2_patron"] = working["Tallos_m2_patron"].fillna(
        working["Tallos/m2"]
    )
    working["Produccion_patron"] = working["Produccion_patron"].fillna(
        working["Produccion"]
    )
    working["Incremento_tallos_patron"] = working[
        "Incremento_tallos_patron"
    ].fillna(0.0)
    working["Incremento_produccion_patron"] = working[
        "Incremento_produccion_patron"
    ].fillna(0.0)
    weight = float(np.clip(pattern_weight, 0.0, 1.0))
    working["Tallos_m2_patron_ponderado"] = (
        (1.0 - weight) * working["Tallos/m2"]
        + weight * working["Tallos_m2_patron"]
    )
    working["Produccion_patron_ponderado"] = (
        (1.0 - weight) * working["Produccion"]
        + weight * working["Produccion_patron"]
    )
    working["sn_alto"] = 0.0
    return working.reset_index(drop=True)


def _load_difference_factors() -> dict[str, Any]:
    evaluation_dir = Path(__file__).with_name("Evaluacion")
    factors_path = evaluation_dir / "factor_diferencia_2025_por_variedad.csv"
    summary_path = evaluation_dir / "factor_diferencia_2025_resumen.csv"
    factors: dict[str, float] = {}
    global_factor = 1.0

    if factors_path.exists():
        factor_df = pd.read_csv(factors_path, encoding="utf-8-sig")
        if {"Bloque&Varid", "Factor_diferencia"}.issubset(factor_df.columns):
            for _, row in factor_df.iterrows():
                value = pd.to_numeric(
                    row["Factor_diferencia"], errors="coerce")
                if pd.notna(value) and np.isfinite(value) and value > 0:
                    factors[str(row["Bloque&Varid"]).strip()] = float(value)

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
        column = "Factor_global_ponderado_sumas"
        if not summary_df.empty and column in summary_df.columns:
            value = pd.to_numeric(summary_df.loc[0, column], errors="coerce")
            if pd.notna(value) and np.isfinite(value) and value > 0:
                global_factor = float(value)
    return {"factores_por_variedad": factors, "factor_global": global_factor}


def _apply_difference_factor(
    predictions: np.ndarray,
    evaluation_df: pd.DataFrame,
    selected_var: str,
) -> tuple[np.ndarray, float, int, str]:
    config = _load_difference_factors()
    factors = config["factores_por_variedad"]
    if selected_var in factors:
        factor = float(factors[selected_var])
        origin = "variedad"
    else:
        factor = float(config["factor_global"])
        origin = "global"

    adjusted = np.asarray(predictions, dtype=float).copy()
    years = pd.to_numeric(evaluation_df["Anio"], errors="coerce").to_numpy()
    weeks = pd.to_numeric(evaluation_df["Semana"], errors="coerce").to_numpy()
    candidates = np.where((years == 2026) & (weeks > 24))[0]
    selected = sorted(candidates, key=lambda index: (
        weeks[index], index), reverse=True)[:4]
    adjusted[selected] *= factor
    return adjusted, factor, len(selected), origin


def train_projection_model(
    df: pd.DataFrame,
    selected_var: str | None,
    reference_series: "pd.Series | None" = None,
    include_chart_df: bool = True,
) -> dict[str, Any]:
    """Proyecta Produccion con el mismo modelo usado por ProyAst Streamlit."""
    validate_required_columns(df, PRODUCTION_REQUIRED_COLUMNS)

    if selected_var:
        subset = df[df["Bloque&Varid"].astype(str) == str(selected_var)].copy()
    else:
        first_var = df["Bloque&Varid"].dropna().astype(str)
        if first_var.empty:
            raise ValueError("No hay variedades disponibles en el archivo.")
        selected_var = first_var.iloc[0]
        subset = df[df["Bloque&Varid"].astype(str) == str(selected_var)].copy()

    pattern_info = find_reference_pattern(df, str(selected_var))
    if pattern_info is None:
        raise ValueError(
            "No hay un patron diferente disponible para la variedad.")
    pattern_var = str(pattern_info["reference_var"])
    pattern_rows = df[df["Bloque&Varid"].astype(str) == pattern_var].copy()
    weekly_pattern = _build_weekly_pattern(pattern_rows)
    evaluation_df = _prepare_production_dataset(subset, weekly_pattern)
    training_df = evaluation_df[evaluation_df["Anio"] >= 2025].copy()
    training_df = _exclude_latest_four_weeks(training_df)
    if len(evaluation_df) == 0:
        raise ValueError("No hay datos validos para generar la proyeccion.")

    target = (
        0.5 * training_df["Produccion"].to_numpy(dtype=float)
        + 0.5 * training_df["Produccion_patron"].to_numpy(dtype=float)
    )
    features = training_df[MODEL_COLUMNS].reset_index(drop=True)
    features["Semana_orden"] = np.arange(len(features), dtype=float)
    prediction_features = evaluation_df[MODEL_COLUMNS].reset_index(drop=True)
    prediction_features["Semana_orden"] = np.arange(
        len(prediction_features), dtype=float
    )
    model = fit_production_model(features, target)
    predictions = model.predict(prediction_features)
    real_production = evaluation_df["Produccion"].to_numpy(dtype=float)
    prediction_mean = float(np.nanmean(predictions))
    real_mean = float(np.nanmean(real_production))
    if prediction_mean != 0 and not np.isclose(prediction_mean, real_mean):
        predictions *= real_mean / prediction_mean
    predictions, factor, affected_weeks, factor_origin = _apply_difference_factor(
        predictions, evaluation_df, str(selected_var)
    )
    metric_df = _exclude_latest_four_weeks(
        evaluation_df.assign(Estimado_modelo=predictions)
    )
    rmse = float(np.sqrt(np.mean(
        (metric_df["Estimado_modelo"] - metric_df["Produccion"]) ** 2
    )))

    chart_df = pd.DataFrame(
        {
            "anio": evaluation_df["Anio"].astype(int),
            "semana": evaluation_df["Semana"].astype(int),
            "produccion_real": real_production.round(2),
            "produccion_patron": evaluation_df["Produccion_patron"].round(2),
            "estimado_modelo": np.round(predictions, 2),
        }
    )

    result = {
        "selected_var": str(selected_var),
        "reference_var": pattern_var,
        "rows_used": int(len(training_df)),
        "rmse": rmse,
        "using_reference_pattern": True,
        "production_avg": real_mean,
        "factor_diferencia_2025": factor,
        "factor_origin": factor_origin,
        "factor_affected_weeks": affected_weeks,
        "preview": chart_df.tail(20).to_dict(orient="records"),
    }
    if include_chart_df:
        result["chart_df"] = chart_df

    return result


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
