from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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
]
MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "max_depth": 16,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": "sqrt",
}
BUFFER_COLUMNS = [
    "Prediccion_base", "m2Variedad", "Tallos/m2", "Semana", "Dif_previo"
]
DEFAULT_MAX_BUFFER_RATE = 0.10
BUFFER_RISK_THRESHOLD = 0.60
DEFAULT_EVALUATED_WEEKS = 16
REAL_PRODUCTION_TARGET_WEIGHT = 0.55
PESO_AJUSTE_REAL_2026 = 1.5
PATTERN_PRODUCTION_TARGET_WEIGHT = 0.45


def calculate_evaluation_area_metrics(
    relative_error: "pd.Series | np.ndarray",
) -> dict[str, float | int]:
    """Calcula las areas firmadas de %dif respecto al eje cero.

    El area positiva corresponde a subestimacion y el area negativa conserva
    su signo porque representa sobreestimacion. Estas areas son la senal
    integral que utiliza el amortiguador como actuador.
    """
    values = pd.to_numeric(
        pd.Series(relative_error), errors="coerce"
    ).dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    area_positive = float(np.maximum(values, 0.0).sum())
    area_negative = float(np.minimum(values, 0.0).sum())
    return {
        "area_positive": area_positive,
        "area_negative": area_negative,
        "area_net": area_positive + area_negative,
        "evaluation_cases": int(values.size),
    }


def load_overestimation_calibration(
    evaluation_path: "Path | None" = None,
) -> dict[str, float | int | str]:
    """Carga la evaluacion real que calibra el actuador del amortiguador."""
    path = evaluation_path or (
        Path(__file__).with_name("Evaluacion")
        / "errores_evaluacion_modelo.csv"
    )
    if not path.exists():
        raise ValueError(
            "No existe una evaluacion real inicial; el amortiguador no puede calcularse."
        )

    try:
        evaluation = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(
            f"No fue posible leer la evaluacion real: {path}"
        ) from exc
    error_column = next(
        (
            column for column in evaluation.columns
            if str(column).strip().casefold() in {"error_relativo", "%dif"}
        ),
        None,
    )
    if error_column is None:
        raise ValueError(
            "La evaluacion real no contiene la columna %dif o error_relativo."
        )

    relative_error = pd.to_numeric(
        evaluation[error_column], errors="coerce"
    ).dropna()
    relative_error = relative_error[relative_error < 1.0]
    if relative_error.empty:
        raise ValueError(
            "La evaluacion real no contiene errores relativos validos."
        )

    absolute_error_on_prediction = (
        relative_error.abs() / (1.0 - relative_error)
    )
    areas = calculate_evaluation_area_metrics(relative_error)
    evaluated_weeks = DEFAULT_EVALUATED_WEEKS
    if {"Anio", "Semana"}.issubset(evaluation.columns):
        evaluated_periods = evaluation[["Anio", "Semana"]].copy()
        evaluated_periods["Anio"] = pd.to_numeric(
            evaluated_periods["Anio"], errors="coerce"
        )
        evaluated_periods["Semana"] = pd.to_numeric(
            evaluated_periods["Semana"], errors="coerce"
        )
        evaluated_periods = evaluated_periods.dropna().drop_duplicates()
        if not evaluated_periods.empty:
            evaluated_weeks = int(len(evaluated_periods))

    return {
        "max_buffer_rate": float(absolute_error_on_prediction.median()),
        "risk_threshold": BUFFER_RISK_THRESHOLD,
        "error_cases": int(len(relative_error)),
        "overestimation_cases": int((relative_error < 0).sum()),
        "underestimation_cases": int((relative_error > 0).sum()),
        "area_positive": areas["area_positive"],
        "area_negative": areas["area_negative"],
        "area_net": areas["area_net"],
        "evaluation_cases": areas["evaluation_cases"],
        "evaluated_weeks": evaluated_weeks,
        "source": str(path),
    }


def save_buffer_evaluation_report(
    excel_path: "str | Path",
    sheet_name: "str | int" = 0,
    destination_path: "str | Path | None" = None,
) -> dict[str, Any]:
    """Recalcula %dif desde un Excel y persiste el informe del amortiguador."""
    evaluation = pd.read_excel(excel_path, sheet_name=sheet_name)
    normalized_columns = {
        str(column).strip().casefold(): column for column in evaluation.columns
    }
    production_column = normalized_columns.get("produccion")
    estimate_column = normalized_columns.get("estimado_modelo")
    if production_column is None or estimate_column is None:
        raise ValueError(
            "El informe requiere Produccion y Estimado_modelo para calcular %dif."
        )

    report = evaluation.copy()
    error_report = calculate_model_error_report(
        report[production_column], report[estimate_column]
    )
    report["%dif"] = error_report["%dif"].to_numpy()
    report["error_relativo"] = report["%dif"]
    area_metrics = calculate_evaluation_area_metrics(report["%dif"])
    report["area_subestimacion"] = area_metrics["area_positive"]
    report["area_sobreestimacion"] = area_metrics["area_negative"]
    report["area_neta"] = area_metrics["area_net"]

    semana_column = normalized_columns.get("semana")
    bloque_column = normalized_columns.get("bloque")
    variedad_column = normalized_columns.get("variedad")
    if semana_column and bloque_column and variedad_column:
        semana_str = pd.to_numeric(
            report[semana_column], errors="coerce"
        ).astype("Int64").astype(str)
        bloque_str = pd.to_numeric(
            report[bloque_column], errors="coerce"
        ).astype("Int64").astype(str).str.zfill(3)
        variedad_str = report[variedad_column].astype(str).str.strip()
        report["Semana&Bloque&Varid"] = (
            semana_str + "&" + bloque_str + "&" + variedad_str
        )

    destination = Path(destination_path) if destination_path else (
        Path(__file__).with_name("Evaluacion")
        / "errores_evaluacion_modelo.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False, encoding="utf-8-sig")

    periods = pd.DataFrame()
    if {"anio", "semana"}.issubset(normalized_columns):
        periods = report[[
            normalized_columns["anio"], normalized_columns["semana"]
        ]].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates()
    return {
        "rows": int(report["%dif"].notna().sum()),
        "evaluated_weeks": int(len(periods)),
        **area_metrics,
        "source": str(destination),
        "values": report["%dif"].dropna().tolist(),
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


def calculate_model_error_report(
    actual_values: "pd.Series | np.ndarray",
    estimated_values: "pd.Series | np.ndarray",
) -> pd.DataFrame:
    """Calcula el error real firmado: %dif = (real - estimado) / real."""
    actual = pd.to_numeric(
        pd.Series(actual_values).reset_index(drop=True), errors="coerce"
    )
    estimated = pd.to_numeric(
        pd.Series(estimated_values).reset_index(drop=True), errors="coerce"
    )
    if len(actual) != len(estimated):
        raise ValueError(
            "Produccion real y estimado deben tener igual longitud."
        )

    report = pd.DataFrame({
        "Produccion": actual,
        "Estimado_modelo": estimated,
    })
    valid = (
        report["Produccion"].notna()
        & report["Estimado_modelo"].notna()
        & ~np.isclose(report["Produccion"], 0.0)
    )
    report["%dif"] = np.nan
    report.loc[valid, "%dif"] = (
        report.loc[valid, "Produccion"]
        - report.loc[valid, "Estimado_modelo"]
    ) / report.loc[valid, "Produccion"]
    return report


def apply_overestimation_buffer(
    production_model: RandomForestRegressor,
    training_features: pd.DataFrame,
    training_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    predictions: np.ndarray,
    m2_variedad: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Estima un amortiguador bilateral sin alterar la prediccion oficial."""
    base_predictions = np.asarray(predictions, dtype=float).reshape(-1)
    history = training_df.reset_index(drop=True).copy()
    history_features = training_features.reset_index(drop=True).copy()
    if len(history) != len(history_features):
        raise ValueError(
            "Datos y variables historicas del amortiguador deben tener igual longitud."
        )

    calibration = load_overestimation_calibration()
    evaluated_weeks = max(
        1,
        int(calibration.get("evaluated_weeks", DEFAULT_EVALUATED_WEEKS)),
    )
    exclude_latest_weeks = 4 if evaluated_weeks >= DEFAULT_EVALUATED_WEEKS else 0

    if {"Anio", "Semana"}.issubset(history.columns):
        order = history.assign(
            __position=np.arange(len(history)),
            __anio=pd.to_numeric(history["Anio"], errors="coerce"),
            __semana=pd.to_numeric(history["Semana"], errors="coerce"),
        ).dropna(subset=["__anio", "__semana"])
        positions = order.sort_values(["__anio", "__semana"])[
            "__position"
        ].to_numpy(dtype=int)
        if len(positions) >= evaluated_weeks:
            positions = positions[-evaluated_weeks:]
            if exclude_latest_weeks and len(positions) > exclude_latest_weeks:
                positions = positions[:-exclude_latest_weeks]
        else:
            positions = positions[-len(positions):]
        history = history.iloc[positions].reset_index(drop=True)
        history_features = history_features.iloc[positions].reset_index(
            drop=True)
    else:
        history = history.tail(evaluated_weeks).reset_index(drop=True)
        history_features = history_features.tail(
            evaluated_weeks
        ).reset_index(drop=True)
        if len(history) >= evaluated_weeks:
            if exclude_latest_weeks and len(history) > exclude_latest_weeks:
                history = history.iloc[:-
                                       exclude_latest_weeks].reset_index(drop=True)
                history_features = history_features.iloc[
                    :-exclude_latest_weeks
                ].reset_index(drop=True)

    historical_predictions = production_model.predict(history_features)
    error_report = calculate_model_error_report(
        history["Produccion"], historical_predictions
    )
    actual = error_report["Produccion"].to_numpy(dtype=float)
    relative_error = error_report["%dif"].to_numpy(dtype=float)
    signed_error = relative_error * actual
    finite_signed_error = signed_error[np.isfinite(signed_error)]
    average_error = float(np.max(finite_signed_error)
                          ) if finite_signed_error.size else 0.0
    valid_area = float(m2_variedad)
    if not np.isfinite(valid_area) or valid_area <= 0:
        raise ValueError("m2Variedad debe ser un numero mayor que cero.")

    valid_errors = np.isfinite(signed_error)
    if not np.all(valid_errors):
        history = history.loc[valid_errors].reset_index(drop=True)
        history_features = history_features.loc[valid_errors].reset_index(
            drop=True
        )
        historical_predictions = historical_predictions[valid_errors]
        signed_error = signed_error[valid_errors]

    if signed_error.size == 0 or not np.any(~np.isclose(signed_error, 0.0)):
        return base_predictions.copy(), np.zeros_like(base_predictions), average_error, 0.0

    max_buffer_rate = float(calibration["max_buffer_rate"])
    risk_threshold = float(calibration["risk_threshold"])
    evaluation_cases = int(calibration.get("evaluation_cases", 0))
    if evaluation_cases <= 0:
        raise ValueError(
            "El amortiguador requiere una evaluacion real con al menos un caso."
        )
    # Actuador integral: el area neta respecto al eje cero conserva la
    # memoria de la evaluacion. Positiva pide reservar; negativa pide liberar.
    area_control = float(calibration["area_net"]) / evaluation_cases

    signed_error_per_m2 = signed_error / valid_area

    # Senal de retroalimentacion (equivalente al retorno de un actuador): el
    # %dif firmado por m2 del periodo anterior. En entrenamiento se usa el
    # valor propio del primer registro (no hay periodo previo disponible);
    # en prediccion se arrastra el ultimo valor real conocido, ya que el
    # error real de las semanas futuras aun no existe.
    dif_previo_train = (
        np.concatenate([signed_error_per_m2[:1], signed_error_per_m2[:-1]])
        if signed_error_per_m2.size else signed_error_per_m2
    )
    ultimo_dif_conocido = (
        float(signed_error_per_m2[-1]) if signed_error_per_m2.size else 0.0
    )

    train_buffer_features = pd.DataFrame({
        "Prediccion_base": historical_predictions,
        "m2Variedad": np.full(len(historical_predictions), valid_area),
        "Tallos/m2": history["Tallos/m2"].to_numpy(dtype=float),
        "Semana": history["Semana"].to_numpy(dtype=float),
        "Dif_previo": dif_previo_train,
    })
    prediction_buffer_features = pd.DataFrame({
        "Prediccion_base": base_predictions,
        "m2Variedad": np.full(len(base_predictions), valid_area),
        "Tallos/m2": evaluation_df["Tallos/m2"].to_numpy(dtype=float),
        "Semana": evaluation_df["Semana"].to_numpy(dtype=float),
        "Dif_previo": np.full(len(base_predictions), ultimo_dif_conocido),
    })
    buffer_model = RandomForestRegressor(**MODEL_PARAMS)
    buffer_model.fit(
        train_buffer_features[BUFFER_COLUMNS], signed_error_per_m2
    )
    estimated_error = buffer_model.predict(
        prediction_buffer_features[BUFFER_COLUMNS]
    ) * valid_area
    estimated_error += base_predictions * area_control
    negative_error_ratio = float(np.mean(signed_error < 0.0))
    positive_error_ratio = float(np.mean(signed_error > 0.0))
    dominant_direction = None
    if negative_error_ratio >= 0.75:
        dominant_direction = -1.0
        estimated_error = -np.abs(estimated_error)
    elif positive_error_ratio >= 0.75:
        dominant_direction = 1.0
        estimated_error = np.abs(estimated_error)

    direction_labels = signed_error > 0
    if dominant_direction is not None:
        direction_probability = np.ones_like(base_predictions)
    elif np.unique(direction_labels).size == 1:
        direction_probability = np.ones_like(base_predictions)
    else:
        risk_model = RandomForestClassifier(
            **MODEL_PARAMS, class_weight="balanced")
        risk_model.fit(
            train_buffer_features[BUFFER_COLUMNS], direction_labels
        )
        positive_class_index = int(np.where(risk_model.classes_ == True)[0][0])
        underestimation_probability = risk_model.predict_proba(
            prediction_buffer_features[BUFFER_COLUMNS]
        )[:, positive_class_index]
        direction_probability = np.where(
            estimated_error >= 0,
            underestimation_probability,
            1.0 - underestimation_probability,
        )

    maximum_buffer = np.maximum(base_predictions, 0.0) * max_buffer_rate
    estimated_buffer = np.clip(
        estimated_error * direction_probability,
        -maximum_buffer,
        maximum_buffer,
    )
    buffer = np.where(
        direction_probability >= risk_threshold, estimated_buffer, 0.0
    )
    # Signo alineado con %dif=(real-estimado)/real: positivo=subestima, negativo=sobreestima.
    prediction_mean = float(np.mean(base_predictions)
                            ) if base_predictions.size else 0.0
    buffer_rate = float(
        np.mean(np.abs(buffer)) / prediction_mean
        if prediction_mean > 0 else 0.0
    )
    return base_predictions.copy(), buffer, average_error, buffer_rate


def calculate_additional_buffer_area(
    buffer_stems: "pd.Series | np.ndarray",
    stems_per_m2: "pd.Series | np.ndarray",
) -> np.ndarray:
    """Convierte el amortiguador firmado en m2, redondeando su magnitud arriba."""
    buffer_values = np.asarray(buffer_stems, dtype=float).reshape(-1)
    productivity = np.asarray(stems_per_m2, dtype=float).reshape(-1)
    if buffer_values.size != productivity.size:
        raise ValueError(
            "Amortiguador y Tallos/m2 deben tener igual longitud.")

    area = np.where(np.isclose(buffer_values, 0.0), 0.0, np.nan)
    valid = (
        np.isfinite(buffer_values)
        & np.isfinite(productivity)
        & (productivity > 0)
    )
    area[valid] = (
        np.sign(buffer_values[valid])
        * np.ceil(np.abs(buffer_values[valid]) / productivity[valid])
    )
    return area


def calculate_buffer_area_from_projection(
    buffer_stems: "pd.Series | np.ndarray",
    projected_stems: "pd.Series | np.ndarray",
    total_area_m2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula productividad proyectada y la fraccion de area amortiguadora."""
    total_area = float(total_area_m2)
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError(
            "El area total de la variedad debe ser mayor que cero.")

    projected_values = np.asarray(projected_stems, dtype=float).reshape(-1)
    projected_productivity = projected_values / total_area
    buffer_area = calculate_additional_buffer_area(
        buffer_stems, projected_productivity
    )
    valid_area = np.isfinite(buffer_area)
    buffer_area[valid_area] = np.sign(buffer_area[valid_area]) * np.minimum(
        np.abs(buffer_area[valid_area]), np.floor(total_area)
    )
    return projected_productivity, buffer_area


def add_buffer_projection_columns(
    projection_df: pd.DataFrame,
    total_area_m2: float,
) -> pd.DataFrame:
    """Agrega el mismo escenario amortiguado a proyecciones individuales o masivas."""
    required = {"Estimado_modelo", "Amortiguador_sobreestimacion"}
    missing = required - set(projection_df.columns)
    if missing:
        raise ValueError(
            "Faltan columnas para calcular el amortiguador: "
            + ", ".join(sorted(missing))
        )

    result = projection_df.copy()
    result["Estimado_con_amortiguador_IA"] = pd.to_numeric(
        result["Estimado_modelo"], errors="coerce"
    )
    projected_productivity, buffer_area = calculate_buffer_area_from_projection(
        result["Amortiguador_sobreestimacion"],
        result["Estimado_modelo"],
        total_area_m2,
    )
    result["Tallos_m2_variedad"] = projected_productivity
    result["M2_amortiguador_adicional"] = buffer_area
    result["M2_variedad_disponibles"] = float(total_area_m2)
    result["Estado_M2_amortiguador"] = np.select(
        [
            result["M2_amortiguador_adicional"].isna()
            & ~np.isclose(result["Amortiguador_sobreestimacion"], 0.0),
            result["M2_amortiguador_adicional"] > 0,
            result["M2_amortiguador_adicional"] < 0,
        ],
        [
            "No calculable: Tallos/m2 es cero",
            "Subestimacion: reservar area",
            "Sobreestimacion: liberar area",
        ],
        default="Sin amortiguador requerido",
    )
    return result


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
    return working.reset_index(drop=True)


def calcular_ajuste_factor_diferencia_2026(
    evaluation_dir: "Path | None" = None,
    peso_ajuste: float = PESO_AJUSTE_REAL_2026,
) -> dict[str, Any]:
    """Corrige el factor estacional de 2025 con el error real observado en 2026.

    Usa `errores_evaluacion_modelo.csv` (Anio==2026) y calcula, por variedad,
    la razon `sum(Produccion) / sum(Estimado_modelo)`: valores <1 indican que
    el modelo esta sobreestimando en la realidad reciente. `peso_ajuste`
    amplifica (>1) o atenua (<1) la desviacion de esa razon respecto a 1.0
    antes de limitar el resultado a [0.5, 1.5], para dar mas o menos peso a
    la correccion real de 2026 frente al factor estacional de 2025.
    """
    directorio = evaluation_dir or Path(__file__).with_name("Evaluacion")
    ajustes: dict[str, float] = {}
    ajuste_global = 1.0
    errores_path = directorio / "errores_evaluacion_modelo.csv"
    if not errores_path.exists():
        return {"ajustes_por_variedad": ajustes, "ajuste_global": ajuste_global}

    try:
        errores_df = pd.read_csv(errores_path, encoding="utf-8-sig")
    except Exception:
        return {"ajustes_por_variedad": ajustes, "ajuste_global": ajuste_global}

    requeridas = {"Anio", "Bloque", "Variedad",
                  "Estimado_modelo", "Produccion"}
    if not requeridas.issubset(errores_df.columns):
        return {"ajustes_por_variedad": ajustes, "ajuste_global": ajuste_global}

    trabajo = errores_df.copy()
    trabajo["Anio"] = pd.to_numeric(trabajo["Anio"], errors="coerce")
    trabajo["Estimado_modelo"] = pd.to_numeric(
        trabajo["Estimado_modelo"], errors="coerce")
    trabajo["Produccion"] = pd.to_numeric(
        trabajo["Produccion"], errors="coerce")
    # 'Bloque&Varid' en este CSV viene prefijado con la semana; se reconstruye
    # el identificador canonico (igual al de factor_diferencia_2025_por_variedad.csv)
    # a partir de 'Bloque' y 'Variedad'.
    bloque_numerico = pd.to_numeric(trabajo["Bloque"], errors="coerce")
    trabajo["__bloque_varid"] = np.where(
        bloque_numerico.notna(),
        bloque_numerico.astype("Int64").astype(str).str.zfill(3)
        + trabajo["Variedad"].astype(str).str.strip(),
        np.nan,
    )
    trabajo = trabajo[trabajo["Anio"] == 2026].dropna(
        subset=["Estimado_modelo", "Produccion", "__bloque_varid"]
    )
    if trabajo.empty:
        return {"ajustes_por_variedad": ajustes, "ajuste_global": ajuste_global}

    limite_inferior, limite_superior = 0.5, 1.5

    def _ponderar(ratio: float) -> float:
        ratio_ponderado = 1.0 + peso_ajuste * (ratio - 1.0)
        return float(np.clip(ratio_ponderado, limite_inferior, limite_superior))

    for variedad, grupo in trabajo.groupby("__bloque_varid"):
        suma_estimado = float(grupo["Estimado_modelo"].sum())
        suma_real = float(grupo["Produccion"].sum())
        if suma_estimado > 0:
            ajustes[str(variedad)] = _ponderar(suma_real / suma_estimado)

    suma_estimado_total = float(trabajo["Estimado_modelo"].sum())
    suma_real_total = float(trabajo["Produccion"].sum())
    if suma_estimado_total > 0:
        ajuste_global = _ponderar(suma_real_total / suma_estimado_total)

    return {"ajustes_por_variedad": ajustes, "ajuste_global": ajuste_global}


def calcular_moda_signo_evaluacion_2026(
    evaluation_dir: "Path | None" = None,
) -> dict[str, Any]:
    """Calcula el signo moda (mas frecuente) del error real de 2026 por variedad.

    Usa el signo de `%dif` (positivo=subestima, negativo=sobreestima) de las
    filas `Anio == 2026` en `errores_evaluacion_modelo.csv`. Por `Bloque&Varid`
    devuelve `1` si predominan semanas de subestimacion, `-1` si predominan de
    sobreestimacion, o `0` si hay empate o no hay datos suficientes. Tambien
    calcula una moda global (agregando todas las variedades) como respaldo.
    """
    directorio = evaluation_dir or Path(__file__).with_name("Evaluacion")
    modas: dict[str, int] = {}
    moda_global = 0
    errores_path = directorio / "errores_evaluacion_modelo.csv"
    if not errores_path.exists():
        return {"moda_por_variedad": modas, "moda_global": moda_global}

    try:
        errores_df = pd.read_csv(errores_path, encoding="utf-8-sig")
    except Exception:
        return {"moda_por_variedad": modas, "moda_global": moda_global}

    requeridas = {"Anio", "Bloque", "Variedad", "%dif"}
    if not requeridas.issubset(errores_df.columns):
        return {"moda_por_variedad": modas, "moda_global": moda_global}

    trabajo = errores_df.copy()
    trabajo["Anio"] = pd.to_numeric(trabajo["Anio"], errors="coerce")
    trabajo["%dif"] = pd.to_numeric(trabajo["%dif"], errors="coerce")
    bloque_numerico = pd.to_numeric(trabajo["Bloque"], errors="coerce")
    trabajo["__bloque_varid"] = np.where(
        bloque_numerico.notna(),
        bloque_numerico.astype("Int64").astype(str).str.zfill(3)
        + trabajo["Variedad"].astype(str).str.strip(),
        np.nan,
    )
    trabajo = trabajo[trabajo["Anio"] == 2026].dropna(
        subset=["%dif", "__bloque_varid"]
    )
    if trabajo.empty:
        return {"moda_por_variedad": modas, "moda_global": moda_global}

    def _moda_signo(serie_dif: pd.Series) -> int:
        positivos = int((serie_dif > 0).sum())
        negativos = int((serie_dif < 0).sum())
        if positivos > negativos:
            return 1
        if negativos > positivos:
            return -1
        return 0

    for variedad, grupo in trabajo.groupby("__bloque_varid"):
        modas[str(variedad)] = _moda_signo(grupo["%dif"])

    moda_global = _moda_signo(trabajo["%dif"])
    return {"moda_por_variedad": modas, "moda_global": moda_global}


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

    ajuste_real_2026 = calcular_ajuste_factor_diferencia_2026(evaluation_dir)
    ajustes_por_variedad = ajuste_real_2026["ajustes_por_variedad"]
    ajuste_global = ajuste_real_2026["ajuste_global"]
    factors = {
        variedad: valor * ajustes_por_variedad.get(variedad, ajuste_global)
        for variedad, valor in factors.items()
    }
    global_factor *= ajuste_global

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
    selected = np.where(
        (years == 2026) & (weeks >= 24) & (weeks <= 52)
    )[0]
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
        REAL_PRODUCTION_TARGET_WEIGHT
        * training_df["Produccion"].to_numpy(dtype=float)
        + PATTERN_PRODUCTION_TARGET_WEIGHT
        * training_df["Produccion_patron"].to_numpy(dtype=float)
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
    m2_values = pd.to_numeric(subset["m2Variedad"], errors="coerce").dropna()
    if m2_values.empty:
        raise ValueError("m2Variedad no contiene valores numericos validos.")
    predictions, buffer, average_overestimation, buffer_rate = apply_overestimation_buffer(
        model,
        prediction_features,
        evaluation_df,
        evaluation_df,
        predictions,
        float(m2_values.iloc[0]),
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
            "amortiguador_sobreestimacion": np.round(buffer, 2),
            "estimado_con_amortiguador_ia": np.round(predictions, 2),
        }
    )
    chart_df["%dif"] = calculate_model_error_report(
        real_production,
        predictions,
    )["%dif"].to_numpy()

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
        "amortiguador_promedio_historico": average_overestimation,
        "amortiguador_porcentaje": buffer_rate * 100.0,
        "preview": chart_df.tail(20).to_dict(orient="records"),
    }
    if include_chart_df:
        result["chart_df"] = chart_df

    return result


def find_reference_pattern(df: pd.DataFrame, selected_var: str) -> dict[str, Any] | None:
    validate_required_columns(df, {"Bloque&Varid", "Tallos/m2"})

    target_rows = df[df["Bloque&Varid"].astype(
        str) == str(selected_var)].copy()

    candidates: list[dict[str, Any]] = []
    grouped = df.dropna(subset=["Bloque&Varid"]).groupby("Bloque&Varid")
    for candidate_var, candidate_rows in grouped:
        candidate_name = str(candidate_var)
        if candidate_name == str(selected_var):
            continue
        if not has_sufficient_pattern_history(target_rows, candidate_rows):
            continue

        mse = calculate_normalized_stems_mse(target_rows, candidate_rows)
        if mse is None:
            continue

        candidate_series = pd.to_numeric(
            candidate_rows["Tallos/m2"], errors="coerce").dropna().reset_index(drop=True)
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


def has_sufficient_pattern_history(
    target_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    required_columns: tuple[str, ...] = ("Tallos/m2",),
) -> bool:
    if not set(required_columns).issubset(target_rows.columns):
        return False
    if not set(required_columns).issubset(candidate_rows.columns):
        return False

    target_length = len(target_rows.dropna(subset=list(required_columns)))
    candidate_length = len(
        candidate_rows.dropna(subset=list(required_columns))
    )
    return candidate_length >= target_length


def calculate_normalized_stems_mse(
    target_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> float | None:
    def prepare_series(rows: pd.DataFrame) -> np.ndarray:
        working = rows.copy()
        if {"Anio", "Semana"}.issubset(working.columns):
            working = working.sort_values(["Anio", "Semana"])
        return pd.to_numeric(
            working["Tallos/m2"], errors="coerce"
        ).dropna().to_numpy(dtype=float)

    target = prepare_series(target_rows)
    candidate = prepare_series(candidate_rows)
    common_length = min(len(target), len(candidate))
    if common_length < 4:
        return None

    target = target[:common_length]
    candidate = candidate[:common_length]
    target_std = float(np.std(target, ddof=0))
    candidate_std = float(np.std(candidate, ddof=0))
    if np.isclose(target_std, 0.0) or np.isclose(candidate_std, 0.0):
        return None

    target_normalized = (target - np.mean(target)) / target_std
    candidate_normalized = (candidate - np.mean(candidate)) / candidate_std
    return float(np.mean((target_normalized - candidate_normalized) ** 2))


def calculate_pattern_signal_to_noise(
    pattern_values: "pd.Series | np.ndarray",
    model_values: "pd.Series | np.ndarray",
    actual_values: "pd.Series | np.ndarray",
) -> tuple[float, float, float]:
    values = pd.DataFrame({
        "pattern": pd.to_numeric(
            pd.Series(pattern_values).reset_index(drop=True), errors="coerce"
        ),
        "model": pd.to_numeric(
            pd.Series(model_values).reset_index(drop=True), errors="coerce"
        ),
        "actual": pd.to_numeric(
            pd.Series(actual_values).reset_index(drop=True), errors="coerce"
        ),
    }).dropna()
    if values.empty:
        return np.nan, np.nan, np.nan

    signal_power = float(np.mean(values["pattern"].to_numpy() ** 2))
    noise_power = float(np.mean(
        (values["model"].to_numpy() - values["actual"].to_numpy()) ** 2
    ))
    if signal_power <= 0:
        sn_ratio_db = np.nan
    elif np.isclose(noise_power, 0.0):
        sn_ratio_db = np.inf
    else:
        sn_ratio_db = float(10.0 * np.log10(signal_power / noise_power))
    return signal_power, noise_power, sn_ratio_db


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
