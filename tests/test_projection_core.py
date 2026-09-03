from projection_core import (
    add_buffer_projection_columns,
    apply_overestimation_buffer,
    build_production_model,
    calculate_additional_buffer_area,
    calculate_buffer_area_from_projection,
    load_overestimation_calibration,
    train_projection_model,
)
import projection_core
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _weekly_data() -> pd.DataFrame:
    rows = []
    for variety, offset in [("001RED", 0.0), ("002RED", 2.0)]:
        for week in range(1, 17):
            stems = 10.0 + week * 0.2 + offset
            rows.append({
                "Anio": 2025,
                "Semana": week,
                "Bloque&Varid": variety,
                "Tallos/m2": stems,
                "Produccion": 900.0 + week * 25.0 + offset * 10.0,
                "m2Variedad": 100.0,
            })
    return pd.DataFrame(rows)


def test_build_production_model_uses_shared_parameters():
    model = build_production_model()

    assert model.n_estimators == 100
    assert model.max_depth == 16
    assert model.random_state == 42
    assert model.max_features == "sqrt"


def test_train_projection_model_predicts_production_with_reference_pattern():
    result = train_projection_model(_weekly_data(), "001RED")

    assert result["selected_var"] == "001RED"
    assert result["reference_var"] == "002RED"
    assert result["using_reference_pattern"] is True
    assert result["rows_used"] == 12
    assert np.isfinite(result["rmse"])
    assert result["factor_origin"] in {"variedad", "global"}
    assert not result["chart_df"].empty
    assert {
        "anio",
        "semana",
        "produccion_real",
        "produccion_patron",
        "estimado_modelo",
        "amortiguador_sobreestimacion",
        "estimado_con_amortiguador_ia",
    } == set(result["chart_df"].columns)
    assert np.isfinite(result["amortiguador_promedio_historico"])
    assert result["amortiguador_porcentaje"] >= 0


class _FixedProductionModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["prediccion_prueba"].to_numpy(dtype=float)


def test_amortiguador_usa_16_semanas_anteriores_a_las_4_ultimas(monkeypatch):
    semanas_usadas = []

    class RecordingProductionModel:
        def predict(self, features):
            semanas_usadas.extend(features["Semana_orden"].tolist())
            return np.full(len(features), 100.0)

    training_df = pd.DataFrame({
        "Anio": [2026] * 25,
        "Semana": list(range(1, 26)),
        "Produccion": [80.0, 120.0] * 12 + [80.0],
        "Tallos/m2": np.linspace(10.0, 20.0, 25),
    })
    training_features = pd.DataFrame({
        "Semana_orden": list(range(25)),
    })
    evaluation_df = pd.DataFrame({
        "Tallos/m2": [15.0],
        "Semana": [26],
    })

    apply_overestimation_buffer(
        RecordingProductionModel(),
        training_features,
        training_df,
        evaluation_df,
        np.array([100.0]),
        100.0,
    )

    assert semanas_usadas == list(range(5, 21))


def test_amortiguador_pondera_magnitud_por_probabilidad_y_umbral(monkeypatch):
    class FixedBufferModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, target):
            return self

        def predict(self, features):
            return np.full(len(features), 0.20)

    class FixedRiskModel:
        classes_ = np.array([False, True])

        def __init__(self, **kwargs):
            pass

        def fit(self, features, target):
            return self

        def predict_proba(self, features):
            return np.array([[0.30, 0.70], [0.45, 0.55]])

    monkeypatch.setattr(
        projection_core, "RandomForestRegressor", FixedBufferModel)
    monkeypatch.setattr(
        projection_core, "RandomForestClassifier", FixedRiskModel)
    monkeypatch.setattr(
        projection_core,
        "load_overestimation_calibration",
        lambda: {"max_buffer_rate": 1.0, "risk_threshold": 0.60},
    )
    training_df = pd.DataFrame({
        "Produccion": [80, 120, 80, 120, 80],
        "Tallos/m2": [10, 11, 12, 13, 14],
        "Semana": [1, 2, 3, 4, 5],
    })
    training_features = pd.DataFrame({
        "prediccion_prueba": [100, 100, 100, 100, 100],
    })
    evaluation_df = pd.DataFrame({
        "Tallos/m2": [12, 13],
        "Semana": [6, 7],
    })

    official, buffer, _, _ = apply_overestimation_buffer(
        _FixedProductionModel(),
        training_features,
        training_df,
        evaluation_df,
        np.array([100.0, 100.0]),
        100.0,
    )

    np.testing.assert_array_equal(official, [100.0, 100.0])
    np.testing.assert_allclose(buffer, [14.0, 0.0])


def test_amortiguador_aprende_error_bilateral_sin_cambiar_prediccion_oficial():
    training_df = pd.DataFrame({
        "Produccion": [80, 120, 70, 140, 90],
        "Tallos/m2": [10, 11, 12, 13, 14],
        "Semana": [1, 2, 3, 4, 5],
    })
    training_features = pd.DataFrame({
        "prediccion_prueba": [100, 100, 100, 100, 100],
    })
    evaluation_df = pd.DataFrame({
        "Tallos/m2": [12, 13],
        "Semana": [6, 7],
    })
    predictions = np.array([110.0, 130.0])

    official, buffer, average_error, buffer_rate = apply_overestimation_buffer(
        _FixedProductionModel(),
        training_features,
        training_df,
        evaluation_df,
        predictions,
        100.0,
    )

    assert average_error == 0.0
    np.testing.assert_array_equal(official, predictions)
    calibration = load_overestimation_calibration()
    assert np.all(np.abs(buffer) <= predictions *
                  calibration["max_buffer_rate"])
    assert np.isfinite(buffer_rate)


def test_calibracion_usa_errores_negativos_y_positivos(tmp_path):
    path = tmp_path / "errores.csv"
    pd.DataFrame({
        "error_relativo": [0.80, 0.50, -0.10, -0.20, -0.30]
    }).to_csv(path, index=False)

    calibration = load_overestimation_calibration(path)

    expected_rate = 0.30 / 1.30
    assert calibration["error_cases"] == 5
    assert calibration["overestimation_cases"] == 3
    assert calibration["underestimation_cases"] == 2
    assert np.isclose(calibration["max_buffer_rate"], expected_rate)
    assert calibration["risk_threshold"] == 0.60


def test_amortiguador_rechaza_m2_variedad_no_valido():
    training_df = pd.DataFrame({
        "Produccion": [80, 80, 80, 80, 80],
        "Tallos/m2": [10, 11, 12, 13, 14],
        "Semana": [1, 2, 3, 4, 5],
    })
    features = pd.DataFrame({"prediccion_prueba": [100] * 5})

    with np.testing.assert_raises_regex(ValueError, "m2Variedad"):
        apply_overestimation_buffer(
            _FixedProductionModel(),
            features,
            training_df,
            pd.DataFrame({"Tallos/m2": [12], "Semana": [6]}),
            np.array([110.0]),
            0.0,
        )


def test_area_adicional_amortiguador_divide_tallos_por_tallos_m2():
    area = calculate_additional_buffer_area(
        np.array([0.0, 101.0, -101.0, 200.0]),
        np.array([10.0, 10.0, 10.0, 0.0]),
    )

    assert area[0] == 0
    assert area[1] == 11
    assert area[2] == -11
    assert np.isnan(area[3])


def test_area_amortiguador_es_fraccion_del_area_total():
    productivity, area = calculate_buffer_area_from_projection(
        buffer_stems=np.array([300.0, 200.0]),
        projected_stems=np.array([1000.0, 2000.0]),
        total_area_m2=100.0,
    )

    np.testing.assert_array_equal(productivity, [10.0, 20.0])
    np.testing.assert_array_equal(area, [30.0, 10.0])
    assert np.all(area <= 100.0)


def test_columnas_amortiguador_son_compartidas_por_proyeccion_masiva():
    projection = pd.DataFrame({
        "Estimado_modelo": [1000.0, 500.0],
        "Amortiguador_sobreestimacion": [110.0, -100.0],
    })

    result = add_buffer_projection_columns(projection, total_area_m2=100.0)

    np.testing.assert_array_equal(
        result["Estimado_con_amortiguador_IA"], result["Estimado_modelo"]
    )
    np.testing.assert_array_equal(result["Tallos_m2_variedad"], [10.0, 5.0])
    np.testing.assert_array_equal(
        result["M2_amortiguador_adicional"], [11.0, -20.0])
    np.testing.assert_array_equal(
        result["M2_variedad_disponibles"], [100.0, 100.0])
    assert result["Estado_M2_amortiguador"].tolist() == [
        "Subestimacion: reservar area",
        "Sobreestimacion: liberar area",
    ]
