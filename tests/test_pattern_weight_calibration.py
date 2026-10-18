import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import projection_core as core


def _data():
    rows = pd.DataFrame({
        "Anio": [2026] * 12,
        "Semana": range(1, 13),
        "Tallos/m2": [10.0] * 12,
        "Produccion": [100.0] * 7 + [120.0] * 5,
    })
    pattern = pd.DataFrame({
        "Anio": [2026] * 12,
        "Semana": range(1, 13),
        "Tallos_m2_patron": [10.0] * 12,
        "Produccion_patron": [50.0, 150.0, 50.0, 150.0, 50.0, 150.0, 100.0]
        + [100.0] * 5,
        "Incremento_tallos_patron": [0.0] * 12,
        "Incremento_produccion_patron": [0.0] * 12,
    })
    adjustments = {
        "periods_by_case": {
            "001rosa": [(2026, week) for week in range(8, 13)]
        }
    }
    return rows, pattern, adjustments


def test_selects_lower_absolute_error_and_keeps_training_chronological(monkeypatch):
    rows, pattern, adjustments = _data()
    training_targets = []
    validation_features = []

    class Model:
        def __init__(self, weight):
            self.weight = weight

        def predict(self, features):
            validation_features.append(features.copy())
            return np.full(len(features), 100 + self.weight * 50)

    def fit(features, target):
        assert features["Semana_orden"].max() == 6
        training_targets.append(target.copy())
        weight = (target[1] - 100) / 50
        return Model(weight)

    monkeypatch.setattr(core, "fit_production_model", fit)
    result = core.calibrate_pattern_target_weight(
        rows, pattern, "1ROSA", adjustments, feature_weight=0.4
    )

    assert result["status"] == "calibrado"
    assert result["weight"] == 0.4
    assert result["mape"] == 0
    assert result["mape"] < result["baseline_mape"]
    assert result["validation_weeks"] == 5
    assert len(training_targets) == 9
    for features in validation_features:
        np.testing.assert_allclose(features["Produccion_patron"], 100)
        np.testing.assert_allclose(features["Produccion_patron_ponderado"], 100)
    assert rows["Produccion"].iloc[-1] == 120


def test_ties_keep_configured_weight(monkeypatch):
    rows, pattern, adjustments = _data()

    class Model:
        def predict(self, features):
            return np.full(len(features), 120.0)

    monkeypatch.setattr(core, "fit_production_model", lambda *args: Model())
    result = core.calibrate_pattern_target_weight(
        rows, pattern, "001ROSA", adjustments, default_weight=0.35
    )
    assert result["weight"] == 0.35
    assert result["mape"] == result["baseline_mape"] == 0


@pytest.mark.parametrize("scenario", ["missing", "short", "zero", "no_history"])
def test_insufficient_evidence_keeps_default_without_fitting(monkeypatch, scenario):
    rows, pattern, adjustments = _data()
    if scenario == "missing":
        adjustments["periods_by_case"] = {"002rosa": [(2026, 8)]}
    elif scenario == "short":
        adjustments["periods_by_case"]["001rosa"] = [(2026, 8), (2026, 9)]
    elif scenario == "zero":
        rows.loc[rows["Semana"] >= 8, "Produccion"] = 0
    else:
        rows = rows[rows["Semana"] >= 5]

    def unexpected_fit(*args):
        pytest.fail("No debe entrenar sin evidencia suficiente")

    monkeypatch.setattr(core, "fit_production_model", unexpected_fit)
    result = core.calibrate_pattern_target_weight(
        rows, pattern, "001ROSA", adjustments
    )
    assert result["weight"] == 0.30
    assert result["status"] != "calibrado"
    assert result["mape"] is None


def test_report_supplies_only_own_recent_periods(tmp_path):
    path = tmp_path / "errors.csv"
    pd.DataFrame({
        "Anio": [2026] * 7,
        "Semana": [1, 2, 3, 4, 5, 6, 10],
        "Bloque&Varid": ["001ROSA"] * 6 + ["002ROSA"],
        "%dif": [0.1] * 7,
    }).to_csv(path, index=False)
    config = core.load_recent_media_adjustments(path)
    assert config["periods_by_case"]["001rosa"] == [
        (2026, week) for week in range(2, 7)
    ]
    assert config["periods_by_case"]["002rosa"] == [(2026, 10)]


def test_actual_random_forest_calibration_is_finite_and_reproducible():
    rows, pattern, adjustments = _data()
    first = core.calibrate_pattern_target_weight(
        rows, pattern, "001ROSA", adjustments
    )
    second = core.calibrate_pattern_target_weight(
        rows, pattern, "001ROSA", adjustments
    )
    assert first == second
    assert 0 <= first["weight"] <= 0.8
    assert np.isfinite(first["mape"])
    assert first["mape"] <= first["baseline_mape"]


def test_duplicate_pattern_period_is_rejected():
    rows, pattern, adjustments = _data()
    pattern = pd.concat([pattern, pattern.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="una sola fila"):
        core.calibrate_pattern_target_weight(
            rows, pattern, "001ROSA", adjustments
        )


def test_validation_actuals_never_change_training_or_prediction_features(monkeypatch):
    rows, pattern, adjustments = _data()
    captured = []

    class Model:
        def predict(self, features):
            captured.append(features.copy())
            return np.full(len(features), 100.0)

    def fit(features, target):
        captured.extend([features.copy(), np.asarray(target).copy()])
        return Model()

    monkeypatch.setattr(core, "fit_production_model", fit)
    core.calibrate_pattern_target_weight(
        rows, pattern, "001ROSA", adjustments, feature_weight=0.4
    )
    original = captured.copy()
    captured.clear()
    rows.loc[rows["Semana"] >= 8, "Produccion"] = 90000
    core.calibrate_pattern_target_weight(
        rows, pattern, "001ROSA", adjustments, feature_weight=0.4
    )
    for before, after in zip(original, captured):
        if isinstance(before, pd.DataFrame):
            pd.testing.assert_frame_equal(before, after)
        else:
            np.testing.assert_array_equal(before, after)


def test_api_training_uses_selected_weight_and_exposes_metadata(monkeypatch):
    rows, pattern, adjustments = _data()
    target_rows = rows.assign(**{"Bloque&Varid": "001ROSA", "m2Variedad": 100})
    pattern_rows = rows.assign(**{
        "Bloque&Varid": "002ROSA",
        "Produccion": pattern["Produccion_patron"],
        "m2Variedad": 100,
    })
    data = pd.concat([target_rows, pattern_rows], ignore_index=True)
    calibration = {
        "weight": 0.4, "status": "calibrado", "validation_weeks": 5,
        "mape": 0.1, "baseline_mape": 0.2,
    }
    monkeypatch.setattr(core, "find_reference_pattern",
                        lambda *args: {"reference_var": "002ROSA"})
    monkeypatch.setattr(core, "load_recent_media_adjustments", lambda: adjustments)
    monkeypatch.setattr(core, "calibrate_pattern_target_weight",
                        lambda *args: calibration)
    monkeypatch.setattr(core, "_apply_difference_factor",
                        lambda values, *args: (values, 1.0, 0, "global"))
    monkeypatch.setattr(core, "apply_overestimation_buffer",
                        lambda *args: (args[4], np.zeros(len(args[4])), 0.0, 0.0))
    fitted_targets = []
    real_fit = core.fit_production_model

    def fit(features, target):
        fitted_targets.append(np.asarray(target))
        return real_fit(features, target)

    monkeypatch.setattr(core, "fit_production_model", fit)
    result = core.train_projection_model(data, "001ROSA", include_chart_df=False)
    prepared = core._prepare_production_dataset(
        target_rows, core._build_weekly_pattern(pattern_rows)
    )
    training = core._exclude_latest_five_weeks(prepared)
    np.testing.assert_allclose(
        fitted_targets[0],
        0.6 * training["Produccion"] + 0.4 * training["Produccion_patron"],
    )
    assert result["pattern_target_calibration"] == calibration
    assert "chart_df" not in result


def test_streamlit_model_cache_depends_on_training_data_and_target():
    source = Path(core.__file__).with_name("ProyAst.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "construir_clave_modelo_calibrado"
    )
    namespace = {"np": np, "pd": pd, "hashlib": hashlib}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<helper>", "exec"),
         namespace)
    key = namespace["construir_clave_modelo_calibrado"]
    features = pd.DataFrame({"Tallos/m2": [1.0, 2.0]})
    baseline = key("001ROSA", "002ROSA", features, [100, 200])
    assert baseline == key("001ROSA", "002ROSA", features.copy(), [100, 200])
    assert baseline != key("001ROSA", "002ROSA", features, [110, 200])
    assert baseline != key("001ROSA", "003ROSA", features, [100, 200])
    features.loc[0, "Tallos/m2"] = 3
    assert baseline != key("001ROSA", "002ROSA", features, [100, 200])
