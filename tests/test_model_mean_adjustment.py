import numpy as np
import pandas as pd

import projection_core
from projection_core import (
    adjust_model_mean_if_below_references,
    align_pattern_to_target_level,
)


def test_adjusts_model_mean_to_midpoint_when_below_both_references():
    predictions = np.array([80.0, 100.0])
    actual = np.array([100.0, 200.0])
    pattern = np.array([120.0, 180.0])

    adjusted, factor = adjust_model_mean_if_below_references(
        predictions, actual, pattern
    )

    expected_mean = (actual.mean() + pattern.mean()) / 2.0
    assert np.isclose(factor, expected_mean / predictions.mean())
    assert np.isclose(adjusted.mean(), expected_mean)
    np.testing.assert_allclose(adjusted / predictions, factor)


def test_leaves_predictions_unchanged_if_not_below_both_references():
    predictions = np.array([100.0, 120.0])
    actual = np.array([100.0, 100.0])
    pattern = np.array([120.0, 120.0])

    adjusted, factor = adjust_model_mean_if_below_references(
        predictions, actual, pattern
    )

    assert factor == 1.0
    np.testing.assert_array_equal(adjusted, predictions)


def test_aligns_pattern_production_and_stems_to_target_means():
    target = pd.DataFrame({
        'Anio': [2025, 2025, 2025],
        'Semana': [1, 2, 3],
        'Tallos/m2': [10.0, 20.0, 30.0],
        'Produccion': [100.0, 200.0, 300.0],
    })
    pattern = pd.DataFrame({
        'Anio': [2025, 2025],
        'Semana': [1, 2],
        'Tallos/m2': [20.0, 40.0],
        'Produccion': [200.0, 400.0],
    })
    weekly_pattern = projection_core._build_weekly_pattern(pattern)

    prepared = projection_core._prepare_production_dataset(
        target, weekly_pattern
    )

    assert np.isclose(
        prepared['Tallos_m2_patron'].mean(), prepared['Tallos/m2'].mean()
    )
    assert np.isclose(
        prepared['Produccion_patron'].mean(), prepared['Produccion'].mean()
    )
    aligned, factor = align_pattern_to_target_level(
        pd.Series([10.0, 20.0]), pd.Series([20.0, 40.0])
    )
    assert factor == 0.5
    np.testing.assert_allclose(aligned, [10.0, 20.0])
