import numpy as np
import pandas as pd
import pytest

from projection_core import (
    RECENT_ADJUSTMENT_MAX_FACTOR,
    RECENT_ADJUSTMENT_MIN_FACTOR,
    apply_recent_media_adjustment,
    load_recent_media_adjustments,
)
from Graf_evaluacion import guardar_calibracion_amortiguador


def test_recent_factor_uses_means_of_four_distinct_weeks(tmp_path):
    path = tmp_path / 'errores.csv'
    pd.DataFrame({
        'Anio': [2026] * 9,
        'Semana': [10, 11, 11, 12, 12, 13, 13, 14, 14],
        'error_relativo': [0.9, -0.5, -0.5, 0.1, 0.3, 0.4, 0.2, 0.0, 0.2],
    }).to_csv(path, index=False)

    adjustments = load_recent_media_adjustments(path)

    # Weeks 11-14 have weekly means [-0.5, 0.2, 0.3, 0.1].
    assert adjustments['available']
    assert np.isclose(adjustments['global_factor'], 1.025)


def test_recent_factor_is_limited_to_configured_range(tmp_path):
    path = tmp_path / 'errores.csv'
    pd.DataFrame({
        'Anio': [2026] * 4,
        'Semana': [31, 32, 33, 34],
        'error_relativo': [0.5] * 4,
    }).to_csv(path, index=False)

    high = load_recent_media_adjustments(path)
    assert high['global_factor'] == RECENT_ADJUSTMENT_MAX_FACTOR == 1.30

    pd.DataFrame({
        'Anio': [2026] * 4,
        'Semana': [31, 32, 33, 34],
        'error_relativo': [-0.5] * 4,
    }).to_csv(path, index=False)
    low = load_recent_media_adjustments(path)
    assert low['global_factor'] == RECENT_ADJUSTMENT_MIN_FACTOR == 0.75


def test_applies_factor_to_all_rows_in_last_four_distinct_weeks():
    evaluation = pd.DataFrame({
        'Anio': [2026] * 6,
        'Semana': [1, 2, 2, 3, 4, 5],
    })
    predictions = np.array([100.0, 200.0, 250.0, 300.0, 400.0, 500.0])
    adjustments = {
        'available': True,
        'global_factor': 1.30,
        'factors_by_case': {'006atomic': 1.30},
    }

    adjusted, factor, origin = apply_recent_media_adjustment(
        predictions, evaluation, '006ATOMIC', adjustments
    )

    assert factor == 1.30
    assert origin == 'case'
    np.testing.assert_allclose(
        adjusted, [100.0, 260.0, 325.0, 390.0, 520.0, 650.0]
    )


def test_graf_evaluacion_preserves_ids_and_calibrates_each_case_individually(
    tmp_path,
):
    source_path = tmp_path / 'evaluacion.csv'
    destination_path = tmp_path / 'errores_evaluacion_modelo.csv'
    rows = []
    for block, variety, errors in [
        ('006', 'ATOMIC', [0.05, 0.10, 0.15, 0.20]),
        ('007', 'SOMERSET', [-0.10, -0.20, -0.30, -0.40]),
    ]:
        for week, error in zip(range(31, 35), errors):
            rows.append({
                'Finca': 'F01',
                'Bloque': block,
                'Variedad': variety,
                'Bloque&Varid': f'{block}{variety}',
                'Anio': 2026,
                'Semana': week,
                '%Dif': error,
            })
    pd.DataFrame(rows).to_csv(source_path, index=False)

    guardar_calibracion_amortiguador(
        source_path,
        '%Dif',
        destination_path=destination_path,
    )
    saved = pd.read_csv(destination_path)
    assert {'Finca', 'Bloque', 'Variedad', 'Bloque&Varid'}.issubset(
        saved.columns
    )

    adjustments = load_recent_media_adjustments(destination_path)
    assert np.isclose(adjustments['factors_by_case']['006atomic'], 1.125)
    assert np.isclose(adjustments['factors_by_case']['007somerset'], 0.75)

    evaluation = pd.DataFrame({
        'Anio': [2026] * 4,
        'Semana': [31, 32, 33, 34],
    })
    predictions = np.full(4, 100.0)
    atomic, atomic_factor, atomic_origin = apply_recent_media_adjustment(
        predictions, evaluation, '006ATOMIC', adjustments
    )
    somerset, somerset_factor, _ = apply_recent_media_adjustment(
        predictions, evaluation, '007SOMERSET', adjustments
    )
    unknown, unknown_factor, unknown_origin = apply_recent_media_adjustment(
        predictions, evaluation, '009UNKNOWN', adjustments
    )

    assert (atomic_factor, atomic_origin) == (1.125, 'case')
    assert somerset_factor == 0.75
    np.testing.assert_allclose(atomic, [112.5] * 4)
    np.testing.assert_allclose(somerset, [75.0] * 4)
    assert unknown_factor == 1.0
    assert unknown_origin == 'case_pending'
    np.testing.assert_array_equal(unknown, predictions)


def test_does_not_fall_back_to_global_when_individual_factors_are_missing():
    predictions = np.full(4, 100.0)
    evaluation = pd.DataFrame({
        'Anio': [2026] * 4,
        'Semana': [31, 32, 33, 34],
    })
    adjustments = {
        'available': True,
        'global_factor': 1.30,
        'factors_by_case': {},
    }

    adjusted, factor, origin = apply_recent_media_adjustment(
        predictions, evaluation, '006ATOMIC', adjustments
    )

    assert factor == 1.0
    assert origin == 'case_pending'
    np.testing.assert_array_equal(adjusted, predictions)


def test_checkbox_calibration_rejects_non_percent_difference_column(tmp_path):
    source = tmp_path / 'datos.csv'
    pd.DataFrame({
        'Bloque': ['006'],
        'Variedad': ['ATOMIC'],
        'Anio': [2026],
        'Semana': [35],
        'Produccion': [100.0],
    }).to_csv(source, index=False)

    with pytest.raises(ValueError, match='%Dif'):
        guardar_calibracion_amortiguador(source, 'Produccion')


def test_checkbox_calibration_rejects_missing_case_identifiers(tmp_path):
    source = tmp_path / 'datos_sin_casos.csv'
    pd.DataFrame({
        '%Dif': [0.10],
        'Anio': [2026],
        'Semana': [35],
    }).to_csv(source, index=False)

    with pytest.raises(ValueError, match='Bloque.*Variedad'):
        guardar_calibracion_amortiguador(source, '%Dif')
