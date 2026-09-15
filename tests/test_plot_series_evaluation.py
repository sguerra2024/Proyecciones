import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_series_evaluation import (
    _area_percentages,
    _cumulative_data_percentage,
)


def test_areas_de_evaluacion_se_muestran_como_porcentajes_que_suman_100():
    positiva, negativa = _area_percentages(0.25, -0.40)

    assert positiva == pytest.approx(100 * 0.25 / 0.65)
    assert negativa == pytest.approx(100 * 0.40 / 0.65)
    assert positiva + negativa == pytest.approx(100.0)


def test_areas_sin_valor_se_muestran_como_cero():
    assert _area_percentages(0.0, 0.0) == (0.0, 0.0)


def test_porcentaje_acumulado_incluye_los_limites_de_menos_y_mas_25():
    valores = [-0.30, -0.25, 0.10, 0.25, 0.40]

    assert _cumulative_data_percentage(valores) == pytest.approx(60.0)


def test_porcentaje_acumulado_sin_datos_es_cero():
    assert _cumulative_data_percentage([]) == 0.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
