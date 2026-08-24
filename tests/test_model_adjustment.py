import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ProyAst import (  # noqa: E402
    ajustar_patron_con_extremos_real,
    ajustar_prediccion_modelo_con_patron,
    alinear_series_para_ajuste,
)


def test_prediccion_no_se_mezcla_con_patron_ni_produccion_real():
    pred = np.array([800.0, 900.0, 1000.0, 1100.0])
    proy = np.array([5000.0, 100.0, 6000.0, 200.0])
    eval_actual_df = pd.DataFrame({
        'Produccion': [100.0, 9000.0, 50.0, 10000.0],
    })

    resultado = ajustar_prediccion_modelo_con_patron(
        pred,
        proy,
        eval_actual_df,
        patron_prediction_weight=1.0,
        sn_alto=True,
        residual_weight=1.0,
    )

    assert np.array_equal(resultado, pred)


def test_prediccion_devuelve_copia_independiente():
    pred = np.array([1000.0, 2000.0])

    resultado = ajustar_prediccion_modelo_con_patron(
        pred,
        np.array([0.0, 0.0]),
        pd.DataFrame({'Produccion': [9000.0, 9000.0]}),
        patron_prediction_weight=1.0,
        sn_alto=True,
        residual_weight=1.0,
    )
    resultado[0] = 0.0

    assert pred[0] == 1000.0


def test_patron_no_se_ajusta_por_extremos_o_tallos():
    trabajo = pd.DataFrame({
        'Anio': [2025, 2025, 2026, 2026],
        'Semana': [1, 2, 1, 2],
        'Produccion': [100.0, 9000.0, 50.0, 10000.0],
        'Produccion_patron': [900.0, 900.0, 940.0, 940.0],
        'Tallos/m2': [12.0, 13.0, 18.0, 19.0],
        'Tallos_m2_patron': [10.0, 11.0, 12.0, 13.0],
    })

    resultado = ajustar_patron_con_extremos_real(
        trabajo,
        refuerzo_tallos_m2=1.0,
    )

    pd.testing.assert_frame_equal(resultado, trabajo)
    assert 'porcentaje_ajuste_modelo' not in resultado.columns


def test_patron_devuelve_copia_independiente():
    trabajo = pd.DataFrame({'Produccion_patron': [900.0, 940.0]})

    resultado = ajustar_patron_con_extremos_real(trabajo)
    resultado.loc[0, 'Produccion_patron'] = 0.0

    assert trabajo.loc[0, 'Produccion_patron'] == 900.0


def test_alineacion_de_series_para_ajuste():
    pred = np.array([1000.0, 2000.0, 3000.0])
    proy = np.array([500.0, 700.0])
    real = np.array([400.0, 600.0, 800.0, 1000.0])

    pred_alineado, proy_alineado, real_alineado = alinear_series_para_ajuste(
        pred,
        proy,
        real,
    )

    assert len(pred_alineado) == len(proy_alineado) == len(real_alineado) == 2
