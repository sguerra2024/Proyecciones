from ProyAst import (
    ajustar_patron_con_extremos_real,
    alinear_series_para_ajuste,
    ajustar_prediccion_modelo_con_patron,
)
import ProyAst as proy_ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_preserva_prediccion_cuando_no_hay_peso_de_patron():
    pred = np.array([1000.0, 1000.0])
    proy = np.array([5000.0, 5000.0])
    eval_actual_df = pd.DataFrame({
        'Produccion': [1000.0, 1100.0],
        'Produccion_lag10': [1000.0, 1100.0],
        'Produccion_lag11': [1000.0, 1100.0],
        'Produccion_lag12': [1000.0, 1100.0],
    })

    ajustado = ajustar_prediccion_modelo_con_patron(
        pred,
        proy,
        eval_actual_df,
        patron_prediction_weight=0.0,
        sn_alto=False,
    )

    assert np.allclose(ajustado, pred)


def test_corrige_sesgo_residual_hacia_la_realidad():
    pred = np.array([800.0, 800.0, 800.0, 800.0])
    proy = np.array([0.0, 0.0, 0.0, 0.0])
    eval_actual_df = pd.DataFrame({
        'Produccion': [1000.0, 1000.0, 1000.0, 1000.0],
        'Produccion_lag10': [1000.0, 1000.0, 1000.0, 1000.0],
        'Produccion_lag11': [1000.0, 1000.0, 1000.0, 1000.0],
        'Produccion_lag12': [1000.0, 1000.0, 1000.0, 1000.0],
    })

    ajustado = ajustar_prediccion_modelo_con_patron(
        pred,
        proy,
        eval_actual_df,
        patron_prediction_weight=0.0,
        sn_alto=False,
        residual_weight=0.25,
    )

    assert np.abs(ajustado - 1000.0).mean() < np.abs(pred - 1000.0).mean()

    def test_lags_no_empujan_la_correccion_final():
        pred = np.array([800.0, 800.0, 800.0, 800.0])
        proy = np.array([0.0, 0.0, 0.0, 0.0])
        eval_actual_df = pd.DataFrame({
            'Produccion': [1000.0, 1000.0, 1000.0, 1000.0],
            'Produccion_lag10': [100.0, 100.0, 100.0, 100.0],
            'Produccion_lag11': [50.0, 50.0, 50.0, 50.0],
            'Produccion_lag12': [25.0, 25.0, 25.0, 25.0],
        })

        ajustado = ajustar_prediccion_modelo_con_patron(
            pred,
            proy,
            eval_actual_df,
            patron_prediction_weight=0.0,
            sn_alto=False,
            residual_weight=0.25,
        )

        assert np.abs(ajustado - 1000.0).mean() < np.abs(pred - 1000.0).mean()


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


def test_no_falla_cuando_la_alineacion_reduce_longitud(monkeypatch):
    monkeypatch.setattr(
        proy_ast,
        'alinear_series_para_ajuste',
        lambda *series: tuple(np.asarray(s, dtype=float)
                              [:2].copy() for s in series),
    )

    trabajo = pd.DataFrame({
        'Produccion': [100.0, 1000.0, 100.0, 100.0, 100.0],
        'Produccion_patron': [90.0, 900.0, 90.0, 90.0, 90.0],
        'Tallos/m2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'Tallos_m2_patron': [9.0, 19.0, 29.0, 39.0, 49.0],
    })

    resultado = proy_ast.ajustar_patron_con_extremos_real(trabajo)

    assert len(resultado) == len(trabajo)
    assert resultado['Produccion_patron'].shape[0] == len(trabajo)
    assert resultado['Tallos_m2_patron'].shape[0] == len(trabajo)


def test_refuerza_el_patron_cuando_2026_tiene_mas_tallos_que_2025():
    trabajo = pd.DataFrame({
        'Anio': [2025, 2025, 2026, 2026],
        'Semana': [1, 2, 1, 2],
        'Tallos/m2': [12.0, 13.0, 18.0, 19.0],
        'Tallos_m2_patron': [10.0, 11.0, 12.0, 13.0],
    })

    resultado = ajustar_patron_con_extremos_real(trabajo)
    media_2025 = resultado.loc[resultado['Anio']
                               == 2025, 'Tallos_m2_patron'].mean()
    media_2026 = resultado.loc[resultado['Anio']
                               == 2026, 'Tallos_m2_patron'].mean()

    assert media_2026 > media_2025


def test_expone_porcentaje_de_ajuste_en_base_a_tallos_2026_vs_2025():
    trabajo = pd.DataFrame({
        'Anio': [2025, 2025, 2026, 2026],
        'Semana': [1, 2, 1, 2],
        'Produccion': [1000.0, 1000.0, 1100.0, 1100.0],
        'Produccion_patron': [900.0, 900.0, 940.0, 940.0],
        'Tallos/m2': [12.0, 13.0, 18.0, 19.0],
        'Tallos_m2_patron': [10.0, 11.0, 12.0, 13.0],
    })

    resultado = ajustar_patron_con_extremos_real(trabajo)

    assert 'porcentaje_ajuste_modelo' in resultado.columns
    assert resultado['porcentaje_ajuste_modelo'].notna().any()
    assert resultado['porcentaje_ajuste_modelo'].max() > 0


def test_refuerzo_de_tallos_se_puede_configurar_con_parametro():
    trabajo = pd.DataFrame({
        'Anio': [2025, 2025, 2026, 2026],
        'Semana': [1, 2, 1, 2],
        'Produccion': [1000.0, 1000.0, 1100.0, 1100.0],
        'Produccion_patron': [900.0, 900.0, 940.0, 940.0],
        'Tallos/m2': [12.0, 13.0, 18.0, 19.0],
        'Tallos_m2_patron': [10.0, 11.0, 12.0, 13.0],
    })

    resultado_basico = ajustar_patron_con_extremos_real(trabajo)
    resultado_fuerte = ajustar_patron_con_extremos_real(
        trabajo,
        refuerzo_tallos_m2=0.20,
    )

    assert resultado_fuerte['Produccion_patron'].iloc[0] != resultado_basico['Produccion_patron'].iloc[0]


def test_refuerzo_mas_alto_acerca_mas_la_proyeccion_a_la_realidad():
    trabajo = pd.DataFrame({
        'Anio': [2025, 2025, 2026, 2026],
        'Semana': [1, 2, 1, 2],
        'Produccion': [1000.0, 1000.0, 1100.0, 1100.0],
        'Produccion_patron': [860.0, 870.0, 900.0, 910.0],
        'Tallos/m2': [12.0, 13.0, 18.0, 19.0],
        'Tallos_m2_patron': [10.0, 11.0, 12.0, 13.0],
    })

    resultado_bajo = ajustar_patron_con_extremos_real(
        trabajo,
        refuerzo_tallos_m2=0.05,
    )
    resultado_alto = ajustar_patron_con_extremos_real(
        trabajo,
        refuerzo_tallos_m2=0.30,
    )

    distancia_baja = np.abs(
        resultado_bajo['Produccion_patron'] - trabajo['Produccion']
    ).mean()
    distancia_alta = np.abs(
        resultado_alto['Produccion_patron'] - trabajo['Produccion']
    ).mean()

    assert distancia_alta < distancia_baja


def test_reacciona_al_pico_y_descenso_del_caso_021leila():
    pred = np.array([5000.0, 5000.0, 5000.0, 5000.0])
    proy = np.array([5000.0, 5000.0, 5000.0, 5000.0])
    eval_actual_df = pd.DataFrame({
        'Produccion': [5750.0, 8825.0, 1700.0, 1000.0],
        'Produccion_lag10': [3975.0, 3975.0, 9450.0, 1700.0],
        'Produccion_lag11': [3975.0, 3975.0, 4950.0, 2175.0],
        'Produccion_lag12': [3000.0, 2950.0, 4725.0, 2150.0],
    })

    ajustado = ajustar_prediccion_modelo_con_patron(
        pred,
        proy,
        eval_actual_df,
        patron_prediction_weight=0.0,
        sn_alto=True,
    )

    assert ajustado[0] == 5000.0
    assert ajustado[1] > ajustado[0]
    assert ajustado[2] <= 5000.0
    assert ajustado[3] <= ajustado[2]
