import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ProyAst import (  # noqa: E402
    aplicar_factor_diferencia_2026_semanas_24_52,
    ajustar_patron_con_extremos_real,
    ajustar_prediccion_modelo_con_patron,
    alinear_series_para_ajuste,
    calcular_factor_correccion_media_anio_en_curso,
    clasificar_sn,
    construir_resumen_ajustes_media,
    construir_objetivo_entrenamiento_con_patron,
)
from projection_core import _apply_difference_factor  # noqa: E402


def test_factor_media_2025_se_aplica_solo_semanas_24_a_52_de_2026(monkeypatch):
    evaluacion = pd.DataFrame({
        'Anio': [2025, 2026, 2026, 2026, 2026],
        'Semana': [24, 23, 24, 52, 53],
    })
    predicciones = np.full(5, 100.0)
    config = {'factores_por_variedad': {'VARIEDAD': 0.8}, 'factor_global': 0.9}

    ajustadas, factor, afectadas, origen = aplicar_factor_diferencia_2026_semanas_24_52(
        predicciones, evaluacion, 'VARIEDAD', config)

    np.testing.assert_array_equal(ajustadas, [100.0, 100.0, 80.0, 80.0, 100.0])
    assert (factor, afectadas, origen) == (0.8, 2, 'variedad')

    monkeypatch.setattr(
        'projection_core._load_difference_factors', lambda: config)
    ajustadas_core, _, afectadas_core, _ = _apply_difference_factor(
        predicciones, evaluacion, 'VARIEDAD')

    np.testing.assert_array_equal(ajustadas_core, ajustadas)
    assert afectadas_core == 2


def test_factor_media_compara_verano_con_inicio_del_mismo_anio():
    promedio = pd.DataFrame({
        'Anio': [2025, 2025, 2026, 2026, 2026, 2026],
        'Semana': [1, 24, 1, 23, 24, 52],
        'Tallos/m2': [300.0, 300.0, 100.0, 100.0, 70.0, 70.0],
    })

    factor = calcular_factor_correccion_media_anio_en_curso(promedio)

    assert np.isclose(factor, 0.7)


def test_resumen_consolida_todos_los_ajustes_de_media():
    export = pd.DataFrame({
        'Variedad_proyectada': ['001ROSA', '001ROSA'],
        'Factor_correccion': [0.9, 0.9],
        'Factor_ajuste_reciente_4_semanas': [0.85, 0.85],
        'Semanas_factor_2026_24_52': [4, 4],
        'Origen_factor_2025': ['variedad', 'variedad'],
        'Semanas_ajuste_reciente': [4, 4],
        'Origen_ajuste_reciente': ['variedad', 'variedad'],
        'Porcentaje_amortiguador_sugerido': [10.0, 10.0],
    })

    resumen = construir_resumen_ajustes_media(export)

    assert len(resumen) == 1
    fila = resumen.iloc[0]
    assert fila['Factor_media_anual'] == 0.9
    assert fila['Factor_media_ultimas_4_semanas'] == 0.85
    assert np.isclose(
        fila['Factor_media_combinado_informativo'],
        0.9 * 0.85,
    )
    assert 'Factor_media_ingresos' not in resumen.columns
    assert 'Factor_media_2025_2026' not in resumen.columns


def test_clasificacion_sn_alerta_con_7_db_o_menos():
    assert clasificar_sn(7.0) == 'alerta'
    for valor in [7.1, 8.0, 8.348, 9.0, 10.0, 10.999]:
        assert clasificar_sn(valor) == 'sin_dato'
    for valor in [11.0, 13.0]:
        assert clasificar_sn(valor) == 'ok'


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


def test_objetivo_entrenamiento_pondera_70_real_y_30_patron():
    entrenamiento = pd.DataFrame({
        'Produccion': [100.0, 200.0],
        'Produccion_patron': [300.0, 400.0],
    })

    objetivo = construir_objetivo_entrenamiento_con_patron(
        entrenamiento,
        patron_train_target_weight=0.30,
    )

    np.testing.assert_array_equal(objetivo, [160.0, 260.0])
