import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ingresos_core as ic  # noqa: E402  (requiere ROOT en sys.path)

HOY = pd.Timestamp('2026-08-09')


def proyeccion_base():
    return pd.DataFrame({
        'Variedad_proyectada': ['FREEDOM', 'MONDIAL'],
        'Anio_Semana': ['2026-32', '2026-32'],
        'Estimado_modelo': [1000.0, 500.0],
    })


def catalogo_base():
    """Archivo unico: variedad, longitud, % y precio de esa longitud."""
    return pd.DataFrame({
        'Fecha': [HOY] * 5,
        'Variedad': ['FREEDOM', 'FREEDOM', 'FREEDOM', 'MONDIAL', 'MONDIAL'],
        'Longitud': ['60cm', '60cm', '50cm', '70cm', '70cm'],
        'Mercado': ['USA', 'EU', 'USA', 'USA', 'EU'],
        'Pct': [60, 60, 40, 100, 100],
        'Precio': [0.50, 0.60, 0.40, 1.00, 0.90],
    })


def preparar(proyeccion=None, catalogo=None, estrategia='mejor_precio',
             mercado_fijo=None, normalizar=False, corte=HOY):
    catalogo_tidy, fuera = ic.normalizar_catalogo(
        catalogo if catalogo is not None else catalogo_base(),
        normalizar=normalizar, fecha_defecto=HOY)
    vigente = ic.catalogo_vigente(catalogo_tidy, fecha_corte=corte)
    precio = ic.elegir_mercado(
        vigente, estrategia=estrategia, mercado_fijo=mercado_fijo)
    return ic.calcular_ingresos(
        proyeccion if proyeccion is not None else proyeccion_base(),
        precio,
        mix_fuera_de_rango=fuera,
    )


def normalizado(catalogo=None, **kwargs):
    kwargs.setdefault('fecha_defecto', HOY)
    return ic.normalizar_catalogo(
        catalogo if catalogo is not None else catalogo_base(), **kwargs)


# --- calculo del ingreso ---------------------------------------------------

def test_ingreso_base_multiplica_tallos_por_pct_y_precio():
    resultado = preparar()

    # FREEDOM: 1000*0.6*0.60 (EU) + 1000*0.4*0.40 = 360 + 160 = 520
    # MONDIAL: 500*1.0*1.00 = 500
    assert resultado.totales['ingreso_base'] == pytest.approx(1020.0)
    assert resultado.totales['tallos_valorados'] == pytest.approx(1500.0)
    assert resultado.diagnostico.cobertura_pct == pytest.approx(100.0)
    assert not resultado.diagnostico.hay_alertas


def test_el_precio_depende_de_la_variedad_no_solo_de_la_longitud():
    catalogo = pd.DataFrame({
        'Fecha': [HOY, HOY],
        'Variedad': ['FREEDOM', 'MONDIAL'],
        'Longitud': ['60cm', '60cm'],
        'Mercado': ['USA', 'USA'],
        'Pct': [100, 100],
        'Precio': [0.50, 0.80],
    })

    resultado = preparar(catalogo=catalogo)
    por_variedad = resultado.por_variedad.set_index('variedad')

    # Misma longitud, dos precios: cada variedad se valora con el suyo.
    assert por_variedad.loc['FREEDOM', 'ingreso_base'] == pytest.approx(500.0)
    assert por_variedad.loc['MONDIAL', 'ingreso_base'] == pytest.approx(400.0)
    assert resultado.totales['ingreso_base'] == pytest.approx(900.0)


def test_escenarios_aplican_variacion_sobre_el_precio_vigente():
    resultado = preparar()

    assert resultado.totales['ingreso_pesimista'] == pytest.approx(
        1020.0 * 0.9)
    assert resultado.totales['ingreso_optimista'] == pytest.approx(
        1020.0 * 1.1)


def test_ranking_por_variedad_ordena_por_ingreso():
    resultado = preparar()

    fila_top = resultado.por_variedad.iloc[0]
    assert fila_top['variedad'] == 'FREEDOM'
    assert fila_top['rank'] == 1
    assert fila_top['ingreso_base'] == pytest.approx(520.0)


# --- mercados -------------------------------------------------------------

def test_mejor_precio_elige_el_mercado_mas_alto_por_variedad_y_longitud():
    catalogo_tidy, _ = normalizado()
    vigente = ic.catalogo_vigente(catalogo_tidy, fecha_corte=HOY)
    precio = ic.elegir_mercado(vigente, estrategia='mejor_precio')

    fila = precio[(precio['variedad'] == 'FREEDOM')
                  & (precio['grado'] == '60CM')].iloc[0]
    assert fila['mercado_elegido'] == 'EU'
    assert fila['precio'] == pytest.approx(0.60)
    # 0.60 vs 0.50 del segundo mercado = +20%
    assert fila['brecha_pct'] == pytest.approx(20.0)
    assert fila['mercados_disponibles'] == 2


def test_mercado_fijo_valora_contra_ese_mercado():
    resultado = preparar(estrategia='mercado_fijo', mercado_fijo='USA')

    # FREEDOM ahora usa 0.50 en 60cm: 300 + 160 = 460; MONDIAL 500
    assert resultado.totales['ingreso_base'] == pytest.approx(960.0)


def test_mercado_fijo_no_tapa_con_otro_mercado_lo_que_no_compra():
    catalogo = pd.DataFrame({
        'Fecha': [HOY] * 2,
        'Variedad': ['FREEDOM', 'FREEDOM'],
        'Longitud': ['60cm', '50cm'],
        'Mercado': ['USA', 'EU'],
        'Pct': [60, 40],
        'Precio': [0.50, 0.40],
    })

    resultado = preparar(catalogo=catalogo, estrategia='mercado_fijo',
                         mercado_fijo='USA')

    # El 50cm solo tiene precio en EU: queda sin valorar y se reporta.
    assert resultado.diagnostico.grados_sin_precio == ['FREEDOM 50CM']
    assert resultado.totales['ingreso_base'] == pytest.approx(300.0)


def test_mercado_fijo_inexistente_falla_con_mensaje_util():
    catalogo_tidy, _ = normalizado()
    vigente = ic.catalogo_vigente(catalogo_tidy, fecha_corte=HOY)

    with pytest.raises(ValueError, match='ASIA'):
        ic.elegir_mercado(vigente, estrategia='mercado_fijo',
                          mercado_fijo='ASIA')


# --- cobertura y diagnostico ---------------------------------------------

def test_variedad_fuera_del_catalogo_no_desaparece_en_silencio():
    proyeccion = pd.concat([
        proyeccion_base(),
        pd.DataFrame({
            'Variedad_proyectada': ['VENDELA'],
            'Anio_Semana': ['2026-32'],
            'Estimado_modelo': [300.0],
        }),
    ], ignore_index=True)

    resultado = preparar(proyeccion=proyeccion)

    assert resultado.diagnostico.variedades_sin_catalogo == ['VENDELA']
    assert resultado.diagnostico.tallos_sin_valorar == pytest.approx(300.0)
    assert resultado.diagnostico.cobertura_pct == pytest.approx(
        100 * 1500 / 1800)
    assert resultado.diagnostico.hay_alertas
    assert any('VENDELA' in msg for msg in resultado.diagnostico.mensajes())


def test_longitud_con_pct_pero_sin_precio_se_reporta_por_variedad():
    catalogo = pd.DataFrame({
        'Fecha': [HOY] * 4,
        'Variedad': ['FREEDOM', 'FREEDOM', 'FREEDOM', 'MONDIAL'],
        'Longitud': ['60cm', '50cm', '40cm', '70cm'],
        'Mercado': ['USA'] * 4,
        'Pct': [50, 30, 20, 100],
        'Precio': [0.50, 0.40, None, 1.00],
    })

    resultado = preparar(catalogo=catalogo)

    assert resultado.diagnostico.grados_sin_precio == ['FREEDOM 40CM']
    # El 20% de FREEDOM queda sin valorar: 1000*0.8 + 500 = 1300
    assert resultado.totales['tallos_valorados'] == pytest.approx(1300.0)


def test_precio_en_cero_cuenta_como_sin_precio():
    catalogo = pd.DataFrame({
        'Fecha': [HOY] * 2,
        'Variedad': ['FREEDOM', 'FREEDOM'],
        'Longitud': ['60cm', '50cm'],
        'Mercado': ['USA', 'USA'],
        'Pct': [70, 30],
        'Precio': [0.50, 0.0],
    })

    resultado = preparar(catalogo=catalogo)

    assert resultado.diagnostico.grados_sin_precio == ['FREEDOM 50CM']
    assert resultado.totales['ingreso_base'] == pytest.approx(350.0)


def test_pct_que_no_suma_cien_se_reporta_sin_normalizar():
    catalogo = catalogo_base()
    catalogo.loc[catalogo['Longitud'] == '50cm', 'Pct'] = 30

    resultado = preparar(catalogo=catalogo)

    assert resultado.diagnostico.mix_fuera_de_rango == {
        'FREEDOM': pytest.approx(90.0)}
    # Sin normalizar, el 10% faltante no se inventa.
    assert resultado.totales['tallos_valorados'] == pytest.approx(1400.0)


def test_los_pct_se_revalidan_sobre_el_catalogo_vigente():
    catalogo = catalogo_base()
    catalogo.loc[catalogo['Longitud'] == '50cm', 'Pct'] = 30
    catalogo_tidy, _ = normalizado(catalogo)
    vigente = ic.catalogo_vigente(catalogo_tidy, fecha_corte=HOY)

    # El aviso no puede depender de haber subido el archivo en esta sesion.
    assert ic.variedades_fuera_de_rango(vigente) == {
        'FREEDOM': pytest.approx(90.0)}
    assert ic.variedades_fuera_de_rango(pd.DataFrame()) == {}


def test_pct_se_puede_normalizar_de_forma_explicita():
    catalogo = pd.DataFrame({
        'Fecha': [HOY] * 2,
        'Variedad': ['FREEDOM', 'FREEDOM'],
        'Longitud': ['60cm', '50cm'],
        'Mercado': ['USA', 'USA'],
        'Pct': [60, 30],
        'Precio': [0.50, 0.40],
    })

    catalogo_tidy, fuera = normalizado(catalogo, normalizar=True)

    assert fuera == {'FREEDOM': pytest.approx(90.0)}
    assert catalogo_tidy['fraccion'].sum() == pytest.approx(1.0)


# --- limpieza del archivo -------------------------------------------------

def test_fila_repetida_gana_la_ultima():
    catalogo = pd.DataFrame({
        'Fecha': [HOY, HOY],
        'Variedad': ['FREEDOM', 'FREEDOM'],
        'Longitud': ['60cm', '60cm'],
        'Mercado': ['USA', 'USA'],
        'Pct': [100, 100],
        'Precio': [0.50, 0.55],
    })

    catalogo_tidy, fuera = normalizado(catalogo)

    assert len(catalogo_tidy) == 1
    assert catalogo_tidy.iloc[0]['precio'] == pytest.approx(0.55)
    assert fuera == {}


def test_la_longitud_se_unifica_aunque_venga_escrita_distinto():
    catalogo = pd.DataFrame({
        'Fecha': [HOY] * 3,
        'Variedad': ['FREEDOM'] * 3,
        'Longitud': ['60', '60 CM', '50cm'],
        'Mercado': ['USA'] * 3,
        'Pct': [40, 40, 60],
        'Precio': [0.50, 0.50, 0.40],
    })

    catalogo_tidy, fuera = normalizado(catalogo)

    assert sorted(catalogo_tidy['grado']) == ['50CM', '60CM']
    assert fuera == {}


def test_el_pct_puesto_en_un_solo_mercado_aplica_a_los_demas():
    catalogo = pd.DataFrame({
        'Fecha': [HOY] * 2,
        'Variedad': ['FREEDOM', 'FREEDOM'],
        'Longitud': ['60cm', '60cm'],
        'Mercado': ['USA', 'EU'],
        'Pct': [100, None],
        'Precio': [0.50, 0.60],
    })

    catalogo_tidy, _ = normalizado(catalogo)

    # La fila de EU no se pierde por tener el % en blanco.
    assert sorted(catalogo_tidy['mercado']) == ['EU', 'USA']
    assert (catalogo_tidy['pct'] == 100).all()


def test_catalogo_vacio_falla_con_mensaje_claro():
    with pytest.raises(ValueError, match='vacio'):
        ic.normalizar_catalogo(pd.DataFrame())


def test_formato_largo_sin_pct_con_una_longitud_asume_100():
    catalogo = pd.DataFrame({
        'Producto': ['ROSA'],
        'Variedad': ['FREEDOM'],
        'Longitud (cm)': ['60'],
        'Precio (USD/tallo)': [0.55],
    })

    catalogo_tidy, fuera = normalizado(catalogo)

    assert len(catalogo_tidy) == 1
    assert catalogo_tidy.iloc[0]['pct'] == pytest.approx(100.0)
    assert catalogo_tidy.iloc[0]['fraccion'] == pytest.approx(1.0)
    assert catalogo_tidy.iloc[0]['precio'] == pytest.approx(0.55)
    assert fuera == {}


def test_formato_largo_sin_pct_con_varias_longitudes_reparte_equilibrado():
    catalogo = pd.DataFrame({
        'Producto': ['ROSA', 'ROSA'],
        'Variedad': ['FREEDOM', 'FREEDOM'],
        'Longitud (cm)': ['50', '60'],
        'Precio (USD/tallo)': [0.40, 0.55],
    })

    catalogo_tidy, fuera = normalizado(catalogo)

    assert len(catalogo_tidy) == 2
    assert sorted(catalogo_tidy['pct'].tolist()) == pytest.approx([50.0, 50.0])
    assert sorted(catalogo_tidy['fraccion'].tolist()
                  ) == pytest.approx([0.5, 0.5])
    assert fuera == {}


# --- formatos aceptados ---------------------------------------------------

def catalogo_un_mercado():
    return pd.DataFrame({
        'Fecha': [HOY] * 3,
        'Variedad': ['FREEDOM', 'FREEDOM', 'MONDIAL'],
        'Longitud': ['50cm', '60cm', '70cm'],
        'Mercado': ['USA'] * 3,
        'Pct': [40, 60, 100],
        'Precio': [0.40, 0.50, 1.00],
    })


def catalogo_ancho():
    return pd.DataFrame({
        'Variedad': ['FREEDOM', 'MONDIAL'],
        'Fecha': [HOY, HOY],
        'Mercado': ['USA', 'USA'],
        'Precio 50': [0.40, np.nan],
        '% 50': [40, 0],
        'Precio 60': [0.50, np.nan],
        '% 60': [60, 0],
        'Precio 70': [np.nan, 1.00],
        '% 70': [0, 100],
    })


def test_formato_se_detecta_por_los_encabezados():
    assert ic.formato_catalogo(catalogo_un_mercado()) == 'largo'
    assert ic.formato_catalogo(catalogo_ancho()) == 'ancho'


def test_formato_ancho_da_el_mismo_ingreso_que_el_largo():
    esperado = preparar(catalogo=catalogo_un_mercado())
    resultado = preparar(catalogo=catalogo_ancho())

    # 1000*(0.4*0.40 + 0.6*0.50) + 500*1.00 = 460 + 500
    assert esperado.totales['ingreso_base'] == pytest.approx(960.0)
    assert resultado.totales['ingreso_base'] == pytest.approx(
        esperado.totales['ingreso_base'])
    assert not resultado.diagnostico.hay_alertas


def test_formato_ancho_sin_porcentajes_explica_que_falta():
    ancho = pd.DataFrame({
        'Variedad': ['FREEDOM'],
        '50': [0.40],
        '60': [0.50],
    })

    with pytest.raises(ValueError, match='porcentaje'):
        ic.normalizar_catalogo(ancho)


def test_formato_ancho_sin_precios_explica_que_falta():
    ancho = pd.DataFrame({
        'Variedad': ['FREEDOM'],
        '% 50': [40],
        '% 60': [60],
    })

    with pytest.raises(ValueError, match='precios'):
        ic.normalizar_catalogo(ancho)


def test_un_anio_en_el_encabezado_no_se_confunde_con_una_longitud():
    largo = pd.DataFrame({
        'Variedad': ['FREEDOM'],
        'Longitud': ['60cm'],
        'Pct': [100],
        'Precio 2026': [0.50],
    })

    catalogo_tidy, _ = normalizado(largo)

    assert ic.formato_catalogo(largo) == 'largo'
    assert catalogo_tidy.iloc[0]['precio'] == pytest.approx(0.50)


# --- vigencia: el precio se arrastra, el % no ----------------------------

def historico_dos_cargas(primera, segunda, fecha_primera, fecha_segunda):
    carga_1, _ = ic.normalizar_catalogo(primera, fecha_defecto=fecha_primera)
    carga_2, _ = ic.normalizar_catalogo(segunda, fecha_defecto=fecha_segunda)
    return pd.concat([carga_1, carga_2], ignore_index=True)


def test_precio_se_arrastra_y_se_marca_su_antiguedad():
    corte = HOY + pd.Timedelta(days=11)
    resultado = preparar(corte=corte)

    assert resultado.diagnostico.precios_desactualizados
    assert all(dias == 11
               for dias in resultado.diagnostico.precios_desactualizados.values())
    assert any('11d' in msg for msg in resultado.diagnostico.mensajes())


def test_una_longitud_que_sale_del_mix_no_queda_viva():
    catalogo = historico_dos_cargas(
        pd.DataFrame({
            'Variedad': ['FREEDOM', 'FREEDOM'],
            'Longitud': ['60cm', '40cm'],
            'Pct': [60, 40],
            'Precio': [0.50, 0.30],
        }),
        pd.DataFrame({
            'Variedad': ['FREEDOM'],
            'Longitud': ['60cm'],
            'Pct': [100],
            'Precio': [0.55],
        }),
        HOY - pd.Timedelta(days=8), HOY,
    )

    vigente = ic.catalogo_vigente(catalogo, fecha_corte=HOY)

    # El 40cm de la carga vieja no se arrastra: el % es una foto completa.
    assert list(vigente['grado']) == ['60CM']
    assert vigente.iloc[0]['pct'] == pytest.approx(100.0)


def test_el_mix_nuevo_conserva_el_precio_arrastrado_de_la_carga_anterior():
    catalogo = historico_dos_cargas(
        pd.DataFrame({
            'Variedad': ['FREEDOM', 'FREEDOM'],
            'Longitud': ['60cm', '50cm'],
            'Pct': [50, 50],
            'Precio': [0.50, 0.40],
        }),
        pd.DataFrame({
            'Variedad': ['FREEDOM', 'FREEDOM'],
            'Longitud': ['60cm', '50cm'],
            'Pct': [50, 50],
            'Precio': [0.55, None],
        }),
        HOY - pd.Timedelta(days=8), HOY,
    )
    vigente = ic.catalogo_vigente(catalogo, fecha_corte=HOY)
    precio = ic.elegir_mercado(vigente)
    resultado = ic.calcular_ingresos(
        pd.DataFrame({
            'Variedad_proyectada': ['FREEDOM'],
            'Anio_Semana': ['2026-32'],
            'Estimado_modelo': [1000.0],
        }),
        precio,
    )

    # 60cm con el precio nuevo, 50cm con el de hace 8 dias.
    assert resultado.totales['ingreso_base'] == pytest.approx(475.0)
    assert resultado.diagnostico.precios_desactualizados == {'FREEDOM 50CM': 8}


def test_cargas_posteriores_al_corte_no_se_usan():
    catalogo = historico_dos_cargas(
        catalogo_base(),
        pd.DataFrame({
            'Variedad': ['MONDIAL'],
            'Longitud': ['70cm'],
            'Mercado': ['USA'],
            'Pct': [100],
            'Precio': [9.99],
        }),
        HOY, HOY + pd.Timedelta(days=5),
    )
    vigente = ic.catalogo_vigente(catalogo, fecha_corte=HOY)
    resultado = ic.calcular_ingresos(
        proyeccion_base(), ic.elegir_mercado(vigente))

    assert resultado.totales['ingreso_base'] == pytest.approx(1020.0)


def test_corte_anterior_a_todo_el_catalogo_falla_con_la_fecha():
    catalogo_tidy, _ = normalizado()

    with pytest.raises(ValueError, match='2026-01-01'):
        ic.catalogo_vigente(catalogo_tidy, fecha_corte='2026-01-01')


def test_tendencia_compara_contra_el_precio_de_hace_n_dias():
    catalogo = historico_dos_cargas(
        pd.DataFrame({
            'Variedad': ['FREEDOM'], 'Longitud': ['60cm'], 'Mercado': ['USA'],
            'Pct': [100], 'Precio': [0.50],
        }),
        pd.DataFrame({
            'Variedad': ['FREEDOM'], 'Longitud': ['60cm'], 'Mercado': ['USA'],
            'Pct': [100], 'Precio': [0.60],
        }),
        HOY - pd.Timedelta(days=7), HOY,
    )

    tendencia = ic.tendencia_precios(catalogo, dias=7, fecha_corte=HOY)
    fila = tendencia.iloc[0]

    assert fila['variedad'] == 'FREEDOM'
    assert fila['grado'] == '60CM'
    assert fila['precio_actual'] == pytest.approx(0.60)
    assert fila['precio_hace_7d'] == pytest.approx(0.50)
    assert fila['delta_pct'] == pytest.approx(20.0)


# --- cruce y mapeo de columnas -------------------------------------------

def test_cruce_vacio_falla_en_vez_de_reportar_cero():
    catalogo = pd.DataFrame({
        'Fecha': [HOY],
        'Variedad': ['OTRA_VARIEDAD'],
        'Longitud': ['60cm'],
        'Mercado': ['USA'],
        'Pct': [100],
        'Precio': [0.50],
    })

    with pytest.raises(ValueError, match='cruce quedo vacio'):
        preparar(catalogo=catalogo)


def test_columnas_se_pueden_mapear_a_mano():
    catalogo = pd.DataFrame({
        'col_a': ['FREEDOM'],
        'col_b': ['60cm'],
        'col_c': [100],
        'col_d': [0.50],
    })
    mapeo = {'variedad': 'col_a', 'grado': 'col_b',
             'pct': 'col_c', 'precio': 'col_d'}

    catalogo_tidy, _ = normalizado(catalogo, mapeo=mapeo)

    assert catalogo_tidy.iloc[0]['variedad'] == 'FREEDOM'
    assert catalogo_tidy.iloc[0]['grado'] == '60CM'
    assert catalogo_tidy.iloc[0]['fraccion'] == pytest.approx(1.0)
    assert catalogo_tidy.iloc[0]['precio'] == pytest.approx(0.50)


def test_columna_faltante_falla_listando_las_disponibles():
    catalogo = pd.DataFrame({'col_a': ['FREEDOM'], 'col_b': ['60cm']})

    with pytest.raises(ValueError, match='col_a'):
        ic.normalizar_catalogo(catalogo)


def test_columna_mapeada_a_un_nombre_inexistente_avisa():
    with pytest.raises(ValueError, match='no existe'):
        ic.normalizar_catalogo(
            catalogo_base(), mapeo={'precio': 'columna_fantasma'})


# --- persistencia ---------------------------------------------------------

def test_acumular_catalogo_conserva_las_cargas_anteriores(tmp_path):
    # Sin columna de fecha en el archivo: la fecha de la carga es la del dia.
    sin_fecha = catalogo_base().drop(columns='Fecha')
    carga_1, _ = normalizado(
        sin_fecha, fecha_defecto=HOY - pd.Timedelta(days=1))
    carga_2, _ = normalizado(sin_fecha, fecha_defecto=HOY)

    ic.acumular_catalogo(carga_1, carpeta=tmp_path)
    ic.acumular_catalogo(carga_2, carpeta=tmp_path)
    historico = ic.cargar_catalogo(carpeta=tmp_path)

    assert historico['fecha'].nunique() == 2
    assert len(historico) == 2 * len(carga_1)


def test_acumular_catalogo_deduplica_la_misma_fecha(tmp_path):
    primero, _ = normalizado(pd.DataFrame({
        'Variedad': ['FREEDOM'], 'Longitud': ['60cm'], 'Mercado': ['USA'],
        'Pct': [100], 'Precio': [0.50],
    }))
    correccion, _ = normalizado(pd.DataFrame({
        'Variedad': ['FREEDOM'], 'Longitud': ['60cm'], 'Mercado': ['USA'],
        'Pct': [100], 'Precio': [0.58],
    }))

    ic.acumular_catalogo(primero, carpeta=tmp_path)
    ic.acumular_catalogo(correccion, carpeta=tmp_path)
    historico = ic.cargar_catalogo(carpeta=tmp_path)

    assert len(historico) == 1
    assert historico.iloc[0]['precio'] == pytest.approx(0.58)


def test_catalogo_vacio_devuelve_estructura_utilizable(tmp_path):
    historico = ic.cargar_catalogo(carpeta=tmp_path)

    assert historico.empty
    assert list(historico.columns) == ic.COLUMNAS_CATALOGO


def test_el_catalogo_guardado_se_puede_volver_a_usar(tmp_path):
    carga, _ = normalizado()
    ic.acumular_catalogo(carga, carpeta=tmp_path)

    recargado = ic.cargar_catalogo(carpeta=tmp_path)
    resultado = ic.calcular_ingresos(
        proyeccion_base(),
        ic.elegir_mercado(ic.catalogo_vigente(recargado, fecha_corte=HOY)),
    )

    assert resultado.totales['ingreso_base'] == pytest.approx(1020.0)


def test_la_plantilla_de_ejemplo_es_un_catalogo_valido():
    catalogo_tidy, fuera = ic.normalizar_catalogo(ic.plantilla_catalogo())

    assert fuera == {}
    assert set(catalogo_tidy['grado']) == {'50CM', '60CM', '70CM'}
    assert catalogo_tidy['precio'].notna().all()
    # La plantilla debe servir tal cual para arrancar un analisis.
    vigente = ic.catalogo_vigente(catalogo_tidy, fecha_corte='2026-08-10')
    assert not ic.elegir_mercado(vigente).empty


# --- resumen para el modelo ----------------------------------------------

def test_resumen_para_llm_entrega_totales_y_advertencias():
    proyeccion = pd.concat([
        proyeccion_base(),
        pd.DataFrame({
            'Variedad_proyectada': ['VENDELA'],
            'Anio_Semana': ['2026-32'],
            'Estimado_modelo': [300.0],
        }),
    ], ignore_index=True)
    resultado = preparar(proyeccion=proyeccion)

    texto = ic.resumen_para_llm(resultado)

    assert 'Total base: 1,020.00' in texto
    assert 'FREEDOM' in texto
    assert 'ADVERTENCIAS DE COBERTURA' in texto
    assert 'VENDELA' in texto
    # El resumen debe ser compacto, no un volcado de filas.
    assert len(texto.splitlines()) < 60


def test_resumen_recorta_los_precios_a_las_variedades_del_ranking():
    resultado = preparar()

    texto = ic.resumen_para_llm(resultado, top=1)

    # Con precio por variedad la tabla crece con el catalogo entero.
    assert 'FREEDOM' in texto
    assert 'MONDIAL' not in texto


# --- probabilidad del ingreso --------------------------------------------

def errores_sinteticos():
    """Errores (real - estimado)/real con la asimetria del caso real."""
    return pd.Series([0.30, 0.20, 0.10, 0.05, 0.0, -0.05, -0.10, -0.40, -1.50])


def test_los_errores_medidos_del_modelo_se_leen_del_proyecto():
    errores = ic.cargar_errores_modelo()

    # Los 400+ casos evaluados son la base empirica de la probabilidad.
    assert len(errores) > 400
    # El error esta acotado en +1 por construccion, pero no por abajo.
    assert errores.max() < 1.0
    assert errores.min() < -1.0
    assert 40.0 < ic.probabilidad_cumplimiento(errores) < 70.0


def test_el_error_se_convierte_en_factor_sobre_el_estimado():
    factor = ic.factor_real_sobre_estimado(
        pd.Series([0.0, 0.5, -1.0, -3.0]))

    # real = estimado / (1 - error)
    assert list(factor.round(4)) == [1.0, 2.0, 0.5, 0.25]
    # Un error de +1 implicaria estimado cero: se recorta en vez de explotar.
    assert np.isfinite(ic.factor_real_sobre_estimado(pd.Series([1.0]))).all()


def test_probabilidad_de_cumplimiento_y_bandas():
    errores = errores_sinteticos()

    # 5 de 9 casos con error >= 0.
    assert ic.probabilidad_cumplimiento(errores) == pytest.approx(100 * 5 / 9)
    assert ic.probabilidad_cumplimiento(pd.Series(dtype=float)) == 0.0

    tabla = ic.tabla_cumplimiento(errores, tolerancias=(0.05, 0.10, 0.30))
    assert list(tabla['tolerancia_pct']) == [5.0, 10.0, 30.0]
    # A mayor tolerancia, mas casos entran: nunca puede bajar.
    assert list(tabla['casos']) == sorted(tabla['casos'])
    assert tabla.iloc[-1]['probabilidad_pct'] == pytest.approx(100 * 7 / 9)


def test_simulacion_es_reproducible_y_cambia_con_la_semilla():
    resultado = preparar()

    a = ic.simular_ingresos(resultado, simulaciones=500, semilla=1)
    b = ic.simular_ingresos(resultado, simulaciones=500, semilla=1)
    c = ic.simular_ingresos(resultado, simulaciones=500, semilla=2)

    # El panel se recarga todo el tiempo: la probabilidad no puede bailar.
    assert a.percentiles == b.percentiles
    assert a.percentiles['p50'] != c.percentiles['p50']


def test_sin_incertidumbre_la_distribucion_colapsa_en_el_deterministico():
    resultado = preparar()

    probabilidad = ic.simular_ingresos(
        resultado, simulaciones=200, sigma_precio_defecto=0.0,
        sigma_tallos_defecto=0.0)

    assert probabilidad.percentiles['p50'] == pytest.approx(1020.0)
    assert probabilidad.desviacion == pytest.approx(0.0)
    assert probabilidad.probabilidad_de_superar(1020.0) == pytest.approx(100.0)


def test_mas_volatilidad_ensancha_el_intervalo():
    resultado = preparar()

    poca = ic.simular_ingresos(resultado, simulaciones=2000,
                               sigma_precio_defecto=0.05, sigma_tallos_defecto=0.0)
    mucha = ic.simular_ingresos(resultado, simulaciones=2000,
                                sigma_precio_defecto=0.25, sigma_tallos_defecto=0.0)

    ancho_poca = poca.percentiles['p95'] - poca.percentiles['p05']
    ancho_mucha = mucha.percentiles['p95'] - mucha.percentiles['p05']
    assert ancho_mucha > ancho_poca * 2


def test_la_correlacion_no_diversifica_el_riesgo_de_forma_ficticia():
    resultado = preparar()

    independiente = ic.simular_ingresos(
        resultado, simulaciones=3000, sigma_precio_defecto=0.20,
        sigma_tallos_defecto=0.0, correlacion_precio=0.0)
    correlacionado = ic.simular_ingresos(
        resultado, simulaciones=3000, sigma_precio_defecto=0.20,
        sigma_tallos_defecto=0.0, correlacion_precio=0.95)

    # Si los precios se mueven juntos el total es mas riesgoso, no menos.
    assert correlacionado.desviacion > independiente.desviacion


def test_la_volatilidad_de_precio_sale_del_historico_del_catalogo():
    catalogo = historico_dos_cargas(
        pd.DataFrame({
            'Variedad': ['FREEDOM'], 'Longitud': ['60cm'], 'Mercado': ['USA'],
            'Pct': [100], 'Precio': [0.50],
        }),
        pd.DataFrame({
            'Variedad': ['FREEDOM'], 'Longitud': ['60cm'], 'Mercado': ['USA'],
            'Pct': [100], 'Precio': [0.60],
        }),
        HOY - pd.Timedelta(days=7), HOY,
    )
    tercera, _ = ic.normalizar_catalogo(pd.DataFrame({
        'Variedad': ['FREEDOM'], 'Longitud': ['60cm'], 'Mercado': ['USA'],
        'Pct': [100], 'Precio': [0.54],
    }), fecha_defecto=HOY + pd.Timedelta(days=7))
    catalogo = pd.concat([catalogo, tercera], ignore_index=True)

    volatilidad = ic.volatilidad_precios(catalogo)

    fila = volatilidad.iloc[0]
    assert fila['variedad'] == 'FREEDOM'
    assert fila['cambios'] == 2
    # Variaciones de +20% y -10%: sigma = std([0.2, -0.1]).
    assert fila['sigma_precio'] == pytest.approx(np.std([0.2, -0.1], ddof=1))
    assert ic.volatilidad_precios(pd.DataFrame()).empty


def test_sin_historico_la_volatilidad_de_precio_queda_declarada_como_supuesto():
    catalogo_tidy, _ = normalizado()
    resultado = preparar()

    probabilidad = ic.simular_ingresos(
        resultado, catalogo=catalogo_tidy, simulaciones=300)

    # Una sola carga no da variacion: no se puede llamar medicion.
    assert 'supuesto' in probabilidad.supuestos['precio']
    assert 'supuesto' in probabilidad.supuestos['tallos']


def test_el_error_del_modelo_se_mide_del_backtest_de_la_proyeccion():
    proyeccion = pd.DataFrame({
        'Variedad_proyectada': ['FREEDOM'] * 4,
        'Anio_Semana': ['2026-28', '2026-29', '2026-30', '2026-31'],
        'Estimado_modelo': [1000.0, 1000.0, 1000.0, 1000.0],
        'Produccion_real': [1100.0, 900.0, 1050.0, 950.0],
    })

    error = ic.volatilidad_proyeccion(proyeccion)

    assert error.iloc[0]['variedad'] == 'FREEDOM'
    assert error.iloc[0]['observaciones'] == 4
    assert error.iloc[0]['sigma_tallos'] > 0
    # Sin columna de produccion real no hay nada que medir.
    assert ic.volatilidad_proyeccion(
        proyeccion.drop(columns='Produccion_real')).empty


def test_la_columna_de_produccion_real_no_altera_la_valoracion():
    proyeccion = proyeccion_base()
    proyeccion['Produccion_real'] = [1100.0, 480.0]

    resultado = preparar(proyeccion=proyeccion)

    # El estimado sigue siendo la columna que se valora, no el real.
    assert resultado.totales['ingreso_base'] == pytest.approx(1020.0)


def test_los_errores_medidos_reemplazan_al_supuesto_normal():
    resultado = preparar()

    probabilidad = ic.simular_ingresos(
        resultado, errores_modelo=errores_sinteticos(), simulaciones=3000,
        sigma_precio_defecto=0.0)

    assert 'empirica' in probabilidad.supuestos['tallos']
    assert probabilidad.casos_evaluados == 9
    assert not probabilidad.cumplimiento.empty
    assert 'cumplimiento' in probabilidad.supuestos


def test_la_simulacion_empirica_conserva_la_asimetria_medida():
    resultado = preparar()
    factores = ic.factor_real_sobre_estimado(errores_sinteticos())

    probabilidad = ic.simular_ingresos(
        resultado, errores_modelo=errores_sinteticos(), simulaciones=6000,
        sigma_precio_defecto=0.0, correlacion_tallos=1.0)

    # Con un unico choque comun el ingreso es base x factor sorteado, asi que la
    # distribucion del ingreso tiene que reproducir la de los factores medidos.
    esperado_media = 1020.0 * float(factores.mean())
    esperado_p50 = 1020.0 * float(factores.median())
    assert probabilidad.media == pytest.approx(esperado_media, rel=0.05)
    assert probabilidad.percentiles['p50'] == pytest.approx(
        esperado_p50, rel=0.05)


def test_la_probabilidad_de_superar_es_monotona():
    resultado = preparar()
    probabilidad = ic.simular_ingresos(resultado, simulaciones=2000)

    p05 = probabilidad.percentiles['p05']
    p95 = probabilidad.percentiles['p95']

    assert probabilidad.probabilidad_de_superar(p05) > 90.0
    assert probabilidad.probabilidad_de_superar(p95) < 10.0
    inferior, superior = probabilidad.intervalo(0.90)
    assert inferior < probabilidad.percentiles['p50'] < superior


def test_el_rango_por_variedad_es_coherente_con_el_total():
    resultado = preparar()
    probabilidad = ic.simular_ingresos(resultado, simulaciones=1000)

    por_variedad = probabilidad.por_variedad
    assert set(por_variedad['variedad']) == {'FREEDOM', 'MONDIAL'}
    assert (por_variedad['p05'] <= por_variedad['p50']).all()
    assert (por_variedad['p50'] <= por_variedad['p95']).all()
    # El esperado por variedad tiene que sumar el esperado total.
    assert por_variedad['esperado'].sum() == pytest.approx(
        probabilidad.media, rel=1e-9)
    assert probabilidad.por_semana['semana'].tolist() == ['2026-32']


def test_simulaciones_insuficientes_se_rechazan():
    resultado = preparar()

    with pytest.raises(ValueError, match='100 simulaciones'):
        ic.simular_ingresos(resultado, simulaciones=10)
    with pytest.raises(ValueError, match='correlacion_precio'):
        ic.simular_ingresos(resultado, simulaciones=200,
                            correlacion_precio=1.5)


def test_el_ingreso_comprometible_baja_al_exigir_mas_confianza():
    resultado = preparar()
    probabilidad = ic.simular_ingresos(
        resultado, errores_modelo=ic.cargar_errores_modelo(), simulaciones=3000)

    al_50 = probabilidad.ingreso_comprometible(0.50)
    al_80 = probabilidad.ingreso_comprometible(0.80)
    al_95 = probabilidad.ingreso_comprometible(0.95)

    # Exigir mas probabilidad de cumplir obliga a prometer menos.
    assert al_50 > al_80 > al_95
    assert al_50 == pytest.approx(probabilidad.percentiles['p50'], rel=1e-9)
    assert al_95 == pytest.approx(probabilidad.percentiles['p05'], rel=1e-9)
    # Y el monto comprometido se alcanza con la probabilidad prometida.
    assert probabilidad.probabilidad_de_superar(
        al_80) == pytest.approx(80.0, abs=1.5)


def test_la_curva_de_confianza_es_monotona():
    resultado = preparar()
    probabilidad = ic.simular_ingresos(resultado, simulaciones=2000)

    curva = probabilidad.curva_confianza()

    assert list(curva['confianza_pct']) == [
        50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0]
    assert list(curva['ingreso']) == sorted(curva['ingreso'], reverse=True)


def test_confianza_fuera_de_rango_se_rechaza():
    probabilidad = ic.simular_ingresos(preparar(), simulaciones=200)

    for invalida in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match='confianza'):
            probabilidad.ingreso_comprometible(invalida)


def test_la_barra_de_volatilidad_ensancha_sin_mover_el_centro():
    resultado = preparar()
    errores = ic.cargar_errores_modelo()

    medida = ic.simular_ingresos(
        resultado, errores_modelo=errores, simulaciones=4000,
        sigma_precio_defecto=0.0)
    duplicada = ic.simular_ingresos(
        resultado, errores_modelo=errores, simulaciones=4000,
        sigma_precio_defecto=0.0, escala_volatilidad_proyeccion=2.0)

    # Estirar la dispersion no debe mover la mediana medida...
    assert duplicada.percentiles['p50'] == pytest.approx(
        medida.percentiles['p50'], rel=0.05)
    # ...pero si abrir las colas y castigar lo comprometible.
    assert duplicada.desviacion > medida.desviacion
    assert duplicada.ingreso_comprometible(
        0.90) < medida.ingreso_comprometible(0.90)
    assert 'escala' in duplicada.supuestos
    assert 'escala' not in medida.supuestos


def test_menos_volatilidad_angosta_el_intervalo():
    resultado = preparar()
    errores = ic.cargar_errores_modelo()

    medida = ic.simular_ingresos(
        resultado, errores_modelo=errores, simulaciones=4000,
        sigma_precio_defecto=0.0)
    reducida = ic.simular_ingresos(
        resultado, errores_modelo=errores, simulaciones=4000,
        sigma_precio_defecto=0.0, escala_volatilidad_proyeccion=0.5)

    assert reducida.desviacion < medida.desviacion
    assert reducida.ingreso_comprometible(
        0.90) > medida.ingreso_comprometible(0.90)


def test_la_escala_tambien_aplica_al_camino_por_supuesto():
    resultado = preparar()

    normal = ic.simular_ingresos(
        resultado, simulaciones=3000, sigma_precio_defecto=0.0,
        sigma_tallos_defecto=0.10)
    escalada = ic.simular_ingresos(
        resultado, simulaciones=3000, sigma_precio_defecto=0.0,
        sigma_tallos_defecto=0.10, escala_volatilidad_proyeccion=2.0)

    assert escalada.desviacion > normal.desviacion * 1.5


def test_escala_de_volatilidad_invalida_se_rechaza():
    resultado = preparar()

    with pytest.raises(ValueError, match='escala_volatilidad_proyeccion'):
        ic.simular_ingresos(resultado, simulaciones=200,
                            escala_volatilidad_proyeccion=0.0)


def test_el_resumen_lleva_las_probabilidades_ya_calculadas():
    resultado = preparar()
    probabilidad = ic.simular_ingresos(
        resultado, errores_modelo=ic.cargar_errores_modelo(), simulaciones=1000)

    texto = ic.resumen_para_llm(resultado, probabilidad=probabilidad)

    assert 'PROBABILIDAD DEL INGRESO' in texto
    assert 'Intervalo 90%' in texto
    assert 'Probabilidad de alcanzar el ingreso base calculado' in texto
    assert 'Cumplimiento historico del modelo' in texto
    assert 'Ingreso comprometible por nivel de confianza' in texto
    assert 'De donde sale la incertidumbre' in texto
    # El modelo no debe recalcular nada.
    assert 'no estimar de nuevo' in texto


def _prompt_generado(monkeypatch, **kwargs):
    import ProyAst

    capturado = {}

    def fake_consultar_llm(prompt, max_tokens=None):
        capturado['prompt'] = prompt
        capturado['max_tokens'] = max_tokens
        return 'respuesta'

    monkeypatch.setattr(ProyAst, 'consultar_llm', fake_consultar_llm)
    df_base = pd.DataFrame({
        'Finca': ['BL25', 'BL25'],
        'Variedad': ['FREEDOM', 'MONDIAL'],
        'Produccion': [1000, 500],
    })
    ProyAst.responder_pregunta_anthropic(
        df_base, 'Que variedad conviene priorizar?', 'BL25', **kwargs)
    return capturado


def test_prompt_lleva_los_ingresos_calculados(monkeypatch):
    resumen = ic.resumen_para_llm(preparar())

    capturado = _prompt_generado(monkeypatch, resumen_ingresos=resumen)

    assert 'Total base: 1,020.00' in capturado['prompt']
    assert 'recalcules' in capturado['prompt'].lower()
    # El Q&A debe pedir un limite de salida mayor al de los resumenes cortos.
    assert capturado['max_tokens'] > 512


def test_prompt_avisa_cuando_no_hay_calculo_de_ingresos(monkeypatch):
    capturado = _prompt_generado(monkeypatch)

    assert 'No hay calculo de ingresos' in capturado['prompt']
