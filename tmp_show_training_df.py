import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

path = 'Produccion Astroflores BL25-26-27-28 a la Semana 29_entrenamiento.xlsx'
df = pd.read_excel(path)

if 'Bloque&Varid' not in df.columns:
    raise SystemExit('No existe Bloque&Varid en el archivo')


def nombre_base_variedad(valor):
    txt = str(valor).strip().upper()
    txt = re.sub(r'^\d+\s*', '', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt


def seleccionar_patron(arr_list, var_proy):
    var_obj = str(var_proy).strip()
    var_obj_norm = nombre_base_variedad(var_obj)

    candidatos = []
    for item in arr_list:
        if not item:
            continue
        raw_name = item[0]
        if isinstance(raw_name, tuple):
            raw_name = raw_name[0]
        nombre_candidato = nombre_base_variedad(raw_name)
        if nombre_candidato and nombre_candidato != var_obj_norm:
            candidatos.append(str(raw_name).strip())

    if len(candidatos) == 0:
        if arr_list:
            primer_item = arr_list[0]
            if isinstance(primer_item, tuple) and len(primer_item) > 0:
                raw_name = primer_item[0]
                if isinstance(raw_name, tuple):
                    raw_name = raw_name[0]
                return str(raw_name).strip()
        raise ValueError('No hay suficientes patrones para comparar.')

    return candidatos[0]


def calcular_sn_y_mse_equivalente(trabajo):
    if trabajo is None or trabajo.empty:
        return np.nan, np.nan
    if 'Produccion' not in trabajo.columns or 'Produccion_patron' not in trabajo.columns:
        return np.nan, np.nan

    produccion_real = pd.to_numeric(trabajo['Produccion'], errors='coerce')
    produccion_patron = pd.to_numeric(
        trabajo['Produccion_patron'], errors='coerce')
    anio_valores = pd.to_numeric(trabajo.get(
        'Anio', pd.Series(np.nan)), errors='coerce')

    if not anio_valores.notna().any():
        return np.nan, np.nan

    mse_2025 = np.nan
    mse_2026 = np.nan
    for anio_ref in [2025, 2026]:
        mask_anio = anio_valores == anio_ref
        if not mask_anio.any():
            continue
        mse_anio = np.mean(
            np.abs(
                produccion_real[mask_anio].to_numpy() -
                produccion_patron[mask_anio].to_numpy()
            )
        )
        if anio_ref == 2025:
            mse_2025 = mse_anio
        else:
            mse_2026 = mse_anio

    if pd.notna(mse_2025) and pd.notna(mse_2026) and mse_2025 != mse_2026:
        diferencia = mse_2026 - mse_2025
        if diferencia != 0:
            mse_equivalente = float(abs(diferencia))
            sn_valor = float(np.log10((mse_equivalente ** 2)))
            return sn_valor, mse_equivalente

    return np.nan, np.nan


def calcular_sn_patron(trabajo):
    sn_valor, _ = calcular_sn_y_mse_equivalente(trabajo)
    return sn_valor


def usar_produccion_patron_como_real(trabajo):
    sn_valor, mse_equivalente = calcular_sn_y_mse_equivalente(trabajo)
    if pd.notna(sn_valor) and np.isfinite(sn_valor) and sn_valor > 13.0:
        return True
    if pd.notna(mse_equivalente) and np.isfinite(mse_equivalente):
        return bool(mse_equivalente > 10**6.5)
    return False


def construir_patron_semanal(df_patron_base):
    patron_weekly = df_patron_base[[
        'Anio', 'Semana', 'Tallos/m2', 'Produccion'
    ]].dropna(subset=['Anio', 'Semana']).copy()
    patron_weekly['Anio'] = pd.to_numeric(
        patron_weekly['Anio'], errors='coerce')
    patron_weekly['Semana'] = pd.to_numeric(
        patron_weekly['Semana'], errors='coerce')
    patron_weekly['Tallos/m2'] = pd.to_numeric(
        patron_weekly['Tallos/m2'], errors='coerce')
    patron_weekly['Produccion'] = pd.to_numeric(
        patron_weekly['Produccion'], errors='coerce')
    patron_weekly = patron_weekly.dropna(subset=['Anio', 'Semana'])
    patron_weekly['Anio'] = patron_weekly['Anio'].astype(int)
    patron_weekly['Semana'] = patron_weekly['Semana'].astype(int)
    patron_weekly = (
        patron_weekly
        .groupby(['Anio', 'Semana'], as_index=False)
        .agg({'Tallos/m2': 'mean', 'Produccion': 'sum'})
        .rename(columns={'Tallos/m2': 'Tallos_m2_patron', 'Produccion': 'Produccion_patron'})
        .sort_values(['Anio', 'Semana'])
        .reset_index(drop=True)
    )
    patron_weekly['Incremento_tallos_patron'] = patron_weekly['Tallos_m2_patron'].diff(
    ).fillna(0.0)
    patron_weekly['Incremento_produccion_patron'] = patron_weekly['Produccion_patron'].diff(
    ).fillna(0.0)
    return patron_weekly


def excluir_ultimas_4_semanas(df_base, columnas_grupo=None):
    if df_base is None or df_base.empty:
        return df_base.copy()

    trabajo = df_base.copy()
    if 'Anio_Semana' in trabajo.columns:
        anio_semana = trabajo['Anio_Semana'].astype(
            str).str.split('-', n=1, expand=True)
        trabajo['__anio_tmp'] = pd.to_numeric(anio_semana[0], errors='coerce')
        trabajo['__semana_tmp'] = pd.to_numeric(
            anio_semana[1], errors='coerce')
    elif {'Anio', 'Semana'}.issubset(trabajo.columns):
        trabajo['__anio_tmp'] = pd.to_numeric(trabajo['Anio'], errors='coerce')
        trabajo['__semana_tmp'] = pd.to_numeric(
            trabajo['Semana'], errors='coerce')
    else:
        return trabajo

    validos = trabajo[trabajo['__anio_tmp'].notna(
    ) & trabajo['__semana_tmp'].notna()].copy()
    invalidos = trabajo[~(trabajo['__anio_tmp'].notna() &
                          trabajo['__semana_tmp'].notna())].copy()

    if validos.empty:
        return trabajo.drop(columns=['__anio_tmp', '__semana_tmp'], errors='ignore')

    grupos_validos = []
    if columnas_grupo:
        grupos_validos = [
            col for col in columnas_grupo if col in validos.columns]

    if grupos_validos:
        orden_desc = validos.sort_values(
            grupos_validos + ['__anio_tmp', '__semana_tmp'], ascending=[True] * len(grupos_validos) + [False, False])
        orden_desc['__rank_ultimas_tmp'] = orden_desc.groupby(
            grupos_validos).cumcount() + 1
    else:
        orden_desc = validos.sort_values(
            ['__anio_tmp', '__semana_tmp'], ascending=[False, False])
        orden_desc['__rank_ultimas_tmp'] = np.arange(len(orden_desc)) + 1

    validos_filtrados = orden_desc[orden_desc['__rank_ultimas_tmp'] > 4].copy()
    resultado = pd.concat([validos_filtrados, invalidos],
                          ignore_index=True, sort=False)
    resultado = resultado.drop(
        columns=['__anio_tmp', '__semana_tmp', '__rank_ultimas_tmp'], errors='ignore')

    if {'Anio', 'Semana'}.issubset(resultado.columns):
        resultado['__anio_sort'] = pd.to_numeric(
            resultado['Anio'], errors='coerce')
        resultado['__semana_sort'] = pd.to_numeric(
            resultado['Semana'], errors='coerce')
        resultado = resultado.sort_values(['__anio_sort', '__semana_sort'])
        resultado = resultado.drop(
            columns=['__anio_sort', '__semana_sort'], errors='ignore')
    elif 'Anio_Semana' in resultado.columns:
        anio_semana_sort = resultado['Anio_Semana'].astype(
            str).str.split('-', n=1, expand=True)
        resultado['__anio_sort'] = pd.to_numeric(
            anio_semana_sort[0], errors='coerce')
        resultado['__semana_sort'] = pd.to_numeric(
            anio_semana_sort[1], errors='coerce')
        resultado = resultado.sort_values(['__anio_sort', '__semana_sort'])
        resultado = resultado.drop(
            columns=['__anio_sort', '__semana_sort'], errors='ignore')

    return resultado.reset_index(drop=True)


def ajustar_patron_con_extremos_real(trabajo):
    if trabajo is None or trabajo.empty:
        return trabajo
    if 'Produccion' not in trabajo.columns or 'Produccion_patron' not in trabajo.columns:
        return trabajo

    produccion_real = pd.to_numeric(trabajo['Produccion'], errors='coerce')
    produccion_patron = pd.to_numeric(
        trabajo['Produccion_patron'], errors='coerce')
    media_real = float(produccion_real.mean()
                       ) if produccion_real.notna().any() else np.nan
    std_real = float(produccion_real.std(ddof=0)
                     ) if produccion_real.notna().any() else np.nan

    if not np.isfinite(media_real) or not np.isfinite(std_real) or std_real <= 0:
        return trabajo

    sn_valor = calcular_sn_patron(trabajo)
    sn_alto = bool(pd.notna(sn_valor) and sn_valor > 13.0)
    factor_intensidad = 1.25 if sn_alto else 1.0

    z_score = (produccion_real - media_real) / std_real
    mask_positiva_moderada = (z_score >= 1.0) & (z_score < 2.0)
    mask_positiva_alta = z_score >= 2.0
    mask_negativa_moderada = (z_score <= -1.0) & (z_score > -2.0)
    mask_negativa_alta = z_score <= -2.0
    if not any([mask_positiva_moderada.any(), mask_positiva_alta.any(), mask_negativa_moderada.any(), mask_negativa_alta.any()]):
        return trabajo

    patron_ajustado = produccion_patron.copy()
    fuerza_serie = np.zeros(len(z_score), dtype=float)

    if mask_positiva_moderada.any():
        fuerza_serie[mask_positiva_moderada] = np.clip(
            (0.12 + 0.08 * (z_score[mask_positiva_moderada] - 1.0)) * factor_intensidad, 0.12, 0.24 + (0.06 if sn_alto else 0.0))
    if mask_positiva_alta.any():
        fuerza_serie[mask_positiva_alta] = np.clip(
            (0.25 + 0.10 * (z_score[mask_positiva_alta] - 2.0)) * factor_intensidad, 0.25, 0.45 + (0.08 if sn_alto else 0.0))
    if mask_negativa_moderada.any():
        fuerza_serie[mask_negativa_moderada] = np.clip(
            (0.10 + 0.05 * (abs(z_score[mask_negativa_moderada]) - 1.0)) * factor_intensidad, 0.10, 0.22 + (0.06 if sn_alto else 0.0))
    if mask_negativa_alta.any():
        fuerza_serie[mask_negativa_alta] = np.clip(
            (0.22 + 0.10 * (abs(z_score[mask_negativa_alta]) - 2.0)) * factor_intensidad, 0.22, 0.40 + (0.08 if sn_alto else 0.0))

    mascara_ajuste = fuerza_serie > 0
    if mascara_ajuste.any():
        if sn_alto:
            patron_ajustado[mascara_ajuste] = produccion_real[mascara_ajuste]
        else:
            patron_ajustado[mascara_ajuste] = (
                (1.0 - fuerza_serie[mascara_ajuste]) * patron_ajustado[mascara_ajuste] + fuerza_serie[mascara_ajuste] * produccion_real[mascara_ajuste])

    factor_descenso = 0.12 + (0.06 if sn_alto else 0.0)
    for idx in np.where(mask_positiva_alta)[0]:
        for offset in [10, 11, 12]:
            future_idx = idx + offset
            if future_idx < len(patron_ajustado):
                patron_ajustado[future_idx] = min(
                    patron_ajustado[future_idx], patron_ajustado[idx] * (1.0 - factor_descenso))

    for idx in np.where(mask_positiva_moderada)[0]:
        for offset in [10, 11, 12]:
            future_idx = idx + offset
            if future_idx < len(patron_ajustado):
                patron_ajustado[future_idx] = min(
                    patron_ajustado[future_idx], patron_ajustado[idx] * (1.0 - (0.08 + (0.04 if sn_alto else 0.0))))

    trabajo['Produccion_patron'] = patron_ajustado
    return trabajo


def preparar_dataset_modelo(df_variedad_base, patron_weekly, patron_feature_weight):
    trabajo = df_variedad_base[[
        'Anio', 'Semana', 'Tallos/m2', 'Produccion']].dropna().reset_index(drop=True)
    trabajo['Anio'] = pd.to_numeric(trabajo['Anio'], errors='coerce')
    trabajo['Semana'] = pd.to_numeric(trabajo['Semana'], errors='coerce')
    trabajo['Tallos/m2'] = pd.to_numeric(trabajo['Tallos/m2'], errors='coerce')
    trabajo['Produccion'] = pd.to_numeric(
        trabajo['Produccion'], errors='coerce')
    trabajo = trabajo.dropna(
        subset=['Anio', 'Semana', 'Tallos/m2', 'Produccion'])
    trabajo['Anio'] = trabajo['Anio'].astype(int)
    trabajo['Semana'] = trabajo['Semana'].astype(int)
    trabajo = trabajo.sort_values(['Anio', 'Semana']).reset_index(drop=True)
    trabajo = trabajo.merge(patron_weekly, on=['Anio', 'Semana'], how='left')
    trabajo['Tallos_m2_patron'] = trabajo['Tallos_m2_patron'].fillna(
        trabajo['Tallos/m2'])
    trabajo['Produccion_patron'] = trabajo['Produccion_patron'].fillna(
        trabajo['Produccion'])
    if usar_produccion_patron_como_real(trabajo):
        trabajo['Produccion'] = trabajo['Produccion_patron'].copy()
    trabajo = ajustar_patron_con_extremos_real(trabajo)
    trabajo['Incremento_tallos_patron'] = trabajo['Incremento_tallos_patron'].fillna(
        0.0)
    trabajo['Incremento_produccion_patron'] = trabajo['Incremento_produccion_patron'].fillna(
        0.0)
    peso_patron = 0.4
    if np.isfinite(patron_feature_weight):
        peso_patron = float(np.clip(patron_feature_weight, 0.0, 1.0))
    if peso_patron <= 0:
        peso_patron = 0.0
    trabajo['Tallos_m2_patron_ponderado'] = (
        (1.0 - peso_patron) * trabajo['Tallos/m2'] + peso_patron * trabajo['Tallos_m2_patron'])
    trabajo['Produccion_patron_ponderado'] = (
        (1.0 - peso_patron) * trabajo['Produccion'] + peso_patron * trabajo['Produccion_patron'])
    trabajo['Produccion_lag10'] = trabajo['Produccion'].shift(
        10).fillna(trabajo['Produccion'].median())
    trabajo['Produccion_lag11'] = trabajo['Produccion'].shift(
        11).fillna(trabajo['Produccion'].median())
    trabajo['Produccion_lag12'] = trabajo['Produccion'].shift(
        12).fillna(trabajo['Produccion'].median())
    trabajo['Produccion_lag13'] = trabajo['Produccion'].shift(
        13).fillna(trabajo['Produccion'].median())
    trabajo['Cambio_produccion_vs_lag10'] = (
        trabajo['Produccion'] - trabajo['Produccion_lag10']).fillna(0.0)
    trabajo['Cambio_produccion_ultimas_3'] = trabajo['Produccion'].diff(
        3).fillna(0.0)
    trabajo['Cambio_relativo_vs_lag10'] = (
        (trabajo['Produccion'] - trabajo['Produccion_lag10']) / trabajo['Produccion_lag10'].replace(0, np.nan)).fillna(0.0)
    trabajo['Pendiente_ultimas_3'] = (
        trabajo['Produccion'].diff(3).fillna(0.0) / 3.0)
    trabajo['Promedio_picos_12_13'] = (
        (trabajo['Produccion_lag12'] + trabajo['Produccion_lag13']) / 2.0)
    trabajo['Relacion_valle_vs_pico_12_13'] = (
        trabajo['Produccion'] / trabajo['Promedio_picos_12_13'].replace(0, np.nan)).fillna(1.0)
    trabajo['Cambio_vs_promedio_picos_12_13'] = (
        trabajo['Produccion'] - trabajo['Promedio_picos_12_13']).fillna(0.0)
    trabajo['Semana_ciclo_12'] = ((trabajo['Semana'] - 1) % 20) + 1
    return trabajo


def calcular_patron_compatible_individual(df_patrones, df_variedad_objetivo, var_proy):
    pivot_table_obj = df_variedad_objetivo.pivot_table(
        values=['Tallos/m2'], columns=['Bloque&Varid'], index=['Anio', 'Semana'], aggfunc='sum')
    arr_2 = np.array(pivot_table_obj)
    arr_list = []
    for name, group in df_patrones.groupby(['Bloque&Varid']):
        try:
            mse = np.mean(abs(group['Tallos/m2'].to_numpy() - arr_2))
            patron_weekly = construir_patron_semanal(group)
            trabajo = df_variedad_objetivo[[
                'Anio', 'Semana', 'Tallos/m2', 'Produccion']].dropna().reset_index(drop=True)
            trabajo['Anio'] = pd.to_numeric(trabajo['Anio'], errors='coerce')
            trabajo['Semana'] = pd.to_numeric(
                trabajo['Semana'], errors='coerce')
            trabajo['Tallos/m2'] = pd.to_numeric(
                trabajo['Tallos/m2'], errors='coerce')
            trabajo['Produccion'] = pd.to_numeric(
                trabajo['Produccion'], errors='coerce')
            trabajo = trabajo.dropna(
                subset=['Anio', 'Semana', 'Tallos/m2', 'Produccion'])
            trabajo['Anio'] = trabajo['Anio'].astype(int)
            trabajo['Semana'] = trabajo['Semana'].astype(int)
            trabajo = trabajo.sort_values(
                ['Anio', 'Semana']).reset_index(drop=True)
            trabajo = trabajo.merge(
                patron_weekly, on=['Anio', 'Semana'], how='left')
            trabajo['Tallos_m2_patron'] = trabajo['Tallos_m2_patron'].fillna(
                trabajo['Tallos/m2'])
            trabajo['Produccion_patron'] = trabajo['Produccion_patron'].fillna(
                trabajo['Produccion'])
            sn_valor, mse_equivalente = calcular_sn_y_mse_equivalente(trabajo)
            arr_list.append((name, mse, sn_valor, mse_equivalente))
        except Exception:
            continue

    if len(arr_list) < 2:
        raise ValueError('No hay suficientes patrones para comparar.')

    arr_list.sort(key=lambda x: x[1])
    patron_seleccionado = seleccionar_patron(arr_list, var_proy)
    sn_seleccionado = next((item[2] for item in arr_list if str(
        item[0]).strip().upper() == patron_seleccionado.upper()), np.nan)
    mse_equivalente_seleccionado = next((item[3] for item in arr_list if str(
        item[0]).strip().upper() == patron_seleccionado.upper()), np.nan)
    usar_patron = not ((pd.notna(sn_seleccionado) and np.isfinite(sn_seleccionado) and sn_seleccionado > 13.0) or (pd.notna(
        mse_equivalente_seleccionado) and np.isfinite(mse_equivalente_seleccionado) and mse_equivalente_seleccionado > 10**6.5))
    return patron_seleccionado, usar_patron


var_proy = '021LEILA'
df_filtered_ = df[df['Bloque&Varid'].isin([var_proy])].copy()
patron_seleccionado, usar_patron_sin_dependencia = calcular_patron_compatible_individual(
    df, df_filtered_, var_proy)
df_patron = df[df['Bloque&Varid'].isin([patron_seleccionado])]
patron_weekly = construir_patron_semanal(df_patron)
entrenamiento_df = preparar_dataset_modelo(
    df_filtered_, patron_weekly, 0.0 if usar_patron_sin_dependencia else 1.5)
entrenamiento_df = entrenamiento_df[entrenamiento_df['Anio'] >= 2025].reset_index(
    drop=True)
entrenamiento_df = excluir_ultimas_4_semanas(entrenamiento_df)

print('patron_seleccionado=', patron_seleccionado)
print('usar_patron_sin_dependencia=', usar_patron_sin_dependencia)
print('shape=', entrenamiento_df.shape)
print(entrenamiento_df.head(30).to_string(index=False))
