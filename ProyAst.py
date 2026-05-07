from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from typing import Any
import subprocess
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import streamlit as st

if __name__ == "__main__" and "--via-run" not in sys.argv:
    subprocess.run([sys.executable, "-m", "streamlit",
                   "run", __file__, "--", "--via-run"], check=False)
    raise SystemExit

models_dir = Path(__file__).with_name("modelos")
models_dir.mkdir(exist_ok=True)

logo_path = Path(__file__).with_name("Denmar.jpeg")
if logo_path.exists():
    st.image(str(logo_path), width=170)

readme_path = Path(__file__).with_name("README.md")
if readme_path.exists():
    with st.expander("Ver README"):
        try:
            st.markdown(readme_path.read_text(encoding="utf-8"))
        except Exception:
            st.code(readme_path.read_text(errors="ignore"))

file_path = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
if file_path is not None:
    df = pd.read_excel(file_path)
    # st.write(df.head())
# 1.- SELECCIONAR Y IMPORTAR PATRONES EN BASE A INFORMACION
    fincas = sorted(df["Finca"].dropna().astype(str).unique().tolist())
    selected_finca = st.selectbox("Finca", fincas)

    df_finca = df[df["Finca"].astype(str) == selected_finca].copy()

    variedades = sorted(
        df_finca["Bloque&Varid"].dropna().astype(str).unique().tolist())
    selected_var = st.selectbox("Bloque&Variedad", variedades)

    df = pd.read_excel(file_path)
    df_info = pd.DataFrame(df)
    # st.write(df.head())
    var_interes = df['Bloque&Varid'].unique()
    # st.write(var_interes)
    df_filtered = df[df['Bloque&Varid'].isin(var_interes)]
    pd.pivot_table = df_filtered.pivot_table(values=['Tallos/m2'],
                                             columns=['Bloque', 'Variedad'],
                                             index=['Anio', 'Semana'],
                                             aggfunc='sum')
    # pd.pivot_table.plot(kind='lintch")e')
# CARACTERISTICAS DEL PATRON"

    # st.write('CARACTERISTICAS DE LA BASE PATRON')
    # st.write('Prom Tallos/m2', df['Tallos/m2'].max())


# 2. CALCULAR EL MENOR MSE

# file_path = "C:\\Users\\Personal\\Downloads\\Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"

    df_1 = pd.read_excel(file_path)
    # st.write(df_1)
    var_proy = selected_var
    # st.write(var_proy)

    df_filtered_ = df_1[df_1['Bloque&Varid'].isin([var_proy])]

    print(df_filtered_)
    y = df_filtered_['Produccion']
    pivot_table_ = df_filtered_.pivot_table(values=['Tallos/m2'],
                                            columns=['Bloque&Varid'],
                                            index=['Anio', 'Semana'],
                                            aggfunc='sum')
# pivot_table_.plot(kind='line')
    df_2 = pivot_table_
    arr_2 = np.array(df_2)
    arr_list = arr_2.tolist()

    arr_list = []

    for name, group in df.groupby(['Bloque&Varid']):
        mse = np.mean(abs(group['Tallos/m2'].to_numpy() - arr_2))
        arr_list.append((name, mse))
        arr_list.sort(key=lambda x: x[1])

    print(arr_list)

# 3.- IMPORTAR BASE DE VARIEDADES A PROYECTAR Y COMPARAR CURVA CON PATRON SELECCIONADO\n",

    df_3 = pd.read_excel(file_path)
    var_patron = arr_list[0]
    var_patron_list = list(var_patron[0])
    var_patron_1 = arr_list[1]
    var_patron_alt = list(var_patron_1[0])
    # st.write(var_proy)
    # st.write(var_patron_list)
    # st.write(var_patron_alt)

    if [var_proy] == var_patron_list:
        combined_varieties = ([var_proy] + var_patron_alt)
    else:
        combined_varieties = ([var_proy] + var_patron_list)

    df_filtered = df_3[df_3['Bloque&Varid'].isin(combined_varieties)]
    # st.write(df_filtered.head())
    pivot_table_3 = df_filtered.pivot_table(values=['Tallos/m2'],
                                            columns=['Bloque&Varid'],
                                            index=['Anio', 'Semana'],
                                            aggfunc='sum')
    # pivot_table_3.plot(kind='line')

# 4. CALCULO DE LA PROYECCION CON PATRON SELECCIONADO Y MODEL0\n"

# file_path = "C:\\Users\\Personal\\Downloads\\Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"

    df = pd.read_excel(file_path)
    if [var_proy] == var_patron_list:
        var_patron_list = var_patron_alt

    df_filtered = df[df['Bloque&Varid'].isin(var_patron_list)]
# df_filtered_ = df[df['Bloque&Varid'].isin(var_proy)]
    index = np.array(df_filtered['Tallos/m2'])
    m2 = np.array(df_filtered_.to_records(index=False))[0]
    m2_1 = np.float64(m2[10])
    print(var_proy, m2[10], 'M2')
    index_1 = np.float64(index)
    print(index_1*m2_1)
    proy = pd.Series(index_1*m2_1)
    # st.write(proy.tail(6))


# Entrenamiento modelo


# file_path = "C:\\Users\\Personal\\Downloads\\Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"

    df_1 = pd.read_excel(file_path)
    y_actual = df_1[df_1['Bloque&Varid'].isin([var_proy])].copy()

    ejecutar_entrenamiento = st.button('Entrenar / Reentrenar ahora')
    if ejecutar_entrenamiento:
        st.session_state['entrenado'] = True
    if not st.session_state.get('entrenado', False):
        st.info(
            'Configura la opcion de historico y presiona "Entrenar / Reentrenar ahora".')
        st.stop()

    entrenamiento_df = (
        y_actual[['Anio', 'Semana', 'Tallos/m2', 'Produccion']]
        .dropna()
        .reset_index(drop=True)
    )
    entrenamiento_df = entrenamiento_df.sort_values(
        ['Anio', 'Semana']
    ).reset_index(drop=True)

    # Ajustar el objetivo de entrenamiento con la regla agronomica
    # para que el modelo la aprenda, no solo se corrija al final.
    prod_train = entrenamiento_df['Produccion'].to_numpy(copy=True)
    running_max_train = -np.inf
    ajustes_train = 0
    for i in range(len(prod_train) - 1):
        if prod_train[i] > running_max_train:
            running_max_train = prod_train[i]
            limite_next = prod_train[i] * 0.57
            if prod_train[i + 1] > limite_next:
                prod_train[i + 1] = limite_next
                ajustes_train += 1
        else:
            running_max_train = max(running_max_train, prod_train[i])
    entrenamiento_df['Produccion_ajustada'] = prod_train

    eval_actual_df = (
        y_actual[['Anio', 'Semana', 'Tallos/m2', 'Produccion']]
        .dropna()
        .sort_values(['Anio', 'Semana'])
        .reset_index(drop=True)
    )
    y_frame = pd.DataFrame(eval_actual_df['Produccion']).reset_index(drop=True)

    # Promedio semanal de Tallos/m2 por cada anio para calcular factor de correccion.
    if {'Anio', 'Semana', 'Tallos/m2'}.issubset(y_actual.columns):
        df_promedio = y_actual[['Anio', 'Semana', 'Tallos/m2']].dropna().copy()
        promedio_semanal = (
            df_promedio
            .groupby(['Anio', 'Semana'], as_index=False)['Tallos/m2']
            .mean()
        )
    else:
        df_promedio = pd.DataFrame(columns=['Anio', 'Semana', 'Tallos/m2'])
        promedio_semanal = pd.DataFrame(
            columns=['Anio', 'Semana', 'Tallos/m2'])
    promedio_semanal_anual = (
        promedio_semanal
        .groupby('Anio', as_index=False)['Tallos/m2']
        .mean()
        .rename(columns={'Tallos/m2': 'promedio_semanal_tallos_m2'})
        .sort_values('Anio')
    )

    factor_correccion = 1.0
    if not promedio_semanal_anual.empty:
        anio_objetivo = promedio_semanal_anual['Anio'].max()
        prom_objetivo = float(
            promedio_semanal_anual.loc[
                promedio_semanal_anual['Anio'] == anio_objetivo,
                'promedio_semanal_tallos_m2'
            ].iloc[0]
        )
        historico = promedio_semanal_anual[
            promedio_semanal_anual['Anio'] != anio_objetivo
        ]['promedio_semanal_tallos_m2']
        prom_historico = float(
            historico.mean()) if not historico.empty else prom_objetivo
        if prom_historico != 0:
            factor_correccion = prom_objetivo / prom_historico

    n_train = len(entrenamiento_df)
    if n_train < 5:
        st.error('No hay suficientes datos para entrenar/reentrenar el modelo.')
        st.stop()

    x_train_df = pd.DataFrame(
        entrenamiento_df['Tallos/m2']).reset_index(drop=True)
    x_train_df['Semana_orden'] = np.arange(len(x_train_df), dtype=float)
    y_train_df = pd.DataFrame(
        entrenamiento_df['Produccion_ajustada']).reset_index(drop=True)

    if len(eval_actual_df) == 0:
        st.error(
            'No hay suficientes datos actuales validos para generar la evaluacion.')
        st.stop()

    x_frame = pd.DataFrame(eval_actual_df['Tallos/m2']).reset_index(drop=True)
    x_frame['Semana_orden'] = np.arange(len(x_frame), dtype=float)
    y_frame = pd.DataFrame(eval_actual_df['Produccion']).reset_index(drop=True)

    split_idx = int(len(x_train_df) * 0.8)
    split_idx = min(max(split_idx, 1), len(x_train_df) - 1)
    X_train, X_test = x_train_df.iloc[:split_idx], x_train_df.iloc[split_idx:]
    y_train, y_test = y_train_df.iloc[:split_idx], y_train_df.iloc[split_idx:]

    modelo = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=12,
        min_samples_leaf=2
    )

    modelo.fit(X_train, y_train.values.ravel())

    model_name = ''.join(ch if ch.isalnum() else '_' for ch in str(var_proy))
    model_file = models_dir / f'rf_{model_name}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(modelo, f)

    st.info('Modelo entrenado con base actual y regla agronomica integrada.')
    st.caption(f'Modelo guardado en: {model_file.name}')

    pred_1 = modelo.predict(x_frame)

    y_pred = pd.DataFrame(pred_1, columns=['Estimado_modelo'])
    y_pred['Estimado_modelo'] = y_pred['Estimado_modelo'] * factor_correccion

    # Regla agronomica estricta: despues de cada nuevo maximo semanal,
    # la semana siguiente debe quedar por debajo del 57% de ese pico.
    pred_vals = y_pred['Estimado_modelo'].to_numpy(copy=True)
    ajustes_pico = 0
    running_max = -np.inf
    for i in range(len(pred_vals) - 1):
        if pred_vals[i] > running_max:
            running_max = pred_vals[i]
            limite_siguiente = pred_vals[i] * 0.57
            if pred_vals[i + 1] > limite_siguiente:
                pred_vals[i + 1] = limite_siguiente
                ajustes_pico += 1
        else:
            running_max = max(running_max, pred_vals[i])

    # Segunda pasada de seguridad para evitar picos consecutivos por redondeos.
    for i in range(1, len(pred_vals)):
        max_hasta_prev = pred_vals[:i].max()
        if pred_vals[i - 1] >= max_hasta_prev and pred_vals[i] >= pred_vals[i - 1]:
            pred_vals[i] = pred_vals[i - 1] * 0.57
            ajustes_pico += 1

    # Regla de valores iguales consecutivos: el segundo se ajusta al 57% del primero.
    ajustes_iguales = 0
    for i in range(1, len(pred_vals)):
        if pred_vals[i] == pred_vals[i - 1]:
            pred_vals[i] = pred_vals[i - 1] * 0.57
            ajustes_iguales += 1

    # Ajuste de media: si la media del modelo difiere de la media de produccion,
    # escalar las predicciones para igualarlas.
    prod_real_vals = y_frame.iloc[:len(pred_vals), 0].to_numpy()
    media_real = prod_real_vals.mean()
    media_modelo = pred_vals.mean()
    if media_modelo != 0 and not np.isclose(media_modelo, media_real):
        pred_vals = pred_vals * (media_real / media_modelo)

    y_pred['Estimado_modelo'] = pred_vals

    # Etiquetas eje X: Anio-Semana alineadas con x_frame
    etiquetas_anio_semana = eval_actual_df.apply(
        lambda r: f"{int(r['Anio'])}-{int(r['Semana']):02d}", axis=1
    )
    n_puntos = len(y_pred)
    etiquetas_x = etiquetas_anio_semana.iloc[:n_puntos].tolist()
    x_pos = range(n_puntos)

    # Marcar cambios de año para lineas verticales divisoras
    cambios_anio = []
    anio_prev = None
    for i, lbl in enumerate(etiquetas_x):
        anio_actual = lbl.split('-')[0]
        if anio_prev and anio_actual != anio_prev:
            cambios_anio.append(i)
        anio_prev = anio_actual

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_pos, y_pred['Estimado_modelo'].reset_index(drop=True),
            label='Modelo', color='orange', linewidth=2)
    ax.plot(x_pos, y_frame.reset_index(drop=True),
            label='Produccion', color='red', linestyle='--')
    ax.plot(x_pos, proy.reset_index(drop=True),
            label='Proy_patron', color='green', linestyle='-')

    for c in cambios_anio:
        ax.axvline(x=c, color='gray', linestyle=':', linewidth=1)

    # Mostrar etiqueta cada 4 semanas para no saturar el eje
    paso = max(1, n_puntos // 20)
    ticks_pos = list(range(0, n_puntos, paso))
    ticks_lbl = [etiquetas_x[i] for i in ticks_pos]
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(ticks_lbl, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Año - Semana')
    ax.set_ylabel('Produccion')
    ax.set_title(f'Proyeccion de produccion - {var_proy}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.write('Factor de correccion aplicado', round(factor_correccion, 4))
    st.write('Promedio semanal Tallos/m2 por anio')
    st.dataframe(promedio_semanal_anual, use_container_width=True)

    n_export = min(
        len(etiquetas_anio_semana),
        len(x_frame),
        len(y_frame),
        len(proy),
        len(y_pred)
    )
    df_export = pd.DataFrame({
        'Variedad_proyectada': [var_proy] * n_export,
        'Anio_Semana': etiquetas_anio_semana.iloc[:n_export].values,
        'Tallos_m2_patron': x_frame.iloc[:n_export, 0].values,
        'Produccion_real': y_frame.iloc[:n_export, 0].values,
        'Proy_patron': proy.iloc[:n_export].values,
        'Estimado_modelo': y_pred.iloc[:n_export, 0].values,
        'Factor_correccion': [factor_correccion] * n_export,
    })
    df_export['Error'] = (
        df_export['Produccion_real'] - df_export['Estimado_modelo']
    )
    df_export['Error_abs'] = df_export['Error'].abs()
    df_export['Error_pct'] = np.where(
        df_export['Produccion_real'] != 0,
        (df_export['Error_abs'] / df_export['Produccion_real']) * 100,
        np.nan
    )
    mse_modelo = mean_squared_error(
        df_export['Produccion_real'],
        df_export['Estimado_modelo']
    )
    mse_patron = mean_squared_error(
        df_export['Produccion_real'],
        df_export['Proy_patron']
    )

    ruta_salida = r'C:\\Users\\Personal\\Desktop\\Proyecto.xlsx'
    if st.button('Exportar datos a Excel'):
        with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='Datos_modelo', index=False)
            df_export[
                ['Anio_Semana', 'Produccion_real', 'Estimado_modelo',
                 'Error', 'Error_abs', 'Error_pct']
            ].to_excel(writer, sheet_name='Errores_modelo', index=False)
            promedio_semanal_anual.to_excel(
                writer, sheet_name='Promedio_anual', index=False)
            pd.DataFrame([
                {'metrica': 'MSE_modelo', 'valor': mse_modelo},
                {'metrica': 'MSE_proy_patron', 'valor': mse_patron},
                {'metrica': 'factor_correccion', 'valor': factor_correccion},
                {'metrica': 'MAE_modelo',
                    'valor': df_export['Error_abs'].mean()},
                {'metrica': 'MAPE_modelo_pct',
                    'valor': df_export['Error_pct'].mean()},
                {'metrica': 'variedad_proyectada', 'valor': var_proy}
            ]).to_excel(writer, sheet_name='Resumen', index=False)

        st.success(
            f'Excel exportado con todos los datos del modelo en: {ruta_salida}')
    etiquetas_tail = etiquetas_anio_semana.iloc[:len(y_pred)].tail(4).values
    y_pred_tail = y_pred.tail(4).round(0).copy()
    y_pred_tail.index = etiquetas_tail
    st.write(y_pred_tail)
else:
    st.info("Por favor, sube el archivo Excel.")
