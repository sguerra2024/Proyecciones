from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import streamlit as st

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
    y = df_1[df_1['Bloque&Varid'].isin([var_proy])]
    y_frame = pd.DataFrame(y['Produccion']).reset_index(drop=True).dropna()

    # Promedio semanal de Tallos/m2 por cada anio para calcular factor de correccion.
    df_promedio = y[['Anio', 'Semana', 'Tallos/m2']].dropna().copy()
    promedio_semanal = (
        df_promedio
        .groupby(['Anio', 'Semana'], as_index=False)['Tallos/m2']
        .mean()
    )
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

    y_len = len(y_frame)
    x = pd.DataFrame(index_1[0:]).dropna()
    x_frame = pd.DataFrame(x)
    x_len = (len(x_frame))
    print(x_frame)
    y_final = y_frame.iloc[:x_len]
    print(y_final)
    X_train, X_test, y_train, y_test = train_test_split(
        x_frame, y_frame, test_size=0.2, random_state=42)

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)

    modelo.fit(X_train, y_train.values.ravel())

    pred_1 = modelo.predict(x_frame)

    y_pred = pd.DataFrame(pred_1, columns=['Estimado_modelo'])
    y_pred['Estimado_modelo'] = y_pred['Estimado_modelo'] * factor_correccion

    # Etiquetas eje X: Anio-Semana alineadas con x_frame
    etiquetas_anio_semana = (
        df_filtered_[['Anio', 'Semana']]
        .dropna()
        .reset_index(drop=True)
        .apply(lambda r: f"{int(r['Anio'])}-{int(r['Semana']):02d}", axis=1)
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

    y_pred.to_excel(r'C:\\Users\\Personal\\Desktop\\Proyecto.xlsx',
                    index=False, startcol=2)
    etiquetas_tail = etiquetas_anio_semana.iloc[:len(y_pred)].tail(4).values
    y_pred_tail = y_pred.tail(4).round(0).copy()
    y_pred_tail.index = etiquetas_tail
    st.write(y_pred_tail)
else:
    st.info("Por favor, sube el archivo Excel.")
