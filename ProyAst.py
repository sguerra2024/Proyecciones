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

logo_path = Path(__file__).with_name("Agromejora.jpg")
if logo_path.exists():
    st.image(str(logo_path), width=220)

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

    st.write('CARACTERISTICAS DE LA BASE PATRON')
    st.write('Temp_Max', df['TMP MAX'].max())
    st.write('Temp_Min', df['TMP MIN'].min())


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

# file_path = "C:\\Users\\Personal\\Downloads\\Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"

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

    modelo.fit(X_train, y_train)

    pred_1 = modelo.predict(x_frame)

    y_pred = pd.DataFrame(pred_1, columns=['Estimado'])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(y_pred['Estimado'].reset_index(drop=True),
            label='Modelo', color='blue', linewidth=2)
    ax.plot(y_frame.reset_index(drop=True),
            label='Produccion', color='red', linestyle='--')
    ax.plot(proy.reset_index(drop=True), label='Proy_patron',
            color='green', linestyle='-')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)
    y_pred.to_excel(r'C:\\Users\\Personal\\Desktop\\Proyecto.xlsx',
                    index=False, startcol=2)
    st.write(y_pred.tail(6).round(0))
else:
    st.info("Por favor, sube un archivo Excel.")
