from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import requests
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# tu modelo de ML

app = FastAPI()

# Agregar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite peticiones desde cualquier dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ProyAst.py")
async def ProyAst(file: UploadFile):
    df = pd.read_excel(file.file)
    # procesar con tu modelo
    resultados = ProyAst.predict(df)
    return {"estimaciones": resultados}

st.markdown("<h1 style='font-size: 22px; text-align: center;'>ESTIMADOS DE UNIDADES PRODUCTIVAS</h1>",
            unsafe_allow_html=True)


# 1.- SELECCIONAR Y IMPORTAR PATRONES EN BASE A INFORMACION

# file_path = "Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"
file_path = st.file_uploader("Sube archivo Excel", type=["xlsx"])

if file_path is not None:
    df = pd.read_excel(file_path)
    df_info = pd.DataFrame(df)
    print(df.head())
    var_interes = df['Bloque&Varid'].unique()
    # print(var_interes)
    # st.write("Variedades disponibles:")
    # st.dataframe(pd.DataFrame(var_interes, columns=['Bloque&Varid']))
    selected_var = st.selectbox(
        "Selecciona la variedad a proyectar", var_interes)

    # Optional: call the private API deployed on Render.
    api_url_default = st.secrets.get(
        "API_URL", "https://proyecciones-api.onrender.com/predict")
    api_key_default = st.secrets.get("API_KEY", "")

    api_url = st.text_input("URL API Render", value=api_url_default)
    api_key = st.text_input("API Key", value=api_key_default, type="password")

    if st.button("Ejecutar por API"):
        try:
            uploaded_bytes = file_path.getvalue()
            response = requests.post(
                api_url,
                headers={"X-API-Key": api_key},
                files={
                    "file": (
                        file_path.name,
                        uploaded_bytes,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
                data={"selected_var": selected_var},
                timeout=120,
            )

            if response.ok:
                result_json = response.json()
                st.success("Respuesta recibida desde API Render")
                st.json(result_json)

                result = result_json.get("result", {})
                preview = result.get("preview", [])
                if preview:
                    st.dataframe(pd.DataFrame(preview))
                st.stop()

            st.error(f"Error API: {response.status_code}")
            st.code(response.text)

        except Exception as exc:
            st.error(f"No fue posible llamar la API: {exc}")

    df_filtered = df[df['Bloque&Varid'].isin(var_interes)]
    cant_varied = df['Bloque&Varid'].count()
    st.write(f"Cantidad total de registros: {cant_varied}")

    # 2. CALCULAR EL MENOR MSE

    # file_path = "C:\\Users\\Personal\\Downloads\\Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"

    df_1 = pd.read_excel(file_path)
    print(df_1)
    var_proy = selected_var
    print(var_proy)

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
    print(var_proy)
    print(var_patron_list)
    print(var_patron_alt)

    if [var_proy] == var_patron_list:
        combined_varieties = ([var_proy] + var_patron_alt)
    else:
        combined_varieties = ([var_proy] + var_patron_list)

    df_filtered = df_3[df_3['Bloque&Varid'].isin(combined_varieties)]
    print(df_filtered.head())
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
    print(index_1 * m2_1)
    proy = pd.Series(index_1 * m2_1)
    print(proy.tail(6))
    # proy.to_excel(r'C:\\Users\\Personal\\Desktop\\Proyecto.xlsx',index=False, startcol=0)

    # Entrenamiento modelo

    # file_path = "C:\\Users\\Personal\\Downloads\\Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"

    df_1 = pd.read_excel(file_path)
    y = df_1[df_1['Bloque&Varid'].isin([var_proy])]
    y_frame = pd.DataFrame(y['Produccion']).reset_index(drop=True).dropna()
    print(len(y_frame))
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

    y_pred = pd.DataFrame((pred_1), columns=['prediction'])
    print(y_pred.tail(6))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y_pred, label='Modelo', color='blue', linewidth=2)
    ax.plot(y_frame, label='Produccion', color='red', linestyle='--')
    ax.plot(proy, label='Proy_patron', color='green', linestyle='-')
    ax.legend()
    ax.grid(True)
    ax.set_title(f"Proyección: {var_proy}")
    st.pyplot(fig)

    y_pred.to_excel(r'C:\\Users\\Personal\\Desktop\\ProyModelo.xlsx',
                    index=True, startcol=2)
    st.write(y_pred, unsafe_allow_html=False)
    # Reducir tamaño visual del selectbox (alto y fuente)
    st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        min-height: 30px !important;
        font-size: 12px !important;
    }
    div[data-baseweb="select"] span {
        font-size: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    

# CARACTERISTICAS DEL PATRON"

    st.write('CARACTERISTICAS PATRON SELECCIONADO')
    st.write('Temp_Max', df['TMP MAX'].tail(50).mean())

else:
    st.write("Por favor, sube un archivo Excel para continuar.")
