from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from fastapi import FastAPI, UploadFile

# tu modelo de ML

app = FastAPI()


@app.post("/ProyAst.py")
async def ProyAst(file: UploadFile):
    df = pd.read_excel(file.file)
    # procesar con tu modelo
    resultados = ProyAst.predict(df)
    return {"estimaciones": resultados}

st.title("ESTIMADOS DE UNIDADES PRODUCTIVAS")
st.write("1.- Sube tu archivo Excel ")
st.write("2.- Selleciona la variedad")

# 1.- SELECCIONAR Y IMPORTAR PATRONES EN BASE A INFORMACION

file_path = "Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"
# file_path = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
df = pd.read_excel(file_path)
df_info = pd.DataFrame(df)
print(df.head())
var_interes = df['Bloque&Varid'].unique()
print(var_interes)
df_filtered = df[df['Bloque&Varid'].isin(var_interes)]
pd.pivot_table = df_filtered.pivot_table(values=['Tallos/m2'],
                                         columns=['Bloque', 'Variedad'],
                                         index=['Anio', 'Semana'],
                                         aggfunc='sum')
# pd.pivot_table.plot(kind='line')
# CARACTERISTICAS DEL PATRON"

print('CARACTERISTICAS DE LA BASE PATRON')
print('Temp_Max', df['TMP MAX'].max())
print('Temp_Min', df['TMP MIN'].min())
print('Anio', df['Anio'].unique())
print('Sem_inicio', df['Semana'].min())
print('Sem_final', df['Semana'].max())

# 2. CALCULAR EL MENOR MSE

# file_path = "C:\\Users\\Personal\\Downloads\\Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"

df_1 = pd.read_excel(file_path)
print(df_1)
var_proy = var_interes[0]
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

y_pred.to_excel(r'C:\\Users\\Personal\\Desktop\\Proyecto.xlsx',
                index=False, startcol=2)
