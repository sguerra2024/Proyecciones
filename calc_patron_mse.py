import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import mean_squared_error

file_path = Path(
    'Produccion Astroflores BL25-26-27-28 a la Semana 29_entrenamiento.xlsx')
df = pd.read_excel(file_path)
var_proy = '021LEILA'


def nombre_base_variedad(valor):
    txt = str(valor).strip().upper()
    txt = re.sub(r'^\d+\s*', '', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt


def seleccionar_patron(arr_list, var_proy):
    var_obj = str(var_proy).strip()
    var_obj_norm = var_obj.upper()
    base_obj = nombre_base_variedad(var_obj)
    candidatos = [
        str(item[0][0]).strip()
        for item in arr_list
        if str(item[0][0]).strip().upper() != var_obj_norm
    ]
    if len(candidatos) == 0:
        raise ValueError('No hay suficientes patrones para comparar.')
    for candidato in candidatos:
        if nombre_base_variedad(candidato) == base_obj and candidato.strip().upper() != var_obj_norm:
            return candidato
    for candidato in candidatos:
        if candidato.strip().upper() != var_obj_norm:
            return candidato
    raise ValueError('No hay un patron distinto de la variedad proyectada.')


def calcular_patron_compatible_individual(df_patrones, df_variedad_objetivo, var_proy):
    pivot_table_obj = df_variedad_objetivo.pivot_table(
        values=['Tallos/m2'],
        columns=['Bloque&Varid'],
        index=['Anio', 'Semana'],
        aggfunc='sum'
    )
    arr_2 = np.array(pivot_table_obj)
    arr_list = []
    for name, group in df_patrones.groupby(['Bloque&Varid']):
        try:
            mse = np.mean(abs(group['Tallos/m2'].to_numpy() - arr_2))
            arr_list.append((name, mse))
        except Exception:
            continue
    arr_list.sort(key=lambda x: x[1])
    return seleccionar_patron(arr_list, var_proy)


df_filtered_ = df[df['Bloque&Varid'].isin([var_proy])]
patron_seleccionado = calcular_patron_compatible_individual(
    df, df_filtered_, var_proy)
print('Patron seleccionado:', patron_seleccionado)

# Replicar la proyección del patrón usada por el flujo individual
m2_col = next((col for col in df_filtered_.columns if str(
    col).strip().lower() == 'm2variedad'), None)
if m2_col is None:
    raise RuntimeError('No se encontró la columna m2Variedad')

m2_1 = np.float64(df_filtered_.iloc[0][m2_col])
df_patron = df[df['Bloque&Varid'].isin([patron_seleccionado])]
index = np.array(df_patron['Tallos/m2'])
index_1 = np.float64(index)
proy = pd.Series(index_1 * m2_1)

# Alinear con la variedad objetivo y calcular MSE por año
actual = df_filtered_.copy()
actual = actual.sort_values(['Anio', 'Semana']).reset_index(drop=True)
proy = pd.Series(proy.to_numpy()[:len(actual)], index=actual.index)

for anio in [2025, 2026]:
    mask = actual['Anio'] == anio
    y_true = actual.loc[mask, 'Produccion'].astype(float).to_numpy()
    y_pred = proy.loc[mask].astype(float).to_numpy()
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f'Anio {anio}: sin datos')
        continue
    mse = mean_squared_error(y_true, y_pred)
    print(f'Anio {anio}: MSE patron = {mse:.6f}')
