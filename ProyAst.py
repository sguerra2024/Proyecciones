from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import streamlit as st


models_dir = Path(__file__).with_name("modelos")
models_dir.mkdir(exist_ok=True)

logo_path = Path(__file__).with_name("Denmar.jpeg")
if logo_path.exists():
    st.image(str(logo_path), width=100)

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
    st.markdown(
        "<p style='color:#F28C28;font-weight:700;margin:0.25rem 0 0.25rem 0;'>"
        "PARA PROYECTAR FINCA:"
        "</p>",
        unsafe_allow_html=True
    )
    run_masiva = st.button('PROYECTAR FINCA')

    df_finca = df[df["Finca"].astype(str) == selected_finca].copy()

    variedades = sorted(
        df_finca["Bloque&Varid"].dropna().astype(str).unique().tolist())
    selected_var = st.selectbox("Bloque&Variedad", variedades)

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

        # Regla para ambos flujos: priorizar misma familia cuando exista.
        for candidato in candidatos:
            if nombre_base_variedad(candidato) == base_obj and candidato.strip().upper() != var_obj_norm:
                return candidato

        # Respaldo defensivo para evitar auto-seleccion por cualquier inconsistencia.
        for candidato in candidatos:
            if candidato.strip().upper() != var_obj_norm:
                return candidato

        raise ValueError(
            'No hay un patron distinto de la variedad proyectada.')

    def proyectar_variedad_masiva(df_base, var_proy):
        df_filtered_ = df_base[df_base['Bloque&Varid'].isin([var_proy])].copy()
        if df_filtered_.empty:
            raise ValueError('Sin datos para la variedad seleccionada.')

        pivot_table_ = df_filtered_.pivot_table(values=['Tallos/m2'],
                                                columns=['Bloque&Varid'],
                                                index=['Anio', 'Semana'],
                                                aggfunc='sum')
        arr_2 = np.array(pivot_table_).reshape(-1)

        arr_list = []
        for name, group in df_base.groupby(['Bloque&Varid']):
            vals = group['Tallos/m2'].to_numpy()
            n_cmp = min(len(vals), len(arr_2))
            if n_cmp == 0:
                continue
            mse = np.mean(abs(vals[:n_cmp] - arr_2[:n_cmp]))
            arr_list.append((name, mse))

        if len(arr_list) < 2:
            raise ValueError('No hay suficientes patrones para comparar.')

        arr_list.sort(key=lambda x: x[1])
        patron_seleccionado = seleccionar_patron(arr_list, var_proy)

        df_patron = df_base[df_base['Bloque&Varid'].isin(
            [patron_seleccionado])]
        if df_patron.empty:
            raise ValueError('No hay datos del patron seleccionado.')

        m2_col = next(
            (col for col in df_filtered_.columns if str(
                col).strip().lower() == 'm2variedad'),
            None
        )
        if m2_col is None:
            raise ValueError(
                'No se encontro la columna m2Variedad en la base de datos.')

        m2_1 = np.float64(df_filtered_.iloc[0][m2_col])
        proy = pd.Series(np.float64(np.array(df_patron['Tallos/m2'])) * m2_1)

        y_actual = df_base[df_base['Bloque&Varid'].isin([var_proy])].copy()
        patron_actual = df_base[df_base['Bloque&Varid'].isin(
            [patron_seleccionado])].copy()

        patron_weekly = patron_actual[[
            'Anio', 'Semana', 'Tallos/m2']].dropna().copy()
        patron_weekly['Anio'] = pd.to_numeric(
            patron_weekly['Anio'], errors='coerce')
        patron_weekly['Semana'] = pd.to_numeric(
            patron_weekly['Semana'], errors='coerce')
        patron_weekly = patron_weekly.dropna(subset=['Anio', 'Semana'])
        patron_weekly['Anio'] = patron_weekly['Anio'].astype(int)
        patron_weekly['Semana'] = patron_weekly['Semana'].astype(int)
        patron_weekly = (
            patron_weekly
            .groupby(['Anio', 'Semana'], as_index=False)['Tallos/m2']
            .mean()
            .rename(columns={'Tallos/m2': 'Tallos_m2_patron'})
        )

        patron_feature_weight = 5
        patron_prediction_weight = 0.45
        peak_decay_train = 0.75
        peak_decay_pred = 0.75

        entrenamiento_df = (
            y_actual[['Anio', 'Semana', 'Tallos/m2', 'Produccion']]
            .dropna()
            .reset_index(drop=True)
        )
        entrenamiento_df['Anio'] = pd.to_numeric(
            entrenamiento_df['Anio'], errors='coerce')
        entrenamiento_df['Semana'] = pd.to_numeric(
            entrenamiento_df['Semana'], errors='coerce')
        entrenamiento_df = entrenamiento_df.dropna(subset=['Anio', 'Semana'])
        entrenamiento_df['Anio'] = entrenamiento_df['Anio'].astype(int)
        entrenamiento_df['Semana'] = entrenamiento_df['Semana'].astype(int)
        entrenamiento_df = entrenamiento_df.sort_values(
            ['Anio', 'Semana']).reset_index(drop=True)
        entrenamiento_df = entrenamiento_df[
            (entrenamiento_df['Anio'] > 2025) |
            ((entrenamiento_df['Anio'] == 2025)
             & (entrenamiento_df['Semana'] >= 1))
        ].reset_index(drop=True)
        entrenamiento_df = entrenamiento_df.merge(
            patron_weekly,
            on=['Anio', 'Semana'],
            how='left'
        )
        entrenamiento_df['Tallos_m2_patron'] = entrenamiento_df[
            'Tallos_m2_patron'
        ].fillna(entrenamiento_df['Tallos/m2'])
        entrenamiento_df['Tallos_m2_patron_ponderado'] = (
            entrenamiento_df['Tallos_m2_patron'] * patron_feature_weight
        )

        prod_train = entrenamiento_df['Produccion'].to_numpy(copy=True)
        running_max_train = -np.inf
        for i in range(len(prod_train) - 1):
            if prod_train[i] > running_max_train:
                running_max_train = prod_train[i]
                limite_next = prod_train[i] * peak_decay_train
                if prod_train[i + 1] > limite_next:
                    prod_train[i + 1] = limite_next
            else:
                running_max_train = max(running_max_train, prod_train[i])
        entrenamiento_df['Produccion_ajustada'] = prod_train

        eval_actual_df = (
            y_actual[['Anio', 'Semana', 'Tallos/m2', 'Produccion']]
            .dropna()
            .sort_values(['Anio', 'Semana'])
            .reset_index(drop=True)
        )
        eval_actual_df = eval_actual_df.merge(
            patron_weekly,
            on=['Anio', 'Semana'],
            how='left'
        )
        eval_actual_df['Tallos_m2_patron'] = eval_actual_df[
            'Tallos_m2_patron'
        ].fillna(eval_actual_df['Tallos/m2'])
        eval_actual_df['Tallos_m2_patron_ponderado'] = (
            eval_actual_df['Tallos_m2_patron'] * patron_feature_weight
        )

        if len(entrenamiento_df) < 5 or len(eval_actual_df) == 0:
            raise ValueError(
                'No hay suficientes datos para entrenar/prediccion.')

        x_train_df = entrenamiento_df[
            ['Tallos/m2', 'Tallos_m2_patron_ponderado']
        ].reset_index(drop=True)
        x_train_df['Semana_orden'] = np.arange(len(x_train_df), dtype=float)
        y_train_df = pd.DataFrame(
            entrenamiento_df['Produccion_ajustada']).reset_index(drop=True)

        x_frame = eval_actual_df[
            ['Tallos/m2', 'Tallos_m2_patron_ponderado']
        ].reset_index(drop=True)
        x_frame['Semana_orden'] = np.arange(len(x_frame), dtype=float)
        y_frame = pd.DataFrame(
            eval_actual_df['Produccion']).reset_index(drop=True)

        split_idx = int(len(x_train_df) * 0.8)
        split_idx = min(max(split_idx, 1), len(x_train_df) - 1)

        model_name = ''.join(
            ch if ch.isalnum() else '_' for ch in str(var_proy))
        train_key = f'entrenado_masivo_{model_name}_cal_v2'

        if train_key not in st.session_state:
            modelo = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=12,
                min_samples_leaf=2
            )
            modelo.fit(x_train_df.iloc[:split_idx],
                       y_train_df.iloc[:split_idx].values.ravel())
            st.session_state[train_key] = modelo
        else:
            modelo = st.session_state[train_key]

        y_pred = pd.DataFrame(modelo.predict(x_frame),
                              columns=['Estimado_modelo'])
        pred_vals = y_pred['Estimado_modelo'].to_numpy(copy=True)
        proy_vals = proy.reset_index(drop=True).to_numpy(copy=True)
        n_blend = min(len(pred_vals), len(proy_vals))
        if n_blend > 0:
            pred_vals[:n_blend] = (
                (1 - patron_prediction_weight) * pred_vals[:n_blend]
                + patron_prediction_weight * proy_vals[:n_blend]
            )

        running_max = -np.inf
        for i in range(len(pred_vals) - 1):
            if pred_vals[i] > running_max:
                running_max = pred_vals[i]
                limite_siguiente = pred_vals[i] * peak_decay_pred
                if pred_vals[i + 1] > limite_siguiente:
                    pred_vals[i + 1] = limite_siguiente
            else:
                running_max = max(running_max, pred_vals[i])

        for i in range(1, len(pred_vals)):
            max_hasta_prev = pred_vals[:i].max()
            if pred_vals[i - 1] >= max_hasta_prev and pred_vals[i] >= pred_vals[i - 1]:
                pred_vals[i] = pred_vals[i - 1] * peak_decay_pred

        prod_real_vals = y_frame.iloc[:len(pred_vals), 0].to_numpy()
        media_real = prod_real_vals.mean()
        media_modelo = pred_vals.mean()
        if media_modelo != 0 and not np.isclose(media_modelo, media_real):
            pred_vals = pred_vals * (media_real / media_modelo)

        y_pred['Estimado_modelo'] = pred_vals

        etiquetas_anio_semana = eval_actual_df.apply(
            lambda r: f"{int(r['Anio'])}-{int(r['Semana']):02d}", axis=1
        )
        n_export = min(
            len(etiquetas_anio_semana),
            len(y_frame),
            len(proy),
            len(y_pred)
        )
        df_export = pd.DataFrame({
            'Variedad_proyectada': [var_proy] * n_export,
            'Anio_Semana': etiquetas_anio_semana.iloc[:n_export].values,
            'Produccion_real': y_frame.iloc[:n_export, 0].values,
            'Proy_patron': proy.iloc[:n_export].values,
            'Estimado_modelo': y_pred.iloc[:n_export, 0].values,
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

        return {
            'df_export': df_export,
            'mse_modelo': mean_squared_error(df_export['Produccion_real'], df_export['Estimado_modelo']),
            'mse_patron': mean_squared_error(df_export['Produccion_real'], df_export['Proy_patron'])
        }

    if run_masiva:
        # La corrida masiva se limita a la finca seleccionada en pantalla.
        df_masivo = df_finca.copy()
        # Respetar el orden original de aparicion en la base.
        variedades_todas = (
            df_masivo['Bloque&Varid']
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if len(variedades_todas) == 0:
            st.warning(
                'No hay Bloque&Varid disponibles en la finca seleccionada.')
            st.stop()

        resultados_export = []
        resumen = []
        errores = []
        progreso = st.progress(0)

        for i, var_item in enumerate(variedades_todas, start=1):
            try:
                resultado = proyectar_variedad_masiva(df_masivo, var_item)
                resultados_export.append(resultado['df_export'])
                resumen.append({
                    'Variedad_proyectada': var_item,
                    'MSE_modelo': resultado['mse_modelo'],
                    'MSE_proy_patron': resultado['mse_patron']
                })
            except Exception as e:
                errores.append({
                    'Variedad_proyectada': var_item,
                    'Error': str(e)
                })
            progreso.progress(i / len(variedades_todas))

        if resultados_export:
            df_export_todo = pd.concat(resultados_export, ignore_index=True)
        else:
            df_export_todo = pd.DataFrame(
                columns=['Variedad_proyectada', 'Anio_Semana', 'Estimado_modelo'])

        # Reordenar resultados al orden original del archivo cargado.
        df_original_order = df_masivo.copy().reset_index(drop=True)
        df_original_order['_orden_original'] = np.arange(
            len(df_original_order))
        # Evitar choque de nombres al hacer merge si la base ya trae esta columna.
        if 'Estimado_modelo' in df_original_order.columns:
            df_original_order = df_original_order.drop(
                columns=['Estimado_modelo'])
        df_original_order['Variedad_proyectada'] = df_original_order['Bloque&Varid'].astype(
            str)
        df_original_order['Anio_Semana'] = df_original_order.apply(
            lambda r: f"{int(r['Anio'])}-{int(r['Semana']):02d}", axis=1
        )

        if 'Estimado_modelo' not in df_export_todo.columns:
            df_export_todo['Estimado_modelo'] = np.nan

        df_estimado_ordenado = df_original_order.merge(
            df_export_todo[['Variedad_proyectada',
                            'Anio_Semana', 'Estimado_modelo']],
            on=['Variedad_proyectada', 'Anio_Semana'],
            how='left'
        ).sort_values('_orden_original')

        # Si no se pudo proyectar una variedad/semana, exportar 0.
        df_estimado_ordenado['Estimado_modelo'] = df_estimado_ordenado[
            'Estimado_modelo'
        ].fillna(0)
        df_estimado_ordenado['Estimado_modelo'] = np.rint(
            df_estimado_ordenado['Estimado_modelo']
        ).astype(np.int64)
        df_export_estimado = df_estimado_ordenado[[
            'Estimado_modelo']].reset_index(drop=True)

        if len(errores) == 0:
            st.success('Proyeccion masiva completada.')
        elif len(errores) == len(variedades_todas):
            st.error(
                'No se pudo proyectar ninguna variedad; '
                'se exportara 0 en Estimado_modelo para todos los registros.'
            )
        else:
            st.warning(
                'Algunas variedades no se pudieron proyectar; '
                'se exportara 0 en Estimado_modelo para esos casos.'
            )
            detalle_fallos = '\n'.join(
                f"- {item['Variedad_proyectada']}: {item['Error']}" for item in errores
            )
            if detalle_fallos:
                st.info(f'Motivos de no proyeccion:\n{detalle_fallos}')

        buffer_masivo = io.BytesIO()
        with pd.ExcelWriter(buffer_masivo, engine='openpyxl') as writer:
            # Exportar unicamente la columna Estimado_modelo.
            df_export_estimado.to_excel(
                writer, sheet_name='Estimado_modelo', index=False)
        st.download_button(
            'Exportar datos a Excel',
            data=buffer_masivo.getvalue(),
            file_name='Proyecto_todas_variedades.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key='descargar_masivo'
        )

        st.stop()

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
    patron_seleccionado = seleccionar_patron(arr_list, var_proy)

    combined_varieties = [var_proy, patron_seleccionado]

    df_filtered = df_3[df_3['Bloque&Varid'].isin(combined_varieties)]
    # st.write(df_filtered.head())
    pivot_table_3 = df_filtered.pivot_table(values=['Tallos/m2'],
                                            columns=['Bloque&Varid'],
                                            index=['Anio', 'Semana'],
                                            aggfunc='sum')
    # st.write('pivot_table_3')
    # st.dataframe(pivot_table_3, width='stretch')
    # pivot_table_3.plot(kind='line')

# 4. CALCULO DE LA PROYECCION CON PATRON SELECCIONADO Y MODEL0\n"

# file_path = "C:\\Users\\Personal\\Downloads\\Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"

    df = pd.read_excel(file_path)
    df_filtered = df[df['Bloque&Varid'].isin([patron_seleccionado])]
# df_filtered_ = df[df['Bloque&Varid'].isin(var_proy)]
    index = np.array(df_filtered['Tallos/m2'])
    m2 = df_filtered_.iloc[0]
    m2_col = next(
        (col for col in df_filtered_.columns if str(
            col).strip().lower() == 'm2variedad'),
        None
    )
    if m2_col is None:
        st.error('No se encontro la columna m2Variedad en la base de datos.')
        st.stop()
    m2_1 = np.float64(m2[m2_col])
    print(var_proy, m2[m2_col], 'M2')
    index_1 = np.float64(index)
    print(index_1*m2_1)
    proy = pd.Series(index_1*m2_1)
    # st.write(proy.tail(6))


# Entrenamiento modelo


# file_path = "C:\\Users\\Personal\\Downloads\\Produccion Astroflores BL25-26-27-28 a la Semana 09.xlsx"

    df_1 = pd.read_excel(file_path)
    y_actual = df_1[df_1['Bloque&Varid'].isin([var_proy])].copy()
    patron_actual = df_1[df_1['Bloque&Varid'].isin(
        [patron_seleccionado])].copy()

    if patron_actual.empty:
        st.error('No hay datos del patron seleccionado para entrenar/prediccion.')
        st.stop()

    patron_weekly = patron_actual[[
        'Anio', 'Semana', 'Tallos/m2']].dropna().copy()
    patron_weekly['Anio'] = pd.to_numeric(
        patron_weekly['Anio'], errors='coerce')
    patron_weekly['Semana'] = pd.to_numeric(
        patron_weekly['Semana'], errors='coerce')
    patron_weekly = patron_weekly.dropna(subset=['Anio', 'Semana'])
    patron_weekly['Anio'] = patron_weekly['Anio'].astype(int)
    patron_weekly['Semana'] = patron_weekly['Semana'].astype(int)
    patron_weekly = (
        patron_weekly
        .groupby(['Anio', 'Semana'], as_index=False)['Tallos/m2']
        .mean()
        .rename(columns={'Tallos/m2': 'Tallos_m2_patron'})
    )

    # Pesos para priorizar el patron en entrenamiento y prediccion.
    patron_feature_weight = 5
    patron_prediction_weight = 0.45
    peak_decay_train = 0.75
    peak_decay_pred = 0.75

    entrenamiento_df = (
        y_actual[['Anio', 'Semana', 'Tallos/m2', 'Produccion']]
        .dropna()
        .reset_index(drop=True)
    )
    entrenamiento_df['Anio'] = pd.to_numeric(
        entrenamiento_df['Anio'], errors='coerce')
    entrenamiento_df['Semana'] = pd.to_numeric(
        entrenamiento_df['Semana'], errors='coerce')
    entrenamiento_df = entrenamiento_df.dropna(subset=['Anio', 'Semana'])
    entrenamiento_df['Anio'] = entrenamiento_df['Anio'].astype(int)
    entrenamiento_df['Semana'] = entrenamiento_df['Semana'].astype(int)
    entrenamiento_df = entrenamiento_df.sort_values(
        ['Anio', 'Semana']
    ).reset_index(drop=True)
    entrenamiento_df = entrenamiento_df[
        (entrenamiento_df['Anio'] > 2025) |
        ((entrenamiento_df['Anio'] == 2025)
         & (entrenamiento_df['Semana'] >= 1))
    ].reset_index(drop=True)
    entrenamiento_df = entrenamiento_df.merge(
        patron_weekly,
        on=['Anio', 'Semana'],
        how='left'
    )
    entrenamiento_df['Tallos_m2_patron'] = entrenamiento_df[
        'Tallos_m2_patron'
    ].fillna(entrenamiento_df['Tallos/m2'])
    entrenamiento_df['Tallos_m2_patron_ponderado'] = (
        entrenamiento_df['Tallos_m2_patron'] * patron_feature_weight
    )

    # Ajustar el objetivo de entrenamiento con la regla agronomica
    # para que el modelo la aprenda, no solo se corrija al final.
    prod_train = entrenamiento_df['Produccion'].to_numpy(copy=True)
    running_max_train = -np.inf
    ajustes_train = 0
    for i in range(len(prod_train) - 1):
        if prod_train[i] > running_max_train:
            running_max_train = prod_train[i]
            limite_next = prod_train[i] * peak_decay_train
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
    eval_actual_df = eval_actual_df.merge(
        patron_weekly,
        on=['Anio', 'Semana'],
        how='left'
    )
    eval_actual_df['Tallos_m2_patron'] = eval_actual_df[
        'Tallos_m2_patron'
    ].fillna(eval_actual_df['Tallos/m2'])
    eval_actual_df['Tallos_m2_patron_ponderado'] = (
        eval_actual_df['Tallos_m2_patron'] * patron_feature_weight
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
    if n_train == 0:
        st.error('No hay datos desde 2025-01 para entrenar el modelo.')
        st.stop()
    if n_train < 5:
        st.error('No hay suficientes datos para entrenar/reentrenar el modelo.')
        st.stop()

    x_train_df = entrenamiento_df[
        ['Tallos/m2', 'Tallos_m2_patron_ponderado']
    ].reset_index(drop=True)
    x_train_df['Semana_orden'] = np.arange(len(x_train_df), dtype=float)
    y_train_df = pd.DataFrame(
        entrenamiento_df['Produccion_ajustada']).reset_index(drop=True)

    if len(eval_actual_df) == 0:
        st.error(
            'No hay suficientes datos actuales validos para generar la evaluacion.')
        st.stop()

    x_frame = eval_actual_df[
        ['Tallos/m2', 'Tallos_m2_patron_ponderado']
    ].reset_index(drop=True)
    x_frame['Semana_orden'] = np.arange(len(x_frame), dtype=float)
    y_frame = pd.DataFrame(eval_actual_df['Produccion']).reset_index(drop=True)

    split_idx = int(len(x_train_df) * 0.8)
    split_idx = min(max(split_idx, 1), len(x_train_df) - 1)
    X_train, X_test = x_train_df.iloc[:split_idx], x_train_df.iloc[split_idx:]
    y_train, y_test = y_train_df.iloc[:split_idx], y_train_df.iloc[split_idx:]
    inicio_train = entrenamiento_df.iloc[0]
    fin_train = entrenamiento_df.iloc[split_idx - 1]

    model_name = ''.join(ch if ch.isalnum() else '_' for ch in str(var_proy))
    model_file = models_dir / f'rf_{model_name}.pkl'
    train_key = f'entrenado_{model_name}_cal_v2'

    if train_key not in st.session_state:
        modelo = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=12,
            min_samples_leaf=2
        )
        modelo.fit(X_train, y_train.values.ravel())
        st.session_state[train_key] = modelo
        with open(model_file, 'wb') as f:
            pickle.dump(modelo, f)
        # st.info('Modelo entrenado automaticamente una vez con el 80% de los datos.')
        st.caption(
            'Rango de entrenamiento usado: '
            f"{int(inicio_train['Anio'])}-{int(inicio_train['Semana']):02d} "
            f"a {int(fin_train['Anio'])}-{int(fin_train['Semana']):02d} "
            f"({split_idx} de {len(entrenamiento_df)} registros)."
        )
        st.caption(f'Modelo guardado en: {model_file.name}')
    else:
        modelo = st.session_state[train_key]
        st.caption(
            'Modelo ya entrenado en esta sesion; se reutiliza para la proyeccion.')
        st.caption(
            'Rango de entrenamiento configurado: '
            f"{int(inicio_train['Anio'])}-{int(inicio_train['Semana']):02d} "
            f"a {int(fin_train['Anio'])}-{int(fin_train['Semana']):02d} "
            f"({split_idx} de {len(entrenamiento_df)} registros)."
        )

    pred_1 = modelo.predict(x_frame)

    y_pred = pd.DataFrame(pred_1, columns=['Estimado_modelo'])
    y_pred['Estimado_modelo'] = y_pred['Estimado_modelo'] * factor_correccion

    # Regla agronomica estricta: despues de cada nuevo maximo semanal,
    # la semana siguiente debe quedar por debajo del 57% de ese pico.
    pred_vals = y_pred['Estimado_modelo'].to_numpy(copy=True)

    # Mezcla dirigida con la proyeccion del patron para dar mas peso en prediccion.
    proy_vals = proy.reset_index(drop=True).to_numpy(copy=True)
    n_blend = min(len(pred_vals), len(proy_vals))
    if n_blend > 0:
        pred_vals[:n_blend] = (
            (1 - patron_prediction_weight) * pred_vals[:n_blend]
            + patron_prediction_weight * proy_vals[:n_blend]
        )

    ajustes_pico = 0
    running_max = -np.inf
    for i in range(len(pred_vals) - 1):
        if pred_vals[i] > running_max:
            running_max = pred_vals[i]
            limite_siguiente = pred_vals[i] * peak_decay_pred
            if pred_vals[i + 1] > limite_siguiente:
                pred_vals[i + 1] = limite_siguiente
                ajustes_pico += 1
        else:
            running_max = max(running_max, pred_vals[i])

    # Segunda pasada de seguridad para evitar picos consecutivos por redondeos.
    for i in range(1, len(pred_vals)):
        max_hasta_prev = pred_vals[:i].max()
        if pred_vals[i - 1] >= max_hasta_prev and pred_vals[i] >= pred_vals[i - 1]:
            pred_vals[i] = pred_vals[i - 1] * peak_decay_pred
            ajustes_pico += 1

    # Regla de valores iguales consecutivos: mantener el valor evita caidas artificiales.
    ajustes_iguales = 0
    for i in range(1, len(pred_vals)):
        if pred_vals[i] == pred_vals[i - 1]:
            pred_vals[i] = pred_vals[i - 1]
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
    st.dataframe(promedio_semanal_anual, width='stretch')

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
        'Tallos_m2_patron': eval_actual_df['Tallos_m2_patron'].iloc[:n_export].values,
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

    buffer_individual = io.BytesIO()
    with pd.ExcelWriter(buffer_individual, engine='openpyxl') as writer:
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

    st.download_button(
        'Exportar datos a Excel',
        data=buffer_individual.getvalue(),
        file_name='Proyecto.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        key='descargar_individual'
    )
    y_pred_tail = y_pred.tail(4).round(0).copy()
    etiquetas_tail = etiquetas_anio_semana.iloc[:len(
        y_pred)].tail(len(y_pred_tail)).values
    y_pred_tail.index = etiquetas_tail
    st.write(y_pred_tail)
else:
    st.info("Por favor, sube el archivo Excel.")
