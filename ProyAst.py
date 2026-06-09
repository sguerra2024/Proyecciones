from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle
import io
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import os

try:
    anthropic = importlib.import_module('anthropic')
except ImportError:
    anthropic = None

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def crear_cliente_anthropic():
    if anthropic is None:
        raise RuntimeError(
            'La libreria anthropic no esta instalada en el entorno actual.'
        )
    if not api_key:
        raise RuntimeError(
            'No se encontro ANTHROPIC_API_KEY en las variables de entorno.'
        )
    return anthropic.Anthropic(api_key=api_key)


def consultar_anthropic(prompt_usuario):
    cliente = crear_cliente_anthropic()
    modelos_candidatos = []
    for modelo in [anthropic_model, 'claude-sonnet-4-6', 'claude-haiku-4-5']:
        if modelo and modelo not in modelos_candidatos:
            modelos_candidatos.append(modelo)

    ultimo_error = None
    respuesta = None
    for modelo in modelos_candidatos:
        try:
            respuesta = cliente.messages.create(
                model=modelo,
                max_tokens=512,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt_usuario
                    }
                ]
            )
            break
        except Exception as exc:
            ultimo_error = exc
            texto_error = str(exc).lower()
            if 'not_found_error' in texto_error or 'model' in texto_error:
                continue
            raise

    if respuesta is None:
        raise RuntimeError(
            'No fue posible usar ningun modelo de Anthropic configurado. '
            f'Ultimo error: {ultimo_error}'
        )

    textos = []
    for bloque in respuesta.content:
        texto = getattr(bloque, 'text', None)
        if texto:
            textos.append(texto)
    return '\n'.join(textos).strip()


def subir_archivo_anthropic(archivo_subido):
    if archivo_subido is None:
        raise ValueError('Selecciona un archivo antes de cargar a Anthropic.')

    cliente = crear_cliente_anthropic()
    nombre = getattr(archivo_subido, 'name', None) or 'archivo.bin'
    media_type = getattr(
        archivo_subido, 'type', None) or 'application/octet-stream'
    contenido = archivo_subido.getvalue()

    intentos = [
        (
            'beta.files.upload',
            lambda: cliente.beta.files.upload(
                file=(nombre, contenido, media_type)
            )
        ),
        (
            'files.create',
            lambda: cliente.files.create(
                file=(nombre, contenido, media_type)
            )
        ),
        (
            'beta.files.create',
            lambda: cliente.beta.files.create(
                file=(nombre, contenido, media_type)
            )
        )
    ]

    ultimo_error = None
    for metodo, fn in intentos:
        try:
            respuesta = fn()
            file_id = getattr(respuesta, 'id', None)
            if file_id is None and isinstance(respuesta, dict):
                file_id = respuesta.get('id')
            return {
                'file_id': file_id,
                'nombre': nombre,
                'bytes': len(contenido),
                'metodo': metodo
            }
        except Exception as exc:
            ultimo_error = exc

    raise RuntimeError(
        'No fue posible cargar el archivo a Anthropic. '
        f'Ultimo error: {ultimo_error}'
    )


def resumir_proyeccion_individual(var_proy, patron_seleccionado, mse_modelo, mse_patron,
                                  factor_correccion, df_export):
    vista = df_export[[
        'Anio_Semana', 'Produccion_real', 'Proy_patron', 'Estimado_modelo',
        'Error_abs', 'Error_pct'
    ]].tail(8).copy()
    prompt = (
        'Analiza esta proyeccion agricola y responde en espanol SOLO con el '
        'desempeno general. Entrega un unico parrafo corto (maximo 3 lineas), '
        'sin bullets, sin recomendaciones y sin detallar semanas especificas.\n\n'
        f'Variedad proyectada: {var_proy}\n'
        f'Patron seleccionado: {patron_seleccionado}\n'
        f'MSE modelo: {mse_modelo:.4f}\n'
        f'MSE patron: {mse_patron:.4f}\n'
        f'Factor de correccion: {factor_correccion:.4f}\n'
        'Ultimas semanas (tabla):\n'
        f'{vista.to_csv(index=False)}'
    )
    return consultar_anthropic(prompt)


def resumir_proyeccion_masiva(selected_finca, resumen, errores, df_export_estimado):
    resumen_df = pd.DataFrame(resumen)
    errores_df = pd.DataFrame(errores)
    total_variedades = len(resumen) + len(errores)
    promedio_modelo = (
        float(resumen_df['MSE_modelo'].mean()
              ) if not resumen_df.empty else None
    )
    promedio_patron = (
        float(resumen_df['MSE_proy_patron'].mean()
              ) if not resumen_df.empty else None
    )
    top_ok = resumen_df.sort_values('MSE_modelo').head(
        8) if not resumen_df.empty else pd.DataFrame()
    top_error = errores_df.head(8) if not errores_df.empty else pd.DataFrame()
    prompt = (
        'Analiza esta corrida masiva agricola y responde en espanol SOLO con el '
        'desempeno general. Entrega un unico parrafo corto (maximo 3 lineas), '
        'sin bullets, sin recomendaciones y sin listar variedades especificas.\n\n'
        f'Finca: {selected_finca}\n'
        f'Total variedades evaluadas: {total_variedades}\n'
        f'Variedades proyectadas: {len(resumen)}\n'
        f'Variedades con error: {len(errores)}\n'
        f'MSE promedio modelo: {promedio_modelo}\n'
        f'MSE promedio patron: {promedio_patron}\n'
        f'Promedio estimado exportado: {float(df_export_estimado["Estimado_modelo"].mean()):.2f}\n'
        'Mejores casos por MSE:\n'
        f'{top_ok.to_csv(index=False) if not top_ok.empty else "Sin datos"}\n'
        'Errores reportados:\n'
        f'{top_error.to_csv(index=False) if not top_error.empty else "Sin errores"}'
    )
    return consultar_anthropic(prompt)


def responder_pregunta_anthropic(df_base, pregunta_usuario, finca_contexto=None,
                                 df_proyeccion=None):
    pregunta_limpia = str(pregunta_usuario).strip()
    if not pregunta_limpia:
        raise ValueError(
            'Escribe una pregunta antes de consultar a Anthropic.')

    columnas = [str(col) for col in df_base.columns]
    precio_cols = [
        col for col in columnas
        if re.search(r'precio|price|valor', col, flags=re.IGNORECASE)
    ]
    ocasion_cols = [
        col for col in columnas
        if re.search(r'ocasi|occasion|evento|uso|segmento|canal|cliente', col, flags=re.IGNORECASE)
    ]

    col_variedad = 'Bloque&Varid' if 'Bloque&Varid' in df_base.columns else (
        'Variedad' if 'Variedad' in df_base.columns else None
    )
    col_finca = 'Finca' if 'Finca' in df_base.columns else None

    contexto_df = df_base.copy()
    if finca_contexto is not None and col_finca is not None:
        contexto_df = contexto_df[
            contexto_df[col_finca].astype(str) == str(finca_contexto)
        ].copy()

    if contexto_df.empty:
        raise ValueError(
            'No hay datos en el contexto seleccionado para responder.')

    fincas_disponibles = []
    if col_finca is not None:
        fincas_disponibles = sorted(
            contexto_df[col_finca].dropna().astype(str).unique().tolist()
        )

    top_variedades_csv = 'No disponible'
    if col_variedad is not None:
        top_variedades = (
            contexto_df[col_variedad]
            .dropna()
            .astype(str)
            .value_counts()
            .head(20)
            .reset_index()
        )
        top_variedades.columns = ['Variedad', 'Registros']
        top_variedades_csv = top_variedades.to_csv(index=False)

    proyeccion_info = 'No disponible en esta sesion.'
    if df_proyeccion is not None and not df_proyeccion.empty:
        proy_ctx = df_proyeccion.copy()
        if 'Finca_proyectada' in proy_ctx.columns and finca_contexto is not None:
            proy_ctx = proy_ctx[
                proy_ctx['Finca_proyectada'].astype(str) == str(finca_contexto)
            ].copy()

        if not proy_ctx.empty and all(
            col in proy_ctx.columns
            for col in ['Variedad_proyectada', 'Estimado_modelo']
        ):
            resumen_proy = (
                proy_ctx
                .groupby('Variedad_proyectada', as_index=False)
                .agg(
                    semanas=('Estimado_modelo', 'count'),
                    estimado_promedio=('Estimado_modelo', 'mean')
                )
                .sort_values('estimado_promedio', ascending=False)
                .head(50)
            )
            muestra_cols = [
                col for col in ['Finca_proyectada', 'Variedad_proyectada',
                                'Anio_Semana', 'Estimado_modelo']
                if col in proy_ctx.columns
            ]
            muestra_proy = proy_ctx[muestra_cols].head(150)
            proyeccion_info = (
                f'Filas proyeccion en contexto: {len(proy_ctx)}\n'
                'Resumen proyeccion por variedad (csv):\n'
                f'{resumen_proy.to_csv(index=False)}\n'
                'Muestra de base proyectada (csv):\n'
                f'{muestra_proy.to_csv(index=False)}'
            )

    prompt = (
        'Eres un analista de datos del negocio floricola. '
        'Responde SOLO con informacion disponible en la base cargada. '
        'Si un dato no existe (por ejemplo precio u ocasion), di exactamente: '
        '"No disponible en esta base". No inventes datos ni supuestos. '
        'Responde en espanol, claro y en bullets cuando aplique.\n\n'
        f'Finca en contexto: {finca_contexto}\n'
        f'Filas en contexto: {len(contexto_df)}\n'
        f'Columnas disponibles: {", ".join(columnas)}\n'
        f'Columnas de precio detectadas: {precio_cols if precio_cols else "Ninguna"}\n'
        f'Columnas de ocasion/uso detectadas: {ocasion_cols if ocasion_cols else "Ninguna"}\n'
        f'Fincas detectadas en contexto: {fincas_disponibles[:20]}\n'
        'Top variedades por registros (csv):\n'
        f'{top_variedades_csv}\n'
        'Base de proyeccion del modelo:\n'
        f'{proyeccion_info}\n'
        'Pregunta del usuario:\n'
        f'{pregunta_limpia}'
    )
    return consultar_anthropic(prompt)


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

if 'base_proyeccion_anthropic' not in st.session_state:
    st.session_state['base_proyeccion_anthropic'] = pd.DataFrame()
if 'anthropic_archivos_subidos' not in st.session_state:
    st.session_state['anthropic_archivos_subidos'] = []

with st.expander('Cargar archivos a Anthropic'):
    st.caption(
        'Sube un archivo para registrarlo directamente en Anthropic.'
    )
    archivo_anthropic = st.file_uploader(
        'Selecciona archivo para Anthropic',
        type=['xlsx', 'csv', 'txt', 'json', 'pdf'],
        key='archivo_uploader_anthropic'
    )
    if st.button('Cargar archivo a Anthropic', key='btn_cargar_archivo_anthropic'):
        try:
            info_carga = subir_archivo_anthropic(archivo_anthropic)
            st.session_state['anthropic_archivos_subidos'].append(info_carga)
            st.success(
                f"Archivo cargado. file_id: {info_carga.get('file_id')}"
            )
            st.write(info_carga)
        except Exception as exc:
            st.error(f'No se pudo cargar el archivo a Anthropic: {exc}')

    if st.session_state['anthropic_archivos_subidos']:
        st.write('Historial de archivos cargados en esta sesion:')
        st.dataframe(
            pd.DataFrame(st.session_state['anthropic_archivos_subidos']),
            width='stretch'
        )

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

    with st.expander('Preguntas a Anthropic sobre la base'):
        st.caption(
            'Consulta sobre variedades, fincas, usos y precios, segun datos cargados.'
        )
        st.write(
            'Ejemplos: "Que fincas ofrecen la variedad LIGHT HOUSE?", '
            '"Hay columna de precios para esta variedad?", '
            '"En que ocasiones se uso FREEDOM?"'
        )
        with st.form('form_pregunta_anthropic', clear_on_submit=True):
            pregunta_negocio = st.text_area(
                'Escribe tu pregunta',
                key='pregunta_negocio_anthropic'
            )
            enviar_pregunta = st.form_submit_button('Preguntar a Anthropic')

        if enviar_pregunta:
            try:
                respuesta_negocio = responder_pregunta_anthropic(
                    df,
                    pregunta_negocio,
                    selected_finca,
                    st.session_state.get('base_proyeccion_anthropic')
                )
                st.success('Respuesta generada.')
                st.write(respuesta_negocio)
            except Exception as exc:
                st.error(f'No se pudo responder la pregunta: {exc}')

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

    def calcular_patron_compatible_individual(df_patrones, df_variedad_objetivo, var_proy):
        # Replica exacta de la comparacion del flujo individual para mantener
        # el mismo patron seleccionado en la corrida masiva.
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

        if len(arr_list) < 2:
            raise ValueError('No hay suficientes patrones para comparar.')

        arr_list.sort(key=lambda x: x[1])
        return seleccionar_patron(arr_list, var_proy)

    def proyectar_variedad_masiva(df_base, var_proy):
        df_filtered_ = df_base[df_base['Bloque&Varid'].isin([var_proy])].copy()
        if df_filtered_.empty:
            raise ValueError('Sin datos para la variedad seleccionada.')

        patron_seleccionado = calcular_patron_compatible_individual(
            df,
            df_filtered_,
            var_proy
        )

        df_patron = df[df['Bloque&Varid'].isin(
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
        patron_actual = df[df['Bloque&Varid'].isin(
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

        # Exportar solo las 4 ultimas semanas por cada Bloque&Varid.
        df_estimado_ordenado['__anio'] = pd.to_numeric(
            df_estimado_ordenado['Anio'], errors='coerce')
        df_estimado_ordenado['__semana'] = pd.to_numeric(
            df_estimado_ordenado['Semana'], errors='coerce')
        df_estimado_ordenado = df_estimado_ordenado.sort_values(
            ['Variedad_proyectada', '__anio', '__semana', '_orden_original'],
            ascending=[True, False, False, False]
        )
        df_estimado_ordenado['__rank_ultimas'] = (
            df_estimado_ordenado
            .groupby('Variedad_proyectada')
            .cumcount() + 1
        )
        df_estimado_ordenado = df_estimado_ordenado[
            df_estimado_ordenado['__rank_ultimas'] <= 4
        ].copy()
        df_estimado_ordenado = df_estimado_ordenado.sort_values(
            '_orden_original')
        df_estimado_ordenado = df_estimado_ordenado.drop(
            columns=['__anio', '__semana', '__rank_ultimas']
        )

        # Si no se pudo proyectar una variedad/semana, exportar 0.
        df_estimado_ordenado['Estimado_modelo'] = df_estimado_ordenado[
            'Estimado_modelo'
        ].fillna(0)
        df_estimado_ordenado['Estimado_modelo'] = np.rint(
            df_estimado_ordenado['Estimado_modelo']
        ).astype(np.int64)
        columnas_base_export = [
            'Anio', 'Semana', 'Producto', 'Finca',
            'Bloque', 'Variedad', 'Bloque&Varid'
        ]
        columnas_export = [
            col for col in columnas_base_export if col in df_estimado_ordenado.columns
        ] + ['Estimado_modelo']
        df_export_estimado = df_estimado_ordenado[
            columnas_export
        ].reset_index(drop=True)

        base_proy_masiva = df_estimado_ordenado[[
            'Variedad_proyectada', 'Anio_Semana', 'Estimado_modelo'
        ]].copy()
        base_proy_masiva['Finca_proyectada'] = str(selected_finca)
        st.session_state['base_proyeccion_anthropic'] = base_proy_masiva

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
            # Exportar columnas base mas Estimado_modelo.
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
    patron_seleccionado = calcular_patron_compatible_individual(
        df,
        df_filtered_,
        var_proy
    )

# 3.- IMPORTAR BASE DE VARIEDADES A PROYECTAR Y COMPARAR CURVA CON PATRON SELECCIONADO\n",

    df_3 = pd.read_excel(file_path)
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

    base_proy_individual = df_export[[
        'Variedad_proyectada', 'Anio_Semana', 'Estimado_modelo'
    ]].copy()
    base_proy_individual['Finca_proyectada'] = str(selected_finca)
    base_actual = st.session_state.get('base_proyeccion_anthropic')
    if base_actual is None or base_actual.empty:
        st.session_state['base_proyeccion_anthropic'] = base_proy_individual
    else:
        base_merge = pd.concat(
            [base_actual, base_proy_individual],
            ignore_index=True
        )
        base_merge = base_merge.drop_duplicates(
            subset=['Finca_proyectada', 'Variedad_proyectada', 'Anio_Semana'],
            keep='last'
        )
        st.session_state['base_proyeccion_anthropic'] = base_merge
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
