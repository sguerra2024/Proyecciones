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
from dotenv import load_dotenv, dotenv_values
import os
import sys
import mimetypes

agents_dir = Path(__file__).with_name("agents")
if agents_dir.exists():
    sys.path.insert(0, str(agents_dir))

try:
    anthropic = importlib.import_module('anthropic')
except ImportError:
    anthropic = None

try:
    openai = importlib.import_module('openai')
except ImportError:
    openai = None

dotenv_path = Path(__file__).with_name('.env')
load_dotenv(dotenv_path=dotenv_path if dotenv_path.exists() else None)

anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
llm_provider = (os.getenv("LLM_PROVIDER") or "anthropic").strip().lower()
openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
github_model = os.getenv("GITHUB_MODEL", openai_model)
PATRON_FEATURE_WEIGHT = float(os.getenv("PATRON_FEATURE_WEIGHT", "1.5"))
PATRON_PREDICTION_WEIGHT = float(os.getenv("PATRON_PREDICTION_WEIGHT", "0.65"))
PICO_NO_CICLO_UMBRAL_REL = float(os.getenv("PICO_NO_CICLO_UMBRAL_REL", "0.06"))
PICO_NO_CICLO_UMBRAL_PENDIENTE = float(
    os.getenv("PICO_NO_CICLO_UMBRAL_PENDIENTE", "0.02"))
PICO_NO_CICLO_MAX_BLEND = float(os.getenv("PICO_NO_CICLO_MAX_BLEND", "0.75"))
MINIMOS_MERCADO_CANTIDAD = 3
PICO_MERCADO_INCREMENTO = float(os.getenv("PICO_MERCADO_INCREMENTO", "0.12"))
REFUERZO_TALLOS_M2 = float(os.getenv("REFUERZO_TALLOS_M2", "0.30"))
TALLOS_2026_BOOST = float(os.getenv("TALLOS_2026_BOOST", "1.0"))
AJUSTE_RESIDUAL_WEIGHT = float(os.getenv("AJUSTE_RESIDUAL_WEIGHT", "0.30"))
PATRON_TRAIN_TARGET_WEIGHT = float(
    os.getenv("PATRON_TRAIN_TARGET_WEIGHT", "0.50")
)


def obtener_valor_env(*claves):
    for clave in claves:
        valor = (os.getenv(clave) or "").strip()
        if valor:
            return valor

    for ruta_env in [Path(__file__).with_name('.env'), Path.cwd() / '.env']:
        if not ruta_env.exists():
            continue
        valores = dotenv_values(ruta_env)
        for clave in claves:
            valor = (valores.get(clave) or "").strip()
            if valor:
                return valor
    return ""


def obtener_llm_provider():
    proveedor = (os.getenv("LLM_PROVIDER")
                 or llm_provider or "anthropic").strip().lower()
    alias = {
        'copilot': 'github',
        'github_models': 'github',
        'gh': 'github',
        'openai_compatible': 'openai'
    }
    return alias.get(proveedor, proveedor)


def obtener_api_key_anthropic():
    return obtener_valor_env("ANTHROPIC_API_KEY", "ANTHROPIC_KEY")


def obtener_api_key_openai():
    return obtener_valor_env("OPENAI_API_KEY")


def obtener_api_key_github():
    return obtener_valor_env("GITHUB_MODELS_TOKEN", "GITHUB_TOKEN")


def modelos_anthropic_candidatos():
    candidatos = []
    for modelo in [
        anthropic_model,
        'claude-3-5-sonnet-latest',
        'claude-3-5-haiku-latest',
        'claude-3-opus-latest',
        'claude-sonnet-4-6',
        'claude-haiku-4-5'
    ]:
        if modelo and modelo not in candidatos:
            candidatos.append(modelo)
    return candidatos


def crear_cliente_anthropic():
    if anthropic is None:
        raise RuntimeError(
            'La libreria anthropic no esta instalada en el entorno actual.'
        )
    api_key = obtener_api_key_anthropic()
    if not api_key:
        raise RuntimeError(
            'No se encontro ANTHROPIC_API_KEY en las variables de entorno.'
        )
    return anthropic.Anthropic(api_key=api_key)


def modelos_openai_candidatos(proveedor):
    candidatos = []
    if proveedor == 'github':
        for modelo in [
            os.getenv("GITHUB_MODEL", ""),
            github_model,
            os.getenv("OPENAI_MODEL", ""),
            'gpt-4.1-mini',
            'gpt-4o-mini'
        ]:
            if modelo and modelo not in candidatos:
                candidatos.append(modelo)
    else:
        for modelo in [
            os.getenv("OPENAI_MODEL", ""),
            openai_model,
            'gpt-4.1-mini',
            'gpt-4o-mini'
        ]:
            if modelo and modelo not in candidatos:
                candidatos.append(modelo)
    return candidatos


def crear_cliente_openai_compatible(proveedor):
    if openai is None:
        raise RuntimeError(
            'La libreria openai no esta instalada en el entorno actual.'
        )
    if not hasattr(openai, 'OpenAI'):
        raise RuntimeError(
            'La version de openai instalada no soporta cliente OpenAI. '
            'Actualiza el paquete openai.'
        )

    if proveedor == 'github':
        api_key = obtener_api_key_github()
        if not api_key:
            raise RuntimeError(
                'No se encontro GITHUB_MODELS_TOKEN (o GITHUB_TOKEN).'
            )
        base_url = (
            os.getenv("GITHUB_MODELS_BASE_URL")
            or "https://models.inference.ai.azure.com"
        ).strip()
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    api_key = obtener_api_key_openai()
    if not api_key:
        raise RuntimeError(
            'No se encontro OPENAI_API_KEY en las variables de entorno.'
        )
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    return openai.OpenAI(api_key=api_key)


def normalizar_error_github_models(exc):
    texto_error = str(exc)
    texto_error_lower = texto_error.lower()
    if (
        'models permission is required' in texto_error_lower
        or ('unauthorized' in texto_error_lower and 'models' in texto_error_lower)
    ):
        return RuntimeError(
            'El token de GitHub no tiene permiso models. '
            'Los permisos de repositorio como read access to administration '
            'and metadata no habilitan GitHub Models. '
            'Actualiza el token con permiso Models y vuelve a intentarlo.'
        )
    return exc


def consultar_openai_compatible(prompt_usuario, proveedor):
    cliente = crear_cliente_openai_compatible(proveedor)
    modelos_candidatos = modelos_openai_candidatos(proveedor)

    ultimo_error = None
    for modelo in modelos_candidatos:
        try:
            respuesta = cliente.chat.completions.create(
                model=modelo,
                max_tokens=512,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt_usuario
                    }
                ]
            )
            contenido = ''
            if respuesta and getattr(respuesta, 'choices', None):
                mensaje = getattr(respuesta.choices[0], 'message', None)
                contenido = (getattr(mensaje, 'content', None) or '').strip()
            if contenido:
                return contenido
            raise RuntimeError('La respuesta del modelo llego vacia.')
        except Exception as exc:
            if proveedor == 'github':
                exc = normalizar_error_github_models(exc)
            ultimo_error = exc
            texto_error = str(exc).lower()
            es_error_modelo = any(
                frag in texto_error for frag in [
                    'model_not_found',
                    'model not found',
                    'invalid model',
                    'unknown model',
                    'not available for your account'
                ]
            )
            if es_error_modelo:
                continue
            raise

    raise RuntimeError(
        'No fue posible usar ningun modelo OpenAI-compatible configurado. '
        f'Ultimo error: {ultimo_error}'
    )


def consultar_llm(prompt_usuario):
    proveedor = obtener_llm_provider()
    if proveedor == 'anthropic':
        return consultar_anthropic(prompt_usuario)
    if proveedor in ['github', 'openai']:
        return consultar_openai_compatible(prompt_usuario, proveedor)
    raise RuntimeError(
        'LLM_PROVIDER no soportado. Usa anthropic, github u openai.'
    )


def obtener_configuracion_proyeccion_masiva(ligera=False):
    if ligera:
        return {
            "modo_liviano": True,
            "n_estimators": 25,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "min_samples_split": 3,
            "max_features": "sqrt",
            "columnas_modelo": [
                'Tallos/m2',
                'Tallos_m2_patron',
                'Produccion_patron',
                'Tallos_m2_patron_ponderado',
                'Produccion_patron_ponderado',
            ],
            "usar_sensibilidad_picos": False,
            "limite_filas_entrenamiento": 40,
        }
    return {
        "modo_liviano": False,
        "n_estimators": 100,
        "max_depth": 16,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_features": "sqrt",
        "columnas_modelo": [
            'Tallos/m2',
            'Tallos_m2_patron',
            'Produccion_patron',
            'Tallos_m2_patron_ponderado',
            'Produccion_patron_ponderado',
            'Incremento_tallos_patron',
            'Incremento_produccion_patron',
            'sn_alto',
        ],
        "usar_sensibilidad_picos": True,
        "limite_filas_entrenamiento": None,
    }


def estado_configuracion_llm():
    proveedor = obtener_llm_provider()
    if proveedor == 'anthropic':
        if anthropic is None:
            return False, 'LLM=anthropic, falta instalar libreria anthropic.'
        if not obtener_api_key_anthropic():
            return False, 'LLM=anthropic, falta ANTHROPIC_API_KEY.'
        return True, f'LLM=anthropic ({modelos_anthropic_candidatos()[0]})'

    if proveedor == 'github':
        if openai is None or not hasattr(openai, 'OpenAI'):
            return False, 'LLM=github, falta instalar/actualizar libreria openai.'
        if not obtener_api_key_github():
            return False, 'LLM=github, falta GITHUB_MODELS_TOKEN o GITHUB_TOKEN.'
        modelo = modelos_openai_candidatos('github')[0]
        return True, f'LLM=github ({modelo})'

    if proveedor == 'openai':
        if openai is None or not hasattr(openai, 'OpenAI'):
            return False, 'LLM=openai, falta instalar/actualizar libreria openai.'
        if not obtener_api_key_openai():
            return False, 'LLM=openai, falta OPENAI_API_KEY.'
        modelo = modelos_openai_candidatos('openai')[0]
        return True, f'LLM=openai ({modelo})'

    return False, 'LLM_PROVIDER no soportado. Usa anthropic, github u openai.'


def consultar_anthropic(prompt_usuario):
    proveedor = obtener_llm_provider()
    if proveedor in ['github', 'openai']:
        return consultar_openai_compatible(prompt_usuario, proveedor)

    cliente = crear_cliente_anthropic()
    modelos_candidatos = modelos_anthropic_candidatos()

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
            es_error_modelo = any(
                frag in texto_error for frag in [
                    'not_found_error',
                    'model not found',
                    'invalid model',
                    'unknown model',
                    'is not available for your account'
                ]
            )
            if es_error_modelo:
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


def es_perfil_analista():
    """Valida si el usuario actual es Analista"""
    for clave in ['perfil', 'perfil_usuario', 'perfil_activo', 'rol', 'role', 'user_profile', 'user_role']:
        valor = st.session_state.get(clave)
        if valor is not None and str(valor).strip().lower() == 'analista':
            return True
    return False


def subir_archivo_anthropic(archivo_subido):
    if obtener_llm_provider() != 'anthropic':
        raise RuntimeError(
            'La carga de archivos solo esta disponible con LLM_PROVIDER=anthropic.'
        )

    if archivo_subido is None:
        raise ValueError('Selecciona un archivo antes de cargar a claude')

    cliente = crear_cliente_anthropic()
    nombre = getattr(archivo_subido, 'name', None) or 'archivo.bin'
    media_type = getattr(archivo_subido, 'type', None)
    if not media_type:
        media_type = mimetypes.guess_type(
            nombre)[0] or 'application/octet-stream'
    contenido = archivo_subido.getvalue()
    if not contenido:
        raise ValueError('El archivo seleccionado esta vacio.')

    # Anthropic puede variar el formato aceptado segun version del SDK/API.
    # Probamos variantes comunes del payload antes de reportar error final.
    def construir_variantes_payload(file_name, file_bytes, mime):
        return [
            ('tuple_name_bytes_mime', (file_name, file_bytes, mime)),
            ('tuple_name_bytes', (file_name, file_bytes)),
            ('dict_content', {
                'name': file_name,
                'content': file_bytes,
                'media_type': mime
            }),
            ('dict_file', {
                'file_name': file_name,
                'bytes': file_bytes,
                'media_type': mime
            }),
        ]

    variantes_archivo = []

    # Para Excel, priorizamos CSV porque Anthropic lo reconoce de forma mas estable.
    nombre_lower = nombre.lower()
    if nombre_lower.endswith('.xlsx') or nombre_lower.endswith('.xls'):
        try:
            df_excel = pd.read_excel(io.BytesIO(contenido), sheet_name=0)
            csv_bytes = df_excel.to_csv(index=False).encode('utf-8')
            nombre_csv = str(Path(nombre).with_suffix('.csv'))
            variantes_archivo.append((nombre_csv, csv_bytes, 'text/csv'))
        except Exception:
            # Si falla la conversion, seguimos con el binario original.
            pass

    # Mantener binario original como respaldo final.
    variantes_archivo.append((nombre, contenido, media_type))

    def ejecutar_metodo(metodo, payload):
        if metodo == 'beta.files.upload':
            return cliente.beta.files.upload(file=payload)
        if metodo == 'files.create':
            return cliente.files.create(file=payload)
        if metodo == 'beta.files.create':
            return cliente.beta.files.create(file=payload)
        raise RuntimeError(f'Metodo no soportado: {metodo}')

    metodos = ['beta.files.upload', 'files.create', 'beta.files.create']

    ultimo_error = None
    errores = []
    for file_name, file_bytes, mime in variantes_archivo:
        for metodo in metodos:
            for payload_tipo, payload in construir_variantes_payload(file_name, file_bytes, mime):
                try:
                    respuesta = ejecutar_metodo(metodo, payload)
                    file_id = getattr(respuesta, 'id', None)
                    if file_id is None and isinstance(respuesta, dict):
                        file_id = respuesta.get('id')
                    return {
                        'file_id': file_id,
                        'nombre': file_name,
                        'bytes': len(file_bytes),
                        'metodo': f'{metodo}:{payload_tipo}'
                    }
                except Exception as exc:
                    ultimo_error = exc
                    errores.append(
                        f'{metodo}:{payload_tipo}:{file_name} -> {exc}'
                    )

    raise RuntimeError(
        'No fue posible cargar el archivo a Anthropic. '
        f'Ultimo error: {ultimo_error}. '
        'Detalle de intentos: '
        + ' | '.join(errores[-4:])
    )


def sincronizar_archivo_llm(archivo_subido, dataframe=None, nombre_archivo=None):
    if archivo_subido is None and dataframe is None:
        raise ValueError('Selecciona un archivo antes de sincronizar.')

    proveedor = obtener_llm_provider()
    nombre = nombre_archivo or getattr(
        archivo_subido, 'name', None) or 'archivo'

    if proveedor == 'anthropic':
        info_carga = subir_archivo_anthropic(archivo_subido)
        info_carga['modo'] = 'remoto'
        return info_carga

    if dataframe is None:
        dataframe = cargar_archivo_a_dataframe(archivo_subido)
    if dataframe is None or dataframe.empty:
        raise RuntimeError(
            'Con GitHub Models/OpenAI el archivo debe poder leerse como tabla '
            'para usarlo como contexto local en la sesion.'
        )

    return {
        'file_id': 'local-session',
        'nombre': nombre,
        'bytes': int(len(dataframe.index)),
        'metodo': 'session-dataframe',
        'modo': 'local',
        'filas': int(len(dataframe.index))
    }


def registrar_sincronizacion_en_sesion(info_carga, df_sync, nombre_sync, id_base=''):
    st.session_state['archivo_anthropic_cargado'] = id_base or nombre_sync
    st.session_state['archivo_anthropic_id'] = info_carga.get('file_id') or ''
    if info_carga.get('modo') == 'remoto':
        st.session_state['estado_subida_anthropic'] = (
            'ok', f"Anthropic ID: {info_carga.get('file_id')}"
        )
    else:
        st.session_state['estado_subida_anthropic'] = (
            'ok',
            f"Contexto local listo: {info_carga.get('nombre')} ({info_carga.get('filas', 0)} filas)"
        )
    if df_sync is not None and not df_sync.empty:
        st.session_state['archivo_sesion_df'] = df_sync
        st.session_state['archivo_sesion_nombre'] = nombre_sync


class ArchivoEnMemoria:
    def __init__(self, name, data, media_type='application/octet-stream'):
        self.name = name
        self._data = data
        self.type = media_type
        self.size = len(data)

    def getvalue(self):
        return self._data


def sincronizar_export_generado_automatico(bytes_export, nombre_export, mime_export, dataframe=None, state=None):
    if not bytes_export:
        return
    target_state = st.session_state if state is None else state
    cache_key = f"__export_sync_cache__{nombre_export}"
    if target_state.get(cache_key) == bytes_export:
        return

    try:
        df_export = dataframe
        if df_export is None:
            df_export = pd.read_excel(io.BytesIO(bytes_export))
        archivo_export = ArchivoEnMemoria(
            nombre_export,
            bytes_export,
            mime_export
        )
        info_carga = sincronizar_archivo_llm(
            archivo_export,
            dataframe=df_export,
            nombre_archivo=nombre_export
        )
        registrar_sincronizacion_en_sesion(
            info_carga,
            df_export,
            nombre_export,
            nombre_export + str(len(bytes_export))
        )
        target_state[cache_key] = bytes_export
    except Exception as exc:
        target_state['estado_subida_anthropic'] = (
            'error', f'Sincronizacion automatica fallida: {exc}'
        )


def construir_cache_patrones_semanales(df_base):
    cache = {}
    if df_base is None or df_base.empty:
        return cache

    for nombre_variedad, grupo in df_base.groupby('Bloque&Varid', dropna=False):
        try:
            patron_weekly = grupo[[
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
                .agg({
                    'Tallos/m2': 'mean',
                    'Produccion': 'sum'
                })
                .rename(columns={
                    'Tallos/m2': 'Tallos_m2_patron',
                    'Produccion': 'Produccion_patron'
                })
                .sort_values(['Anio', 'Semana'])
                .reset_index(drop=True)
            )
            patron_weekly['Incremento_tallos_patron'] = (
                patron_weekly['Tallos_m2_patron'].diff().fillna(0.0)
            )
            patron_weekly['Incremento_produccion_patron'] = (
                patron_weekly['Produccion_patron'].diff().fillna(0.0)
            )
            cache[str(nombre_variedad)] = patron_weekly
        except Exception:
            continue
    return cache


def cargar_archivo_a_dataframe(archivo_subido):
    if archivo_subido is None:
        return pd.DataFrame()

    nombre = str(getattr(archivo_subido, 'name', '') or '').lower()
    contenido = archivo_subido.getvalue()
    if not contenido:
        return pd.DataFrame()

    try:
        if nombre.endswith('.xlsx') or nombre.endswith('.xls'):
            return pd.read_excel(io.BytesIO(contenido))
        if nombre.endswith('.csv'):
            return pd.read_csv(io.BytesIO(contenido))
        if nombre.endswith('.txt'):
            return pd.read_csv(io.BytesIO(contenido), sep=None, engine='python')
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()


def resumir_proyeccion_individual(var_proy, patron_seleccionado,
                                  factor_correccion, df_export):
    pares_modelo = df_export[[
        'Produccion_real', 'Estimado_modelo'
    ]].dropna()
    pares_patron = df_export[[
        'Produccion_real', 'Proy_patron'
    ]].dropna()
    mse_modelo = mean_squared_error(
        pares_modelo['Produccion_real'],
        pares_modelo['Estimado_modelo']
    ) if not pares_modelo.empty else np.nan
    mse_patron = mean_squared_error(
        pares_patron['Produccion_real'],
        pares_patron['Proy_patron']
    ) if not pares_patron.empty else np.nan
    vista = df_export[[
        'Anio_Semana', 'Produccion_real', 'Proy_patron', 'Estimado_modelo',
        'Error_abs', 'Error_pct'
    ]].tail(8).copy()
    prompt = (
        'Analiza esta proyeccion agricola y responde en espanol SOLO con el '
        'desempeno de analisis. Entrega un unico parrafo corto (maximo 3 lineas), '
        'sin bullets, sin recomendaciones y sin detallar semanas especificas.\n\n'
        f'Variedad proyectada: {var_proy}\n'
        f'Patron seleccionado: {patron_seleccionado}\n'
        f'MSE modelo: {mse_modelo:.4f}\n'
        f'MSE patron: {mse_patron:.4f}\n'
        f'Factor de correccion: {factor_correccion:.4f}\n'
        'Ultimas semanas (tabla):\n'
        f'{vista.to_csv(index=False)}'
    )
    return consultar_llm(prompt)


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
        # 'sin bullets, sin recomendaciones y sin listar variedades especificas.\n\n'
        f'Finca: {selected_finca}\n'
        f'Total variedades evaluadas: {total_variedades}\n'
        f'Variedades proyectadas: {len(resumen)}\n'
        f'Variedades con error: {len(errores)}\n'
        # f'MSE promedio modelo: {promedio_modelo}\n'
        # f'MSE promedio patron: {promedio_patron}\n'
        f'Promedio estimado exportado: {float(df_export_estimado["Estimado_modelo"].mean()):.2f}\n'
        'Mejores casos por MSE:\n'
        f'{top_ok.to_csv(index=False) if not top_ok.empty else "Sin datos"}\n'
        'Errores reportados:\n'
        f'{top_error.to_csv(index=False) if not top_error.empty else "Sin errores"}'
    )
    return consultar_llm(prompt)


def responder_pregunta_anthropic(df_base, pregunta_usuario, finca_contexto=None,
                                 df_proyeccion=None, df_archivo_sincronizado=None,
                                 nombre_archivo_sincronizado=''):
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

    archivo_sync_info = 'No hay archivo sincronizado activo en la sesion.'
    if df_archivo_sincronizado is not None and not df_archivo_sincronizado.empty:
        sync_df = df_archivo_sincronizado.copy()
        col_finca_sync = 'Finca' if 'Finca' in sync_df.columns else None
        if finca_contexto is not None and col_finca_sync is not None:
            sync_filtrado = sync_df[
                sync_df[col_finca_sync].astype(str) == str(finca_contexto)
            ].copy()
            if sync_filtrado.empty:
                sync_filtrado = sync_df
        else:
            sync_filtrado = sync_df

        sample_sync = sync_filtrado.head(150)
        archivo_sync_info = (
            f'Archivo sincronizado activo: {nombre_archivo_sincronizado or "sin nombre"}\n'
            f'Filas disponibles: {len(sync_filtrado)}\n'
            f'Columnas disponibles: {", ".join([str(c) for c in sync_filtrado.columns])}\n'
            'Muestra de archivo sincronizado (csv):\n'
            f'{sample_sync.to_csv(index=False)}'
        )

    prompt = (
        'Eres un analista de datos del negocio floricola. '
        'Responde SOLO con informacion disponible en la base cargada, '
        'la base de proyeccion y el archivo sincronizado activo de esta sesion. '
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
        'Archivo sincronizado de la sesion:\n'
        f'{archivo_sync_info}\n'
        'Pregunta del usuario:\n'
        f'{pregunta_limpia}'
    )
    return consultar_llm(prompt)


def alinear_series_para_ajuste(*series):
    arrays = []
    for serie in series:
        if serie is None:
            arrays.append(np.array([], dtype=float))
            continue
        arrays.append(np.asarray(serie, dtype=float))

    longitudes = [len(arr) for arr in arrays]
    if not longitudes:
        return tuple()

    n_common = min(longitudes) if longitudes else 0
    return tuple(arr[:n_common].copy() if len(arr) else arr.copy() for arr in arrays)


def _ajustar_patron_con_extremos_real_modulo(trabajo, refuerzo_tallos_m2=None):
    if trabajo is None or trabajo.empty:
        return trabajo
    if 'Produccion' not in trabajo.columns or 'Produccion_patron' not in trabajo.columns:
        return trabajo

    produccion_real_base = np.asarray(
        pd.to_numeric(trabajo['Produccion'], errors='coerce'),
        dtype=float,
    )
    produccion_patron_base = np.asarray(
        pd.to_numeric(trabajo['Produccion_patron'], errors='coerce'),
        dtype=float,
    )
    n_common = min(len(produccion_real_base), len(produccion_patron_base))
    produccion_real = produccion_real_base[:n_common] if n_common > 0 else np.array([
    ], dtype=float)
    produccion_patron = produccion_patron_base[:n_common] if n_common > 0 else np.array([
    ], dtype=float)
    mascara_real_valida = np.isfinite(produccion_real)
    media_real = float(np.nanmean(produccion_real[mascara_real_valida])) if np.any(
        mascara_real_valida) else np.nan
    std_real = float(np.nanstd(produccion_real[mascara_real_valida], ddof=0)) if np.any(
        mascara_real_valida) else np.nan

    if not np.isfinite(media_real) or not np.isfinite(std_real) or std_real <= 0:
        return trabajo

    anio_valores = pd.to_numeric(trabajo.get(
        'Anio', pd.Series(np.nan)), errors='coerce')
    sn_valor = np.nan
    intensidad_tallos = 1.0
    if anio_valores.notna().any():
        mse_2025 = np.nan
        mse_2026 = np.nan
        for anio_ref in [2025, 2026]:
            mask_anio = anio_valores == anio_ref
            if not mask_anio.any():
                continue
            mse_anio = np.mean(
                np.abs(
                    produccion_real_base[mask_anio.to_numpy()].astype(float) -
                    produccion_patron_base[mask_anio.to_numpy()].astype(float)
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

        if 'Tallos/m2' in trabajo.columns and 'Tallos_m2_patron' in trabajo.columns:
            tallos_real = pd.to_numeric(trabajo['Tallos/m2'], errors='coerce')
            tallos_patron = pd.to_numeric(
                trabajo['Tallos_m2_patron'], errors='coerce')
            tallos_2025_real = tallos_real[anio_valores == 2025]
            tallos_2026_real = tallos_real[anio_valores == 2026]
            tallos_2025_patron = tallos_patron[anio_valores == 2025]
            tallos_2026_patron = tallos_patron[anio_valores == 2026]
            if not tallos_2025_real.empty and not tallos_2026_real.empty:
                media_2025_real = float(np.nanmean(tallos_2025_real))
                media_2026_real = float(np.nanmean(tallos_2026_real))
                if np.isfinite(media_2025_real) and np.isfinite(media_2026_real):
                    delta_tallos = media_2026_real - media_2025_real
                    if delta_tallos > 0:
                        intensidad_tallos = 1.0 + \
                            min(delta_tallos / max(media_2025_real, 1.0), 0.6)
                    elif delta_tallos < 0:
                        intensidad_tallos = max(
                            0.6, 1.0 + delta_tallos / max(abs(media_2025_real), 1.0))

    sn_alto = bool(pd.notna(sn_valor) and sn_valor > 13.0)
    factor_refuerzo_tallos = REFUERZO_TALLOS_M2 if refuerzo_tallos_m2 is None else float(
        refuerzo_tallos_m2)
    factor_intensidad = 1.15 * intensidad_tallos if sn_alto else intensidad_tallos

    z_score = (produccion_real - media_real) / std_real
    mask_positiva_moderada = (z_score >= 1.0) & (z_score < 2.0)
    mask_positiva_alta = z_score >= 2.0
    mask_negativa_moderada = (z_score <= -1.0) & (z_score > -2.0)
    mask_negativa_alta = z_score <= -2.0
    if not any([
        mask_positiva_moderada.any(),
        mask_positiva_alta.any(),
        mask_negativa_moderada.any(),
        mask_negativa_alta.any()
    ]):
        return trabajo

    patron_ajustado = produccion_patron.copy()
    patron_ajustado_full = produccion_patron_base.copy()
    fuerza_serie = np.zeros(len(z_score), dtype=float)

    if mask_positiva_moderada.any():
        fuerza_serie[mask_positiva_moderada] = np.clip(
            (0.12 + 0.08 *
             (z_score[mask_positiva_moderada] - 1.0)) * factor_intensidad,
            0.12,
            0.24 + (0.06 if sn_alto else 0.0)
        )
    if mask_positiva_alta.any():
        fuerza_serie[mask_positiva_alta] = np.clip(
            (0.25 + 0.10 *
             (z_score[mask_positiva_alta] - 2.0)) * factor_intensidad,
            0.25,
            0.45 + (0.08 if sn_alto else 0.0)
        )
    if mask_negativa_moderada.any():
        fuerza_serie[mask_negativa_moderada] = np.clip(
            (0.10 + 0.05 *
             (abs(z_score[mask_negativa_moderada]) - 1.0)) * factor_intensidad,
            0.10,
            0.22 + (0.06 if sn_alto else 0.0)
        )
    if mask_negativa_alta.any():
        fuerza_serie[mask_negativa_alta] = np.clip(
            (0.22 + 0.10 *
             (abs(z_score[mask_negativa_alta]) - 2.0)) * factor_intensidad,
            0.22,
            0.40 + (0.08 if sn_alto else 0.0)
        )

    if 'Tallos/m2' in trabajo.columns and 'Tallos_m2_patron' in trabajo.columns:
        tallos_real = pd.to_numeric(
            trabajo['Tallos/m2'], errors='coerce').to_numpy(dtype=float)
        tallos_patron = pd.to_numeric(
            trabajo['Tallos_m2_patron'], errors='coerce').to_numpy(dtype=float)
        mascara_tallos_en_alza = np.isfinite(
            tallos_real) & np.isfinite(tallos_patron)
        if mascara_tallos_en_alza.any():
            delta_tallos = tallos_real[mascara_tallos_en_alza] - \
                tallos_patron[mascara_tallos_en_alza]
            if np.any(delta_tallos > 0):
                factor_refuerzo = factor_refuerzo_tallos
                fuerza_serie[mascara_tallos_en_alza] = np.clip(
                    fuerza_serie[mascara_tallos_en_alza] + factor_refuerzo * np.clip(delta_tallos / np.maximum(
                        np.abs(tallos_patron[mascara_tallos_en_alza]), 1.0), 0.0, 0.60),
                    0.0,
                    0.85
                )

    mascara_ajuste = fuerza_serie > 0
    if mascara_ajuste.any():
        if sn_alto:
            patron_ajustado[mascara_ajuste] = produccion_real[mascara_ajuste]
        else:
            patron_ajustado[mascara_ajuste] = (
                (1.0 - fuerza_serie[mascara_ajuste]) *
                patron_ajustado[mascara_ajuste]
                + fuerza_serie[mascara_ajuste] *
                produccion_real[mascara_ajuste]
            )

    if n_common > 0:
        patron_ajustado_full[:n_common] = patron_ajustado

    factor_descenso = 0.12 + (0.06 if sn_alto else 0.0)
    for idx in np.where(mask_positiva_alta)[0]:
        for offset in [10, 11, 12]:
            future_idx = idx + offset
            if future_idx < len(patron_ajustado):
                patron_ajustado[future_idx] = min(
                    patron_ajustado[future_idx],
                    patron_ajustado[idx] * (1.0 - factor_descenso)
                )

    for idx in np.where(mask_positiva_moderada)[0]:
        for offset in [10, 11, 12]:
            future_idx = idx + offset
            if future_idx < len(patron_ajustado):
                patron_ajustado[future_idx] = min(
                    patron_ajustado[future_idx],
                    patron_ajustado[idx] *
                    (1.0 - (0.08 + (0.04 if sn_alto else 0.0)))
                )

    trabajo = trabajo.copy()
    trabajo['Produccion_patron'] = patron_ajustado_full
    trabajo['porcentaje_ajuste_modelo'] = np.nan

    if 'Tallos/m2' in trabajo.columns and 'Tallos_m2_patron' in trabajo.columns:
        tallos_real_full = np.asarray(
            pd.to_numeric(trabajo['Tallos/m2'], errors='coerce'),
            dtype=float,
        )
        tallos_patron_full = np.asarray(
            pd.to_numeric(trabajo['Tallos_m2_patron'], errors='coerce'),
            dtype=float,
        )
        tallos_real = tallos_real_full[:n_common] if n_common > 0 else np.array(
            [], dtype=float)
        tallos_patron = tallos_patron_full[:n_common] if n_common > 0 else np.array([
        ], dtype=float)
        tallos_ajustados = tallos_patron.copy()
        tallos_ajustados_full = tallos_patron_full.copy()
        if mascara_ajuste.any():
            if sn_alto:
                tallos_ajustados[mascara_ajuste] = tallos_real[mascara_ajuste]
            else:
                tallos_ajustados[mascara_ajuste] = (
                    (1.0 - fuerza_serie[mascara_ajuste]) *
                    tallos_ajustados[mascara_ajuste]
                    + fuerza_serie[mascara_ajuste] *
                    tallos_real[mascara_ajuste]
                )
        factor_descenso_tallos = 0.12 + (0.06 if sn_alto else 0.0)
        for idx in np.where(mask_positiva_alta)[0]:
            for offset in [10, 11, 12]:
                future_idx = idx + offset
                if future_idx < len(tallos_ajustados):
                    tallos_ajustados[future_idx] = min(
                        tallos_ajustados[future_idx],
                        tallos_ajustados[idx] * (1.0 - factor_descenso_tallos)
                    )
        for idx in np.where(mask_positiva_moderada)[0]:
            for offset in [10, 11, 12]:
                future_idx = idx + offset
                if future_idx < len(tallos_ajustados):
                    tallos_ajustados[future_idx] = min(
                        tallos_ajustados[future_idx],
                        tallos_ajustados[idx] *
                        (1.0 - (0.08 + (0.04 if sn_alto else 0.0)))
                    )
        tallos_ajustados_full[:n_common] = tallos_ajustados
        trabajo['Tallos_m2_patron'] = tallos_ajustados_full

    if n_common > 0:
        porcentaje_ajuste_full = np.full(len(trabajo), np.nan, dtype=float)
        porcentaje_base = np.abs(produccion_patron - produccion_real)
        porcentaje_base = np.where(
            np.isfinite(produccion_patron) & np.isfinite(produccion_real),
            (porcentaje_base / np.maximum(np.abs(produccion_real), 1.0)) * 100.0,
            np.nan,
        )

        factor_tallos = np.ones(n_common, dtype=float)
        if 'Tallos/m2' in trabajo.columns and 'Tallos_m2_patron' in trabajo.columns:
            tallos_real_full = np.asarray(
                pd.to_numeric(trabajo['Tallos/m2'], errors='coerce'),
                dtype=float,
            )
            tallos_patron_full = np.asarray(
                pd.to_numeric(trabajo['Tallos_m2_patron'], errors='coerce'),
                dtype=float,
            )
            tallos_real = tallos_real_full[:n_common] if n_common > 0 else np.array(
                [], dtype=float)
            tallos_patron = tallos_patron_full[:n_common] if n_common > 0 else np.array([
            ], dtype=float)
            anio_valores = pd.to_numeric(trabajo.get(
                'Anio', pd.Series(np.nan)), errors='coerce')
            if len(anio_valores) == len(tallos_real):
                anio_truncado = np.asarray(anio_valores, dtype=float)
            elif len(anio_valores) > 0 and len(tallos_real) > 0:
                n_match = min(len(anio_valores), len(tallos_real))
                anio_truncado = np.asarray(
                    anio_valores.iloc[:n_match], dtype=float)
                if len(anio_truncado) < len(tallos_real):
                    anio_truncado = np.resize(anio_truncado, len(tallos_real))
            else:
                anio_truncado = np.full(len(tallos_real), np.nan, dtype=float)
            tallos_2025_real = tallos_real[anio_truncado == 2025]
            tallos_2026_real = tallos_real[anio_truncado == 2026]
            if tallos_2025_real.size and tallos_2026_real.size:
                media_2025_real = float(np.nanmean(tallos_2025_real))
                media_2026_real = float(np.nanmean(tallos_2026_real))
                if np.isfinite(media_2025_real) and np.isfinite(media_2026_real):
                    delta_tallos = media_2026_real - media_2025_real
                    if delta_tallos > 0:
                        factor_tallos = 1.0 + np.clip(
                            TALLOS_2026_BOOST * delta_tallos /
                            max(abs(media_2025_real), 1.0),
                            0.0,
                            0.75,
                        )
                    elif delta_tallos < 0:
                        factor_tallos = np.maximum(
                            0.6,
                            1.0 + delta_tallos /
                            max(abs(media_2025_real), 1.0),
                        )

        porcentaje_ajuste = np.clip(
            porcentaje_base * factor_tallos, 0.0, 500.0)
        porcentaje_ajuste_full[:n_common] = porcentaje_ajuste
        trabajo['porcentaje_ajuste_modelo'] = porcentaje_ajuste_full

    return trabajo


ajustar_patron_con_extremos_real = _ajustar_patron_con_extremos_real_modulo


def ajustar_prediccion_modelo_con_patron(
    pred_vals,
    proy_vals,
    eval_actual_df,
    patron_prediction_weight=0.0,
    sn_alto=False,
    residual_weight=0.0
):
    pred = np.array(pred_vals, dtype=float, copy=True)
    proy = np.array(proy_vals, dtype=float, copy=True)
    n_blend = min(len(pred), len(proy))
    if n_blend <= 0:
        return pred

    def _get_numeric_array(df, col_name):
        if df is None or df.empty:
            return np.full(0, np.nan, dtype=float)
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors='coerce').to_numpy(copy=False)
        return np.full(len(df), np.nan, dtype=float)

    produccion_hist = _get_numeric_array(eval_actual_df, 'Produccion')

    media_hist = float(np.nanmean(produccion_hist)
                       ) if produccion_hist.size else np.nan
    std_hist = float(np.nanstd(produccion_hist, ddof=0)
                     ) if produccion_hist.size else np.nan
    patron_weight = float(np.clip(patron_prediction_weight, 0.0, 0.55))
    real_weight = 0.30 if sn_alto else 0.16
    residual_weight = float(np.clip(residual_weight, 0.0, 0.60))

    for i in range(n_blend):
        if i == 0:
            continue

        valor_real = produccion_hist[i] if i < len(produccion_hist) else np.nan

        if np.isfinite(valor_real) and np.isfinite(media_hist) and np.isfinite(std_hist) and std_hist > 0:
            z_score = (float(valor_real) - media_hist) / std_hist
            if abs(z_score) > 1.0:
                pred[i] = (1.0 - real_weight) * pred[i] + \
                    real_weight * float(valor_real)

        if np.isfinite(proy[i]) and patron_weight > 0:
            pred[i] = (1.0 - patron_weight) * pred[i] + patron_weight * proy[i]

    if produccion_hist.size:
        n_residual = min(len(pred), len(produccion_hist))
        if n_residual > 0:
            if residual_weight > 0:
                pred[1:n_residual] = (
                    (1.0 - residual_weight) * pred[1:n_residual]
                    + residual_weight * produccion_hist[1:n_residual]
                )

            if n_residual >= 4:
                refuerzo_ultimas = 0.08
                ultimas_idx = slice(n_residual - 4, n_residual)
                if n_residual > 4:
                    pred[ultimas_idx] = pred[ultimas_idx] * \
                        (1.0 + refuerzo_ultimas)
                else:
                    pred[ultimas_idx] = pred[ultimas_idx]

    return pred


def construir_objetivo_entrenamiento_con_patron(
    entrenamiento_df,
    patron_train_target_weight=0.0
):
    produccion_real = pd.to_numeric(
        entrenamiento_df.get('Produccion', pd.Series(dtype=float)),
        errors='coerce'
    ).to_numpy(dtype=float)
    if 'Produccion_patron' not in entrenamiento_df.columns:
        return produccion_real

    produccion_patron = pd.to_numeric(
        entrenamiento_df['Produccion_patron'],
        errors='coerce'
    ).to_numpy(dtype=float)
    peso_patron_objetivo = float(
        np.clip(patron_train_target_weight, 0.0, 0.80))
    if peso_patron_objetivo <= 0:
        return produccion_real

    return (
        (1.0 - peso_patron_objetivo) * produccion_real
        + peso_patron_objetivo * produccion_patron
    )


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

llm_ok, llm_estado = estado_configuracion_llm()
if llm_ok:
    st.caption(f'Conexion IA activa: {llm_estado}')
else:
    st.warning(f'Conexion IA pendiente: {llm_estado}')

if 'base_proyeccion_anthropic' not in st.session_state:
    st.session_state['base_proyeccion_anthropic'] = pd.DataFrame()
if 'respuesta_pregunta_claude' not in st.session_state:
    st.session_state['respuesta_pregunta_claude'] = ''
if 'error_pregunta_claude' not in st.session_state:
    st.session_state['error_pregunta_claude'] = ''
if 'dashboard_finca_activo' not in st.session_state:
    st.session_state['dashboard_finca_activo'] = False
if 'dashboard_archivo_id' not in st.session_state:
    st.session_state['dashboard_archivo_id'] = ''
if 'archivo_anthropic_cargado' not in st.session_state:
    st.session_state['archivo_anthropic_cargado'] = None
if 'archivo_anthropic_id' not in st.session_state:
    st.session_state['archivo_anthropic_id'] = ''
if 'estado_subida_anthropic' not in st.session_state:
    st.session_state['estado_subida_anthropic'] = ''
if 'dashboard_export_bytes' not in st.session_state:
    st.session_state['dashboard_export_bytes'] = None
if 'dashboard_export_name' not in st.session_state:
    st.session_state['dashboard_export_name'] = ''
if 'dashboard_export_mime' not in st.session_state:
    st.session_state['dashboard_export_mime'] = ''
if 'archivo_sesion_df' not in st.session_state:
    st.session_state['archivo_sesion_df'] = pd.DataFrame()
if 'archivo_sesion_nombre' not in st.session_state:
    st.session_state['archivo_sesion_nombre'] = ''
if 'tabla_mse_patron_masivo' not in st.session_state:
    st.session_state['tabla_mse_patron_masivo'] = pd.DataFrame()
if 'mostrar_dashboard_ia' not in st.session_state:
    st.session_state['mostrar_dashboard_ia'] = False


def preparar_estado_para_nuevo_archivo_base(state=None, nuevo_archivo_id=None, preservar_archivo_sesion=True):
    """Resetea solo el estado asociado a la base proyectada, preservando el contexto sincronizado."""
    target_state = st.session_state if state is None else state
    if nuevo_archivo_id is not None:
        target_state['dashboard_archivo_id'] = nuevo_archivo_id
    target_state['dashboard_finca_activo'] = False
    target_state['base_proyeccion_anthropic'] = pd.DataFrame()
    target_state['dashboard_export_bytes'] = None
    target_state['dashboard_export_name'] = ''
    target_state['dashboard_export_mime'] = ''
    if not preservar_archivo_sesion:
        target_state['archivo_sesion_df'] = pd.DataFrame()
        target_state['archivo_sesion_nombre'] = ''
    return target_state


def consulta_solicita_dashboard(texto_consulta):
    texto = (texto_consulta or '').strip().lower()
    if not texto:
        return False

    palabras_clave = [
        'dashboard',
        'tablero',
        'panel',
        'grafico',
        'graficos',
        'visualizacion',
        'indicador',
        'indicadores',
        'kpi',
        'kpis'
    ]
    return any(palabra in texto for palabra in palabras_clave)


@st.fragment
def render_subida_archivo_anthropic(file_path):
    proveedor = obtener_llm_provider()
    _ = file_path
    titulo = (
        'Sincronizar archivo con Anthropic'
        if proveedor == 'anthropic'
        else 'Sincronizar archivo con GitHub Models/OpenAI'
    )
    mensaje = (
        'Puedes sincronizar cualquier archivo local o de red para analisis.'
        if proveedor == 'anthropic'
        else 'Puedes sincronizar cualquier archivo local o de red como contexto '
             'para preguntas y resumenes con GitHub Models/OpenAI.'
    )

    with st.expander(titulo):
        st.caption(mensaje)

        st.write('**Seleccionar archivo a sincronizar**')
        archivo_personalizado = st.file_uploader(
            'Selecciona cualquier archivo',
            key='archivo_personalizado_anthropic'
        )
        if archivo_personalizado is not None:
            st.caption(f'Archivo seleccionado: {archivo_personalizado.name}')
            if st.button('Sincronizar archivo seleccionado', key='btn_subir_personalizado'):
                try:
                    df_personalizado = cargar_archivo_a_dataframe(
                        archivo_personalizado
                    )
                    info_carga = sincronizar_archivo_llm(
                        archivo_personalizado,
                        dataframe=df_personalizado,
                        nombre_archivo=getattr(
                            archivo_personalizado,
                            'name',
                            'archivo_sincronizado'
                        )
                    )
                    archivo_actual_id = (
                        getattr(archivo_personalizado, 'name', '')
                        + str(archivo_personalizado.size if hasattr(archivo_personalizado, 'size') else '')
                    )
                    registrar_sincronizacion_en_sesion(
                        info_carga,
                        df_personalizado,
                        getattr(archivo_personalizado, 'name',
                                'archivo_sincronizado'),
                        archivo_actual_id
                    )
                except Exception as exc:
                    st.session_state['estado_subida_anthropic'] = (
                        'error', str(exc))

        st.divider()
        st.caption(
            'La proyeccion individual y la proyeccion masiva se sincronizan '
            'automaticamente cuando se generan desde el modelo.'
        )

        # Mostrar estado
        st.divider()
        estado_subida = st.session_state.get('estado_subida_anthropic')
        if isinstance(estado_subida, tuple) and len(estado_subida) == 2:
            tipo, detalle = estado_subida
            if tipo == 'ok':
                st.success(f'✓ {detalle}')
            elif tipo == 'error':
                st.error(f'✗ Error: {detalle}')


@st.cache_data(show_spinner=False)
def leer_excel_subido(archivo_excel):
    df = pd.read_excel(archivo_excel)
    # Optimización: convertir columnas numéricas a tipos más eficientes
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            # Solo convertir si no hay NaN
            if df[col].notna().all():
                df[col] = df[col].astype('float32')
    return df


def construir_prompt_dashboard_anthropic(df_base, base_modelo, instruccion_extra=''):
    def top_resumen(df_origen, columna, valor_col='Produccion', top_n=5):
        if columna is None or columna not in df_origen.columns:
            return 'N/D'
        trabajo = df_origen.copy()
        trabajo[columna] = trabajo[columna].astype(str)
        if valor_col in trabajo.columns:
            out = (
                trabajo.groupby(columna, as_index=False)
                .agg(Valor=(valor_col, lambda s: pd.to_numeric(s, errors='coerce').sum(skipna=True)))
                .sort_values('Valor', ascending=False)
                .head(top_n)
            )
        else:
            out = trabajo[columna].value_counts().head(top_n).reset_index()
            out.columns = [columna, 'Valor']
        return out.to_csv(index=False)

    resumen = [
        'Genera un resumen ejecutivo corto del dashboard agricola.',
        'Separa la respuesta por: ANIO, SEMANAS, FINCA, PRODUCTO, VARIEDAD y DESVIACIONES.',
        'Responde en espanol, directo y accionable.'
    ]

    if instruccion_extra and instruccion_extra.strip():
        resumen.append(f'Instruccion adicional: {instruccion_extra.strip()}')

    resumen.append(
        f'GENERAL: registros={len(df_base)}, fincas={int(df_base["Finca"].nunique()) if "Finca" in df_base.columns else 0}, '
        f'variedades={int(df_base["Bloque&Varid"].nunique()) if "Bloque&Varid" in df_base.columns else 0}'
    )

    if {'Anio', 'Semana'}.issubset(df_base.columns):
        serie = df_base.copy()
        serie['Anio'] = pd.to_numeric(serie['Anio'], errors='coerce')
        serie['Semana'] = pd.to_numeric(serie['Semana'], errors='coerce')
        serie = serie.dropna(subset=['Anio', 'Semana'])
        if not serie.empty:
            resumen.append('ANIO:')
            resumen.append(
                serie.groupby('Anio', as_index=False)
                .agg(Produccion=('Produccion', lambda s: pd.to_numeric(s, errors='coerce').sum(skipna=True))
                     if 'Produccion' in serie.columns else ('Anio', 'size'))
                .sort_values('Anio')
                .tail(8)
                .to_csv(index=False)
            )
            serie['Anio_Semana'] = serie.apply(
                lambda r: f"{int(r['Anio'])}-{int(r['Semana']):02d}", axis=1
            )
            resumen.append('SEMANAS:')
            resumen.append(
                serie.groupby('Anio_Semana', as_index=False)
                .agg(Produccion=('Produccion', lambda s: pd.to_numeric(s, errors='coerce').sum(skipna=True))
                     if 'Produccion' in serie.columns else ('Anio_Semana', 'size'))
                .sort_values('Anio_Semana')
                .tail(12)
                .to_csv(index=False)
            )

    resumen.append('FINCA:')
    resumen.append(top_resumen(df_base, 'Finca'))
    resumen.append('PRODUCTO:')
    resumen.append(top_resumen(df_base, 'Producto'))
    col_var = 'Bloque&Varid' if 'Bloque&Varid' in df_base.columns else (
        'Variedad' if 'Variedad' in df_base.columns else None)
    resumen.append('VARIEDAD:')
    resumen.append(top_resumen(df_base, col_var))

    if base_modelo is not None and not base_modelo.empty:
        resumen.append(f'PROYECCION: filas={len(base_modelo)}')

    resumen.append(
        'DESVIACIONES: indica si el modelo sobreestima o subestima y en que frentes.')
    return '\n'.join(resumen)


def render_dashboard_base(df_base, usar_expander=True):
    contenedor_dashboard = (
        st.expander('Dashboard de la base', expanded=False)
        if usar_expander else st.container()
    )
    with contenedor_dashboard:
        if 'Anio' not in df_base.columns:
            st.info('No hay columna Anio para separar acumulados.')
            return

        trabajo = df_base.copy()
        trabajo['Anio'] = pd.to_numeric(trabajo['Anio'], errors='coerce')
        trabajo = trabajo.dropna(subset=['Anio']).copy()
        trabajo['Anio'] = trabajo['Anio'].astype(int)

        st.subheader('Totales del modelo vs produccion')
        col_finca = 'Finca' if 'Finca' in trabajo.columns else None
        col_producto = 'Producto' if 'Producto' in trabajo.columns else None
        col_var = 'Bloque&Varid' if 'Bloque&Varid' in trabajo.columns else (
            'Variedad' if 'Variedad' in trabajo.columns else None)

        base_modelo = st.session_state.get('base_proyeccion_anthropic')
        col_var = 'Bloque&Varid' if 'Bloque&Varid' in trabajo.columns else (
            'Variedad' if 'Variedad' in trabajo.columns else None
        )
        if (
            base_modelo is not None
            and not base_modelo.empty
            and col_var is not None
            and 'Produccion' in trabajo.columns
            and {'Variedad_proyectada', 'Anio_Semana', 'Estimado_modelo'}.issubset(base_modelo.columns)
            and {'Anio', 'Semana'}.issubset(trabajo.columns)
        ):
            columnas_real = [col_var, 'Anio', 'Semana', 'Produccion']
            if col_finca is not None:
                columnas_real.append(col_finca)
            if col_producto is not None:
                columnas_real.append(col_producto)

            real_tmp = trabajo[columnas_real].copy()
            real_tmp['Anio'] = pd.to_numeric(real_tmp['Anio'], errors='coerce')
            real_tmp['Semana'] = pd.to_numeric(
                real_tmp['Semana'], errors='coerce')
            real_tmp = real_tmp.dropna(subset=['Anio', 'Semana'])
            real_tmp['Anio'] = real_tmp['Anio'].astype(int)
            real_tmp['Semana'] = real_tmp['Semana'].astype(int)
            real_tmp['Anio_Semana'] = real_tmp.apply(
                lambda r: f"{int(r['Anio'])}-{int(r['Semana']):02d}", axis=1
            )
            real_tmp = real_tmp.rename(
                columns={col_var: 'Variedad_proyectada'})
            real_tmp['Variedad_proyectada'] = real_tmp['Variedad_proyectada'].astype(
                str)

            modelo_tmp = base_modelo[[
                'Variedad_proyectada', 'Anio_Semana', 'Estimado_modelo'
            ]].copy()
            modelo_tmp['Variedad_proyectada'] = modelo_tmp['Variedad_proyectada'].astype(
                str)

            columnas_merge = ['Variedad_proyectada',
                              'Anio_Semana', 'Produccion', 'Anio']
            if col_finca is not None:
                columnas_merge.append(col_finca)
            if col_producto is not None:
                columnas_merge.append(col_producto)

            comparativo = modelo_tmp.merge(
                real_tmp[columnas_merge],
                on=['Variedad_proyectada', 'Anio_Semana'],
                how='left'
            )
            comparativo['Estimado_modelo'] = pd.to_numeric(
                comparativo['Estimado_modelo'], errors='coerce'
            ).fillna(0)
            comparativo['Produccion'] = pd.to_numeric(
                comparativo['Produccion'], errors='coerce'
            ).fillna(0)

            comparativo['Variedad_mostrar'] = comparativo['Variedad_proyectada'].astype(
                str
            )

            st.markdown('**Filtros dinamicos (pop menu)**')
            filtro_cols = st.columns(4)

            opciones_anio = ['Todos'] + [
                str(x) for x in sorted(
                    comparativo['Anio'].dropna().astype(int).unique().tolist()
                )
            ]
            anio_sel = filtro_cols[0].selectbox(
                'Año', opciones_anio, key='dash_filtro_anio'
            )

            if col_finca is not None:
                opciones_finca = ['Todos'] + sorted(
                    comparativo[col_finca].dropna().astype(
                        str).unique().tolist()
                )
                finca_sel = filtro_cols[1].selectbox(
                    'Finca', opciones_finca, key='dash_filtro_finca'
                )
            else:
                filtro_cols[1].caption('Finca no disponible')
                finca_sel = 'Todos'

            if col_producto is not None:
                opciones_producto = ['Todos'] + sorted(
                    comparativo[col_producto].dropna().astype(
                        str).unique().tolist()
                )
                producto_sel = filtro_cols[2].selectbox(
                    'Producto', opciones_producto, key='dash_filtro_producto'
                )
            else:
                filtro_cols[2].caption('Producto no disponible')
                producto_sel = 'Todos'

            opciones_variedad = ['Todos'] + sorted(
                comparativo['Variedad_mostrar'].dropna().astype(
                    str).unique().tolist()
            )
            variedad_sel = filtro_cols[3].selectbox(
                'Variedad', opciones_variedad, key='dash_filtro_variedad'
            )

            comparativo_filtrado = comparativo.copy()
            if anio_sel != 'Todos':
                comparativo_filtrado = comparativo_filtrado[
                    comparativo_filtrado['Anio'].astype(int) == int(anio_sel)
                ]
            if col_finca is not None and finca_sel != 'Todos':
                comparativo_filtrado = comparativo_filtrado[
                    comparativo_filtrado[col_finca].astype(str) == finca_sel
                ]
            if col_producto is not None and producto_sel != 'Todos':
                comparativo_filtrado = comparativo_filtrado[
                    comparativo_filtrado[col_producto].astype(
                        str) == producto_sel
                ]
            if variedad_sel != 'Todos':
                comparativo_filtrado = comparativo_filtrado[
                    comparativo_filtrado['Variedad_mostrar'].astype(
                        str) == variedad_sel
                ]

            if comparativo_filtrado.empty:
                st.warning('No hay datos para los filtros seleccionados.')
            else:
                total_real = float(comparativo_filtrado['Produccion'].sum())
                total_modelo = float(
                    comparativo_filtrado['Estimado_modelo'].sum())
                brecha = (
                    (total_modelo - total_real) / total_real
                    if total_real != 0 else 0
                )
                m1, m2, m3 = st.columns(3)
                m1.metric('Total Produccion', f"{total_real:,.0f}")
                m2.metric('Total Modelo', f"{total_modelo:,.0f}")
                m3.metric('Brecha Modelo-Real', f"{brecha:.2%}")

                def render_totales_horizontal(df_src, columna_dim, titulo, top_n=15):
                    if columna_dim is None:
                        return
                    totales_dim = (
                        df_src
                        .dropna(subset=[columna_dim])
                        .groupby(columna_dim, as_index=False)
                        .agg(
                            Total_produccion=('Produccion', 'sum'),
                            Total_modelo=('Estimado_modelo', 'sum')
                        )
                    )
                    if columna_dim == 'Anio':
                        totales_dim = totales_dim.sort_values(columna_dim)
                    else:
                        totales_dim['Diferencia_abs'] = (
                            totales_dim['Total_modelo'] -
                            totales_dim['Total_produccion']
                        ).abs()
                        totales_dim = totales_dim.sort_values(
                            'Diferencia_abs', ascending=False
                        ).head(top_n)
                    if totales_dim.empty:
                        return

                    st.markdown(f'**{titulo}**')
                    fig_h = max(2.8, min(6, len(totales_dim) * 0.35))
                    fig, ax = plt.subplots(figsize=(10, fig_h))
                    y_pos = np.arange(len(totales_dim))
                    bar_h = 0.08
                    ax.barh(
                        y_pos - bar_h / 2,
                        totales_dim['Total_produccion'],
                        bar_h,
                        label='Produccion Real',
                        color='blue',
                        alpha=0.85
                    )
                    ax.barh(
                        y_pos + bar_h / 2,
                        totales_dim['Total_modelo'],
                        bar_h,
                        label='Estimado Modelo',
                        color='red',
                        alpha=0.85
                    )
                    ax.set_xlabel('Total', fontsize=11, fontweight='bold')
                    ax.set_ylabel(titulo, fontsize=11, fontweight='bold')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(totales_dim[columna_dim].astype(str))
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

                render_totales_horizontal(
                    comparativo_filtrado, 'Anio', 'Totales por Año', top_n=8)
                render_totales_horizontal(
                    comparativo_filtrado, col_finca, 'Totales por Finca', top_n=8)
                render_totales_horizontal(
                    comparativo_filtrado, col_producto, 'Totales por Producto', top_n=8)
                render_totales_horizontal(
                    comparativo_filtrado, 'Variedad_mostrar', 'Totales por Variedad', top_n=8)
        else:
            st.info(
                'Corre PROYECTAR FINCA para visualizar graficos comparativos del modelo.')

        st.divider()


@st.fragment
def render_preguntas_claude(df_base, selected_finca):
    with st.expander('Preguntas a la IA (Anthropic/GitHub Models/OpenAI)', expanded=True):
        st.caption(
            'Consulta sobre variedades, fincas, usos y precios, segun datos cargados.'
        )

        if st.session_state.get('mostrar_dashboard_ia', False):
            render_dashboard_base(df_base, usar_expander=False)
            st.divider()

        st.write()
        with st.form('form_pregunta_IA', clear_on_submit=True):
            pregunta_negocio = st.text_area(
                'Escribe tu pregunta',
                key='pregunta_negocio_anthropic'
            )
            enviar_pregunta = st.form_submit_button('Preguntale a la IA')

        if enviar_pregunta:
            st.session_state['mostrar_dashboard_ia'] = consulta_solicita_dashboard(
                pregunta_negocio
            )
            try:
                respuesta_negocio = responder_pregunta_anthropic(
                    df_base,
                    pregunta_negocio,
                    selected_finca,
                    st.session_state.get('base_proyeccion_anthropic'),
                    st.session_state.get('archivo_sesion_df'),
                    st.session_state.get('archivo_sesion_nombre', '')
                )
                st.session_state['respuesta_pregunta_claude'] = respuesta_negocio
                st.session_state['error_pregunta_claude'] = ''
            except Exception as exc:
                st.session_state['error_pregunta_claude'] = str(exc)
                st.session_state['respuesta_pregunta_claude'] = ''

        if st.session_state.get('error_pregunta_claude'):
            st.error(
                f"No se pudo responder la pregunta: {st.session_state['error_pregunta_claude']}"
            )
        elif st.session_state.get('respuesta_pregunta_claude'):
            st.success('Respuesta generada.')
            st.write(st.session_state['respuesta_pregunta_claude'])


file_path = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
if file_path is not None:
    df = leer_excel_subido(file_path)
    mostrar_analisis_avanzado = False
    archivo_actual_id = (
        getattr(file_path, 'name', '')
        + str(file_path.size if hasattr(file_path, 'size') else '')
    )

    if st.session_state.get('dashboard_archivo_id') != archivo_actual_id:
        preparar_estado_para_nuevo_archivo_base(
            st.session_state,
            archivo_actual_id,
            preservar_archivo_sesion=True
        )

    if 'Finca' not in df.columns:
        st.warning(
            'La columna requerida "Finca" no existe en este archivo. '
            'Puedes usar sincronizacion y consultas con la IA, '
            'pero no proyeccion/dashboard con este esquema.'
        )
        st.caption('Columnas detectadas: ' + ', '.join(df.columns.tolist()))
        st.divider()
        st.markdown("<h3 style='text-align:center; margin-top:2rem;'>Análisis Avanzado</h3>",
                    unsafe_allow_html=True)
        render_subida_archivo_anthropic(file_path)
        render_preguntas_claude(df, None)
        st.stop()

    # Compatibilidad: algunas bases traen la variedad con otro encabezado.
    if 'Bloque&Varid' not in df.columns:
        col_alt_var = next(
            (
                col for col in ['Bloque&Variedad', 'BloqueVarid']
                if col in df.columns
            ),
            None
        )
        if col_alt_var is not None:
            df['Bloque&Varid'] = df[col_alt_var].astype(str)
        elif {'Bloque', 'Variedad'}.issubset(df.columns):
            df['Bloque&Varid'] = (
                df['Bloque'].astype(str).str.strip()
                + ' '
                + df['Variedad'].astype(str).str.strip()
            )
        else:
            st.error(
                'No se encontro la columna Bloque&Varid (ni equivalentes) '
                'requerida para proyectar.'
            )
            st.caption('Columnas detectadas: ' +
                       ', '.join(df.columns.tolist()))
            st.divider()
            st.markdown("<h3 style='text-align:center; margin-top:2rem;'>Análisis Avanzado</h3>",
                        unsafe_allow_html=True)
            render_subida_archivo_anthropic(file_path)
            render_preguntas_claude(df, None)
            st.stop()

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
    progreso_placeholder = st.empty()

    if st.session_state.get('dashboard_finca_activo') and not run_masiva:
        mostrar_analisis_avanzado = True
        st.divider()
        st.markdown("<h3 style='text-align:center; margin-top:2rem;'>Análisis Avanzado</h3>",
                    unsafe_allow_html=True)
        if st.button('Volver a casos individuales', key='btn_volver_individual'):
            st.session_state['tabla_mse_patron_masivo'] = pd.DataFrame()
            st.session_state['tabla_mse_patron'] = pd.DataFrame()
            st.session_state['base_proyeccion_anthropic'] = pd.DataFrame()
            st.session_state['dashboard_finca_activo'] = False
            st.session_state['dashboard_export_bytes'] = None
            st.session_state['dashboard_export_name'] = ''
            st.session_state['dashboard_export_mime'] = ''
            st.rerun()
        render_subida_archivo_anthropic(file_path)
        render_preguntas_claude(df, selected_finca)

    df_finca = df[df["Finca"].astype(str) == selected_finca].copy()

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
                patron_weekly = construir_patron_semanal(group)
                trabajo = (
                    df_variedad_objetivo[[
                        'Anio', 'Semana', 'Tallos/m2', 'Produccion']]
                    .dropna()
                    .reset_index(drop=True)
                )
                trabajo['Anio'] = pd.to_numeric(
                    trabajo['Anio'], errors='coerce')
                trabajo['Semana'] = pd.to_numeric(
                    trabajo['Semana'], errors='coerce')
                trabajo['Tallos/m2'] = pd.to_numeric(
                    trabajo['Tallos/m2'], errors='coerce')
                trabajo['Produccion'] = pd.to_numeric(
                    trabajo['Produccion'], errors='coerce')
                trabajo = trabajo.dropna(
                    subset=['Anio', 'Semana', 'Tallos/m2', 'Produccion']
                )
                trabajo['Anio'] = trabajo['Anio'].astype(int)
                trabajo['Semana'] = trabajo['Semana'].astype(int)
                trabajo = trabajo.sort_values(
                    ['Anio', 'Semana']).reset_index(drop=True)
                trabajo = trabajo.merge(
                    patron_weekly,
                    on=['Anio', 'Semana'],
                    how='left'
                )
                trabajo['Tallos_m2_patron'] = trabajo['Tallos_m2_patron'].fillna(
                    trabajo['Tallos/m2']
                )
                trabajo['Produccion_patron'] = trabajo['Produccion_patron'].fillna(
                    trabajo['Produccion']
                )
                sn_valor, mse_equivalente = calcular_sn_y_mse_equivalente(
                    trabajo)
                arr_list.append((name, mse, sn_valor, mse_equivalente))
            except Exception:
                continue

        if len(arr_list) < 2:
            raise ValueError('No hay suficientes patrones para comparar.')

        arr_list.sort(key=lambda x: x[1])
        patron_seleccionado = seleccionar_patron(arr_list, var_proy)
        sn_seleccionado = next(
            (item[2] for item in arr_list if str(
                item[0]).strip().upper() == patron_seleccionado.upper()),
            np.nan
        )
        mse_equivalente_seleccionado = next(
            (item[3] for item in arr_list if str(
                item[0]).strip().upper() == patron_seleccionado.upper()),
            np.nan
        )
        usar_patron = not (
            (
                pd.notna(sn_seleccionado)
                and np.isfinite(sn_seleccionado)
                and sn_seleccionado > 13.0
            )
            or (
                pd.notna(mse_equivalente_seleccionado)
                and np.isfinite(mse_equivalente_seleccionado)
                and mse_equivalente_seleccionado > 10**6.5
            )
        )
        return patron_seleccionado, usar_patron

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
            .agg({
                'Tallos/m2': 'mean',
                'Produccion': 'sum'
            })
            .rename(columns={
                'Tallos/m2': 'Tallos_m2_patron',
                'Produccion': 'Produccion_patron'
            })
            .sort_values(['Anio', 'Semana'])
            .reset_index(drop=True)
        )
        patron_weekly['Incremento_tallos_patron'] = (
            patron_weekly['Tallos_m2_patron'].diff().fillna(0.0)
        )
        patron_weekly['Incremento_produccion_patron'] = (
            patron_weekly['Produccion_patron'].diff().fillna(0.0)
        )
        return patron_weekly

    def excluir_ultimas_4_semanas(df_base, columnas_grupo=None):
        if df_base is None or df_base.empty:
            return df_base.copy()

        trabajo = df_base.copy()
        if 'Anio_Semana' in trabajo.columns:
            anio_semana = trabajo['Anio_Semana'].astype(
                str).str.split('-', n=1, expand=True)
            trabajo['__anio_tmp'] = pd.to_numeric(
                anio_semana[0], errors='coerce')
            trabajo['__semana_tmp'] = pd.to_numeric(
                anio_semana[1], errors='coerce')
        elif {'Anio', 'Semana'}.issubset(trabajo.columns):
            trabajo['__anio_tmp'] = pd.to_numeric(
                trabajo['Anio'], errors='coerce')
            trabajo['__semana_tmp'] = pd.to_numeric(
                trabajo['Semana'], errors='coerce')
        else:
            return trabajo

        validos = trabajo[
            trabajo['__anio_tmp'].notna() & trabajo['__semana_tmp'].notna()
        ].copy()
        invalidos = trabajo[
            ~(trabajo['__anio_tmp'].notna() & trabajo['__semana_tmp'].notna())
        ].copy()

        if validos.empty:
            return trabajo.drop(columns=['__anio_tmp', '__semana_tmp'], errors='ignore')

        grupos_validos = []
        if columnas_grupo:
            grupos_validos = [
                col for col in columnas_grupo if col in validos.columns
            ]

        if grupos_validos:
            orden_desc = validos.sort_values(
                grupos_validos + ['__anio_tmp', '__semana_tmp'],
                ascending=[True] * len(grupos_validos) + [False, False]
            )
            orden_desc['__rank_ultimas_tmp'] = (
                orden_desc.groupby(grupos_validos).cumcount() + 1
            )
        else:
            orden_desc = validos.sort_values(
                ['__anio_tmp', '__semana_tmp'],
                ascending=[False, False]
            )
            orden_desc['__rank_ultimas_tmp'] = np.arange(len(orden_desc)) + 1

        validos_filtrados = orden_desc[
            orden_desc['__rank_ultimas_tmp'] > 4
        ].copy()

        resultado = pd.concat(
            [validos_filtrados, invalidos],
            ignore_index=True,
            sort=False
        )
        resultado = resultado.drop(
            columns=['__anio_tmp', '__semana_tmp', '__rank_ultimas_tmp'],
            errors='ignore'
        )

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

    def construir_tabla_mse_patron(df_export, finca_val, variedad_val):
        columnas_necesarias = {
            'Anio_Semana',
            'Estimado_modelo',
            'Produccion_real',
            'Proy_patron',
            'Tallos_m2_patron'
        }
        if not columnas_necesarias.issubset(df_export.columns):
            return pd.DataFrame(
                columns=['Finca', 'Bloque&Varid', 'Anio', 'Log10_MSE', 'S/N']
            )

        tabla = df_export[[
            'Anio_Semana',
            'Estimado_modelo',
            'Produccion_real',
            'Proy_patron',
            'Tallos_m2_patron'
        ]].copy()
        tabla = excluir_ultimas_4_semanas(tabla)
        tabla['Anio'] = pd.to_numeric(
            tabla['Anio_Semana'].astype(str).str.split('-').str[0],
            errors='coerce'
        )
        tabla['Estimado_modelo'] = pd.to_numeric(
            tabla['Estimado_modelo'], errors='coerce')
        tabla['Produccion_real'] = pd.to_numeric(
            tabla['Produccion_real'], errors='coerce')
        tabla['Proy_patron'] = pd.to_numeric(
            tabla['Proy_patron'], errors='coerce')
        tabla['Tallos_m2_patron'] = pd.to_numeric(
            tabla['Tallos_m2_patron'], errors='coerce')
        tabla = tabla.dropna(subset=['Anio'])
        tabla['Anio'] = tabla['Anio'].astype(int)
        peso_patron_mse = PATRON_PREDICTION_WEIGHT
        peso_modelo_mse = 1.0 - PATRON_PREDICTION_WEIGHT
        tabla['promedio_ponderado_modelo_patron'] = (
            (
                tabla['Estimado_modelo'] * peso_modelo_mse
                + tabla['Proy_patron'] * peso_patron_mse
            )
            / (peso_modelo_mse + peso_patron_mse)
        )

        resultados = []
        for anio_val, grupo in tabla.groupby('Anio'):
            pares_validos = grupo[[
                'promedio_ponderado_modelo_patron', 'Produccion_real'
            ]].dropna()
            tallos_m2_anual = pd.to_numeric(
                grupo['Tallos_m2_patron'],
                errors='coerce'
            ).mean()
            if pares_validos.empty:
                mse_valor = np.nan
            else:
                mse_valor = mean_squared_error(
                    pares_validos['promedio_ponderado_modelo_patron'],
                    pares_validos['Produccion_real']
                )

            resultados.append({
                'Finca': finca_val,
                'Bloque&Varid': variedad_val,
                'Anio': int(anio_val),
                'MSE': mse_valor,
                'Tallos_m2_anual': tallos_m2_anual
            })

        tabla_mse = pd.DataFrame(resultados)
        if not tabla_mse.empty:
            tabla_mse['MSE'] = pd.to_numeric(
                tabla_mse['MSE'], errors='coerce')
            tabla_mse['Tallos_m2_anual'] = pd.to_numeric(
                tabla_mse['Tallos_m2_anual'], errors='coerce')
            tabla_mse['Log10_MSE'] = np.where(
                tabla_mse['MSE'] > 0,
                np.log10(tabla_mse['MSE']),
                np.nan
            )
            tabla_mse['Log10_MSE'] = tabla_mse['Log10_MSE'].round(3)

            # S/N unico solicitado: log10((MSE_2026 - MSE_2025)^2)
            tabla_mse = tabla_mse.sort_values('Anio').reset_index(drop=True)
            tabla_mse['S/N'] = np.nan

            fila_2025 = tabla_mse[tabla_mse['Anio'] == 2025]
            fila_2026 = tabla_mse[tabla_mse['Anio'] == 2026]
            if not fila_2025.empty and not fila_2026.empty:
                mse_2025 = pd.to_numeric(
                    fila_2025['MSE'].iloc[0], errors='coerce')
                mse_2026 = pd.to_numeric(
                    fila_2026['MSE'].iloc[0], errors='coerce')

                if (
                    pd.notna(mse_2025)
                    and pd.notna(mse_2026)
                ):
                    diferencia_mse_cuadrada = (mse_2026 - mse_2025) ** 2
                    if diferencia_mse_cuadrada > 0:
                        sn_valor = np.log10(diferencia_mse_cuadrada)
                        tabla_mse.loc[
                            tabla_mse['Anio'].isin([2025, 2026]),
                            'S/N'
                        ] = sn_valor

            tabla_mse['S/N'] = pd.to_numeric(
                tabla_mse['S/N'], errors='coerce'
            ).round(3)

            tabla_mse = tabla_mse.sort_values(
                ['Anio', 'MSE'],
                ascending=[False, False],
                na_position='last'
            ).reset_index(drop=True)
            tabla_mse = tabla_mse[[
                'Finca', 'Bloque&Varid', 'Anio', 'Log10_MSE', 'S/N'
            ]]
        return tabla_mse

    def estilizar_tabla_mse_alerta(df_tabla):
        if df_tabla is None or df_tabla.empty:
            return df_tabla

        col_sn = None
        for col in ['S/N', 'S/N_prom_2025_2026']:
            if col in df_tabla.columns:
                col_sn = col
                break
        if col_sn is None:
            return df_tabla

        def estilo_sn(valor):
            valor_num = pd.to_numeric(valor, errors='coerce')
            if pd.notna(valor_num) and valor_num > 13:
                return 'background-color: #FFA500; color: #000000; font-weight: 700;'
            return ''

        return df_tabla.style.map(estilo_sn, subset=[col_sn])

    def resumir_indicador_promedio_2025_2026(df_tabla):
        columnas_salida = [
            'Finca', 'Bloque&Varid', 'S/N_prom_2025_2026'
        ]
        if df_tabla is None or df_tabla.empty:
            return pd.DataFrame(columns=columnas_salida)
        if not {'Finca', 'Bloque&Varid', 'Anio', 'Log10_MSE', 'S/N'}.issubset(df_tabla.columns):
            return pd.DataFrame(columns=columnas_salida)

        base = df_tabla.copy()
        base['Anio'] = pd.to_numeric(base['Anio'], errors='coerce')
        base['Log10_MSE'] = pd.to_numeric(base['Log10_MSE'], errors='coerce')
        base['S/N'] = pd.to_numeric(base['S/N'], errors='coerce')
        base = base[base['Anio'].isin([2025, 2026])].copy()
        if base.empty:
            return pd.DataFrame(columns=columnas_salida)

        resumen = (
            base
            .groupby(['Finca', 'Bloque&Varid'], as_index=False)
            .agg(
                S_N_prom=('S/N', 'mean')
            )
            .rename(columns={'S_N_prom': 'S/N_prom_2025_2026'})
        )
        resumen['S/N_prom_2025_2026'] = pd.to_numeric(
            resumen['S/N_prom_2025_2026'], errors='coerce'
        ).round(3)
        resumen = resumen.sort_values(
            ['S/N_prom_2025_2026', 'Finca', 'Bloque&Varid'],
            ascending=[False, True, True],
            na_position='last'
        ).reset_index(drop=True)
        return resumen

    def calcular_sn_y_mse_equivalente(trabajo):
        if trabajo is None or trabajo.empty:
            return np.nan, np.nan
        if 'Produccion' not in trabajo.columns or 'Produccion_patron' not in trabajo.columns:
            return np.nan, np.nan

        produccion_real = pd.to_numeric(trabajo['Produccion'], errors='coerce')
        produccion_patron = pd.to_numeric(
            trabajo['Produccion_patron'], errors='coerce')
        anio_valores = pd.to_numeric(
            trabajo.get('Anio', pd.Series(np.nan)), errors='coerce')

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

    def ajustar_patron_con_extremos_real(trabajo):
        return _ajustar_patron_con_extremos_real_modulo(trabajo)

    def preparar_dataset_modelo(df_variedad_base, patron_weekly, patron_feature_weight):
        trabajo = (
            df_variedad_base[['Anio', 'Semana', 'Tallos/m2', 'Produccion']]
            .dropna()
            .reset_index(drop=True)
        )
        trabajo['Anio'] = pd.to_numeric(trabajo['Anio'], errors='coerce')
        trabajo['Semana'] = pd.to_numeric(trabajo['Semana'], errors='coerce')
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
            patron_weekly,
            on=['Anio', 'Semana'],
            how='left'
        )
        trabajo['Tallos_m2_patron'] = trabajo['Tallos_m2_patron'].fillna(
            trabajo['Tallos/m2']
        )
        trabajo['Produccion_patron'] = trabajo['Produccion_patron'].fillna(
            trabajo['Produccion']
        )
        trabajo = ajustar_patron_con_extremos_real(trabajo)
        trabajo['Incremento_tallos_patron'] = trabajo[
            'Incremento_tallos_patron'
        ].fillna(0.0)
        trabajo['Incremento_produccion_patron'] = trabajo[
            'Incremento_produccion_patron'
        ].fillna(0.0)
        peso_patron = 0.4
        if np.isfinite(patron_feature_weight):
            peso_patron = float(np.clip(patron_feature_weight, 0.0, 1.0))
        if peso_patron <= 0:
            peso_patron = 0.0
        trabajo['Tallos_m2_patron_ponderado'] = (
            (1.0 - peso_patron) * trabajo['Tallos/m2']
            + peso_patron * trabajo['Tallos_m2_patron']
        )
        trabajo['Produccion_patron_ponderado'] = (
            (1.0 - peso_patron) * trabajo['Produccion']
            + peso_patron * trabajo['Produccion_patron']
        )

        sn_valor = calcular_sn_patron(trabajo)
        trabajo['sn_alto'] = (
            1.0 if (
                pd.notna(sn_valor)
                and np.isfinite(sn_valor)
                and sn_valor > 13.0
            ) else 0.0
        )
        return trabajo

    columnas_modelo = [
        'Tallos/m2',
        'Tallos_m2_patron',
        'Produccion_patron',
        'Tallos_m2_patron_ponderado',
        'Produccion_patron_ponderado',
        'Incremento_tallos_patron',
        'Incremento_produccion_patron',
        'sn_alto',
    ]

    def ajustar_prediccion_con_sensibilidad_picos(
        pred_vals,
        proy_vals,
        eval_actual_df,
        patron_prediction_weight,
        sn_alto=False,
        residual_weight=0.0
    ):
        return ajustar_prediccion_modelo_con_patron(
            pred_vals,
            proy_vals,
            eval_actual_df,
            patron_prediction_weight=patron_prediction_weight,
            sn_alto=sn_alto,
            residual_weight=residual_weight,
        )

    def preparar_features_base(df_base, cache_patrones=None):
        features_cache = {}
        if df_base is None or df_base.empty:
            return features_cache

        for variedad in df_base['Bloque&Varid'].dropna().astype(str).unique():
            try:
                df_variedad = df_base[df_base['Bloque&Varid'].astype(
                    str) == variedad].copy()
                patron_seleccionado, usar_patron_sin_dependencia = calcular_patron_compatible_individual(
                    df,
                    df_variedad,
                    variedad
                )
                patron_actual = df[df['Bloque&Varid'].astype(
                    str) == str(patron_seleccionado)].copy()
                patron_weekly = None
                if cache_patrones is not None and str(patron_seleccionado) in cache_patrones:
                    patron_weekly = cache_patrones[str(
                        patron_seleccionado)].copy()
                if patron_weekly is None:
                    patron_weekly = construir_patron_semanal(patron_actual)
                entrenamiento_df = preparar_dataset_modelo(
                    df_variedad,
                    patron_weekly,
                    0.0 if usar_patron_sin_dependencia else PATRON_FEATURE_WEIGHT
                )
                entrenamiento_df = entrenamiento_df[entrenamiento_df['Anio'] >= 2025].reset_index(
                    drop=True)
                entrenamiento_df = excluir_ultimas_4_semanas(entrenamiento_df)
                features_cache[variedad] = {
                    'df_variedad': df_variedad,
                    'patron_seleccionado': patron_seleccionado,
                    'usar_patron_sin_dependencia': usar_patron_sin_dependencia,
                    'patron_weekly': patron_weekly,
                    'entrenamiento_df': entrenamiento_df,
                }
            except Exception:
                continue
        return features_cache

    def proyectar_variedad_masiva(df_base, var_proy, cache_patrones=None, features_cache=None, config_proyeccion=None):
        df_filtered_ = df_base[df_base['Bloque&Varid'].isin([var_proy])].copy()
        if df_filtered_.empty:
            raise ValueError('Sin datos para la variedad seleccionada.')

        feature_data = None
        if features_cache is not None and str(var_proy) in features_cache:
            feature_data = features_cache[str(var_proy)]

        if feature_data is None:
            patron_seleccionado, usar_patron_sin_dependencia = calcular_patron_compatible_individual(
                df,
                df_filtered_,
                var_proy
            )
            df_patron = df[df['Bloque&Varid'].isin(
                [patron_seleccionado])]
            patron_actual = df[df['Bloque&Varid'].isin(
                [patron_seleccionado])].copy()
            patron_weekly = None
            if cache_patrones is not None and str(patron_seleccionado) in cache_patrones:
                patron_weekly = cache_patrones[str(patron_seleccionado)].copy()
            if patron_weekly is None:
                patron_weekly = construir_patron_semanal(patron_actual)
            entrenamiento_df = preparar_dataset_modelo(
                df_filtered_,
                patron_weekly,
                0.0 if usar_patron_sin_dependencia else PATRON_FEATURE_WEIGHT
            )
            entrenamiento_df = entrenamiento_df[entrenamiento_df['Anio'] >= 2025].reset_index(
                drop=True)
            entrenamiento_df = excluir_ultimas_4_semanas(entrenamiento_df)
            feature_data = {
                'df_variedad': df_filtered_,
                'patron_seleccionado': patron_seleccionado,
                'usar_patron_sin_dependencia': usar_patron_sin_dependencia,
                'patron_weekly': patron_weekly,
                'entrenamiento_df': entrenamiento_df,
                'df_patron': df_patron,
            }
        else:
            patron_seleccionado = feature_data['patron_seleccionado']
            usar_patron_sin_dependencia = feature_data['usar_patron_sin_dependencia']
            df_patron = df[df['Bloque&Varid'].astype(
                str) == str(patron_seleccionado)]
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
        patron_weekly = feature_data['patron_weekly']

        patron_feature_weight = 0.0 if usar_patron_sin_dependencia else PATRON_FEATURE_WEIGHT
        patron_prediction_weight = 0.0 if usar_patron_sin_dependencia else PATRON_PREDICTION_WEIGHT

        entrenamiento_df = feature_data['entrenamiento_df'].copy()

        prod_train = construir_objetivo_entrenamiento_con_patron(
            entrenamiento_df,
            PATRON_TRAIN_TARGET_WEIGHT
        )
        entrenamiento_df['Produccion_ajustada'] = prod_train

        eval_actual_df = preparar_dataset_modelo(
            y_actual,
            patron_weekly,
            patron_feature_weight
        )

        if len(entrenamiento_df) < 5 or len(eval_actual_df) == 0:
            raise ValueError(
                'No hay suficientes datos para entrenar/prediccion.')

        config_proyeccion = config_proyeccion or obtener_configuracion_proyeccion_masiva(
            ligera=False)
        columnas_modelo = config_proyeccion.get(
            'columnas_modelo', columnas_modelo)
        if config_proyeccion.get('modo_liviano'):
            entrenamiento_df = entrenamiento_df.tail(
                min(len(entrenamiento_df), config_proyeccion.get(
                    'limite_filas_entrenamiento', 40))
            ).reset_index(drop=True)

        x_train_df = entrenamiento_df[columnas_modelo].reset_index(drop=True)
        x_train_df['Semana_orden'] = np.arange(len(x_train_df), dtype=float)
        y_train_df = pd.DataFrame(
            entrenamiento_df['Produccion_ajustada']).reset_index(drop=True)

        x_frame = eval_actual_df[columnas_modelo].reset_index(drop=True)
        x_frame['Semana_orden'] = np.arange(len(x_frame), dtype=float)
        y_frame = pd.DataFrame(
            eval_actual_df['Produccion']).reset_index(drop=True)

        split_idx = len(x_train_df)
        split_idx = max(split_idx, 1)

        model_name = ''.join(
            ch if ch.isalnum() else '_' for ch in str(var_proy))
        train_key = f'entrenado_masivo_{model_name}_cal_v2'

        if train_key not in st.session_state:
            modelo = RandomForestRegressor(
                n_estimators=config_proyeccion.get('n_estimators', 100),
                random_state=42,
                max_depth=config_proyeccion.get('max_depth', 16),
                min_samples_leaf=config_proyeccion.get('min_samples_leaf', 1),
                min_samples_split=config_proyeccion.get(
                    'min_samples_split', 2),
                max_features=config_proyeccion.get('max_features', 'sqrt')
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
        prod_real_vals = y_frame.iloc[:len(proy_vals), 0].to_numpy(copy=True)
        pred_vals, proy_vals, prod_real_vals = alinear_series_para_ajuste(
            pred_vals,
            proy_vals,
            prod_real_vals,
        )
        media_real = float(np.nanmean(prod_real_vals)
                           ) if prod_real_vals.size else np.nan
        std_real = float(np.nanstd(prod_real_vals, ddof=0)
                         ) if prod_real_vals.size else np.nan
        if np.isfinite(media_real) and np.isfinite(std_real) and std_real > 0:
            z_score = (prod_real_vals - media_real) / std_real
            mask_extremo = np.abs(z_score) > 2.0
            mask_positiva_mod = (z_score >= 1.0) & (z_score <= 2.0)
            proy_vals_adj = proy_vals.copy()
            if mask_extremo.any():
                fuerza = np.clip(
                    0.20 + 0.15 * (np.abs(z_score[mask_extremo]) - 2.0), 0.20, 0.45)
                proy_vals_adj[mask_extremo] = (
                    (1.0 - fuerza) * proy_vals_adj[mask_extremo]
                    + fuerza * prod_real_vals[mask_extremo]
                )
            if mask_positiva_mod.any():
                fuerza_mod = 0.20
                proy_vals_adj[mask_positiva_mod] = (
                    (1.0 - fuerza_mod) * proy_vals_adj[mask_positiva_mod]
                    + fuerza_mod * prod_real_vals[mask_positiva_mod]
                )
                for idx in np.where(mask_positiva_mod)[0]:
                    for offset in [10, 11, 12]:
                        future_idx = idx + offset
                        if 0 <= future_idx < len(proy_vals_adj):
                            proy_vals_adj[future_idx] = min(
                                proy_vals_adj[future_idx],
                                proy_vals_adj[idx] * (1.0 - 0.12)
                            )
            proy_vals = proy_vals_adj

        if config_proyeccion.get('modo_liviano') or not config_proyeccion.get('usar_sensibilidad_picos', True):
            pred_vals = pred_vals
        else:
            sn_alto = bool(
                pd.notna(calcular_sn_patron(eval_actual_df))
                and calcular_sn_patron(eval_actual_df) > 13.0
            )
            pred_vals = ajustar_prediccion_con_sensibilidad_picos(
                pred_vals,
                proy_vals,
                eval_actual_df,
                patron_prediction_weight,
                sn_alto=sn_alto,
                residual_weight=AJUSTE_RESIDUAL_WEIGHT
            )

        prod_real_vals = y_frame.iloc[:len(pred_vals), 0].to_numpy()
        media_real = float(np.nanmean(prod_real_vals)
                           ) if prod_real_vals.size else np.nan
        media_modelo = float(np.nanmean(pred_vals)
                             ) if pred_vals.size else np.nan
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
            'Tallos_m2_patron': eval_actual_df['Tallos_m2_patron'].iloc[:n_export].values,
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

        df_metricas = excluir_ultimas_4_semanas(df_export)
        pares_modelo = df_metricas[[
            'Produccion_real', 'Estimado_modelo']].dropna()
        pares_patron = df_metricas[['Produccion_real', 'Proy_patron']].dropna()

        return {
            'df_export': df_export,
            'mse_modelo': mean_squared_error(
                pares_modelo['Produccion_real'],
                pares_modelo['Estimado_modelo']
            ) if not pares_modelo.empty else np.nan,
            'mse_patron': mean_squared_error(
                pares_patron['Produccion_real'],
                pares_patron['Proy_patron']
            ) if not pares_patron.empty else np.nan
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
        tablas_mse_masivo = []
        resumen = []
        errores = []
        progreso = progreso_placeholder.progress(0)
        cache_patrones = construir_cache_patrones_semanales(df_masivo)
        features_cache = preparar_features_base(
            df_masivo, cache_patrones=cache_patrones)
        config_proyeccion = obtener_configuracion_proyeccion_masiva(
            ligera=True)

        for i, var_item in enumerate(variedades_todas, start=1):
            try:
                resultado = proyectar_variedad_masiva(
                    df_masivo,
                    var_item,
                    cache_patrones=cache_patrones,
                    features_cache=features_cache,
                    config_proyeccion=config_proyeccion,
                )
                resultados_export.append(resultado['df_export'])
                tabla_mse_item = construir_tabla_mse_patron(
                    resultado['df_export'],
                    selected_finca,
                    var_item
                )
                if not tabla_mse_item.empty:
                    tablas_mse_masivo.append(tabla_mse_item)
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

        df_sn_por_caso = pd.DataFrame(columns=['Bloque&Varid', 'S/N'])
        if tablas_mse_masivo:
            df_mse_patron = pd.concat(tablas_mse_masivo, ignore_index=True)
            df_mse_patron = resumir_indicador_promedio_2025_2026(df_mse_patron)
            if 'S/N_prom_2025_2026' in df_mse_patron.columns:
                df_sn_por_caso = (
                    df_mse_patron[['Bloque&Varid', 'S/N_prom_2025_2026']]
                    .rename(columns={'S/N_prom_2025_2026': 'S/N'})
                    .drop_duplicates(subset=['Bloque&Varid'])
                    .reset_index(drop=True)
                )
            st.session_state['tabla_mse_patron_masivo'] = df_mse_patron.copy()
            with st.expander(
                'MSE entre calculo del modelo y patron seleccionado - masivo',
                expanded=True
            ):
                tabla_mse_visual = estilizar_tabla_mse_alerta(df_mse_patron)
                st.dataframe(
                    tabla_mse_visual,
                    use_container_width=True,
                    hide_index=True
                )
            st.session_state['tabla_mse_patron'] = df_mse_patron

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
        # Exportar solo registros del 2026.
        df_estimado_ordenado = df_estimado_ordenado[
            df_estimado_ordenado['__anio'] == 2026
        ].copy()
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

        # No exportar filas sin estimado o con estimado igual a 0.
        df_estimado_ordenado['Estimado_modelo'] = pd.to_numeric(
            df_estimado_ordenado['Estimado_modelo'], errors='coerce')
        df_estimado_ordenado = df_estimado_ordenado[
            df_estimado_ordenado['Estimado_modelo'].notna()
            & (df_estimado_ordenado['Estimado_modelo'] > 0)
        ].copy()
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
        st.session_state['dashboard_finca_activo'] = True

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
            df_mse_export = pd.DataFrame(resumen)
            if not df_mse_export.empty:
                df_mse_export = df_mse_export.rename(columns={
                    'Variedad_proyectada': 'Bloque&Varid',
                    'MSE_modelo': 'MSE'
                })
                df_mse_export = df_mse_export.merge(
                    df_sn_por_caso,
                    on='Bloque&Varid',
                    how='left'
                )
                columnas_mse = [
                    col for col in ['Bloque&Varid', 'MSE', 'MSE_proy_patron', 'S/N']
                    if col in df_mse_export.columns
                ]
                df_mse_export = df_mse_export[columnas_mse]
            else:
                df_mse_export = pd.DataFrame(
                    columns=['Bloque&Varid', 'MSE', 'MSE_proy_patron', 'S/N']
                )
            df_mse_export.to_excel(
                writer, sheet_name='MSE_por_BloqueVarid', index=False
            )
        st.download_button(
            'Exportar datos a Excel',
            data=buffer_masivo.getvalue(),
            file_name='Proyecto_todas_variedades.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key='descargar_masivo'
        )

        st.session_state['dashboard_export_bytes'] = buffer_masivo.getvalue()
        st.session_state['dashboard_export_name'] = 'Proyecto_todas_variedades.xlsx'
        st.session_state['dashboard_export_mime'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        sincronizar_export_generado_automatico(
            st.session_state['dashboard_export_bytes'],
            st.session_state['dashboard_export_name'],
            st.session_state['dashboard_export_mime'],
            dataframe=df_export_estimado,
            state=st.session_state
        )

        st.divider()
        st.markdown("<h3 style='text-align:center; margin-top:2rem;'>Análisis Avanzado</h3>",
                    unsafe_allow_html=True)
        render_subida_archivo_anthropic(file_path)
        render_preguntas_claude(df, selected_finca)

        st.stop()

    # Una sola lectura reutilizada para todo el flujo individual
    df = leer_excel_subido(file_path)

    # Seleccion individual ubicada antes del bloque de IA.
    df_finca_ind = df[df["Finca"].astype(str) == selected_finca].copy()
    variedades = sorted(
        df_finca_ind["Bloque&Varid"].dropna().astype(str).unique().tolist()
    )
    if len(variedades) == 0:
        st.warning('No hay Bloque&Varid disponibles para la finca seleccionada.')
        st.stop()
    selected_var = st.selectbox("Bloque&Variedad", variedades)

    var_proy = selected_var
    df_filtered_ = df[df['Bloque&Varid'].isin([var_proy])]

    # Seleccionar patrón una sola vez
    patron_seleccionado, usar_patron_sin_dependencia = calcular_patron_compatible_individual(
        df,
        df_filtered_,
        var_proy
    )

    # Preparar datos para el patrón
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
    # Reutilizar df ya cargado
    y_actual = df[df['Bloque&Varid'].isin([var_proy])].copy()
    patron_actual = df[df['Bloque&Varid'].isin(
        [patron_seleccionado])].copy()

    if patron_actual.empty:
        st.error('No hay datos del patron seleccionado para entrenar/prediccion.')
        st.stop()

    patron_weekly = construir_patron_semanal(patron_actual)

    # Pesos para priorizar el patron en entrenamiento y prediccion.
    patron_feature_weight = 0.0 if usar_patron_sin_dependencia else PATRON_FEATURE_WEIGHT
    patron_prediction_weight = 0.0 if usar_patron_sin_dependencia else PATRON_PREDICTION_WEIGHT
    entrenamiento_df = preparar_dataset_modelo(
        y_actual,
        patron_weekly,
        patron_feature_weight
    )
    # Entrenar con todo el historial desde 2025 en adelante (incluye 2026+).
    entrenamiento_df = entrenamiento_df[
        entrenamiento_df['Anio'] >= 2025
    ].reset_index(drop=True)
    entrenamiento_df = excluir_ultimas_4_semanas(entrenamiento_df)

    prod_train = construir_objetivo_entrenamiento_con_patron(
        entrenamiento_df,
        PATRON_TRAIN_TARGET_WEIGHT
    )
    entrenamiento_df['Produccion_ajustada'] = prod_train

    eval_actual_df = preparar_dataset_modelo(
        y_actual,
        patron_weekly,
        patron_feature_weight
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

    factor_correccion = 1.4
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

    x_train_df = entrenamiento_df[columnas_modelo].reset_index(drop=True)
    x_train_df['Semana_orden'] = np.arange(len(x_train_df), dtype=float)
    y_train_df = pd.DataFrame(
        entrenamiento_df['Produccion_ajustada']).reset_index(drop=True)

    if len(eval_actual_df) == 0:
        st.error(
            'No hay suficientes datos actuales validos para generar la evaluacion.')
        st.stop()

    x_frame = eval_actual_df[columnas_modelo].reset_index(drop=True)
    x_frame['Semana_orden'] = np.arange(len(x_frame), dtype=float)
    y_frame = pd.DataFrame(eval_actual_df['Produccion']).reset_index(drop=True)

    split_idx = len(x_train_df)
    split_idx = max(split_idx, 1)
    X_train = x_train_df.iloc[:split_idx]
    y_train = y_train_df.iloc[:split_idx]
    inicio_train = entrenamiento_df.iloc[0]
    fin_train = entrenamiento_df.iloc[split_idx - 1]

    model_name = ''.join(ch if ch.isalnum() else '_' for ch in str(var_proy))
    model_file = models_dir / f'rf_{model_name}.pkl'
    train_key = f'entrenado_{model_name}_cal_v2'

    if train_key not in st.session_state:
        modelo = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=20,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features='sqrt'
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

    # Regla agronomica (completamente vectorizada para velocidad)
    pred_vals = y_pred['Estimado_modelo'].to_numpy(copy=True)

    # Mezcla dirigida con la proyeccion del patron (vectorizada)
    proy_vals = proy.reset_index(drop=True).to_numpy(copy=True)
    sn_alto = bool(
        pd.notna(calcular_sn_patron(eval_actual_df))
        and calcular_sn_patron(eval_actual_df) > 13.0
    )
    pred_vals = ajustar_prediccion_con_sensibilidad_picos(
        pred_vals,
        proy_vals,
        eval_actual_df,
        patron_prediction_weight,
        sn_alto=sn_alto
    )

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

    y_modelo_plot = y_pred['Estimado_modelo'].reset_index(drop=True).to_numpy()
    y_produccion_plot = y_frame.iloc[:n_puntos,
                                     0].reset_index(drop=True).to_numpy()
    y_patron_plot = proy.iloc[:n_puntos].reset_index(drop=True).to_numpy()
    y_patron_ajustado_plot = np.asarray(proy_vals[:n_puntos]).reshape(-1)

    media_produccion_real = float(np.nanmean(y_produccion_plot))
    desv_produccion_real = float(np.nanstd(y_produccion_plot, ddof=0))
    lim_inf_produccion_real = media_produccion_real - desv_produccion_real
    lim_sup_produccion_real = media_produccion_real + desv_produccion_real

    fig, ax = plt.subplots(figsize=(12, 5))

    def plot_linea_segura(x_vals, y_vals, **kwargs):
        x_arr = np.asarray(list(x_vals)).reshape(-1)
        y_arr = np.asarray(y_vals).reshape(-1)
        if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
            ax.plot([], [], **kwargs)
            return False
        ax.plot(x_arr, y_arr, **kwargs)
        return True

    hubo_desajuste = False
    if not plot_linea_segura(
        x_pos, y_modelo_plot, label='Modelo', color='orange', linewidth=2
    ):
        hubo_desajuste = True
    if not plot_linea_segura(
        x_pos, y_produccion_plot, label='Produccion', color='red', linestyle='--'
    ):
        hubo_desajuste = True
    if not plot_linea_segura(
        x_pos, y_patron_plot, label='Proy_patron', color='green', linestyle='-'
    ):
        hubo_desajuste = True
    if not plot_linea_segura(
        x_pos, y_patron_ajustado_plot, color='purple', linewidth=2, linestyle='-.'
    ):
        hubo_desajuste = True

    ax.plot(
        list(x_pos),
        [media_produccion_real] * len(x_pos),
        color='gray',
        linewidth=1.0,
        alpha=0.7,
        label='Media Producción Real'
    )
    ax.fill_between(
        list(x_pos),
        [lim_inf_produccion_real] * len(x_pos),
        [lim_sup_produccion_real] * len(x_pos),
        color='gray',
        alpha=0.12,
        label='±1σ'
    )

    if hubo_desajuste:
        st.warning(
            'Se detecto un desajuste entre dimensiones de X/Y. '
            'Se dibujo linea vacia para evitar error de grafico.'
        )

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
    with st.expander('Promedio semanal Tallos/m2 por anio', expanded=False):
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

    df_mse_patron = construir_tabla_mse_patron(
        df_export,
        selected_finca,
        var_proy
    )
    df_mse_patron = resumir_indicador_promedio_2025_2026(df_mse_patron)
    with st.expander(
        'MSE entre calculo del modelo y patron seleccionado - individual',
        expanded=True
    ):
        tabla_mse_visual = estilizar_tabla_mse_alerta(df_mse_patron)
        st.dataframe(tabla_mse_visual,
                     use_container_width=True, hide_index=True)
    st.session_state['tabla_mse_patron'] = df_mse_patron

    df_metricas_ind = excluir_ultimas_4_semanas(df_export)
    pares_modelo_ind = df_metricas_ind[[
        'Produccion_real', 'Estimado_modelo']].dropna()
    pares_patron_ind = df_metricas_ind[[
        'Produccion_real', 'Proy_patron']].dropna()
    mse_modelo = mean_squared_error(
        pares_modelo_ind['Produccion_real'],
        pares_modelo_ind['Estimado_modelo']
    ) if not pares_modelo_ind.empty else np.nan
    mse_patron = mean_squared_error(
        pares_patron_ind['Produccion_real'],
        pares_patron_ind['Proy_patron']
    ) if not pares_patron_ind.empty else np.nan

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
                'valor': df_metricas_ind['Error_abs'].mean() if not df_metricas_ind.empty else np.nan},
            {'metrica': 'MAPE_modelo_pct',
                'valor': df_metricas_ind['Error_pct'].mean() if not df_metricas_ind.empty else np.nan},
            {'metrica': 'variedad_proyectada', 'valor': var_proy}
        ]).to_excel(writer, sheet_name='Resumen', index=False)

    st.download_button(
        'Exportar datos a Excel',
        data=buffer_individual.getvalue(),
        file_name='Proyecto.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        key='descargar_individual'
    )
    st.session_state['dashboard_export_bytes'] = buffer_individual.getvalue()
    st.session_state['dashboard_export_name'] = 'Proyecto.xlsx'
    st.session_state['dashboard_export_mime'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    sincronizar_export_generado_automatico(
        st.session_state['dashboard_export_bytes'],
        st.session_state['dashboard_export_name'],
        st.session_state['dashboard_export_mime']
    )
    y_pred_tail = y_pred.tail(4).round(0).copy()
    etiquetas_tail = etiquetas_anio_semana.iloc[:len(
        y_pred)].tail(len(y_pred_tail)).values
    y_pred_tail.index = etiquetas_tail
    st.write(y_pred_tail)

    if not mostrar_analisis_avanzado:
        st.divider()
        st.markdown("<h3 style='text-align:center; margin-top:2rem;'>Análisis Avanzado</h3>",
                    unsafe_allow_html=True)
        render_subida_archivo_anthropic(file_path)
        render_preguntas_claude(df, selected_finca)

else:
    st.info("Por favor, sube el archivo Excel.")
