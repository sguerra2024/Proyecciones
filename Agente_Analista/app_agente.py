"""
App Streamlit para probar y ajustar el AgenteAnalista.

Permite:
- Chatear con el agente viendo qué capítulos de teoría se usaron y el
  prompt exacto enviado al modelo.
- Analizar series de datos con pandas: el agente recibe estadísticas
  ya calculadas (media, mediana, desviación, min, max, conteo).
- Editar los capítulos de la base de conocimiento (crear, editar,
  borrar, importar .md).
- Ajustar el system prompt en vivo.

Ejecutar con:
    streamlit run Agente_Analista/app_agente.py
"""
from conocimiento import BaseConocimiento
from agent_manager import AgenteAnalista, calcular_estadisticas
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))


RUTA_BD = Path(__file__).resolve().parent / 'data' / 'teoria.db'


@st.cache_resource
def cargar_base():
    """Base de conocimiento compartida entre reruns de Streamlit."""
    return BaseConocimiento(RUTA_BD)


def obtener_agente():
    """Instancia el agente una vez por sesión.

    Devuelve (agente, error). Si falta la API key, devuelve (None, error)
    y las pestañas que no consultan al LLM siguen funcionando.
    """
    if 'agente' not in st.session_state:
        try:
            st.session_state['agente'] = AgenteAnalista(
                ruta_conocimiento=RUTA_BD)
        except RuntimeError as exc:
            st.session_state['agente_error'] = str(exc)
            st.session_state['agente'] = None
    return st.session_state['agente'], st.session_state.get('agente_error')


def consultar_con_traza(agente, pregunta):
    """
    Consulta al agente capturando el prompt exacto enviado al modelo,
    para mostrarlo en la pestaña de chat.
    """
    contexto_teoria = ''
    if agente.conocimiento is not None:
        contexto_teoria = agente.conocimiento.formatear_contexto(pregunta)
    partes = [contexto_teoria] if contexto_teoria else []
    pregunta_completa = ('\n\n'.join(partes) + f'\n\nPregunta: {pregunta}'
                         if partes else pregunta)

    respuesta = agente.cliente.chat.completions.create(
        model=agente.modelo,
        max_tokens=1024,
        messages=[
            {'role': 'system', 'content': agente.SYSTEM_PROMPT},
            {'role': 'user', 'content': pregunta_completa},
        ],
    )
    textos = []
    for opcion in respuesta.choices or []:
        mensaje = getattr(opcion, 'message', None)
        texto = getattr(mensaje, 'content', None) if mensaje else None
        if texto:
            textos.append(texto)

    capitulos = []
    if agente.conocimiento is not None:
        capitulos = agente.conocimiento.capitulos_relevantes(pregunta)

    return {
        'respuesta': '\n'.join(textos).strip(),
        'prompt_usuario': pregunta_completa,
        'system_prompt': agente.SYSTEM_PROMPT,
        'capitulos': [(c['numero'], c['titulo']) for c in capitulos],
    }


def pestana_chat(agente, error):
    st.header('Chat con el agente')
    if agente is None:
        st.warning(f'Chat no disponible: {error}')
        st.info('Configura OPENROUTER_API_KEY y reinicia la app. '
                'Las demás pestañas funcionan sin clave.')
        return

    if 'historial' not in st.session_state:
        st.session_state['historial'] = []

    for turno in st.session_state['historial']:
        with st.chat_message('user'):
            st.write(turno['pregunta'])
        with st.chat_message('assistant'):
            st.write(turno['respuesta'])
            if turno['capitulos']:
                st.caption('Capítulos usados: ' + ', '.join(
                    f"{n}. {t}" for n, t in turno['capitulos']))
            with st.expander('Ver prompt enviado'):
                st.markdown('**System prompt:**')
                st.code(turno['system_prompt'], language='text')
                st.markdown('**Mensaje del usuario:**')
                st.code(turno['prompt_usuario'], language='text')

    pregunta = st.chat_input('Escribe tu consulta...')
    if not pregunta:
        return
    with st.chat_message('user'):
        st.write(pregunta)
    with st.chat_message('assistant'):
        with st.spinner('Consultando...'):
            try:
                turno = consultar_con_traza(agente, pregunta)
            except Exception as exc:
                st.error(f'Error al consultar: {exc}')
                return
        st.write(turno['respuesta'])
        turno['pregunta'] = pregunta
        st.session_state['historial'].append(turno)
        st.rerun()


def _parsear_serie(texto):
    """Convierte texto con números separados por coma, espacio o salto
    de línea en una lista de floats."""
    import re
    tokens = re.split(r'[,;\s]+', texto.strip())
    valores = [float(t) for t in tokens if t]
    if not valores:
        raise ValueError('No se encontraron números en el texto.')
    return valores


def pestana_datos(agente, error):
    st.header('Análisis de datos con pandas')
    st.caption('Las estadísticas se calculan localmente con pandas; el '
               'agente solo las interpreta, no las estima.')

    descripcion = st.text_input(
        'Descripción de la serie',
        placeholder='Ej. precios semanales de la variedad Freedom')
    texto_serie = st.text_area(
        'Valores (separados por coma, espacio o salto de línea)',
        placeholder='12.5, 13.1, 11.8, 14.0, 13.6',
        height=120)

    if not texto_serie.strip():
        return
    try:
        valores = _parsear_serie(texto_serie)
    except ValueError as exc:
        st.error(f'Serie inválida: {exc}')
        return

    try:
        estadisticas = calcular_estadisticas(valores)
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return

    cols = st.columns(3)
    cols[0].metric('Media', estadisticas['media'])
    cols[1].metric('Mediana', estadisticas['mediana'])
    cols[2].metric('Desv. estándar', estadisticas['desviacion_estandar'])
    cols = st.columns(3)
    cols[0].metric('Mínimo', estadisticas['minimo'])
    cols[1].metric('Máximo', estadisticas['maximo'])
    cols[2].metric('Conteo', estadisticas['conteo'])

    if agente is None:
        st.info(f'Sin consulta al LLM ({error}). Solo se muestran los '
                'cálculos locales.')
        return

    col_serie, col_error = st.columns(2)
    if col_serie.button('📈 Interpretar serie histórica'):
        with st.spinner('Consultando...'):
            try:
                respuesta = agente.analizar_serie_historica(
                    descripcion or 'Serie sin descripción', valores)
            except Exception as exc:
                st.error(f'Error al consultar: {exc}')
                return
        st.markdown('**Interpretación del agente:**')
        st.write(respuesta)
    if col_error.button('⚠️ Analizar como error de estimado'):
        with st.spinner('Consultando...'):
            try:
                respuesta = agente.analizar_error_estimado(
                    descripcion or 'Error sin descripción', valores)
            except Exception as exc:
                st.error(f'Error al consultar: {exc}')
                return
        st.markdown('**Análisis del agente:**')
        st.write(respuesta)


def pestana_editor(base):
    st.header('Editor de capítulos de teoría')

    capitulos = base.listar_capitulos()
    opciones = ['➕ Nuevo capítulo'] + [
        f"{c['numero']}. {c['titulo']}" for c in capitulos]
    seleccion = st.selectbox('Capítulo', opciones)

    if seleccion == '➕ Nuevo capítulo':
        numero = max([c['numero'] for c in capitulos], default=0) + 1
        titulo, resumen, contenido, claves = '', '', '', ''
    else:
        numero = int(seleccion.split('.')[0])
        fila = base.conexion.execute(
            'SELECT * FROM capitulos WHERE numero = ?', (numero,)
        ).fetchone()
        titulo = fila['titulo']
        resumen = fila['resumen']
        contenido = fila['contenido']
        claves = fila['palabras_clave']

    with st.form('form_capitulo'):
        st.number_input('Número', value=numero, disabled=True)
        titulo = st.text_input('Título', value=titulo)
        resumen = st.text_input('Resumen (una línea)', value=resumen)
        claves = st.text_input(
            'Palabras clave (separadas por espacios)', value=claves,
            help='Pesan x3 al calcular relevancia')
        contenido = st.text_area('Contenido', value=contenido, height=350)
        col_guardar, col_borrar = st.columns(2)
        guardar = col_guardar.form_submit_button('💾 Guardar')
        borrar = col_borrar.form_submit_button(
            '🗑️ Eliminar',
            disabled=seleccion == '➕ Nuevo capítulo')

    if guardar:
        if not titulo.strip() or not contenido.strip():
            st.warning('Título y contenido son obligatorios.')
        else:
            base.agregar_capitulo(numero, titulo.strip(), contenido.strip(),
                                  resumen.strip(), claves.strip())
            st.success(f'Capítulo {numero} guardado.')
            st.rerun()
    if borrar:
        base.conexion.execute(
            'DELETE FROM capitulos WHERE numero = ?', (numero,))
        base.conexion.commit()
        st.success(f'Capítulo {numero} eliminado.')
        st.rerun()

    st.divider()
    st.subheader('Importar desde archivo .md')
    archivo = st.file_uploader(
        'Archivo con capítulos (encabezados "#")', type=['md', 'txt'])
    if archivo and st.button('Importar'):
        ruta_tmp = Path(archivo.name)
        ruta_tmp.write_bytes(archivo.getvalue())
        try:
            importados = base.importar_texto(ruta_tmp)
            st.success(f'Se importaron {importados} capítulo(s).')
            st.rerun()
        finally:
            ruta_tmp.unlink(missing_ok=True)


def pestana_prompt(agente, base):
    st.header('System prompt del agente')
    st.caption('Los cambios aplican solo a esta sesión; para hacerlos '
               'permanentes edita SYSTEM_PROMPT en agent_manager.py.')
    prompt_actual = (agente.SYSTEM_PROMPT if agente is not None
                     else AgenteAnalista.SYSTEM_PROMPT)
    nuevo = st.text_area('Prompt de sistema', value=prompt_actual,
                         height=400)
    if st.button('Aplicar en esta sesión', disabled=agente is None):
        agente.SYSTEM_PROMPT = nuevo
        st.success('System prompt actualizado para esta sesión.')
    if agente is None:
        st.info('Sin API key se muestra el prompt por defecto; '
                'no se puede aplicar a una sesión de agente.')
    if st.button('Restaurar original', disabled=agente is None):
        st.session_state.pop('agente', None)
        st.rerun()


def main():
    st.set_page_config(page_title='Agente Analista', layout='wide')
    st.title('🤖 Agente Analista — ajuste de contenidos y respuestas')

    base = cargar_base()
    agente, error = obtener_agente()

    with st.sidebar:
        st.metric('Modelo', agente.modelo if agente else 'sin clave')
        st.metric('Capítulos en base', len(base.listar_capitulos()))

    tab_chat, tab_datos, tab_editor, tab_prompt = st.tabs(
        ['💬 Chat', '📊 Datos', '📚 Capítulos', '⚙️ System prompt'])
    with tab_chat:
        pestana_chat(agente, error)
    with tab_datos:
        pestana_datos(agente, error)
    with tab_editor:
        pestana_editor(base)
    with tab_prompt:
        pestana_prompt(agente, base)


main()
