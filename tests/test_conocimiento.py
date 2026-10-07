"""Pruebas para la base de conocimiento de teoría del agente."""
from conocimiento import BaseConocimiento


def _base_en_memoria():
    return BaseConocimiento(':memory:')


def test_agregar_y_listar_capitulos():
    base = _base_en_memoria()
    base.agregar_capitulo(1, 'Definición del problema',
                          'Un problema bien definido está medio resuelto.',
                          palabras_clave='definición problema enunciado')
    base.agregar_capitulo(2, 'Causa raíz',
                          'Los cinco porqués para llegar a la causa raíz.',
                          palabras_clave='causa raíz porqués')
    capitulos = base.listar_capitulos()
    assert len(capitulos) == 2
    assert capitulos[0]['titulo'] == 'Definición del problema'
    base.cerrar()


def test_reemplazar_capitulo_mismo_numero():
    base = _base_en_memoria()
    base.agregar_capitulo(1, 'Versión vieja', 'contenido viejo')
    base.agregar_capitulo(1, 'Versión nueva', 'contenido nuevo')
    capitulos = base.listar_capitulos()
    assert len(capitulos) == 1
    assert capitulos[0]['titulo'] == 'Versión nueva'
    base.cerrar()


def test_capitulos_relevantes_por_palabras_clave():
    base = _base_en_memoria()
    base.agregar_capitulo(1, 'Definición del problema',
                          'Cómo enunciar el problema correctamente.',
                          palabras_clave='problema enunciado alcance')
    base.agregar_capitulo(2, 'Lluvia de ideas',
                          'Generar soluciones sin juzgar.',
                          palabras_clave='ideas creatividad soluciones')
    relevantes = base.capitulos_relevantes(
        '¿Cómo genero más ideas de soluciones para este proyecto?')
    assert relevantes
    assert relevantes[0]['titulo'] == 'Lluvia de ideas'
    base.cerrar()


def test_capitulos_relevantes_sin_coincidencias():
    base = _base_en_memoria()
    base.agregar_capitulo(
        1, 'PDCA', 'Ciclo planificar-hacer-verificar-actuar.')
    assert base.capitulos_relevantes('zzzz qqqq xxxx') == []
    base.cerrar()


def test_formatear_contexto_incluye_capitulo():
    base = _base_en_memoria()
    base.agregar_capitulo(3, 'Causa raíz',
                          'Pregunta por qué cinco veces.',
                          palabras_clave='causa raíz error')
    contexto = base.formatear_contexto('¿Por qué ocurre este error?')
    assert 'Capítulo 3: Causa raíz' in contexto
    assert 'TEORÍA DE RESOLUCIÓN DE PROBLEMAS' in contexto
    base.cerrar()


def test_importar_texto_desde_markdown(tmp_path):
    archivo = tmp_path / 'teoria.md'
    archivo.write_text(
        '# Primer capítulo\nContenido del primer capítulo.\n\n'
        '## Sección interna\nDetalle de la sección.\n\n'
        '# Segundo capítulo\nContenido del segundo capítulo.\n',
        encoding='utf-8'
    )
    base = _base_en_memoria()
    importados = base.importar_texto(archivo)
    assert importados == 2
    capitulos = base.listar_capitulos()
    assert capitulos[0]['titulo'] == 'Primer capítulo'
    assert capitulos[1]['titulo'] == 'Segundo capítulo'
    base.cerrar()


def test_agente_usa_conocimiento_en_consulta():
    """El agente inyecta teoría relevante en el prompt enviado al LLM."""
    from unittest.mock import MagicMock, patch
    import agent_manager

    base = _base_en_memoria()
    base.agregar_capitulo(1, 'Causa raíz',
                          'Pregunta por qué cinco veces.',
                          palabras_clave='causa raíz error falla')

    agente = agent_manager.AgenteAnalista.__new__(
        agent_manager.AgenteAnalista)
    agente.conocimiento = base
    agente.modelo = 'modelo-prueba'

    respuesta_falsa = MagicMock()
    mensaje = MagicMock()
    mensaje.content = 'respuesta del modelo'
    opcion = MagicMock()
    opcion.message = mensaje
    respuesta_falsa.choices = [opcion]
    agente.cliente = MagicMock()
    agente.cliente.chat.completions.create.return_value = respuesta_falsa

    agente.consultar('¿Cuál es la causa raíz de esta falla?')

    llamada = agente.cliente.chat.completions.create.call_args
    prompt_usuario = llamada.kwargs['messages'][1]['content']
    assert 'Capítulo 1: Causa raíz' in prompt_usuario
    base.cerrar()
