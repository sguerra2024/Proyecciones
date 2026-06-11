"""
Gestor de agentes especializados para análisis de datos y mercados.
Integración con Anthropic Claude para consultas avanzadas.
"""
import importlib
import os
from pathlib import Path

anthropic = None
try:
    anthropic = importlib.import_module('anthropic')
except ImportError:
    pass


class AgenteAnalistasMercados:
    """Agente especializado en análisis de datos, mercados y gestión de cambios."""

    SYSTEM_PROMPT = """Eres un analista de datos y mercados experto, especializado en maximizar beneficios empresariales.

Tus capacidades incluyen:

1. ANÁLISIS FINANCIERO Y KPIs:
   - Análisis de rentabilidad y márgenes de ganancia
   - Cálculo de KPIs financieros clave
   - Proyecciones de beneficios y flujo de caja

2. INVESTIGACIÓN DE MERCADOS:
   - Análisis de precios y elasticidad de demanda
   - Identificación de tendencias emergentes
   - Segmentación de mercados y competencia

3. GESTIÓN DE CAMBIOS Y ERRORES:
   - Cálculo del costo real de cambios solicitados por cliente
   - Cálculo del impacto de errores en estimados
   - Presentación de opciones adaptadas con impacto económico
   - Identificación de patrones recurrentes

4. MEDIDAS PREVENTIVAS:
   - Propuestas basadas en datos históricos
   - Estrategias para minimizar errores futuros
   - Mejora continua de procesos

INSTRUCCIONES:
- Prioriza recomendaciones por potencial de impacto económico
- Traduce análisis técnicos en decisiones claras y accionables
- Siempre cuantifica el impacto financiero
- Presenta alternativas con sus costos/beneficios asociados
- Sé directo: datos concretos antes que teoría"""

    def __init__(self, api_key=None):
        """Inicializa el agente con credenciales de Anthropic."""
        if anthropic is None:
            raise RuntimeError(
                'La libreria anthropic no esta instalada. '
                'Instala con: pip install anthropic'
            )
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY', '').strip()
        if not self.api_key:
            raise RuntimeError(
                'No se encontro ANTHROPIC_API_KEY. '
                'Define la variable de entorno o pasa api_key como argumento.'
            )
        self.modelo = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-6')
        self.cliente = anthropic.Anthropic(api_key=self.api_key)

    def consultar(self, pregunta, contexto_adicional=None):
        """
        Consulta al agente especializado.

        Args:
            pregunta (str): Pregunta o solicitud de análisis
            contexto_adicional (str): Datos o contexto adicional para el análisis

        Returns:
            str: Respuesta del agente
        """
        if contexto_adicional:
            pregunta_completa = f"{contexto_adicional}\n\nPregunta: {pregunta}"
        else:
            pregunta_completa = pregunta

        try:
            respuesta = self.cliente.messages.create(
                model=self.modelo,
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {
                        'role': 'user',
                        'content': pregunta_completa
                    }
                ]
            )
            textos = []
            for bloque in respuesta.content:
                texto = getattr(bloque, 'text', None)
                if texto:
                    textos.append(texto)
            return '\n'.join(textos).strip()
        except Exception as exc:
            raise RuntimeError(
                f'Error al consultar agente especializado: {exc}'
            )

    def analizar_cambio_cliente(self, descripcion_cambio, impacto_estimado=None):
        """
        Analiza el costo e impacto de un cambio solicitado por el cliente.

        Args:
            descripcion_cambio (str): Descripción del cambio
            impacto_estimado (dict): Datos de impacto (costo, tiempo, etc.)

        Returns:
            str: Análisis detallado con opciones y recomendaciones
        """
        contexto = f"Cambio solicitado: {descripcion_cambio}"
        if impacto_estimado:
            contexto += f"\nDatos de impacto: {impacto_estimado}"

        pregunta = (
            "Por favor, analiza el impacto económico de este cambio. "
            "Incluye: costo real, opciones alternativas, impacto en beneficios, "
            "y recomendaciones para minimizar costos futuros."
        )
        return self.consultar(pregunta, contexto)

    def analizar_error_estimado(self, error_descripcion, desviacion):
        """
        Analiza un error en estimados y su impacto financiero.

        Args:
            error_descripcion (str): Descripción del error
            desviacion (float o dict): Desviación del estimado (monto o porcentaje)

        Returns:
            str: Análisis de impacto y estrategias preventivas
        """
        contexto = f"Error en estimado: {error_descripcion}\nDesviación: {desviacion}"
        pregunta = (
            "Analiza el impacto financiero de este error. "
            "Incluye: pérdida estimada, causas raíz, patrones recurrentes, "
            "y medidas preventivas para el futuro."
        )
        return self.consultar(pregunta, contexto)

    def investigar_mercado(self, producto, region=None):
        """
        Realiza investigación de mercado para un producto.

        Args:
            producto (str): Producto o servicio a analizar
            region (str): Región geográfica (opcional)

        Returns:
            str: Análisis de mercado con precios, tendencias y oportunidades
        """
        contexto = f"Producto: {producto}"
        if region:
            contexto += f"\nRegión: {region}"

        pregunta = (
            "Por favor, realiza un análisis de mercado. "
            "Incluye: análisis de precios actuales, elasticidad de demanda, "
            "tendencias emergentes, oportunidades no explotadas, y recomendaciones."
        )
        return self.consultar(pregunta, contexto)

    def optimizar_beneficios(self, datos_negocio):
        """
        Analiza oportunidades para maximizar beneficios.

        Args:
            datos_negocio (dict): Datos del negocio (ingresos, costos, etc.)

        Returns:
            str: Recomendaciones priorizadas para maximizar beneficios
        """
        contexto = f"Datos del negocio: {datos_negocio}"
        pregunta = (
            "Analiza estas datos y propón estrategias priorizadas por impacto económico "
            "para maximizar beneficios. Cuantifica cada impacto potencial."
        )
        return self.consultar(pregunta, contexto)
