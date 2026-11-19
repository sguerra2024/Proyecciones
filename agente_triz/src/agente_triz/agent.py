"""API reutilizable del agente TRIZ mediante OpenRouter."""

import os
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .conocimiento import BaseConocimiento


def calcular_estadisticas(datos, decimales=2):
    """Calcula estadísticas descriptivas para valores numéricos."""
    if pd is None:
        raise RuntimeError(
            "La librería pandas no está instalada. Instala con: pip install pandas"
        )
    if isinstance(datos, pd.Series):
        serie = datos
    elif isinstance(datos, dict):
        serie = pd.Series(list(datos.values()), dtype="float64")
    else:
        serie = pd.Series(list(datos), dtype="float64")
    serie = pd.to_numeric(serie, errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()
    if serie.empty:
        raise ValueError("No hay datos numéricos válidos para calcular estadísticas.")
    return {
        "media": round(float(serie.mean()), decimales),
        "mediana": round(float(serie.median()), decimales),
        "desviacion_estandar": (
            round(float(serie.std()), decimales) if len(serie) > 1 else 0.0
        ),
        "minimo": round(float(serie.min()), decimales),
        "maximo": round(float(serie.max()), decimales),
        "conteo": int(serie.count()),
    }


class AgenteTRIZ:
    """Agente de resolución de problemas que consulta OpenRouter."""

    SYSTEM_PROMPT = """Eres un agente especializado en resolución sistemática de problemas,
análisis de datos y toma de decisiones. Cuando el contexto incluya teoría TRIZ,
úsala como referencia y explica cómo se relaciona con el problema.

INSTRUCCIONES:
- Define el problema y distingue hechos, supuestos y datos faltantes.
- Si se busca eliminar un efecto negativo o corregir una función insuficiente,
  describe el efecto observado, la función a mejorar y el cambio verificable
  esperado en el sistema, sin asumir la causa ni imponer una solución de antemano.
- Al generar soluciones, considera elevar el problema específico a un problema
  genérico y luego mapear las alternativas de vuelta al sistema concreto,
  conservando sus restricciones y validándolas con evidencia.
- Para acciones perjudiciales, considera si conviene transferirlas a otra
  persona, sistema o proceso; si el sistema puede realizarlas con pequeñas
  modificaciones; o si son realmente necesarias. Trata cada opción como
  hipótesis y verifica riesgos y efectos secundarios.
- Si una función no alcanza el resultado deseado, identifica si existe una
  contradicción técnica (mejorar una característica empeora otra) o física
  (una característica debe satisfacer requisitos opuestos), y usa el conflicto
  validado para generar alternativas.
- Aplica principios inventivos solo después de definir el problema y verificar
  que requiere un enfoque inventivo; vincula cada principio con el conflicto,
  genera alternativas y propone cómo validarlas. No los fuerces si no aplican.
- Propón alternativas accionables y priorízalas por impacto y esfuerzo.
- Cuantifica impactos solo cuando los datos entregados permitan hacerlo.
- No inventes cifras, causas ni información de mercado; identifica lo que deba verificarse.
- Usa las estadísticas proporcionadas sin recalcularlas ni alterarlas."""

    def __init__(self, api_key=None, ruta_conocimiento=None):
        """Crea el cliente OpenRouter y carga una base de conocimiento opcional."""
        if OpenAI is None:
            raise RuntimeError(
                "La librería openai no está instalada. Instala con: pip install openai"
            )
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "No se encontró OPENROUTER_API_KEY. Define la variable de entorno "
                "o pasa api_key como argumento."
            )
        self.modelo = os.getenv(
            "OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5"
        )
        self.cliente = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.conocimiento = None
        if ruta_conocimiento is None:
            self.conocimiento = BaseConocimiento(":memory:")
            data_path = Path(__file__).resolve().parent / "data"
            siguiente_numero = 1
            for teoria_path in sorted(data_path.glob("*.md")):
                importados = self.conocimiento.importar_texto(
                    teoria_path, numero_inicial=siguiente_numero
                )
                siguiente_numero += importados
        elif str(ruta_conocimiento) == ":memory:" or Path(
            ruta_conocimiento
        ).exists():
            ruta_bd = ruta_conocimiento
            self.conocimiento = BaseConocimiento(ruta_bd)

    def cerrar(self):
        """Cierra la conexión de conocimiento local, si fue inicializada."""
        if self.conocimiento is not None:
            self.conocimiento.cerrar()
            self.conocimiento = None

    def consultar(self, pregunta, contexto_adicional=None):
        """Consulta al modelo, agregando capítulos relevantes si están disponibles."""
        return self.enviar_consulta(self.preparar_consulta(pregunta, contexto_adicional))

    def preparar_consulta(
        self, pregunta: str, contexto_adicional: Optional[str] = None
    ) -> str:
        """Prepara el mensaje en el hilo propietario de la conexión SQLite."""
        contexto_teoria = ""
        if self.conocimiento is not None:
            contexto_teoria = self.conocimiento.formatear_contexto(pregunta)
        partes = [
            contexto for contexto in (contexto_teoria, contexto_adicional) if contexto
        ]
        return (
            "\n\n".join(partes) + f"\n\nPregunta: {pregunta}"
            if partes
            else pregunta
        )

    def enviar_consulta(self, pregunta_completa: str) -> str:
        """Envía un mensaje preparado sin acceder a la base de conocimiento."""
        try:
            respuesta = self.cliente.chat.completions.create(
                model=self.modelo,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": pregunta_completa},
                ],
            )
            textos = []
            for opcion in respuesta.choices or []:
                mensaje = getattr(opcion, "message", None)
                texto = getattr(mensaje, "content", None) if mensaje else None
                if texto:
                    textos.append(texto)
            return "\n".join(textos).strip()
        except Exception as exc:
            raise RuntimeError(f"Error al consultar agente TRIZ: {exc}") from exc

    def analizar_sistema(self, descripcion_cambio, impacto_estimado=None):
        contexto = f"Cambio solicitado: {descripcion_cambio}"
        if impacto_estimado:
            contexto += f"\nDatos de impacto: {impacto_estimado}"
        return self.consultar(
            "Analiza el problema y el impacto de este cambio. Propón opciones, "
            "compara beneficios y costos con los datos disponibles y recomienda "
            "cómo reducir riesgos y costos futuros.",
            contexto,
        )

    def analizar_error_estimado(self, error_descripcion, desviacion):
        if isinstance(desviacion, (list, tuple)) or (
            pd is not None and isinstance(desviacion, pd.Series)
        ) or isinstance(desviacion, dict):
            estadisticas = calcular_estadisticas(desviacion)
            contexto = (
                f"Error en estimado: {error_descripcion}\n"
                f"Estadísticas de desviación: {estadisticas}"
            )
        else:
            contexto = (
                f"Error en estimado: {error_descripcion}\n"
                f"Desviación: {desviacion}"
            )
        return self.consultar(
            "Analiza el error: identifica posibles causas que deban verificarse, "
            "patrones recurrentes si los datos los respaldan y medidas preventivas.",
            contexto,
        )

    def analizar_serie_historica(self, descripcion, serie):
        estadisticas = calcular_estadisticas(serie)
        contexto = f"Serie: {descripcion}\nEstadísticas calculadas: {estadisticas}"
        return self.consultar(
            "Interpreta estas estadísticas, señala riesgos u oportunidades "
            "respaldados por ellas y recomienda acciones.",
            contexto,
        )

    def investigar_mercado(self, producto, region=None):
        contexto = f"Producto: {producto}"
        if region:
            contexto += f"\nRegión: {region}"
        return self.consultar(
            "Analiza el mercado de este producto. Distingue los datos conocidos "
            "de aquello que requiera información actualizada externa; no inventes "
            "precios ni tendencias.",
            contexto,
        )

    def optimizar_beneficios(self, datos_negocio):
        contexto = f"Datos del negocio: {datos_negocio}"
        return self.consultar(
            "Propón estrategias para mejorar la rentabilidad, ordenadas por "
            "impacto y esfuerzo. Cuantifica solo si los datos permiten calcularlo.",
            contexto,
        )


# Compatibilidad de nombre para aplicaciones que usaban el agente anterior.
AgenteAnalista = AgenteTRIZ
