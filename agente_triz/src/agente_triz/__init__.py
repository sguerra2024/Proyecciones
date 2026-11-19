"""Agente TRIZ independiente e integrable en aplicaciones Python."""

from .agent import AgenteAnalista, AgenteTRIZ, calcular_estadisticas
from .conocimiento import BaseConocimiento
from .dialogo import construir_prompt_solucion, mostrar_dialogo_solucion
from .inspiracion import (
    agregar_referencia_inspiracion,
    cargar_banco_inspiracion,
    filtrar_banco_por_vias,
)

__all__ = [
    "AgenteAnalista",
    "AgenteTRIZ",
    "BaseConocimiento",
    "agregar_referencia_inspiracion",
    "cargar_banco_inspiracion",
    "construir_prompt_solucion",
    "calcular_estadisticas",
    "filtrar_banco_por_vias",
    "mostrar_dialogo_solucion",
]
