"""Configuración de pytest: hace importables los módulos del proyecto."""
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parent.parent

for _ruta in (_RAIZ, _RAIZ / 'Agente_Analista'):
    _ruta_str = str(_ruta)
    if _ruta_str not in sys.path:
        sys.path.insert(0, _ruta_str)
