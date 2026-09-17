"""Compatibilidad: usa la implementación compartida de AGROMEJORA_DATA."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SOURCE = Path(__file__).resolve(
).parents[1] / "AGROMEJORA_DATA" / "plot_series_evaluation.py"
_SPEC = spec_from_file_location("_plot_series_evaluation_shared", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"No se pudo cargar {_SOURCE}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_area_percentages = _MODULE._area_percentages
_cumulative_data_percentage = _MODULE._cumulative_data_percentage
_read_dataframe = _MODULE._read_dataframe
calculate_evaluation = _MODULE.calculate_evaluation
generar_grafica = _MODULE.generar_grafica
evaluar_archivo = _MODULE.evaluar_archivo

__all__ = [
    "_area_percentages",
    "_cumulative_data_percentage",
    "calculate_evaluation",
    "generar_grafica",
    "evaluar_archivo",
]
