import ProyAst
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_preparar_estado_para_nuevo_archivo_base_preserva_contexto_sesion():
    state = {
        "archivo_sesion_df": pd.DataFrame({"col": [1, 2]}),
        "archivo_sesion_nombre": "archivo_sincronizado.xlsx",
        "base_proyeccion_anthropic": pd.DataFrame({"valor": [9]}),
    }

    ProyAst.preparar_estado_para_nuevo_archivo_base(state, "nuevo.xlsx")

    assert state["dashboard_archivo_id"] == "nuevo.xlsx"
    assert state["base_proyeccion_anthropic"].empty
    assert state["archivo_sesion_df"].equals(pd.DataFrame({"col": [1, 2]}))
    assert state["archivo_sesion_nombre"] == "archivo_sincronizado.xlsx"
