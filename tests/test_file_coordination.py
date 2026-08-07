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


def test_sincronizacion_export_evita_reenvios_duplicados(monkeypatch):
    llamadas = []

    def fake_sincronizar_archivo_llm(archivo_subido, dataframe=None, nombre_archivo=None):
        llamadas.append((nombre_archivo, dataframe is not None))
        return {
            "file_id": "file-1",
            "nombre": nombre_archivo,
            "bytes": 1,
            "metodo": "session-dataframe",
            "modo": "local",
            "filas": 1,
        }

    monkeypatch.setattr(ProyAst, "sincronizar_archivo_llm",
                        fake_sincronizar_archivo_llm)
    monkeypatch.setattr(
        ProyAst, "registrar_sincronizacion_en_sesion", lambda *args, **kwargs: None)

    state = {}
    ProyAst.sincronizar_export_generado_automatico(
        b"contenido-repetido",
        "export.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        dataframe=pd.DataFrame({"a": [1]}),
        state=state,
    )
    ProyAst.sincronizar_export_generado_automatico(
        b"contenido-repetido",
        "export.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        dataframe=pd.DataFrame({"a": [1]}),
        state=state,
    )

    assert len(llamadas) == 1


def test_construir_cache_patrones_semanales_reutiliza_patrones():
    df = pd.DataFrame(
        {
            "Bloque&Varid": ["A", "A", "B", "B"],
            "Anio": [2025, 2025, 2025, 2025],
            "Semana": [1, 2, 1, 2],
            "Tallos/m2": [10, 20, 30, 40],
            "Produccion": [100, 200, 300, 400],
        }
    )

    cache = ProyAst.construir_cache_patrones_semanales(df)

    assert set(cache) == {"A", "B"}
    assert cache["A"].columns.tolist()[:2] == ["Anio", "Semana"]
    assert cache["A"]["Tallos_m2_patron"].tolist() == [10.0, 20.0]


def test_obtener_configuracion_proyeccion_masiva_ligera():
    config = ProyAst.obtener_configuracion_proyeccion_masiva(ligera=True)

    assert config["modo_liviano"] is True
    assert config["n_estimators"] == 25
    assert config["max_depth"] == 8
    assert "Tallos/m2" in config["columnas_modelo"]
