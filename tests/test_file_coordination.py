import io
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ProyAst  # noqa: E402  (requiere ROOT en sys.path)


@pytest.fixture
def st_falso(monkeypatch):
    """Reemplaza streamlit por un stub con session_state de diccionario."""
    fake = types.SimpleNamespace(session_state={})
    monkeypatch.setattr(ProyAst, "st", fake)
    return fake


def _info_carga(modo="local", file_id="local-session", nombre="export.xlsx", filas=3):
    return {
        "file_id": file_id,
        "nombre": nombre,
        "bytes": filas,
        "metodo": "session-dataframe",
        "modo": modo,
        "filas": filas,
    }


# --- preparar_estado_para_nuevo_archivo_base -------------------------------

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


def test_preparar_estado_para_nuevo_archivo_base_limpia_contexto_si_se_pide():
    state = {
        "archivo_sesion_df": pd.DataFrame({"col": [1, 2]}),
        "archivo_sesion_nombre": "archivo_sincronizado.xlsx",
    }

    ProyAst.preparar_estado_para_nuevo_archivo_base(
        state, "nuevo.xlsx", preservar_archivo_sesion=False)

    assert state["archivo_sesion_df"].empty
    assert state["archivo_sesion_nombre"] == ""


def test_preparar_estado_para_nuevo_archivo_base_resetea_export_y_dashboard():
    state = {
        "dashboard_archivo_id": "anterior.xlsx",
        "dashboard_finca_activo": True,
        "dashboard_export_bytes": b"algo",
        "dashboard_export_name": "viejo.xlsx",
        "dashboard_export_mime": "application/vnd.ms-excel",
    }

    devuelto = ProyAst.preparar_estado_para_nuevo_archivo_base(state)

    # Sin nuevo_archivo_id el identificador vigente no se toca.
    assert state["dashboard_archivo_id"] == "anterior.xlsx"
    assert state["dashboard_finca_activo"] is False
    assert state["dashboard_export_bytes"] is None
    assert state["dashboard_export_name"] == ""
    assert state["dashboard_export_mime"] == ""
    assert devuelto is state


# --- sincronizar_export_generado_automatico --------------------------------

@pytest.fixture
def sync_espia(monkeypatch):
    """Registra las llamadas a sincronizar_archivo_llm y neutraliza el registro en sesion."""
    llamadas = []

    def fake_sincronizar_archivo_llm(archivo_subido, dataframe=None, nombre_archivo=None):
        llamadas.append((nombre_archivo, dataframe is not None))
        return _info_carga(nombre=nombre_archivo, filas=1)

    monkeypatch.setattr(ProyAst, "sincronizar_archivo_llm",
                        fake_sincronizar_archivo_llm)
    monkeypatch.setattr(
        ProyAst, "registrar_sincronizacion_en_sesion", lambda *args, **kwargs: None)
    return llamadas


MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _sincronizar(contenido, state, nombre="export.xlsx"):
    ProyAst.sincronizar_export_generado_automatico(
        contenido,
        nombre,
        MIME_XLSX,
        dataframe=pd.DataFrame({"a": [1]}),
        state=state,
    )


def test_sincronizacion_export_evita_reenvios_duplicados(sync_espia):
    state = {}
    _sincronizar(b"contenido-repetido", state)
    _sincronizar(b"contenido-repetido", state)

    assert len(sync_espia) == 1


def test_sincronizacion_export_reenvia_si_cambia_el_contenido(sync_espia):
    state = {}
    _sincronizar(b"version-1", state)
    _sincronizar(b"version-2", state)

    assert len(sync_espia) == 2


def test_sincronizacion_export_distingue_por_nombre_de_archivo(sync_espia):
    state = {}
    _sincronizar(b"mismo-contenido", state, nombre="uno.xlsx")
    _sincronizar(b"mismo-contenido", state, nombre="dos.xlsx")

    assert [nombre for nombre, _ in sync_espia] == ["uno.xlsx", "dos.xlsx"]


def test_sincronizacion_export_ignora_contenido_vacio(sync_espia):
    state = {}
    _sincronizar(b"", state)

    assert sync_espia == []
    assert state == {}


def test_sincronizacion_export_reporta_error_sin_propagar(monkeypatch):
    def fake_sincronizar_archivo_llm(*args, **kwargs):
        raise RuntimeError("credenciales invalidas")

    monkeypatch.setattr(ProyAst, "sincronizar_archivo_llm",
                        fake_sincronizar_archivo_llm)

    state = {}
    _sincronizar(b"contenido", state)

    estado, mensaje = state["estado_subida_anthropic"]
    assert estado == "error"
    assert "credenciales invalidas" in mensaje
    # El fallo no debe cachearse: el proximo intento tiene que reintentar.
    assert "__export_sync_cache__export.xlsx" not in state


# --- sincronizar_archivo_llm ----------------------------------------------

def test_sincronizar_archivo_llm_exige_archivo_o_dataframe():
    with pytest.raises(ValueError):
        ProyAst.sincronizar_archivo_llm(None)


def test_sincronizar_archivo_llm_local_usa_el_dataframe_recibido(monkeypatch):
    monkeypatch.setattr(ProyAst, "obtener_llm_provider", lambda: "github")

    info = ProyAst.sincronizar_archivo_llm(
        None,
        dataframe=pd.DataFrame({"a": [1, 2, 3]}),
        nombre_archivo="base.xlsx",
    )

    assert info["modo"] == "local"
    assert info["nombre"] == "base.xlsx"
    assert info["filas"] == 3
    assert info["metodo"] == "session-dataframe"


def test_sincronizar_archivo_llm_local_rechaza_tabla_vacia(monkeypatch):
    monkeypatch.setattr(ProyAst, "obtener_llm_provider", lambda: "github")

    with pytest.raises(RuntimeError, match="tabla"):
        ProyAst.sincronizar_archivo_llm(
            None, dataframe=pd.DataFrame(), nombre_archivo="vacio.xlsx")


def test_sincronizar_archivo_llm_anthropic_marca_modo_remoto(monkeypatch):
    monkeypatch.setattr(ProyAst, "obtener_llm_provider", lambda: "anthropic")
    monkeypatch.setattr(ProyAst, "subir_archivo_anthropic",
                        lambda archivo: {"file_id": "file-abc", "nombre": "base.xlsx"})

    info = ProyAst.sincronizar_archivo_llm(
        ProyAst.ArchivoEnMemoria("base.xlsx", b"datos"))

    assert info["modo"] == "remoto"
    assert info["file_id"] == "file-abc"


def test_proveedor_predeterminado_sincroniza_remoto_con_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(ProyAst, "llm_provider", "anthropic")
    monkeypatch.setattr(
        ProyAst,
        "subir_archivo_anthropic",
        lambda archivo: {"file_id": "file-default", "nombre": archivo.name},
    )

    assert ProyAst.obtener_llm_provider() == "anthropic"
    info = ProyAst.sincronizar_archivo_llm(
        ProyAst.ArchivoEnMemoria("presentacion.xlsx", b"datos"))

    assert info["modo"] == "remoto"
    assert info["file_id"] == "file-default"


# --- registrar_sincronizacion_en_sesion -----------------------------------

def test_registrar_sincronizacion_en_sesion_modo_local(st_falso):
    df = pd.DataFrame({"a": [1, 2, 3]})

    ProyAst.registrar_sincronizacion_en_sesion(
        _info_carga(modo="local", nombre="base.xlsx", filas=3), df, "base.xlsx")

    sesion = st_falso.session_state
    assert sesion["archivo_anthropic_cargado"] == "base.xlsx"
    assert sesion["archivo_anthropic_id"] == "local-session"
    estado, mensaje = sesion["estado_subida_anthropic"]
    assert estado == "ok"
    assert "Contexto local listo" in mensaje
    assert "3 filas" in mensaje
    assert sesion["archivo_sesion_df"].equals(df)
    assert sesion["archivo_sesion_nombre"] == "base.xlsx"


def test_registrar_sincronizacion_en_sesion_modo_remoto_usa_id_base(st_falso):
    ProyAst.registrar_sincronizacion_en_sesion(
        _info_carga(modo="remoto", file_id="file-xyz"),
        pd.DataFrame({"a": [1]}),
        "base.xlsx",
        id_base="base.xlsx123",
    )

    sesion = st_falso.session_state
    assert sesion["archivo_anthropic_cargado"] == "base.xlsx123"
    assert sesion["estado_subida_anthropic"] == (
        "ok", "Anthropic ID: file-xyz")


def test_registrar_sincronizacion_en_sesion_no_sobreescribe_con_tabla_vacia(st_falso):
    df_previo = pd.DataFrame({"a": [1]})
    st_falso.session_state["archivo_sesion_df"] = df_previo
    st_falso.session_state["archivo_sesion_nombre"] = "previo.xlsx"

    ProyAst.registrar_sincronizacion_en_sesion(
        _info_carga(), pd.DataFrame(), "nuevo.xlsx")

    assert st_falso.session_state["archivo_sesion_df"] is df_previo
    assert st_falso.session_state["archivo_sesion_nombre"] == "previo.xlsx"


# --- cargar_archivo_a_dataframe -------------------------------------------

def test_cargar_archivo_a_dataframe_lee_csv():
    archivo = ProyAst.ArchivoEnMemoria("datos.csv", b"a,b\n1,2\n3,4\n")

    df = ProyAst.cargar_archivo_a_dataframe(archivo)

    assert df.columns.tolist() == ["a", "b"]
    assert df["a"].tolist() == [1, 3]


def test_cargar_archivo_a_dataframe_lee_xlsx():
    buffer = io.BytesIO()
    pd.DataFrame({"Semana": [1, 2], "Produccion": [10, 20]}).to_excel(
        buffer, index=False)
    archivo = ProyAst.ArchivoEnMemoria("datos.xlsx", buffer.getvalue())

    df = ProyAst.cargar_archivo_a_dataframe(archivo)

    assert df["Produccion"].tolist() == [10, 20]


def test_cargar_archivo_a_dataframe_devuelve_vacio_en_casos_no_soportados():
    assert ProyAst.cargar_archivo_a_dataframe(None).empty
    assert ProyAst.cargar_archivo_a_dataframe(
        ProyAst.ArchivoEnMemoria("datos.csv", b"")).empty
    assert ProyAst.cargar_archivo_a_dataframe(
        ProyAst.ArchivoEnMemoria("informe.pdf", b"%PDF-1.4")).empty
    # Un xlsx corrupto no debe romper la carga.
    assert ProyAst.cargar_archivo_a_dataframe(
        ProyAst.ArchivoEnMemoria("roto.xlsx", b"no-es-excel")).empty


def test_prompt_anthropic_incluye_analisis_amortiguador(monkeypatch):
    capturado = {}

    def fake_consultar_llm(prompt, max_tokens=None):
        capturado["prompt"] = prompt
        return "respuesta"

    monkeypatch.setattr(ProyAst, "consultar_llm", fake_consultar_llm)
    base = pd.DataFrame({
        "Finca": ["BL25"],
        "Bloque&Varid": ["001RED"],
        "Produccion": [1000],
    })
    semanas = ["2025-52"] + [f"2026-{semana:02d}" for semana in range(1, 14)]
    proyeccion = pd.DataFrame({
        "Finca_proyectada": ["BL25"] * 14,
        "Variedad_proyectada": ["001RED"] * 14,
        "Anio_Semana": semanas,
        "Estimado_modelo": [1000] * 14,
        "Tallos_por_m2": [999.0] + list(range(1, 14)),
        "M2 Amortiguador": list(range(1, 15)),
        "Proyeccion_con_amortiguador_IA": [800] * 14,
    })

    ProyAst.responder_pregunta_anthropic(
        base,
        "Cuantos M2 de amortiguador necesito?",
        "BL25",
        df_proyeccion=proyeccion,
    )

    assert "area_calculada_m2" in capturado["prompt"]
    assert "promedio_tallos_m2_ultimas_12_semanas" in capturado["prompt"]
    assert "producto_m2_por_promedio_tallos_m2" in capturado["prompt"]
    assert ",14,7.5" in capturado["prompt"]
    assert ",105.0" in capturado["prompt"]
    assert "999.0" not in capturado["prompt"]
    assert "Estimado_modelo" not in capturado["prompt"]
    assert "Proyeccion_con_amortiguador_IA" not in capturado["prompt"]
    assert "Estado M2 Amortiguador" not in capturado["prompt"]
    assert (
        "ES EL AREA QUE EL TECNICO DE CULTIVO DEBE ADMINISTAR PARA CUBRIR "
        "EL ERROR DEL MODELO TANTO EN POSITIVO COMO EN NEGATIVO"
    ) in capturado["prompt"]


def test_pregunta_funcion_amortiguador_devuelve_respuesta_definida(monkeypatch):
    def fake_consultar_llm(*args, **kwargs):
        pytest.fail("No debe consultar el LLM para esta respuesta definida")

    monkeypatch.setattr(ProyAst, "consultar_llm", fake_consultar_llm)

    respuesta = ProyAst.responder_pregunta_anthropic(
        pd.DataFrame({"Finca": ["BL25"]}),
        "¿Qué función tiene el amortiguador?",
        "BL25",
    )

    assert respuesta == (
        "ES EL AREA QUE EL TECNICO DE CULTIVO DEBE ADMINISTAR PARA CUBRIR "
        "EL ERROR DEL MODELO TANTO EN POSITIVO COMO EN NEGATIVO. "
        "PARA ADMINISTRAR ESTE AMORTIGUADOR, SE RECOMIENDA QUE CADA TECNICO "
        "EXPONGA UNA IDEA Y QUE ESTA SE REGISTRE."
    )


def test_recomendacion_amortiguador_devuelve_respuesta_definida(monkeypatch):
    def fake_consultar_llm(*args, **kwargs):
        pytest.fail("No debe consultar el LLM para esta respuesta definida")

    monkeypatch.setattr(ProyAst, "consultar_llm", fake_consultar_llm)

    respuesta = ProyAst.responder_pregunta_anthropic(
        pd.DataFrame({"Finca": ["BL25"]}),
        "¿Qué recomiendas para administrar este amortiguador?",
        "BL25",
    )

    assert respuesta == (
        "PARA ADMINISTRAR ESTE AMORTIGUADOR, SE RECOMIENDA QUE CADA TECNICO "
        "EXPONGA UNA IDEA Y QUE ESTA SE REGISTRE."
    )


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Qué es el amortiguador M2?",
        "Que significa amortiguador m2",
        "Explícame el amortiguador",
        "Define el amortiguador M2",
    ],
)
def test_definicion_amortiguador_reconoce_variantes(monkeypatch, pregunta):
    def fake_consultar_llm(*args, **kwargs):
        pytest.fail("No debe consultar el LLM para esta respuesta definida")

    monkeypatch.setattr(ProyAst, "consultar_llm", fake_consultar_llm)

    respuesta = ProyAst.responder_pregunta_anthropic(
        pd.DataFrame({"Finca": ["ASTROFLORES"]}),
        pregunta,
        "ASTROFLORES",
    )

    assert respuesta.startswith(
        "ES EL AREA QUE EL TECNICO DE CULTIVO DEBE ADMINISTAR"
    )
    assert "TANTO EN POSITIVO COMO EN NEGATIVO" in respuesta


def test_salida_amortiguada_solo_expone_m2_y_proyeccion():
    proyeccion = pd.DataFrame({
        "Finca": ["BL25"],
        "Bloque&Varid": ["001RED"],
        "Estimado_modelo": [1000.0],
        "Amortiguador_sobreestimacion": [110.0],
        "M2_amortiguador_adicional": [11.2],
        "Estimado_con_amortiguador_IA": [890.4],
    })

    salida = ProyAst.simplificar_salida_amortiguada(
        proyeccion,
        ["Finca", "Bloque&Varid"],
    )

    assert salida.columns.tolist() == [
        "Finca",
        "Bloque&Varid",
        "M2 Amortiguador",
        "Proyeccion_con_amortiguador_IA",
    ]
    assert salida.iloc[0].tolist() == ["BL25", "001RED", 11, 890]


def test_export_masivo_solo_incluye_proyeccion_original():
    proyeccion = pd.DataFrame({
        "Anio": [2026],
        "Semana": [35],
        "Bloque&Varid": ["001RED"],
        "Estimado_modelo": [1000.4],
        "Estimado_con_amortiguador_IA": [890.4],
    })

    salida = ProyAst.preparar_salida_proyeccion_masiva(
        proyeccion,
        ["Anio", "Semana", "Bloque&Varid"],
    )

    assert salida.columns.tolist() == [
        "Anio",
        "Semana",
        "Bloque&Varid",
        "Estimado_modelo",
    ]
    assert salida.iloc[0].tolist() == [2026, 35, "001RED", 1000]


# --- construir_cache_patrones_semanales -----------------------------------

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


def test_construir_cache_patrones_semanales_agrega_duplicados_y_calcula_incrementos():
    df = pd.DataFrame(
        {
            "Bloque&Varid": ["A"] * 4,
            "Anio": [2025, 2025, 2025, 2025],
            "Semana": [2, 1, 1, 2],
            "Tallos/m2": [30, 10, 20, 50],
            "Produccion": [300, 100, 200, 500],
        }
    )

    patron = ProyAst.construir_cache_patrones_semanales(df)["A"]

    # Ordenado por Anio/Semana, promedio de tallos y suma de produccion por semana.
    assert patron["Semana"].tolist() == [1, 2]
    assert patron["Tallos_m2_patron"].tolist() == [15.0, 40.0]
    assert patron["Produccion_patron"].tolist() == [300.0, 800.0]
    assert patron["Incremento_tallos_patron"].tolist() == [0.0, 25.0]
    assert patron["Incremento_produccion_patron"].tolist() == [0.0, 500.0]


def test_construir_cache_patrones_semanales_descarta_semanas_no_numericas():
    df = pd.DataFrame(
        {
            "Bloque&Varid": ["A", "A"],
            "Anio": [2025, 2025],
            "Semana": [1, "sin dato"],
            "Tallos/m2": [10, 20],
            "Produccion": [100, 200],
        }
    )

    patron = ProyAst.construir_cache_patrones_semanales(df)["A"]

    assert patron["Semana"].tolist() == [1]


def test_construir_cache_patrones_semanales_sin_datos():
    assert ProyAst.construir_cache_patrones_semanales(None) == {}
    assert ProyAst.construir_cache_patrones_semanales(pd.DataFrame()) == {}


# --- deteccion y normalizacion de errores del proveedor -------------------

def test_es_error_modelo_inexistente_detecta_404():
    assert ProyAst.es_error_modelo_inexistente(
        RuntimeError("Error code: 404")) is True
    assert ProyAst.es_error_modelo_inexistente(
        RuntimeError("Model not found")) is True
    assert ProyAst.es_error_modelo_inexistente(
        RuntimeError("timeout")) is False


def test_es_error_modelo_inexistente_cubre_variantes_del_proveedor():
    for mensaje in [
        "model_not_found",
        "Invalid model requested",
        "unknown model: gpt-x",
        "The model is not available for your account",
        "not_found_error",
    ]:
        assert ProyAst.es_error_modelo_inexistente(
            RuntimeError(mensaje)) is True, mensaje

    assert ProyAst.es_error_modelo_inexistente(
        RuntimeError("Error code: 429 - rate limit")) is False


def test_normalizar_error_github_models_detecta_retiro():
    error = ProyAst.normalizar_error_github_models(
        RuntimeError(
            "Error code: 410 - github_models_retirement_brownout"
        )
    )

    assert "proceso de retiro" in str(error)
    assert "proveedor GitHub" in str(error)


def test_normalizar_error_github_models_detecta_410_con_retirement():
    error = ProyAst.normalizar_error_github_models(
        RuntimeError("Error code: 410 - Models retirement in progress"))

    assert "proceso de retiro" in str(error)


def test_normalizar_error_github_models_detecta_falta_de_permiso_models():
    error = ProyAst.normalizar_error_github_models(
        RuntimeError("Error code: 401 - models permission is required"))

    assert "permiso models" in str(error)


def test_normalizar_error_github_models_devuelve_el_error_original():
    original = RuntimeError("Error code: 500 - internal server error")

    assert ProyAst.normalizar_error_github_models(original) is original


def test_consultar_llm_no_cambia_de_github_si_esta_en_retiro(monkeypatch):
    llamadas = []
    monkeypatch.setattr(ProyAst, "obtener_llm_provider", lambda: "github")

    def fake_consultar_openai_compatible(prompt, proveedor):
        llamadas.append((prompt, proveedor))
        raise RuntimeError(
            "GitHub Models no esta disponible temporalmente por su proceso "
            "de retiro. La consulta se mantuvo en el proveedor GitHub."
        )

    monkeypatch.setattr(
        ProyAst, "consultar_openai_compatible", fake_consultar_openai_compatible)

    with pytest.raises(RuntimeError, match="proveedor GitHub"):
        ProyAst.consultar_llm("analiza estos datos")

    assert llamadas == [("analiza estos datos", "github")]
