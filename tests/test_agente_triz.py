from pathlib import Path
import sys
from threading import get_ident
from time import monotonic
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_SOURCE = ROOT / "agente_triz" / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from agente_triz import (
    AgenteTRIZ,
    BaseConocimiento,
    agregar_referencia_inspiracion,
    cargar_banco_inspiracion,
    construir_prompt_solucion,
    calcular_estadisticas,
    filtrar_banco_por_vias,
    mostrar_dialogo_solucion,
)
import agente_triz.agent as agent_module
import agente_triz.__main__ as entrypoint
from agente_triz.principios import (
    DESCRIPCIONES_PRINCIPIOS,
    PRINCIPIOS_INVENTIVOS,
)


def test_calcular_estadisticas_acepta_valores_de_diccionario():
    resultado = calcular_estadisticas({"semana_1": 10, "semana_2": 20})

    assert resultado["media"] == 15
    assert resultado["conteo"] == 2


def test_catalogo_contiene_40_principios_unicos():
    ids = [identificador for identificador, _ in PRINCIPIOS_INVENTIVOS]

    assert len(ids) == 40
    assert len(set(ids)) == 40
    assert set(ids) == set(DESCRIPCIONES_PRINCIPIOS)
    assert all(DESCRIPCIONES_PRINCIPIOS.values())
    assert PRINCIPIOS_INVENTIVOS[4][1] == "5 Consolidación"
    assert PRINCIPIOS_INVENTIVOS[12][1] == "13 Haga lo contrario"
    assert "Divida el objeto en partes independientes" in DESCRIPCIONES_PRINCIPIOS[
        "01_segmentacion"
    ]


def test_base_conocimiento_formatea_capitulo_relevante():
    base = BaseConocimiento(":memory:")
    base.agregar_capitulo(
        1,
        "Causa raíz",
        "Investiga por qué ocurre la falla.",
        palabras_clave="causa raíz falla",
    )

    contexto = base.formatear_contexto("¿Cómo identificar la causa raíz?")

    assert "Capítulo 1: Causa raíz" in contexto
    base.cerrar()


def test_consultar_incluye_contexto_y_devuelve_respuesta():
    agente = AgenteTRIZ.__new__(AgenteTRIZ)
    agente.conocimiento = None
    agente.modelo = "modelo-prueba"
    agente.cliente = MagicMock()
    opcion = MagicMock()
    opcion.message.content = "Respuesta de prueba"
    agente.cliente.chat.completions.create.return_value.choices = [opcion]

    respuesta = agente.consultar("Resuelve el problema.", "Contexto de prueba")

    assert respuesta == "Respuesta de prueba"
    llamada = agente.cliente.chat.completions.create.call_args.kwargs
    assert llamada["model"] == "modelo-prueba"
    assert "Contexto de prueba" in llamada["messages"][1]["content"]


def test_dialogo_consulta_en_segundo_plano_con_contexto_sqlite(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    from tkinter import ttk
    from tkinter import filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Pantalla Tkinter no disponible: {exc}")

    hilo_ventana = get_ident()
    hilos_envio = []
    cliente = MagicMock()
    opcion = MagicMock()
    opcion.message.content = "Alternativas de prueba"

    def simular_envio(**kwargs):
        hilos_envio.append(get_ident())
        return MagicMock(choices=[opcion])

    cliente.chat.completions.create.side_effect = simular_envio
    monkeypatch.setattr(agent_module, "OpenAI", lambda **kwargs: cliente)
    agente = AgenteTRIZ(api_key="test-key", ruta_conocimiento=":memory:")
    agente.conocimiento.agregar_capitulo(
        1, "Causa raíz", "Distingue causas de síntomas.", palabras_clave="causa raíz"
    )

    def widgets(parent):
        for child in parent.winfo_children():
            yield child
            yield from widgets(child)

    vistos = []
    destino = tmp_path / "analisis.docx"
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kwargs: str(destino))
    monkeypatch.setattr(messagebox, "showinfo", lambda *args, **kwargs: None)
    limite = monotonic() + 5

    def iniciar():
        dialogo = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
        children = list(widgets(dialogo))
        problema, _, resultado = [w for w in children if isinstance(w, ScrolledText)]
        botones = {w.cget("text"): w for w in children if isinstance(w, ttk.Button)}
        exportar = next(w for w in children if isinstance(w, ttk.Menubutton))
        menu = exportar.nametowidget(str(exportar.cget("menu")))
        botones["Exportar análisis"] = exportar
        visor = next(w for w in children if isinstance(w, tk.Canvas))
        dialogo.geometry("520x440+20+40")
        dialogo.update_idletasks()
        posiciones = [
            (b.winfo_rooty(), b.winfo_height()) for b in botones.values()
        ]
        cabe_en_ventana = all(
            dialogo.winfo_rooty() <= y
            and y + alto <= dialogo.winfo_rooty() + dialogo.winfo_height()
            for y, alto in posiciones
        )
        visor.yview_moveto(1)
        dialogo.update_idletasks()
        pie_fijo = posiciones == [
            (b.winfo_rooty(), b.winfo_height()) for b in botones.values()
        ]
        geometria.extend([cabe_en_ventana, visor.yview()[0] > 0, pie_fijo])
        estados_exportacion.append(str(exportar.cget("state")))
        opciones.extend(menu.entrycget(i, "label") for i in range(3))
        problema.insert("1.0", "Investigar la causa raíz de un error.")
        botones["Explorar soluciones"].invoke()

        def esperar():
            texto = resultado.get("1.0", "end").strip()
            if texto or monotonic() >= limite:
                vistos.append(texto)
                estados_exportacion.append(str(exportar.cget("state")))
                menu.invoke(1)
                botones["Cerrar"].invoke()
            else:
                root.after(20, esperar)

        esperar()

    geometria = []
    estados_exportacion = []
    opciones = []
    inicio = root.after_idle(iniciar)
    alarma = root.after(6000, root.destroy)
    try:
        assert mostrar_dialogo_solucion(
            agente, parent=root, ruta_banco=tmp_path / "banco.json"
        ) == "Alternativas de prueba"
        assert geometria == [True, True, True]
        assert estados_exportacion == ["disabled", "normal"]
        assert opciones == ["PDF (.pdf)", "Word (.docx)", "Excel (.xlsx)"]
        from docx import Document

        assert Document(str(destino)).paragraphs[1].text == "Alternativas de prueba"
        assert vistos == ["Alternativas de prueba"]
        assert len(hilos_envio) == 1 and hilos_envio[0] != hilo_ventana
        llamada = cliente.chat.completions.create.call_args.kwargs
        assert "Distingue causas de síntomas." in llamada["messages"][1]["content"]
    finally:
        agente.cerrar()
        try:
            root.after_cancel(inicio)
            root.after_cancel(alarma)
            root.destroy()
        except tk.TclError:
            pass


def test_constructor_carga_conocimiento_incluido(monkeypatch):
    monkeypatch.setattr(agent_module, "OpenAI", lambda **kwargs: MagicMock())
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    agente = AgenteTRIZ()

    assert agente.conocimiento is not None
    titulos = {
        capitulo["titulo"] for capitulo in agente.conocimiento.listar_capitulos()
    }
    assert "Naturaleza y tipos de problemas" in titulos
    assert (
        "Fundamentos TRIZ: enfoque sistemático para resolver problemas inventivos"
        in titulos
    )
    contexto = agente.conocimiento.formatear_contexto(
        "principios inventivos para resolver una contradicción definida"
    )
    assert "herramienta 4" in contexto.lower()
    assert "solo después de definir el problema" in contexto.lower()
    assert "contradicción técnica o física" in contexto.lower()
    agente.cerrar()


def test_calcular_estadisticas_rechaza_serie_vacia():
    with pytest.raises(ValueError, match="No hay datos numéricos"):
        calcular_estadisticas([])


def test_prompt_dialogo_incluye_problema_enlaces_y_vias_seleccionadas():
    prompt = construir_prompt_solucion(
        "Evitar que una máquina dañe el producto.",
        "https://example.com/video\n\nhttps://example.com/article",
        ["problema_generico", "contradicciones"],
    )

    assert "Evitar que una máquina dañe el producto." in prompt
    assert "https://example.com/video" in prompt
    assert "https://example.com/article" in prompt
    assert "uno genérico" in prompt
    assert "contradicción técnica o física" in prompt
    assert "no afirmes haberlos abierto" in prompt.lower()
    assert "transferir" not in prompt


def test_prompt_dialogo_requiere_problema_y_al_menos_una_via():
    with pytest.raises(ValueError, match="Describe el problema"):
        construir_prompt_solucion("", "", ["problema_generico"])

    with pytest.raises(ValueError, match="Selecciona al menos una vía"):
        construir_prompt_solucion("Problema de prueba", "", [])


def test_banco_filtra_recursos_por_vias(tmp_path):
    banco_path = tmp_path / "banco.json"
    banco_path.write_text(
        """{
          "version": 1,
          "entradas": [
            {
              "id": "video-1",
              "titulo": "Separar componentes",
              "url": "https://example.com/video",
              "herramientas_triz": ["problema_generico", "contradicciones"],
              "principios_inventivos": ["01_segmentacion"]
            },
            {
              "id": "articulo-1",
              "titulo": "Revisar la acción",
              "url": "https://example.com/article",
              "herramientas_triz": ["transferir_accion"]
            },
            {
              "id": "principio-1",
              "titulo": "Separar por etapas",
              "url": "https://example.com/principle",
              "herramientas_triz": ["principios_inventivos"],
              "principios_inventivos": ["13_inversion"]
            }
          ]
        }""",
        encoding="utf-8",
    )

    banco = cargar_banco_inspiracion(banco_path)
    filtrados = filtrar_banco_por_vias(
        banco, ["contradicciones", "principios_inventivos"], ["13_inversion"]
    )

    assert [entrada["id"] for entrada in filtrados] == ["video-1", "principio-1"]


def test_agregar_referencia_persiste_principios_seleccionados(tmp_path):
    banco_path = tmp_path / "nested" / "banco.json"
    entrada = agregar_referencia_inspiracion(
        "Idea de segmentación",
        "https://example.com/idea",
        ["01_segmentacion", "13_inversion"],
        "Separar la acción en dos etapas.",
        banco_path,
    )

    guardadas = cargar_banco_inspiracion(banco_path)

    assert guardadas == [entrada]
    assert guardadas[0]["herramientas_triz"] == ["principios_inventivos"]
    assert guardadas[0]["principios_inventivos"] == [
        "01_segmentacion",
        "13_inversion",
    ]


def test_banco_rechaza_vias_desconocidas(tmp_path):
    banco_path = tmp_path / "banco.json"
    banco_path.write_text(
        """{
          "entradas": [{
            "titulo": "Referencia",
            "url": "https://example.com",
            "herramientas_triz": ["via_inexistente"]
          }]
        }""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vías TRIZ desconocidas"):
        cargar_banco_inspiracion(banco_path)


def test_main_abre_dialogo_y_cierra_agente(monkeypatch):
    agente = MagicMock()
    monkeypatch.setattr(entrypoint, "AgenteTRIZ", lambda: agente)
    mostrar = MagicMock()
    monkeypatch.setattr(entrypoint, "mostrar_dialogo_solucion", mostrar)

    assert entrypoint.main() == 0

    mostrar.assert_called_once_with(agente)
    agente.cerrar.assert_called_once_with()


def test_main_informa_error_y_devuelve_codigo_no_cero(monkeypatch):
    monkeypatch.setattr(
        entrypoint,
        "AgenteTRIZ",
        lambda: (_ for _ in ()).throw(RuntimeError("falta API key")),
    )
    error = MagicMock()
    monkeypatch.setattr(entrypoint.messagebox, "showerror", error)
    monkeypatch.setattr(entrypoint.tk, "Tk", lambda: MagicMock())

    assert entrypoint.main() == 1

    assert "falta API key" in error.call_args.args[1]
