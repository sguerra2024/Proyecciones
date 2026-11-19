"""Carga y filtrado local del banco de referencias inspiradoras."""

import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4
from urllib.parse import urlparse

from .principios import ETIQUETAS_PRINCIPIOS
from .vias import VIAS_SOLUCION


RUTA_BANCO_PLANTILLA = (
    Path(__file__).resolve().parent / "data" / "banco_inspiracion.json"
)
RUTA_BANCO_USUARIO = (
    Path(os.getenv("APPDATA", Path.home()))
    / "AgenteTRIZ"
    / "banco_inspiracion.json"
)


def _resolver_ruta_banco(ruta_banco=None):
    if ruta_banco is not None:
        path = Path(ruta_banco)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                RUTA_BANCO_PLANTILLA.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        return path
    path = RUTA_BANCO_USUARIO
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            RUTA_BANCO_PLANTILLA.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return path


def cargar_banco_inspiracion(ruta_banco=None):
    """Carga entradas de inspiración desde JSON sin acceder a sus enlaces."""
    path = _resolver_ruta_banco(ruta_banco)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el banco de inspiración: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"No se pudo leer el banco de inspiración {path}: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("entradas"), list):
        raise ValueError(
            "El banco de inspiración debe ser un objeto JSON con una lista 'entradas'."
        )

    entradas = []
    for indice, entrada in enumerate(data["entradas"], start=1):
        if not isinstance(entrada, dict):
            raise ValueError(f"La entrada {indice} del banco debe ser un objeto.")
        if not {"titulo", "url", "herramientas_triz"}.issubset(entrada):
            raise ValueError(
                f"La entrada {indice} debe incluir titulo, url y herramientas_triz."
            )

        titulo = str(entrada["titulo"]).strip()
        url = str(entrada["url"]).strip()
        herramientas = entrada["herramientas_triz"]
        if not titulo or not url or not isinstance(herramientas, list):
            raise ValueError(
                f"La entrada {indice} debe tener título, URL y herramientas_triz en lista."
            )
        if not all(isinstance(herramienta, str) for herramienta in herramientas):
            raise ValueError(
                f"Todas las herramientas_triz de la entrada {indice} deben ser texto."
            )
        principios = entrada.get("principios_inventivos", [])
        if not isinstance(principios, list) or not all(
            isinstance(principio, str) for principio in principios
        ):
            raise ValueError(
                f"principios_inventivos de la entrada {indice} debe ser una lista de texto."
            )
        if titulo.casefold().startswith("reemplazar por") or "example.org" in url:
            continue
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(
                f"La URL de la entrada {indice} debe usar http o https: {url}"
            )
        desconocidas = set(herramientas) - set(VIAS_SOLUCION)
        if desconocidas:
            raise ValueError(
                f"La entrada {indice} contiene vías TRIZ desconocidas: "
                + ", ".join(sorted(desconocidas))
            )
        principios_desconocidos = set(principios) - set(ETIQUETAS_PRINCIPIOS)
        if principios_desconocidos:
            raise ValueError(
                f"La entrada {indice} contiene principios inventivos desconocidos: "
                + ", ".join(sorted(principios_desconocidos))
            )
        entradas.append({**entrada, "titulo": titulo, "url": url})
    return entradas


def filtrar_banco_por_vias(
    entradas, vias_seleccionadas, principios_activos=()
):
    """Devuelve en orden original las referencias que coinciden con las vías."""
    vias = set(vias_seleccionadas or ())
    principios = set(principios_activos or ())
    if not vias:
        return []
    resultados = []
    for entrada in entradas:
        herramientas = set(entrada["herramientas_triz"])
        coincide_otra_via = bool(
            (vias - {"principios_inventivos"}) & herramientas
        )
        coincide_principio = (
            "principios_inventivos" in vias
            and "principios_inventivos" in herramientas
            and (
                not principios
                or principios.intersection(
                    entrada.get("principios_inventivos", [])
                )
            )
        )
        if coincide_otra_via or coincide_principio:
            resultados.append(entrada)
    return resultados


def agregar_referencia_inspiracion(
    titulo,
    url,
    principios_inventivos,
    notas_originales="",
    ruta_banco=None,
):
    """Guarda en el banco local un enlace asociado a principios activos."""
    titulo = str(titulo or "").strip()
    url = str(url or "").strip()
    principios = list(dict.fromkeys(principios_inventivos or ()))
    if not titulo:
        raise ValueError("El título del enlace es obligatorio.")
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Ingresa un enlace completo que empiece con http:// o https://.")
    if not principios:
        raise ValueError("Selecciona al menos un principio inventivo.")
    desconocidos = set(principios) - set(ETIQUETAS_PRINCIPIOS)
    if desconocidos:
        raise ValueError(
            "Principios inventivos desconocidos: " + ", ".join(sorted(desconocidos))
        )

    path = _resolver_ruta_banco(ruta_banco)
    entradas = cargar_banco_inspiracion(path)
    entrada = {
        "id": str(uuid4()),
        "tipo": "enlace",
        "titulo": titulo,
        "url": url,
        "idioma": "",
        "pregunta_inspiradora": "",
        "notas_originales": str(notas_originales or "").strip(),
        "etiquetas": [],
        "herramientas_triz": ["principios_inventivos"],
        "principios_inventivos": principios,
        "fuente": {
            "autor": "",
            "organizacion": "",
            "licencia": "",
            "url_licencia": "",
            "atribucion_requerida": "",
        },
    }
    entradas.append(entrada)
    data = {
        "version": 1,
        "descripcion": "Banco local de enlaces de inspiración etiquetados por vía y principio TRIZ.",
        "entradas": entradas,
    }
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as archivo:
            temporal = Path(archivo.name)
            json.dump(data, archivo, ensure_ascii=False, indent=2)
            archivo.write("\n")
        temporal.replace(path)
    except OSError:
        if temporal is not None:
            temporal.unlink(missing_ok=True)
        raise
    return entrada
