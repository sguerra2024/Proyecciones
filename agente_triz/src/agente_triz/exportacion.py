"""Exportación local de respuestas de Agente TRIZ."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional
from xml.sax.saxutils import escape


FORMATOS = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".xlsx": "Excel",
}
TITULO = "Análisis de Agente TRIZ"


def _guardar_pdf(ruta: Path, texto: str) -> None:
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    estilos = getSampleStyleSheet()
    elementos = [Paragraph(TITULO, estilos["Title"])]
    for linea in texto.split("\n"):
        contenido = escape(linea.expandtabs(4))
        elementos.append(Paragraph(contenido or "&#160;", estilos["Normal"]))
    SimpleDocTemplate(str(ruta), title=TITULO).build(elementos)


def _guardar_word(ruta: Path, texto: str) -> None:
    from docx import Document

    documento = Document()
    documento.add_heading(TITULO, level=0)
    for linea in texto.split("\n"):
        documento.add_paragraph(linea)
    documento.save(str(ruta))


def _guardar_excel(ruta: Path, texto: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    libro = Workbook()
    hoja = libro.active
    assert hoja is not None
    hoja.title = "Análisis TRIZ"
    hoja.append([TITULO])
    hoja["A1"].font = Font(bold=True, size=14)
    hoja.freeze_panes = "A2"
    hoja.column_dimensions["A"].width = 110
    for linea in texto.split("\n"):
        for inicio in range(0, max(1, len(linea)), 32767):
            hoja.append([linea[inicio:inicio + 32767]])
            celda = hoja.cell(row=hoja.max_row, column=1)
            celda.data_type = "s"
            celda.alignment = Alignment(wrap_text=True, vertical="top")
    libro.save(ruta)


def exportar_analisis(ruta: Path, texto: str) -> None:
    """Escribe el formato indicado por la extensión sin dejar archivos parciales."""
    if not texto.strip():
        raise ValueError("No hay un análisis para exportar.")
    escritores = {".pdf": _guardar_pdf, ".docx": _guardar_word, ".xlsx": _guardar_excel}
    escritor = escritores.get(ruta.suffix.lower())
    if escritor is None:
        raise ValueError("Elige un archivo PDF (.pdf), Word (.docx) o Excel (.xlsx).")
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    with NamedTemporaryFile(dir=ruta.parent, suffix=ruta.suffix, delete=False) as archivo:
        temporal = Path(archivo.name)
    try:
        escritor(temporal, texto)
        temporal.replace(ruta)
    finally:
        temporal.unlink(missing_ok=True)


def crear_menu_exportacion(parent, obtener_analisis: Callable[[], Optional[str]]):
    """Crea un menú de guardado para el análisis mostrado en la interfaz."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    ventana = parent.winfo_toplevel()

    def guardar(extension: str) -> None:
        texto = obtener_analisis()
        if not texto or not texto.strip():
            messagebox.showwarning(TITULO, "Genera un análisis antes de exportar.", parent=ventana)
            return
        formato = FORMATOS[extension]
        try:
            destino = filedialog.asksaveasfilename(
                parent=ventana,
                title=f"Exportar análisis a {formato}",
                initialfile=f"Analisis_TRIZ{extension}",
                defaultextension=extension,
                filetypes=[(formato, f"*{extension}")],
            )
            if not destino:
                return
            ruta = Path(destino)
            if ruta.suffix.lower() != extension:
                raise ValueError(f"El archivo debe tener la extensión {extension}.")
            exportar_analisis(ruta, texto)
        except PermissionError as exc:
            messagebox.showerror(
                TITULO,
                f"No se pudo guardar el archivo. Ciérralo si está abierto o elige "
                f"otra carpeta con permisos de escritura.\n\n{exc}",
                parent=ventana,
            )
            return
        except (ImportError, OSError, ValueError, tk.TclError) as exc:
            messagebox.showerror(TITULO, f"No se pudo exportar el análisis:\n{exc}", parent=ventana)
            return
        messagebox.showinfo(TITULO, f"Análisis guardado en:\n{ruta}", parent=ventana)

    boton = ttk.Menubutton(parent, text="Exportar análisis", state="disabled")
    menu = tk.Menu(boton, tearoff=False)
    for extension, formato in FORMATOS.items():
        menu.add_command(
            label=f"{formato} ({extension})",
            command=lambda extension=extension: guardar(extension),
        )
    boton.configure(menu=menu)
    return boton
