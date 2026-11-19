"""Punto de entrada para abrir el diálogo de Agente TRIZ."""

import sys
import tkinter as tk
from tkinter import messagebox

from .agent import AgenteTRIZ
from .dialogo import mostrar_dialogo_solucion


def main():
    """Inicia el agente y abre la ventana de exploración de soluciones."""
    agente = None
    try:
        agente = AgenteTRIZ()
        mostrar_dialogo_solucion(agente)
    except (RuntimeError, ImportError, tk.TclError) as exc:
        mensaje = f"No se pudo iniciar Agente TRIZ: {exc}"
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Agente TRIZ", mensaje, parent=root)
            root.destroy()
        except tk.TclError:
            print(mensaje, file=sys.stderr)
        return 1
    finally:
        if agente is not None:
            agente.cerrar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
