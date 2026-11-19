"""Diálogo de escritorio para enfocar un problema y explorar vías TRIZ."""

from queue import Empty, Queue
from sqlite3 import Error as SQLiteError
from threading import Thread

from .inspiracion import (
    agregar_referencia_inspiracion,
    cargar_banco_inspiracion,
    filtrar_banco_por_vias,
)
from .principios import (
    DESCRIPCIONES_PRINCIPIOS,
    ETIQUETAS_PRINCIPIOS,
    PRINCIPIOS_INVENTIVOS,
)
from .vias import VIAS_SOLUCION


def construir_prompt_solucion(
    problema, enlaces, vias_seleccionadas, principios_activos=()
):
    """Valida los campos y prepara la consulta para explorar las vías elegidas."""
    problema = str(problema or "").strip()
    if not problema:
        raise ValueError("Describe el problema antes de solicitar alternativas.")

    vias = [
        clave for clave in VIAS_SOLUCION
        if clave in set(vias_seleccionadas or ())
    ]
    if not vias:
        raise ValueError("Selecciona al menos una vía de solución.")

    enlaces_limpios = [
        linea.strip()
        for linea in str(enlaces or "").splitlines()
        if linea.strip()
    ]
    secciones = [
        "Enfoque del problema:\n" + problema,
        "Vías de solución seleccionadas:\n"
        + "\n".join(VIAS_SOLUCION[via] for via in vias),
    ]
    nombres_principios = [
        ETIQUETAS_PRINCIPIOS[principio]
        for principio in principios_activos
        if principio in ETIQUETAS_PRINCIPIOS
    ]
    if nombres_principios:
        secciones.append(
            "Principios inventivos activos para orientar la inspiración:\n"
            + "\n".join(f"- {nombre}" for nombre in nombres_principios)
        )
    if enlaces_limpios:
        secciones.append(
            "Enlaces de inspiración proporcionados por el usuario:\n"
            + "\n".join(f"- {enlace}" for enlace in enlaces_limpios)
            + "\nTrata los enlaces como referencias e ideas sugeridas por el "
            "usuario. No afirmes haberlos abierto, visto ni verificado. Si su "
            "contenido no está incluido en la conversación, pide un resumen o "
            "explica que no puedes evaluar su contenido solo a partir del URL."
        )
    secciones.append(
        "Guía la exploración para estimular la imaginación humana: muestra "
        "preguntas y analogías que ayuden a descubrir caminos, además de "
        "alternativas concretas. Separa hechos de hipótesis, relaciona cada "
        "alternativa con el problema definido y sus restricciones, y propone "
        "cómo probarla. No presentes una inspiración como solución validada."
    )
    return "\n\n".join(secciones)


def mostrar_dialogo_solucion(agente, parent=None, ruta_banco=None):
    """Abre un diálogo modal y devuelve el análisis, o None si se cancela.

    Tkinter es parte de la biblioteca estándar. Si no se proporciona una
    ventana padre, el diálogo crea una raíz propia para poder ejecutarse solo.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter no está disponible en este entorno de Python."
        ) from exc

    from tkinter.scrolledtext import ScrolledText
    from .exportacion import crear_menu_exportacion

    banco = cargar_banco_inspiracion(ruta_banco)
    propia = parent is None
    root = tk.Tk() if propia else parent
    if propia:
        root.withdraw()

    dialogo = tk.Toplevel(root)
    dialogo.title("Agente TRIZ — enfoque y vías de solución")
    ancho = min(800, dialogo.winfo_screenwidth() - 40)
    alto = min(620, dialogo.winfo_screenheight() - 100)
    dialogo.geometry(f"{ancho}x{alto}")
    dialogo.minsize(min(480, ancho), min(340, alto))
    if not propia:
        dialogo.transient(root)

    pie = ttk.Frame(dialogo, padding=(10, 4))
    pie.pack(side="bottom", fill="x")
    cuerpo = ttk.Frame(dialogo)
    cuerpo.pack(fill="both", expand=True)
    visor = tk.Canvas(cuerpo, highlightthickness=0, borderwidth=0)
    barra = ttk.Scrollbar(cuerpo, orient="vertical", command=visor.yview)
    visor.configure(yscrollcommand=barra.set)
    barra.pack(side="right", fill="y")
    visor.pack(side="left", fill="both", expand=True)
    contenedor = ttk.Frame(visor, padding=(10, 6))
    panel = visor.create_window(0, 0, window=contenedor, anchor="nw")
    contenedor.columnconfigure(0, weight=1)
    contenedor.rowconfigure(2, weight=1)

    def ajustar_contenido(event):
        visor.itemconfigure(panel, width=event.width)
        for etiqueta in etiquetas_ajustables:
            etiqueta.configure(wraplength=max(1, event.width - 44))

    ttk.Label(
        contenedor,
        text="1. Enfoca el problema",
        font=("", 11, "bold"),
    ).grid(row=0, column=0, sticky="w")
    instrucciones = ttk.Label(
        contenedor,
        text=(
            "Describe el efecto, la función afectada, el cambio esperado y "
            "las restricciones conocidas."
        ),
        wraplength=ancho - 44,
    )
    instrucciones.grid(row=1, column=0, sticky="w", pady=(1, 3))
    problema = ScrolledText(contenedor, height=3, width=1, wrap="word")
    problema.grid(row=2, column=0, sticky="nsew", pady=(0, 6))

    ttk.Label(
        contenedor,
        text="2. Selecciona las vías de solución",
        font=("", 10, "bold"),
    ).grid(row=3, column=0, sticky="w", pady=(0, 4))
    variables = {
        "problema_generico": tk.BooleanVar(value=True),
        "transferir_accion": tk.BooleanVar(value=True),
        "contradicciones": tk.BooleanVar(value=True),
        "principios_inventivos": tk.BooleanVar(value=True),
    }
    for fila, (clave, descripcion) in enumerate(VIAS_SOLUCION.items(), start=4):
        ttk.Checkbutton(
            contenedor,
            text=descripcion,
            variable=variables[clave],
            command=lambda: actualizar_enlaces(),
        ).grid(row=fila, column=0, sticky="w")

    fila_principios = 4 + len(VIAS_SOLUCION)
    ttk.Label(
        contenedor,
        text="3. Principios inventivos activos (filtro opcional)",
        font=("", 10, "bold"),
    ).grid(row=fila_principios, column=0, sticky="w", pady=(8, 2))
    ttk.Label(
        contenedor,
        text="Selecciona uno o más para filtrar los enlaces asociados.",
        wraplength=760,
    ).grid(row=fila_principios + 1, column=0, sticky="w", pady=(0, 4))
    filtro_frame = ttk.Frame(contenedor)
    filtro_frame.grid(
        row=fila_principios + 2, column=0, sticky="ew", pady=(0, 8)
    )
    filtro_frame.columnconfigure(0, weight=1)
    principios_activos = tk.Listbox(
        filtro_frame,
        selectmode="extended",
        height=3,
        exportselection=False,
    )
    principios_scroll = ttk.Scrollbar(
        filtro_frame, orient="vertical", command=principios_activos.yview
    )
    principios_activos.configure(yscrollcommand=principios_scroll.set)
    principios_activos.grid(row=0, column=0, sticky="ew")
    principios_scroll.grid(row=0, column=1, sticky="ns")
    for _, nombre in PRINCIPIOS_INVENTIVOS:
        principios_activos.insert("end", nombre)
    detalle_principios = ttk.Label(
        filtro_frame,
        text="Selecciona un principio para consultar su descripción.",
        justify="left",
        wraplength=ancho - 44,
    )
    detalle_principios.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def actualizar_principios_seleccionados(event=None):
        if event is not None and event.widget is not principios_activos:
            return
        indices = principios_activos.curselection()
        if indices:
            detalle_principios.configure(
                text="\n\n".join(
                    f"{PRINCIPIOS_INVENTIVOS[indice][1]}: "
                    f"{DESCRIPCIONES_PRINCIPIOS[PRINCIPIOS_INVENTIVOS[indice][0]]}"
                    for indice in indices
                )
            )
        else:
            detalle_principios.configure(
                text="Selecciona un principio para consultar su descripción."
            )
        actualizar_enlaces()

    principios_activos.bind(
        "<<ListboxSelect>>", actualizar_principios_seleccionados
    )

    fila_enlaces = fila_principios + 3
    ttk.Label(
        contenedor,
        text="4. Enlaces del banco para las vías y principios activos",
        font=("", 10, "bold"),
    ).grid(row=fila_enlaces, column=0, sticky="w", pady=(4, 2))
    aviso_enlaces = ttk.Label(
        contenedor,
        text="Los enlaces se muestran como referencias; no se abren ni verifican automáticamente.",
        wraplength=ancho - 44,
    )
    aviso_enlaces.grid(row=fila_enlaces + 1, column=0, sticky="w", pady=(0, 3))
    etiquetas_ajustables = (instrucciones, detalle_principios, aviso_enlaces)
    visor.bind("<Configure>", ajustar_contenido)
    contenedor.bind(
        "<Configure>", lambda _: visor.configure(scrollregion=visor.bbox(panel))
    )
    enlaces = ScrolledText(
        contenedor, height=2, width=1, wrap="word", state="disabled"
    )
    enlaces.grid(row=fila_enlaces + 2, column=0, sticky="nsew", pady=(0, 4))

    fila_resultado = fila_enlaces + 3
    ttk.Label(
        contenedor,
        text="Análisis e ideas para explorar",
        font=("", 10, "bold"),
    ).grid(row=fila_resultado, column=0, sticky="w", pady=(10, 4))
    contenedor.rowconfigure(fila_resultado + 1, weight=1)
    resultado = ScrolledText(
        contenedor, height=4, width=1, wrap="word", state="disabled"
    )
    resultado.grid(row=fila_resultado + 1, column=0, sticky="nsew")

    estado = tk.StringVar(value="Listo")
    ttk.Label(pie, textvariable=estado).pack(anchor="w")
    acciones = ttk.Frame(pie)
    acciones.pack(anchor="e", pady=(3, 0))
    respuesta_final = {"texto": None}
    recursos_filtrados = []
    resultados_worker = Queue()
    cerrar_poller = {"after_id": None}

    def poner_resultado(texto):
        resultado.configure(state="normal")
        resultado.delete("1.0", "end")
        resultado.insert("1.0", texto)
        resultado.configure(state="disabled")

    def actualizar_enlaces(event=None):
        nonlocal recursos_filtrados
        if event is not None and event.widget is not principios_activos:
            return
        vias_activas = [
            clave for clave, variable in variables.items() if variable.get()
        ]
        indices_principios = principios_activos.curselection()
        ids_principios = [
            PRINCIPIOS_INVENTIVOS[indice][0]
            for indice in indices_principios
        ]
        recursos_filtrados = filtrar_banco_por_vias(
            banco, vias_activas, ids_principios
        )
        if recursos_filtrados:
            lineas = []
            for recurso in recursos_filtrados:
                lineas.append(f"{recurso['titulo']} ({recurso.get('tipo', 'enlace')})")
                lineas.append(recurso["url"])
                etiquetas = [
                    ETIQUETAS_PRINCIPIOS[principio]
                    for principio in recurso.get("principios_inventivos", [])
                    if principio in ETIQUETAS_PRINCIPIOS
                ]
                if etiquetas:
                    lineas.append("Principios: " + ", ".join(etiquetas))
                resumen = str(recurso.get("notas_originales", "")).strip()
                if resumen:
                    lineas.append(f"Idea: {resumen}")
                lineas.append("")
            texto = "\n".join(lineas).rstrip()
        elif not banco:
            texto = (
                "El banco está vacío. Agrega enlaces en "
                "la ventana de carga de enlaces."
            )
        else:
            texto = "No hay enlaces etiquetados para las vías seleccionadas."
        enlaces.configure(state="normal")
        enlaces.delete("1.0", "end")
        enlaces.insert("1.0", texto)
        enlaces.configure(state="disabled")

    def consultar():
        try:
            prompt = construir_prompt_solucion(
                problema.get("1.0", "end"),
                "\n".join(
                    f"{recurso['titulo']}: {recurso['url']}"
                    + (
                        f"\nIdea: {recurso['notas_originales']}"
                        if str(recurso.get("notas_originales", "")).strip()
                        else ""
                    )
                    for recurso in recursos_filtrados
                ),
                [
                    clave for clave, variable in variables.items()
                    if variable.get()
                ],
                [
                    PRINCIPIOS_INVENTIVOS[indice][0]
                    for indice in principios_activos.curselection()
                ],
            )
        except ValueError as exc:
            messagebox.showwarning("Información requerida", str(exc), parent=dialogo)
            return

        try:
            pregunta_completa = agente.preparar_consulta(prompt)
        except SQLiteError as exc:
            estado.set("No se pudo generar el análisis")
            poner_resultado(f"Error al preparar el contexto del agente:\n{exc}")
            return

        boton_consultar.configure(state="disabled")
        boton_exportar.configure(state="disabled")
        estado.set("Generando alternativas…")
        poner_resultado("")

        def ejecutar_consulta():
            try:
                resultados_worker.put(("ok", agente.enviar_consulta(pregunta_completa)))
            except RuntimeError as exc:
                resultados_worker.put(("error", str(exc)))

        Thread(target=ejecutar_consulta, daemon=True).start()
        revisar_resultado()

    def abrir_carga_enlace():
        editor = tk.Toplevel(dialogo)
        editor.title("Agregar enlace y principios inventivos")
        editor.geometry("620x650")
        editor.minsize(520, 520)
        editor.transient(dialogo)
        editor.grab_set()
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(7, weight=1)

        ttk.Label(editor, text="Título del recurso").grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 2)
        )
        titulo_input = ttk.Entry(editor)
        titulo_input.grid(row=1, column=0, sticky="ew", padx=12)
        ttk.Label(editor, text="Enlace (URL)").grid(
            row=2, column=0, sticky="w", padx=12, pady=(8, 2)
        )
        url_input = ttk.Entry(editor)
        url_input.grid(row=3, column=0, sticky="ew", padx=12)
        ttk.Label(editor, text="Idea o resumen propio (opcional)").grid(
            row=4, column=0, sticky="w", padx=12, pady=(8, 2)
        )
        notas_input = ScrolledText(editor, height=4, wrap="word")
        notas_input.grid(row=5, column=0, sticky="ew", padx=12)
        ttk.Label(
            editor,
            text="Selecciona los principios inventivos activos relacionados:",
        ).grid(row=6, column=0, sticky="nw", padx=12, pady=(8, 2))

        seleccion_frame = ttk.Frame(editor)
        seleccion_frame.grid(row=7, column=0, sticky="nsew", padx=12)
        seleccion_frame.columnconfigure(0, weight=1)
        seleccion_frame.rowconfigure(0, weight=1)
        principios_enlace = tk.Listbox(
            seleccion_frame,
            selectmode="extended",
            height=12,
            exportselection=False,
        )
        principios_enlace_scroll = ttk.Scrollbar(
            seleccion_frame,
            orient="vertical",
            command=principios_enlace.yview,
        )
        principios_enlace.configure(
            yscrollcommand=principios_enlace_scroll.set
        )
        principios_enlace.grid(row=0, column=0, sticky="nsew")
        principios_enlace_scroll.grid(row=0, column=1, sticky="ns")
        for _, nombre in PRINCIPIOS_INVENTIVOS:
            principios_enlace.insert("end", nombre)

        def guardar_enlace():
            indices = principios_enlace.curselection()
            ids = [PRINCIPIOS_INVENTIVOS[indice][0] for indice in indices]
            try:
                entrada = agregar_referencia_inspiracion(
                    titulo_input.get(),
                    url_input.get(),
                    ids,
                    notas_input.get("1.0", "end"),
                    ruta_banco,
                )
            except (OSError, ValueError) as exc:
                messagebox.showerror(
                    "No se pudo guardar el enlace", str(exc), parent=editor
                )
                return
            banco.append(entrada)
            actualizar_enlaces()
            messagebox.showinfo(
                "Enlace guardado",
                "El enlace quedó asociado a los principios seleccionados.",
                parent=editor,
            )
            editor.destroy()

        ttk.Button(
            editor, text="Guardar enlace", command=guardar_enlace
        ).grid(row=8, column=0, sticky="e", padx=12, pady=12)
        titulo_input.focus_set()

    def revisar_resultado():
        try:
            tipo, contenido = resultados_worker.get_nowait()
        except Empty:
            cerrar_poller["after_id"] = dialogo.after(100, revisar_resultado)
            return

        boton_consultar.configure(state="normal")
        if tipo == "ok":
            respuesta_final["texto"] = contenido
            poner_resultado(contenido)
            if contenido.strip():
                boton_exportar.configure(state="normal")
                estado.set("Análisis listo")
            else:
                estado.set("El agente no devolvió un análisis")
        else:
            estado.set("No se pudo generar el análisis")
            poner_resultado(f"Error al consultar el agente:\n{contenido}")

    def cerrar():
        if cerrar_poller["after_id"] is not None:
            try:
                dialogo.after_cancel(cerrar_poller["after_id"])
            except tk.TclError:
                pass
        dialogo.destroy()
        if propia:
            root.quit()

    boton_consultar = ttk.Button(
        acciones,
        text="Explorar soluciones",
        command=consultar,
    )
    boton_consultar.pack(side="left", padx=(0, 8))
    ttk.Button(
        acciones,
        text="Agregar enlace",
        command=abrir_carga_enlace,
    ).pack(side="left", padx=(0, 8))
    boton_exportar = crear_menu_exportacion(acciones, lambda: respuesta_final["texto"])
    boton_exportar.pack(side="left", padx=(0, 8))
    ttk.Button(acciones, text="Cerrar", command=cerrar).pack(side="left")
    dialogo.protocol("WM_DELETE_WINDOW", cerrar)
    actualizar_enlaces()
    dialogo.grab_set()
    problema.focus_set()

    if propia:
        root.mainloop()
        try:
            root.destroy()
        except tk.TclError:
            pass
    else:
        parent.wait_window(dialogo)

    return respuesta_final["texto"]
