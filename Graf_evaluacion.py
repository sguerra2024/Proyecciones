import matplotlib.pyplot as plt
import argparse
import math
import queue
import re
import sqlite3
import sys
import threading
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')

from evaluation_history import save_recent_evaluation_history
from model_configuration import (
    ModelConfiguration,
    load_model_configuration,
    save_model_configuration,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SERIES_CANDIDATES = [
    SCRIPT_DIR / 'temp_series_area2.py',
    SCRIPT_DIR.parent / 'temp_series_area2.py',
    SCRIPT_DIR.parents[1] / 'PycharmProjects' / 'temp_series_area2.py',
    Path.cwd() / 'temp_series_area2.py',
]
DEFAULT_SERIES_PATH = next(
    (path for path in SERIES_CANDIDATES if path.exists()),
    SERIES_CANDIDATES[0],
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / 'ultima_serie.pdf'
SUPPORTED_INPUT_EXTENSIONS = {'.txt', '.csv', '.py', '.xlsx', '.xls', '.xlsm'}
EXCEL_EXTENSIONS = {'.xlsx', '.xls', '.xlsm'}
CHART_TYPES = {
    'Línea': 'line',
    'Barras': 'bar',
    'Dispersión': 'scatter',
    'Escalones': 'step',
    'Área': 'area',
}


def _area_percentages(area_positive, area_negative):
    """Convierte las áreas firmadas en porcentajes del área absoluta total."""
    positive = max(float(area_positive), 0.0)
    negative = abs(min(float(area_negative), 0.0))
    total = positive + negative
    if total == 0:
        return 0.0, 0.0
    return 100.0 * positive / total, 100.0 * negative / total


def _cumulative_data_percentage(values, limit=0.15):
    """Porcentaje de observaciones dentro del rango [-limit, limit]."""
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return 0.0
    included = sum(1 for value in numeric_values if abs(value) <= limit)
    return 100.0 * included / len(numeric_values)


TEMAS_TRIZ = {
    'Ideas para reducir el error': (
        'Propone ideas TRIZ para reducir errores de un modelo Random Forest '
        'de proyeccion de produccion con un patron historico.'
    ),
    'Peso del patrón': (
        'Propone experimentos para elegir un peso global del patron historico '
        'entre 0 y 80% en el objetivo de entrenamiento de un Random Forest.'
    ),
    'Parámetros del modelo': (
        'Explica como experimentar con numero de arboles, profundidad maxima, '
        'minimo de muestras por hoja y minimo de muestras para dividir nodos '
        'de un Random Forest de regresion para reducir el error.'
    ),
    'Validación de ideas': (
        'Disena una validacion cronologica sin fuga de informacion para '
        'comparar pesos del patron y parametros de un Random Forest.'
    ),
    'Lean Production': (
        'Propone recomendaciones de Lean Production para el flujo de '
        'proyeccion, evaluacion y revision semanal de produccion. Distingue '
        'hipotesis de desperdicio de hechos observados y sugiere un piloto '
        'con indicadores de tiempo de ciclo, retrabajo o calidad del pronostico, '
        'segun corresponda. No presupongas fallas en el proceso ni apliques '
        'cambios al modelo.'
    ),
}


def _habilitar_paquete_triz():
    source = SCRIPT_DIR / 'agente_triz' / 'src'
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def consultar_triz_general(tema):
    if tema not in TEMAS_TRIZ:
        raise ValueError('Selecciona uno de los temas generales TRIZ.')
    _habilitar_paquete_triz()
    from dotenv import load_dotenv
    from agente_triz import AgenteTRIZ

    load_dotenv(SCRIPT_DIR / '.env', override=False)
    agente = AgenteTRIZ(ruta_conocimiento=':memory:')
    try:
        respuesta = agente.consultar(
            TEMAS_TRIZ[tema],
            contexto_adicional=(
                'Consulta general sin datos de evaluacion. No conoces el error '
                'observado ni los parametros activos. No inventes diagnosticos '
                'ni valores optimos. Formula hipotesis y experimentos usando '
                'TRIZ, distinguiendo hechos de supuestos. La meta de diseno '
                'es al menos 60% de observaciones semanales con '
                '%Dif=(real-modelo)/real entre -15% y +15%. Los cambios '
                'requieren aprobacion humana. No garantices mejoras.'
            ),
        )
        if not respuesta or not respuesta.strip():
            raise RuntimeError('El agente TRIZ no devolvió una respuesta.')
        return respuesta
    finally:
        agente.cerrar()


class AgenteEvaluacionTRIZ:
    """Diagnostico local y consultas TRIZ generales sin transmitir evaluaciones."""

    def __init__(self, values, column=None, rango_semanas=None):
        self.values = tuple(float(value) for value in values)
        if not self.values or not all(math.isfinite(v) for v in self.values):
            raise ValueError('El análisis requiere valores numéricos finitos.')
        self.column = column
        self.rango_semanas = rango_semanas
        self.es_error = ''.join(str(column).split()).casefold() in {
            '%dif', 'error_relativo'
        }

    def analizar(self):
        if not self.es_error:
            return (
                f'Se graficaron {len(self.values)} observaciones. '
                'Para interpretar subestimación, sobreestimación y la meta '
                'del 60%, selecciona explícitamente %Dif o error_relativo '
                '(proporción decimal: 0.15 = 15%). No se infiere que otra '
                'columna o una serie sin etiqueta represente errores.'
            )
        included = sum(abs(v) <= 0.15 for v in self.values)
        percentage = _cumulative_data_percentage(self.values)
        positive = sum(max(v, 0.0) for v in self.values)
        negative = sum(min(v, 0.0) for v in self.values)
        positive_pct, negative_pct = _area_percentages(positive, negative)
        required = (3 * len(self.values) + 4) // 5
        if abs(negative) > positive:
            bias = 'Predomina el área negativa: sobreestimación del modelo.'
        elif positive > abs(negative):
            bias = 'Predomina el área positiva: subestimación del modelo.'
        else:
            bias = 'Las áreas están equilibradas; esto no garantiza precisión.'
        return '\n'.join([
            f'Evaluación de {len(self.values)} observaciones de {self.column}.',
            f'Dentro de ±15% (incluye los extremos): {included}/'
            f'{len(self.values)} = {percentage:.1f}%.',
            'Meta: al menos 60%. '
            + ('CUMPLIDA.' if included >= required else 'NO CUMPLIDA.'),
            f'Faltan {max(required - included, 0)} observaciones dentro del '
            'intervalo para alcanzar la meta con este mismo total.',
            f'Área positiva: {positive:.3f} ({positive_pct:.1f}%); '
            f'área negativa: {negative:.3f} ({negative_pct:.1f}%).',
            bias,
            'Las áreas son sumas discretas de errores, no el porcentaje de '
            'datos dentro de ±15%. No se compensan errores de signos opuestos.',
            'Este es un diagnóstico histórico, no una validación de cambios '
            'ni una garantía de precisión futura. Este diagnóstico no '
            'modifica la configuración.',
        ])

    def responder(self, pregunta):
        return consultar_triz_general(pregunta)


def mostrar_ajustes_modelo(parent):
    import tkinter as tk
    from tkinter import messagebox

    try:
        config = load_model_configuration() or ModelConfiguration()
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        messagebox.showerror('Configuración del modelo', str(exc), parent=parent)
        return
    window = tk.Toplevel(parent)
    window.title('Ajustes globales del modelo — aprobación Admin')
    window.transient(parent)
    tk.Label(
        window, wraplength=560, justify='left',
        text=(
            'Aplicación local a las próximas proyecciones individuales, masivas '
            'y API que compartan esta base de acceso. El peso global reemplaza '
            'la selección automática por caso. No modifica la gráfica actual '
            'ni el amortiguador. No hay mejora validada: vuelve a proyectar '
            'y evaluar después del cambio.'
        ),
    ).grid(row=0, column=0, columnspan=2, padx=10, pady=10)
    fields = (
        ('pattern_weight', 'Peso del patrón (%) [0–80]', config.pattern_weight * 100),
        ('n_estimators', 'Número de árboles [10–1000]', config.n_estimators),
        ('max_depth', 'Profundidad máxima [1–64]', config.max_depth),
        ('min_samples_leaf', 'Mínimo de muestras por hoja [1–100]', config.min_samples_leaf),
        ('min_samples_split', 'Mínimo de muestras para dividir [2–200]', config.min_samples_split),
    )
    entries = {}
    for row, (key, label, value) in enumerate(fields, start=1):
        tk.Label(window, text=label).grid(row=row, column=0, sticky='w', padx=10)
        entry = tk.Entry(window)
        entry.insert(0, str(value))
        entry.grid(row=row, column=1, padx=10, pady=3)
        entries[key] = entry
    tk.Label(window, text='Usuario administrador').grid(row=6, column=0, padx=10)
    username = tk.Entry(window)
    username.insert(0, 'Admin')
    username.grid(row=6, column=1, padx=10, pady=3)
    tk.Label(window, text='Contraseña de ProyAst').grid(row=7, column=0, padx=10)
    password = tk.Entry(window, show='*')
    password.grid(row=7, column=1, padx=10, pady=3)

    def aplicar():
        try:
            proposed = ModelConfiguration(
                pattern_weight=float(entries['pattern_weight'].get()) / 100,
                n_estimators=int(entries['n_estimators'].get()),
                max_depth=int(entries['max_depth'].get()),
                min_samples_leaf=int(entries['min_samples_leaf'].get()),
                min_samples_split=int(entries['min_samples_split'].get()),
            )
            proposed.validate()
            summary = '\n'.join(
                f'{label}: {entries[key].get()}' for key, label, _ in fields
            )
            if not messagebox.askyesno(
                'Aprobar configuración global',
                summary + '\n\nSin mejora validada. ¿Aplicar a próximas proyecciones?',
                parent=window,
            ):
                return
            save_model_configuration(proposed, username.get(), password.get())
        except (ValueError, PermissionError, OSError, sqlite3.Error) as exc:
            messagebox.showerror('No se aplicaron los cambios', str(exc), parent=window)
            return
        finally:
            password.delete(0, 'end')
        messagebox.showinfo(
            'Configuración aprobada',
            'Cambios guardados. Vuelve a proyectar en ProyAst y evalúa el nuevo %Dif.',
            parent=window,
        )
        window.destroy()

    tk.Button(window, text='Aprobar y aplicar', command=aplicar).grid(
        row=8, column=0, padx=10, pady=10
    )
    tk.Button(window, text='Cancelar', command=window.destroy).grid(row=8, column=1)


def crear_panel_dialogo_ia(parent, agente):
    """Muestra el diagnostico local y consulta TRIZ sin bloquear Tkinter."""
    import tkinter as tk
    from tkinter import scrolledtext, ttk

    _habilitar_paquete_triz()
    from agente_triz.exportacion import crear_menu_exportacion

    panel = tk.LabelFrame(parent, text='Diálogo con Agente TRIZ')
    aviso = tk.Label(
        panel,
        text=(
            'TRIZ envía solo temas generales a OpenRouter, no los datos de '
            'la evaluación. Las ideas no se aplican automáticamente.'
        ),
        anchor='w',
        justify='left',
        wraplength=580,
    )
    aviso.pack(fill='x', padx=6, pady=(2, 0))
    aviso.bind('<Configure>', lambda event: aviso.configure(
        wraplength=max(1, event.width - 4)
    ))
    historial = scrolledtext.ScrolledText(
        panel, height=4, width=1, wrap='word', state='disabled'
    )

    def escribir(autor, texto):
        historial.configure(state='normal')
        historial.insert('end', f'{autor}:\n{texto}\n\n')
        historial.configure(state='disabled')
        historial.see('end')

    diagnostico = agente.analizar()
    escribir('Diagnóstico local (no enviado a TRIZ)', diagnostico)
    entrada = tk.Frame(panel)
    entrada.pack(side='bottom', fill='x', padx=6, pady=(0, 4))
    tk.Label(entrada, text='Tema TRIZ:').pack(side='left')
    pregunta = ttk.Combobox(
        entrada, values=list(TEMAS_TRIZ), state='readonly', width=24
    )
    pregunta.set(next(iter(TEMAS_TRIZ)))
    pregunta.pack(side='left', fill='x', expand=True, padx=5)

    resultados = queue.Queue()
    ocupado = False
    pendiente = None
    analisis = diagnostico

    def cerrar_panel(event):
        if event.widget is panel and pendiente is not None:
            panel.after_cancel(pendiente)

    panel.bind('<Destroy>', cerrar_panel)

    def consultar(tema):
        try:
            resultados.put(('Agente TRIZ', agente.responder(tema)))
        except Exception as exc:
            resultados.put(('Error de TRIZ (sin cambios al modelo)', str(exc)))

    def recibir():
        nonlocal ocupado, pendiente, analisis
        pendiente = None
        try:
            autor, texto = resultados.get_nowait()
        except queue.Empty:
            pendiente = panel.after(100, recibir)
            return
        escribir(autor, texto)
        if autor == 'Agente TRIZ' and texto.strip():
            analisis = texto
        ocupado = False
        enviar_button.configure(state='normal')

    def enviar():
        nonlocal ocupado, pendiente
        if ocupado:
            return
        ocupado = True
        enviar_button.configure(state='disabled')
        tema = pregunta.get()
        escribir('Consulta general a TRIZ', tema)
        escribir('Estado', 'Consultando OpenRouter...')
        threading.Thread(target=consultar, args=(tema,), daemon=True).start()
        pendiente = panel.after(100, recibir)

    enviar_button = tk.Button(entrada, text='Consultar TRIZ', command=enviar)
    enviar_button.pack(side='left')
    pregunta.bind('<Return>', lambda _event: enviar())
    sugerencias = tk.Frame(panel)
    sugerencias.pack(side='bottom', fill='x', padx=6, pady=(0, 3))
    tk.Button(
        sugerencias, text='Resumen local',
        command=lambda: escribir('Diagnóstico local', agente.analizar()),
    ).pack(side='left', padx=(0, 5))
    tk.Button(
        sugerencias, text='Ajustar peso y parámetros',
        command=lambda: mostrar_ajustes_modelo(parent),
    ).pack(side='left', padx=(0, 5))
    exportar_button = crear_menu_exportacion(sugerencias, lambda: analisis)
    exportar_button.configure(state='normal')
    exportar_button.pack(side='left', padx=(0, 5))
    historial.pack(fill='both', expand=True, padx=6, pady=3)
    return panel


def seleccionar_archivo_y_columna():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception:
        print('No se pudo abrir el selector de archivos. Instala tkinter o usa argumentos por consola.')
        return '', None, None

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        folder_path = filedialog.askdirectory(
            title='Selecciona la carpeta que contiene los datos',
        )
        if not folder_path:
            return '', None, None

        files = sorted(
            path for path in Path(folder_path).iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
        )
        if not files:
            messagebox.showerror(
                'Carpeta sin archivos compatibles',
                'La carpeta no contiene archivos TXT, CSV, PY, XLSX, XLS o XLSM.',
                parent=root,
            )
            return '', None, None

        dialog = tk.Toplevel(root)
        dialog.title('Configurar gráfica')
        dialog.geometry('680x570')
        dialog.resizable(False, False)
        dialog.attributes('-topmost', True)
        dialog.columnconfigure(0, weight=1)

        selected_file = tk.StringVar(value=str(files[0]))
        selected_column = tk.StringVar()
        selected_sheet = tk.StringVar()
        selected_chart = tk.StringVar(value='Línea')
        usar_calibracion = tk.BooleanVar(value=True)
        eliminar_dif = tk.BooleanVar(value=False)
        folder_label = tk.StringVar(value=f'Carpeta: {folder_path}')
        result = {'column': None, 'chart_type': None,
                  'sheet': None, 'calibrar_amortiguador': False,
                  'eliminar_dif': False}

        tk.Label(
            dialog,
            textvariable=folder_label,
            anchor='w',
        ).grid(row=0, column=0, padx=18, pady=(16, 6), sticky='ew')
        tk.Label(
            dialog,
            text='Archivos disponibles en la carpeta (selecciona uno):',
            anchor='w',
        ).grid(row=1, column=0, padx=18, pady=(8, 4), sticky='ew')

        file_frame = tk.Frame(dialog)
        file_frame.grid(row=2, column=0, padx=18, pady=4, sticky='nsew')
        dialog.rowconfigure(2, weight=1)
        file_frame.columnconfigure(0, weight=1)
        file_frame.rowconfigure(0, weight=1)

        file_tree = ttk.Treeview(
            file_frame,
            columns=('nombre', 'tipo', 'tamano'),
            show='headings',
            height=7,
            selectmode='browse',
        )
        file_tree.heading('nombre', text='Archivo Excel / serie')
        file_tree.heading('tipo', text='Tipo')
        file_tree.heading('tamano', text='Tamaño')
        file_tree.column('nombre', width=420, anchor='w')
        file_tree.column('tipo', width=90, anchor='center')
        file_tree.column('tamano', width=90, anchor='e')
        file_tree.grid(row=0, column=0, sticky='nsew')

        file_scroll = ttk.Scrollbar(
            file_frame, orient='vertical', command=file_tree.yview
        )
        file_scroll.grid(row=0, column=1, sticky='ns')
        file_tree.configure(yscrollcommand=file_scroll.set)

        def refresh_file_tree(new_folder):
            nonlocal files
            new_files = sorted(
                path for path in Path(new_folder).iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
            )
            if not new_files:
                messagebox.showwarning(
                    'Carpeta sin archivos compatibles',
                    'La carpeta no contiene archivos TXT, CSV, PY, XLSX, XLS o XLSM.',
                    parent=dialog,
                )
                return False

            file_tree.delete(*file_tree.get_children())
            files = new_files
            for path in files:
                suffix = path.suffix.lower()
                tipo = (
                    'Excel' if suffix in EXCEL_EXTENSIONS
                    else suffix.upper().lstrip('.')
                )
                tamano = f'{path.stat().st_size / 1024:.1f} KB'
                file_tree.insert(
                    '', 'end', iid=str(path),
                    values=(path.name, tipo, tamano),
                )

            selected_file.set(str(files[0]))
            first_item = file_tree.get_children()[0]
            file_tree.selection_set(first_item)
            file_tree.focus(first_item)
            folder_label.set(f'Carpeta: {new_folder}')
            return True

        def explore_folder():
            new_folder = filedialog.askdirectory(
                title='Explorar otra carpeta',
                parent=dialog,
            )
            if new_folder:
                refresh_file_tree(new_folder)
                column_selector['values'] = []
                selected_column.set('')
                estado_carga.set(
                    'Archivo sin cargar. Presiona "Cargar archivo".')

        tk.Button(
            dialog,
            text='Explorar archivos...',
            command=explore_folder,
            width=18,
        ).place(relx=0.72, rely=0.018, anchor='nw')

        refresh_file_tree(folder_path)

        estado_carga = tk.StringVar(
            value='Archivo sin cargar. Presiona "Cargar archivo".'
        )
        carga_frame = tk.Frame(dialog)
        carga_frame.grid(row=3, column=0, padx=18, pady=(6, 4), sticky='ew')
        carga_frame.columnconfigure(0, weight=1)
        tk.Label(
            carga_frame, textvariable=estado_carga, anchor='w', fg='#555555'
        ).grid(row=0, column=0, sticky='ew')
        tk.Button(
            carga_frame,
            text='Cargar archivo',
            command=lambda: load_columns(),
            width=16,
        ).grid(row=0, column=1, padx=(8, 0))

        hoja_frame = tk.Frame(dialog)
        hoja_frame.grid(row=4, column=0, padx=18, pady=(2, 4), sticky='ew')
        hoja_frame.columnconfigure(1, weight=1)
        tk.Label(hoja_frame, text='Hoja:', anchor='w').grid(
            row=0, column=0, sticky='w'
        )
        sheet_selector = ttk.Combobox(
            hoja_frame, textvariable=selected_sheet, state='disabled'
        )
        sheet_selector.grid(row=0, column=1, sticky='ew', padx=(8, 0))

        tk.Label(dialog, text='Columna a graficar:', anchor='w').grid(
            row=5, column=0, padx=18, pady=4, sticky='ew'
        )
        column_selector = ttk.Combobox(
            dialog, textvariable=selected_column, state='disabled'
        )
        column_selector.grid(row=6, column=0, padx=18, pady=4, sticky='ew')

        checks_frame = tk.Frame(dialog)
        checks_frame.grid(row=7, column=0, padx=18, pady=(0, 4), sticky='w')
        ttk.Checkbutton(
            checks_frame,
            text='Guardar %Dif conservando hasta 5 semanas por caso',
            variable=usar_calibracion,
        ).pack(anchor='w')
        ttk.Checkbutton(
            checks_frame,
            text='Borrar CSV con %Dif guardados (tiene prioridad sobre guardar)',
            variable=eliminar_dif,
        ).pack(anchor='w')

        tk.Label(dialog, text='Tipo de gráfica:', anchor='w').grid(
            row=8, column=0, padx=18, pady=4, sticky='ew'
        )
        chart_selector = ttk.Combobox(
            dialog,
            textvariable=selected_chart,
            values=list(CHART_TYPES),
            state='readonly',
        )
        chart_selector.grid(row=9, column=0, padx=18, pady=4, sticky='ew')

        def _elegir_hoja_por_defecto(excel_file, hojas):
            mejor_hoja = hojas[0]
            mejor_score = -1
            for nombre in hojas:
                try:
                    columnas = pd.read_excel(
                        excel_file, sheet_name=nombre, nrows=0
                    ).columns
                except Exception:
                    continue
                score = sum(
                    1 for col in columnas
                    if not str(col).startswith('Unnamed:')
                )
                if score > mejor_score:
                    mejor_score = score
                    mejor_hoja = nombre
            return mejor_hoja

        def load_columns(*_):
            path = Path(selected_file.get())
            columns = []
            try:
                if path.suffix.lower() in EXCEL_EXTENSIONS:
                    excel_file = pd.ExcelFile(path)
                    hojas = excel_file.sheet_names
                    if not hojas:
                        raise ValueError('El archivo Excel no tiene hojas.')
                    hoja_actual = selected_sheet.get()
                    if hoja_actual not in hojas:
                        hoja_actual = _elegir_hoja_por_defecto(
                            excel_file, hojas)
                        selected_sheet.set(hoja_actual)
                    sheet_selector['values'] = hojas
                    sheet_selector.configure(
                        state='readonly' if len(hojas) > 1 else 'disabled'
                    )
                    dataframe = pd.read_excel(
                        excel_file, sheet_name=hoja_actual)
                    columns = [str(column) for column in dataframe.columns]
                elif path.suffix.lower() == '.csv':
                    selected_sheet.set('')
                    sheet_selector['values'] = []
                    sheet_selector.configure(state='disabled')
                    dataframe = pd.read_csv(path)
                    columns = [str(column) for column in dataframe.columns]
                else:
                    selected_sheet.set('')
                    sheet_selector['values'] = []
                    sheet_selector.configure(state='disabled')
            except Exception as exc:
                messagebox.showerror(
                    'Error al cargar archivo', str(exc), parent=dialog
                )
                estado_carga.set('No se pudo cargar el archivo.')
                column_selector.configure(state='disabled')
                column_selector['values'] = []
                selected_column.set('')
                return
            column_selector['values'] = columns
            selected_column.set(columns[0] if columns else '')
            if columns:
                column_selector.configure(state='readonly')
                sufijo_hoja = (
                    f" (hoja '{selected_sheet.get()}')"
                    if selected_sheet.get() else ''
                )
                estado_carga.set(
                    f'Archivo cargado: {path.name}{sufijo_hoja} '
                    f'({len(columns)} columnas).'
                )
            else:
                column_selector.configure(state='disabled')
                estado_carga.set(
                    f'{path.name} no tiene columnas seleccionables.'
                )

        sheet_selector.bind('<<ComboboxSelected>>', lambda _e: load_columns())

        def select_file(_event=None):
            selection = file_tree.selection()
            if selection:
                selected_file.set(selection[0])
                selected_sheet.set('')
                sheet_selector['values'] = []
                sheet_selector.configure(state='disabled')
                column_selector.configure(state='disabled')
                column_selector['values'] = []
                selected_column.set('')
                estado_carga.set(
                    'Archivo sin cargar. Presiona "Cargar archivo".')

        def select_file_por_clic(event):
            fila = file_tree.identify_row(event.y)
            if fila:
                file_tree.selection_set(fila)
                file_tree.focus(fila)
                selected_file.set(fila)
                selected_sheet.set('')
                sheet_selector['values'] = []
                sheet_selector.configure(state='disabled')
                column_selector.configure(state='disabled')
                column_selector['values'] = []
                selected_column.set('')
                estado_carga.set(
                    'Archivo sin cargar. Presiona "Cargar archivo".')

        file_tree.bind('<<TreeviewSelect>>', select_file)
        file_tree.bind('<ButtonRelease-1>', select_file_por_clic)
        file_tree.bind('<<TreeviewOpen>>', select_file)
        file_tree.bind('<Up>', lambda _e: dialog.after(1, select_file))
        file_tree.bind('<Down>', lambda _e: dialog.after(1, select_file))
        first_item = file_tree.get_children()[0]
        file_tree.selection_set(first_item)
        file_tree.focus(first_item)

        def accept():
            path = Path(selected_file.get())
            requiere_columna = path.suffix.lower() in EXCEL_EXTENSIONS | {
                '.csv'}
            if requiere_columna and not selected_column.get():
                messagebox.showwarning(
                    'Carga el archivo',
                    'Presiona "Cargar archivo" y elige una columna antes de graficar.',
                    parent=dialog,
                )
                return
            result['column'] = selected_column.get() or None
            result['chart_type'] = CHART_TYPES[selected_chart.get()]
            result['sheet'] = selected_sheet.get() or None
            result['calibrar_amortiguador'] = bool(usar_calibracion.get())
            result['eliminar_dif'] = bool(eliminar_dif.get())
            dialog.destroy()

        def cancel():
            dialog.destroy()

        buttons = tk.Frame(dialog)
        buttons.grid(row=10, column=0, pady=14)
        tk.Button(buttons, text='Graficar', command=accept, width=12).pack(
            side='left', padx=5
        )
        tk.Button(buttons, text='Cancelar', command=cancel, width=12).pack(
            side='left', padx=5
        )
        dialog.protocol('WM_DELETE_WINDOW', cancel)
        dialog.grab_set()
        root.wait_window(dialog)
        return (
            selected_file.get() if result['chart_type'] else '',
            result['column'],
            result['chart_type'],
            result['sheet'],
            result['calibrar_amortiguador'],
            result['eliminar_dif'],
        )
    except Exception as exc:
        print(f'No se pudo abrir el selector de archivos: {exc}')
        return '', None, None, None, False, False
    finally:
        root.destroy()


def seleccionar_archivo():
    file_path, _, _, _, _, _ = seleccionar_archivo_y_columna()
    return file_path


def extraer_valores_desde_archivo(file_path, column=None, sheet_name=None):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'No se encontró el archivo: {file_path}')

    suffix = path.suffix.lower()
    if suffix in EXCEL_EXTENSIONS | {'.csv'}:
        df = (
            pd.read_excel(path, sheet_name=sheet_name if sheet_name else 0)
            if suffix in EXCEL_EXTENSIONS
            else pd.read_csv(path)
        )
        if column is not None:
            if column not in df.columns:
                raise ValueError(
                    f'No existe la columna seleccionada: {column}')
            series = pd.to_numeric(df[column], errors='coerce').dropna()
            if series.empty:
                raise ValueError(
                    f'La columna seleccionada no contiene valores numéricos: {column}')
            return series.astype(float).tolist()

        for column_name in df.columns:
            serie_num = pd.to_numeric(
                df[column_name], errors='coerce').dropna()
            if not serie_num.empty:
                return serie_num.astype(float).tolist()
        raise ValueError(f'No se encontraron columnas numéricas en: {path}')

    content = path.read_text(encoding='utf-8', errors='ignore')
    matches = re.findall(r'[-+]?\d+(?:[.,]\d+)?', content)
    if not matches:
        raise ValueError(f'No se encontraron valores numéricos en: {path}')
    return [float(match.replace(',', '.')) for match in matches]


def guardar_calibracion_amortiguador(
    file_path, column, sheet_name=None, destination_path=None
):
    """Persiste el %Dif graficado y claves para el ajuste reciente por caso."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in EXCEL_EXTENSIONS:
        dataframe = pd.read_excel(
            path, sheet_name=sheet_name if sheet_name else 0)
    elif suffix == '.csv':
        dataframe = pd.read_csv(path)
    else:
        raise ValueError(
            'Solo se puede calibrar el amortiguador desde Excel o CSV.'
        )

    if column not in dataframe.columns:
        raise ValueError(f'No existe la columna seleccionada: {column}')

    nombre_columna = ''.join(str(column).split()).casefold()
    if nombre_columna not in {'%dif', 'error_relativo'}:
        raise ValueError(
            'Para el ajuste reciente debes graficar la columna %Dif '
            '(o error_relativo), no otra métrica.'
        )

    normalized_columns = {
        ''.join(str(col).split()).casefold(): col for col in dataframe.columns
    }
    columna_anio = normalized_columns.get('anio')
    columna_semana = normalized_columns.get('semana')
    columna_bloque = normalized_columns.get('bloque')
    columna_variedad = normalized_columns.get('variedad')
    columna_caso = normalized_columns.get('bloque&varid')
    if columna_anio is None or columna_semana is None:
        raise ValueError(
            'El archivo debe incluir Anio y Semana para seleccionar '
            'las cinco semanas recientes.'
        )
    if not (
        (columna_bloque is not None and columna_variedad is not None)
        or columna_caso is not None
    ):
        raise ValueError(
            'El archivo debe incluir Bloque y Variedad, o Bloque&Varid, '
            'para calibrar el ajuste individual.'
        )

    serie = pd.to_numeric(dataframe[column], errors='coerce')
    valido = serie.notna()
    if not valido.any():
        raise ValueError(
            f'La columna seleccionada no contiene valores numéricos: {column}'
        )

    reporte = pd.DataFrame({'error_relativo': serie[valido].to_numpy()})
    reporte['%dif'] = reporte['error_relativo']

    for nombre_canonico in ['Finca', 'Bloque', 'Variedad', 'Bloque&Varid']:
        columna_origen = normalized_columns.get(
            ''.join(nombre_canonico.split()).casefold()
        )
        if columna_origen is not None:
            reporte[nombre_canonico] = dataframe.loc[
                valido, columna_origen
            ].to_numpy()

    reporte['Anio'] = pd.to_numeric(
        dataframe.loc[valido, columna_anio], errors='coerce'
    ).to_numpy()
    reporte['Semana'] = pd.to_numeric(
        dataframe.loc[valido, columna_semana], errors='coerce'
    ).to_numpy()

    destino = Path(destination_path) if destination_path else (
        Path(__file__).with_name('Evaluacion') /
        'errores_evaluacion_modelo.csv'
    )
    historial = save_recent_evaluation_history(reporte, destino)

    periodos = pd.DataFrame()
    if {'Anio', 'Semana'}.issubset(historial.columns):
        periodos = historial[['Anio', 'Semana']].dropna().drop_duplicates()

    return {
        'rows': int(historial['error_relativo'].notna().sum()),
        'evaluated_weeks': int(len(periodos)),
        'source': str(destino),
    }


def eliminar_calibracion_amortiguador(destination_path=None):
    """Elimina los CSV con %Dif y deja el ajuste del amortiguador en 0.

    Borra todo archivo de la carpeta de evaluacion cuya cabecera contenga
    %dif o error_relativo y recrea errores_evaluacion_modelo.csv vacio
    (solo encabezados). Al no haber datos, los cargadores devuelven
    available=False y ProyAst fija el amortiguador en 0.
    """
    destino = Path(destination_path) if destination_path else (
        Path(__file__).with_name('Evaluacion') /
        'errores_evaluacion_modelo.csv'
    )
    eliminados = []
    candidatos = [destino]
    if destino.parent.exists():
        candidatos.extend(
            ruta for ruta in sorted(destino.parent.glob('*.csv'))
            if ruta != destino
        )
    for ruta in candidatos:
        if not ruta.exists():
            continue
        lineas = ruta.read_text(encoding='utf-8-sig').splitlines()
        encabezado = lineas[0].casefold() if lineas else ''
        es_destino = ruta == destino
        contiene_dif = '%dif' in encabezado or 'error_relativo' in encabezado
        if (es_destino or contiene_dif) and ruta.exists():
            ruta.unlink()
            eliminados.append(str(ruta))

    destino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=[
        'error_relativo', '%dif', 'Finca', 'Bloque', 'Variedad',
        'Bloque&Varid', 'Anio', 'Semana',
    ]).to_csv(destino, index=False, encoding='utf-8-sig')

    return {
        'deleted': len(eliminados),
        'files': eliminados,
        'source': str(destino),
    }


def obtener_rango_semanas(file_path, sheet_name=None):
    """Obtiene el mínimo y máximo de la columna Semana del archivo."""
    path = Path(file_path)
    if path.suffix.lower() not in EXCEL_EXTENSIONS | {'.csv'}:
        return None

    dataframe = (
        pd.read_excel(path, sheet_name=sheet_name if sheet_name else 0)
        if path.suffix.lower() in EXCEL_EXTENSIONS
        else pd.read_csv(path)
    )
    semana_column = next(
        (
            column for column in dataframe.columns
            if str(column).strip().casefold() == 'semana'
        ),
        None,
    )
    if semana_column is None:
        return None

    semanas = pd.to_numeric(dataframe[semana_column], errors='coerce').dropna()
    if semanas.empty:
        return None

    minimo = float(semanas.min())
    maximo = float(semanas.max())

    def mostrar_numero(valor):
        return str(int(valor)) if valor.is_integer() else f'{valor:g}'

    return mostrar_numero(minimo), mostrar_numero(maximo)


def mostrar_grafica_en_dialogo(
    fig, title='Gráfica de evaluación', agente=None
):
    """Muestra la figura en una ventana Tkinter después de guardarla."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except Exception as exc:
        print(f'No se pudo abrir la vista previa: {exc}')
        return False

    try:
        window = tk.Tk()
        window.title(title)
        ancho = min(960, window.winfo_screenwidth() - 40)
        alto = min(650, window.winfo_screenheight() - 100)
        window.geometry(f'{ancho}x{alto}')
        window.minsize(min(600, ancho), min(420, alto))
        botones = tk.Frame(window)
        botones.pack(side='bottom', fill='x', pady=(2, 4))
        contenido = tk.PanedWindow(window, orient='vertical', sashwidth=8)
        contenido.pack(fill='both', expand=True)
        grafica = tk.Frame(contenido)
        canvas = FigureCanvasTkAgg(fig, master=grafica)
        canvas.draw()
        canvas.get_tk_widget().configure(width=ancho, height=alto // 2)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)
        contenido.add(grafica, minsize=90, stretch='always')
        if agente is not None:
            panel = crear_panel_dialogo_ia(contenido, agente)
            panel.update_idletasks()
            contenido.add(panel, minsize=panel.winfo_reqheight(), stretch='never')

        def guardar_grafica():
            destino = filedialog.asksaveasfilename(
                title='Guardar gráfica de evaluación',
                defaultextension='.png',
                filetypes=[
                    ('Imagen PNG', '*.png'),
                    ('Documento PDF', '*.pdf'),
                    ('Todos los archivos', '*.*'),
                ],
                parent=window,
            )
            if destino:
                fig.savefig(destino, dpi=200, bbox_inches='tight')
                messagebox.showinfo(
                    'Gráfica guardada',
                    f'La gráfica se guardó en:\n{destino}',
                    parent=window,
                )

        tk.Button(
            botones, text='Guardar gráfica', command=guardar_grafica, width=16
        ).pack(side='left', padx=5)
        tk.Button(
            botones, text='Cerrar', command=window.destroy, width=12
        ).pack(side='left', padx=5)
        window.mainloop()
        return True
    except Exception as exc:
        print(f'No se pudo mostrar la gráfica en diálogo: {exc}')
        return False


def generar_grafica(
    file_path=None,
    output_path=None,
    column=None,
    chart_type='line',
    mostrar_dialogo=False,
    sheet_name=None,
):
    if not file_path:
        file_path = str(DEFAULT_SERIES_PATH)

    values = extraer_valores_desde_archivo(
        file_path, column=column, sheet_name=sheet_name)
    values = sorted(values, reverse=True)
    x_values = list(range(1, len(values) + 1))
    rango_semanas = obtener_rango_semanas(file_path, sheet_name=sheet_name)
    agente = AgenteEvaluacionTRIZ(values, column, rango_semanas)

    fig, ax = plt.subplots(figsize=(12, 5))
    if chart_type == 'bar':
        ax.bar(x_values, values, color='#1f77b4', width=0.8)
    elif chart_type == 'scatter':
        ax.scatter(x_values, values, color='#1f77b4', s=24)
    elif chart_type == 'step':
        ax.step(x_values, values, color='#1f77b4', linewidth=1.5, where='mid')
    elif chart_type == 'area':
        ax.fill_between(x_values, values, 0, color='#1f77b4', alpha=0.45)
        ax.plot(x_values, values, color='#1f77b4', linewidth=1.2)
    else:
        ax.plot(x_values, values, color='#1f77b4', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)

    titulo_base = 'Grafica de Evaluacion diferencias (Real-proy)/Real'
    if rango_semanas:
        semana_minima, semana_maxima = rango_semanas
        titulo = f'{titulo_base} - SEMANAS {semana_minima} A {semana_maxima}'
    else:
        titulo = titulo_base
    ax.set_title(titulo)
    ax.set_xlabel('Número de caso')
    ax.set_ylabel('Valor')
    ax.grid(True, alpha=0.3)

    interval = 15
    ax.set_xticks(x_values[::interval] if len(
        x_values) > interval else x_values)
    ax.set_xticklabels(x_values[::interval] if len(
        x_values) > interval else x_values)
    ax.set_yticks([i / 5 for i in range(-10, 11)])
    ax.spines['bottom'].set_position(('data', 0))

    def discrete_area(series):
        return sum(series)

    positive_parts = [max(v, 0.0) for v in values]
    negative_parts = [min(v, 0.0) for v in values]
    area_positive = discrete_area(positive_parts)
    area_negative = discrete_area(negative_parts)
    area_positive_pct, area_negative_pct = _area_percentages(
        area_positive, area_negative
    )
    cumulative_15_pct = _cumulative_data_percentage(values)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    counts = {}
    previous_threshold = 0.0
    for threshold in thresholds:
        counts[f'±{int(threshold * 100)}%'] = sum(
            1 for v in values if previous_threshold < abs(v) <= threshold
        )
        previous_threshold = threshold
    total_graficado = len(values)

    metrics_text = '\n'.join([
        f'Área positiva / subestimación: {area_positive:.3f}',
        f'Área negativa / sobreestimación: {area_negative:.3f}',
        f'Áreas: positiva {area_positive_pct:.1f}% | negativa {area_negative_pct:.1f}%',
        f'Datos acumulados entre ±15%: {cumulative_15_pct:.1f}%',
        *[
            f'Datos entre {label}: {count} '
            f'({count / total_graficado * 100 if total_graficado else 0:.1f}%)'
            for label, count in counts.items()
        ],
    ])

    if chart_type == 'line':
        ax.fill_between(
            x_values, values, 0,
            where=[value >= 0 for value in values],
            color='#2ca02c', alpha=0.18,
            interpolate=True,
        )
        ax.fill_between(
            x_values, values, 0,
            where=[value < 0 for value in values],
            color='#d62728', alpha=0.18,
            interpolate=True,
        )

    ax.text(
        0.02,
        0.02,
        metrics_text,
        transform=ax.transAxes,
        va='bottom',
        ha='left',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
    )

    fig.tight_layout()

    if not output_path:
        output_path = str(DEFAULT_OUTPUT_PATH)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f'Gráfica guardada en: {output_path}')
    if mostrar_dialogo:
        mostrar_grafica_en_dialogo(
            fig,
            title=f'Gráfica ordenada de mayor a menor - {Path(file_path).name}',
            agente=agente,
        )
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Genera una gráfica de evaluación desde una serie o Excel')
    parser.add_argument('archivo', nargs='?', default='',
                        help='Ruta del archivo de entrada (serie o Excel)')
    parser.add_argument('salida', nargs='?', default='',
                        help='Ruta del PDF de salida')
    parser.add_argument('--dialogo', action='store_true',
                        help='Abrir selector de archivos para elegir la entrada')
    parser.add_argument('--columna', default='',
                        help='Nombre de la columna numérica que se desea graficar')
    parser.add_argument('--tipo-grafica', choices=list(CHART_TYPES.values()),
                        default='line', help='Tipo de gráfica')
    parser.add_argument('--hoja', default='',
                        help='Nombre de la hoja de Excel a usar (si aplica)')
    args = parser.parse_args()

    archivo_seleccionado = args.archivo
    columna_seleccionada = args.columna or None
    tipo_grafica = args.tipo_grafica
    hoja_seleccionada = args.hoja or None
    calibrar_amortiguador = False
    eliminar_dif_guardado = False
    if args.dialogo or not archivo_seleccionado:
        (
            archivo_seleccionado, columna_dialogo, grafica_dialogo,
            hoja_dialogo, calibrar_dialogo, eliminar_dialogo,
        ) = seleccionar_archivo_y_columna()
        columna_seleccionada = columna_dialogo or columna_seleccionada
        tipo_grafica = grafica_dialogo or tipo_grafica
        hoja_seleccionada = hoja_dialogo or hoja_seleccionada
        calibrar_amortiguador = bool(calibrar_dialogo)
        eliminar_dif_guardado = bool(eliminar_dialogo)

    if not archivo_seleccionado:
        archivo_seleccionado = str(DEFAULT_SERIES_PATH)

    output_path = Path(args.salida) if args.salida else DEFAULT_OUTPUT_PATH

    if eliminar_dif_guardado:
        try:
            info_eliminacion = eliminar_calibracion_amortiguador()
            mensaje_eliminacion = (
                'Ajuste del amortiguador reiniciado a 0: '
                f"{info_eliminacion['deleted']} archivo(s) con %Dif "
                f"eliminados -> {info_eliminacion['source']}"
            )
            print(mensaje_eliminacion)
            try:
                import tkinter as tk
                from tkinter import messagebox
                aviso = tk.Tk()
                aviso.withdraw()
                messagebox.showinfo(
                    'Calibracion eliminada', mensaje_eliminacion)
                aviso.destroy()
            except Exception:
                pass
        except Exception as exc:
            mensaje_error = f'No se pudo eliminar la calibracion: {exc}'
            print(mensaje_error, file=sys.stderr)
            if args.dialogo:
                from tkinter import messagebox
                messagebox.showerror('Error al borrar evaluaciones', mensaje_error)
            raise
    elif calibrar_amortiguador and columna_seleccionada:
        try:
            info_calibracion = guardar_calibracion_amortiguador(
                archivo_seleccionado,
                columna_seleccionada,
                sheet_name=hoja_seleccionada,
            )
            mensaje_calibracion = (
                'Ajuste reciente individual actualizado desde %Dif: '
                f"{info_calibracion['rows']} filas, "
                f"{info_calibracion['evaluated_weeks']} semanas evaluadas -> "
                f"{info_calibracion['source']}"
            )
            print(mensaje_calibracion)
            try:
                import tkinter as tk
                from tkinter import messagebox
                aviso = tk.Tk()
                aviso.withdraw()
                messagebox.showinfo(
                    'Calibracion actualizada', mensaje_calibracion)
                aviso.destroy()
            except Exception:
                pass
        except Exception as exc:
            mensaje_error = f'No se pudo actualizar la calibracion: {exc}'
            print(mensaje_error, file=sys.stderr)
            if args.dialogo:
                from tkinter import messagebox
                messagebox.showerror('Error al guardar evaluaciones', mensaje_error)
            raise

    generar_grafica(
        file_path=archivo_seleccionado,
        output_path=output_path,
        column=columna_seleccionada,
        chart_type=tipo_grafica,
        mostrar_dialogo=True,
        sheet_name=hoja_seleccionada,
    )
