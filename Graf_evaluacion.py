import matplotlib.pyplot as plt
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')

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


def _cumulative_data_percentage(values, limit=0.25):
    """Porcentaje de observaciones dentro del rango [-limit, limit]."""
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return 0.0
    included = sum(1 for value in numeric_values if abs(value) <= limit)
    return 100.0 * included / len(numeric_values)


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
        dialog.geometry('680x470')
        dialog.resizable(False, False)
        dialog.attributes('-topmost', True)
        dialog.columnconfigure(0, weight=1)

        selected_file = tk.StringVar(value=str(files[0]))
        selected_column = tk.StringVar()
        selected_chart = tk.StringVar(value='Línea')
        folder_label = tk.StringVar(value=f'Carpeta: {folder_path}')
        result = {'column': None, 'chart_type': None}

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

        tk.Button(
            dialog,
            text='Explorar archivos...',
            command=explore_folder,
            width=18,
        ).place(relx=0.72, rely=0.018, anchor='nw')

        refresh_file_tree(folder_path)

        tk.Label(dialog, text='Columna numérica:', anchor='w').grid(
            row=3, column=0, padx=18, pady=4, sticky='ew'
        )
        column_selector = ttk.Combobox(
            dialog, textvariable=selected_column, state='readonly'
        )
        column_selector.grid(row=4, column=0, padx=18, pady=4, sticky='ew')

        tk.Label(dialog, text='Tipo de gráfica:', anchor='w').grid(
            row=5, column=0, padx=18, pady=4, sticky='ew'
        )
        chart_selector = ttk.Combobox(
            dialog,
            textvariable=selected_chart,
            values=list(CHART_TYPES),
            state='readonly',
        )
        chart_selector.grid(row=6, column=0, padx=18, pady=4, sticky='ew')

        def load_columns(*_):
            path = Path(selected_file.get())
            columns = []
            if path.suffix.lower() in EXCEL_EXTENSIONS | {'.csv'}:
                try:
                    dataframe = (
                        pd.read_excel(path)
                        if path.suffix.lower() in EXCEL_EXTENSIONS
                        else pd.read_csv(path)
                    )
                    columns = [
                        str(column) for column in dataframe.columns
                        if not pd.to_numeric(
                            dataframe[column], errors='coerce'
                        ).dropna().empty
                    ]
                except Exception as exc:
                    messagebox.showerror(
                        'Error al cargar archivo', str(exc), parent=dialog
                    )
            column_selector['values'] = columns
            selected_column.set(columns[0] if columns else '')

        def select_file(_event=None):
            selection = file_tree.selection()
            if selection:
                selected_file.set(selection[0])
                load_columns()

        file_tree.bind('<<TreeviewSelect>>', select_file)
        first_item = file_tree.get_children()[0]
        file_tree.selection_set(first_item)
        file_tree.focus(first_item)
        load_columns()

        def accept():
            if column_selector['values'] and not selected_column.get():
                messagebox.showwarning(
                    'Selecciona una columna',
                    'Elige una columna numérica para graficar.',
                    parent=dialog,
                )
                return
            result['column'] = selected_column.get() or None
            result['chart_type'] = CHART_TYPES[selected_chart.get()]
            dialog.destroy()

        def cancel():
            dialog.destroy()

        buttons = tk.Frame(dialog)
        buttons.grid(row=7, column=0, pady=14)
        tk.Button(buttons, text='Graficar', command=accept, width=12).pack(
            side='left', padx=5
        )
        tk.Button(buttons, text='Cancelar', command=cancel, width=12).pack(
            side='left', padx=5
        )
        dialog.protocol('WM_DELETE_WINDOW', cancel)
        dialog.grab_set()
        root.wait_window(dialog)
        return selected_file.get() if result['chart_type'] else '', result['column'], result['chart_type']
    except Exception as exc:
        print(f'No se pudo abrir el selector de archivos: {exc}')
        return '', None, None
    finally:
        root.destroy()


def seleccionar_archivo():
    file_path, _, _ = seleccionar_archivo_y_columna()
    return file_path


def extraer_valores_desde_archivo(file_path, column=None):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'No se encontró el archivo: {file_path}')

    suffix = path.suffix.lower()
    if suffix in EXCEL_EXTENSIONS | {'.csv'}:
        df = pd.read_excel(
            path) if suffix in EXCEL_EXTENSIONS else pd.read_csv(path)
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


def obtener_rango_semanas(file_path):
    """Obtiene el mínimo y máximo de la columna Semana del archivo."""
    path = Path(file_path)
    if path.suffix.lower() not in EXCEL_EXTENSIONS | {'.csv'}:
        return None

    dataframe = (
        pd.read_excel(path)
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


def mostrar_grafica_en_dialogo(fig, title='Gráfica de evaluación'):
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
        window.geometry('1100x650')
        window.minsize(800, 500)
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=8)

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

        botones = tk.Frame(window)
        botones.pack(pady=(0, 10))
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
):
    if not file_path:
        file_path = str(DEFAULT_SERIES_PATH)

    values = extraer_valores_desde_archivo(file_path, column=column)
    values = sorted(values, reverse=True)
    x_values = list(range(1, len(values) + 1))
    rango_semanas = obtener_rango_semanas(file_path)

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

    title_column = f' - {column}' if column else ''
    if rango_semanas:
        semana_minima, semana_maxima = rango_semanas
        titulo = (
            f'EVALUACION MODELO VS REALES - SEMANAS '
            f'{semana_minima} A {semana_maxima}{title_column}'
        )
    else:
        titulo = f'EVALUACION MODELO VS REALES{title_column}'
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
    cumulative_25_pct = _cumulative_data_percentage(values)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    counts = {}
    previous_threshold = 0.0
    for threshold in thresholds:
        counts[f'±{int(threshold * 100)}%'] = sum(
            1 for v in values if previous_threshold < abs(v) <= threshold
        )
        previous_threshold = threshold

    metrics_text = '\n'.join([
        f'Área positiva / subestimación: {area_positive:.3f}',
        f'Área negativa / sobreestimación: {area_negative:.3f}',
        f'Áreas: positiva {area_positive_pct:.1f}% | negativa {area_negative_pct:.1f}%',
        f'Datos acumulados entre ±25%: {cumulative_25_pct:.1f}%',
        *[f'Datos entre {label}: {count}' for label, count in counts.items()],
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
        )
    else:
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
    args = parser.parse_args()

    archivo_seleccionado = args.archivo
    columna_seleccionada = args.columna or None
    tipo_grafica = args.tipo_grafica
    if args.dialogo or not archivo_seleccionado:
        archivo_seleccionado, columna_dialogo, grafica_dialogo = seleccionar_archivo_y_columna()
        columna_seleccionada = columna_dialogo or columna_seleccionada
        tipo_grafica = grafica_dialogo or tipo_grafica

    if not archivo_seleccionado:
        archivo_seleccionado = str(DEFAULT_SERIES_PATH)

    output_path = Path(args.salida) if args.salida else DEFAULT_OUTPUT_PATH

    generar_grafica(
        file_path=archivo_seleccionado,
        output_path=output_path,
        column=columna_seleccionada,
        chart_type=tipo_grafica,
        mostrar_dialogo=True,
    )
