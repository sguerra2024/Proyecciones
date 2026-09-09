from projection_core import save_buffer_evaluation_report
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')


REFERENCE_CHART_TYPE = 'bars_with_sequence_line'
EVALUATION_COLUMN_INDEX = 10
EVALUATION_COLUMN_NAME = '%dif'


def _parse_values(raw_text):
    values = []
    for line in raw_text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue

        parts = clean_line.replace(';', ' ').split()
        for part in parts:
            values.append(float(part.replace(',', '.')))

    if not values:
        raise ValueError('No se detectaron datos numéricos para evaluar.')

    return values


def _extract_values_from_source_file(source_path):
    content = Path(source_path).read_text(encoding='utf-8')
    match = re.search(r"text\s*=\s*r'''(.*?)'''", content, re.S)
    if not match:
        raise RuntimeError(
            'No se encontró el contenido de la serie en el archivo fuente.')

    return _parse_values(match.group(1))


def _extract_values_from_excel(excel_path, column_selector=EVALUATION_COLUMN_INDEX, sheet_name=0):
    data_frame = pd.read_excel(excel_path, sheet_name=sheet_name)

    if isinstance(column_selector, int):
        if column_selector <= 0:
            raise ValueError(
                'El indice de columna debe ser mayor o igual a 1.')
        if column_selector > len(data_frame.columns):
            raise ValueError(
                f'El archivo tiene {len(data_frame.columns)} columnas y se solicito la columna {column_selector}.'
            )
        raw_series = data_frame.iloc[:, column_selector - 1]
        selected_column_label = f'{column_selector} ({data_frame.columns[column_selector - 1]})'
    else:
        column_name = str(column_selector).strip()
        if not column_name:
            raise ValueError(
                'Debes indicar una columna valida (indice o nombre).')
        if column_name not in data_frame.columns:
            raise ValueError(
                f'No se encontro la columna "{column_name}" en el archivo.'
            )
        raw_series = data_frame[column_name]
        selected_column_label = column_name

    series = pd.to_numeric(
        raw_series,
        errors='coerce',
    ).dropna()
    values = series.tolist()
    if not values:
        raise ValueError(
            f'La columna {selected_column_label} no tiene valores numericos validos.'
        )

    return values


def _extract_semana_min_max_from_excel(excel_path, sheet_name=0):
    data_frame = pd.read_excel(excel_path, sheet_name=sheet_name)
    normalized_columns = {
        str(column).strip().lower(): column for column in data_frame.columns
    }

    if 'semana' not in normalized_columns:
        return None, None

    semana_series = pd.to_numeric(
        data_frame[normalized_columns['semana']],
        errors='coerce',
    ).dropna()

    if semana_series.empty:
        return None, None

    return int(semana_series.min()), int(semana_series.max())


def plot_series_evaluation_from_values(values, output_prefix, title='Evaluacion proyecciones Modelo vs real semanas:', interval=20, show_before_save=True, semana_min=None, semana_max=None):
    ordered_values = sorted(values, reverse=True)
    x_values = list(range(1, len(ordered_values) + 1))

    def discrete_area(series):
        return sum(series)

    positive_parts = [max(v, 0.0) for v in ordered_values]
    negative_parts = [max(-v, 0.0) for v in ordered_values]
    area_positive = discrete_area(positive_parts)
    area_negative = discrete_area(negative_parts)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    counts = {}
    previous_threshold = 0.0
    for threshold in thresholds:
        counts[f'±{int(threshold * 100)}%'] = sum(
            1 for v in ordered_values if previous_threshold < abs(v) <= threshold
        )
        previous_threshold = threshold

    fig, ax = plt.subplots(figsize=(12, 5))
    if REFERENCE_CHART_TYPE != 'bars_with_sequence_line':
        raise ValueError(
            'El tipo de grafica permitido es barras +/- con linea de secuencia.')

    positive_bars = [v if v > 0 else 0.0 for v in ordered_values]
    negative_bars = [v if v < 0 else 0.0 for v in ordered_values]

    # Tipo de grafica fijo: barras para positivos/negativos y linea de secuencia por caso.
    ax.bar(x_values, positive_bars, width=0.85, color='#43a047',
           alpha=0.55, label='Intervalos positivos')
    ax.bar(x_values, negative_bars, width=0.85, color='#e53935',
           alpha=0.55, label='Intervalos negativos')
    ax.plot(x_values, ordered_values, color='#1f3a93',
            linewidth=1.2, label='Secuencia por caso')
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Número de caso')
    ax.set_ylabel('Valor')
    ax.grid(True, alpha=0.3)

    x_tick_step = max(1, interval)
    x_ticks = list(range(1, len(x_values) + 1, x_tick_step))
    if x_values and x_ticks[-1] != x_values[-1]:
        x_ticks.append(x_values[-1])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    max_abs_value = max(abs(min(ordered_values)), abs(
        max(ordered_values))) if ordered_values else 1.0
    max_abs_value = max(max_abs_value, 1.0)
    ax.set_ylim(-max_abs_value, max_abs_value)

    ax.spines['bottom'].set_position(('data', 0))
    ax.legend(loc='upper right')

    metrics_lines = [
        f'Area positiva: {area_positive:.3f}',
        f'Area negativa: {area_negative:.3f}',
    ]
    if semana_min is not None and semana_max is not None:
        metrics_lines.append(f'Semana min/max: {semana_min} - {semana_max}')

    metrics_lines.extend(
        [f'Datos entre {label}: {count}' for label, count in counts.items()]
    )
    metrics_text = '\n'.join(metrics_lines)
    # Recuadro fuera del area de la grafica: centro-abajo (center down).
    fig.text(
        0.5,
        0.04,
        metrics_text,
        va='bottom',
        ha='center',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
    )

    fig.tight_layout(rect=(0, 0.24, 1, 1))

    if show_before_save:
        # Muestra la grafica para revision manual antes de escribir archivos.
        plt.show(block=True)

    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{output_prefix}.png', dpi=200)
    fig.savefig(f'{output_prefix}.pdf', dpi=200)
    plt.close(fig)


def plot_series_evaluation(source_path, output_prefix, title='Evaluacion proyecciones Modelo vs real semanas:', interval=20):
    values = _extract_values_from_source_file(source_path)
    plot_series_evaluation_from_values(
        values=values,
        output_prefix=output_prefix,
        title=title,
        interval=interval,
        semana_min=None,
        semana_max=None,
    )


def launch_data_entry_dialog(default_output_prefix, default_title='Evaluacion proyecciones (real - modelo)/real'):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title('Evaluacion de Serie')

    title_var = tk.StringVar(value=default_title)
    output_var = tk.StringVar(value='')
    interval_var = tk.StringVar(value='20')
    excel_path_var = tk.StringVar(value='')
    excel_column_var = tk.StringVar(value=str(EVALUATION_COLUMN_INDEX))
    excel_sheet_var = tk.StringVar(value='0')
    semana_range_var = tk.StringVar(value='Semanas evaluacion: -')

    header = tk.Label(
        root,
        text='Selecciona el archivo. Tipo: barras +/- y linea de secuencia.',
        anchor='w',
        justify='left',
    )
    header.pack(fill='x', padx=12, pady=(12, 6))

    text_data = tk.Text(root, wrap='word', height=2)
    text_data.pack(fill='both', expand=True, padx=12, pady=(0, 10))

    form = tk.Frame(root)
    form.pack(fill='x', padx=12, pady=(0, 10))

    tk.Label(form, text='Titulo:').grid(row=0, column=0, sticky='w', pady=3)
    tk.Entry(form, textvariable=title_var, width=85).grid(
        row=0, column=1, columnspan=2, sticky='we', pady=3
    )

    tk.Label(form, text='Intervalo eje X:').grid(
        row=1, column=0, sticky='w', pady=3)
    tk.Entry(form, textvariable=interval_var, width=12).grid(
        row=1, column=1, sticky='w', pady=3)

    tk.Label(form, text='Archivo Excel:').grid(
        row=2, column=0, sticky='w', pady=3)
    tk.Entry(form, textvariable=excel_path_var, width=70).grid(
        row=2, column=1, sticky='we', pady=3
    )

    tk.Label(form, text='Columna (1-based o nombre):').grid(
        row=3, column=0, sticky='w', pady=3
    )
    tk.Entry(form, textvariable=excel_column_var, width=12).grid(
        row=3, column=1, sticky='w', pady=3
    )

    tk.Label(form, text='Hoja (nombre o indice):').grid(
        row=4, column=0, sticky='w', pady=3)
    tk.Entry(form, textvariable=excel_sheet_var, width=20).grid(
        row=4, column=1, sticky='w', pady=3
    )

    tk.Label(form, textvariable=semana_range_var).grid(
        row=5, column=0, columnspan=2, sticky='w', pady=3
    )

    tk.Label(form, text='Archivo salida PDF (auto):').grid(
        row=6, column=0, sticky='w', pady=3
    )
    tk.Entry(form, textvariable=output_var, width=70, state='readonly').grid(
        row=6, column=1, sticky='we', pady=3)

    def sync_dialog_title_with_excel_name():
        excel_path = excel_path_var.get().strip()
        if excel_path:
            root.title(f'Evaluacion de Serie - {Path(excel_path).stem}')
        else:
            root.title('Evaluacion de Serie')

    def _sanitize_filename(value):
        sanitized = re.sub(r'[<>:"/\\|?*]+', '_', value).strip()
        sanitized = sanitized.rstrip('.')
        return sanitized or 'evaluacion'

    def update_output_pdf_path():
        excel_path = excel_path_var.get().strip()
        if not excel_path:
            output_var.set('')
            return

        title_for_file = title_var.get().strip() or default_title
        safe_name = _sanitize_filename(title_for_file)
        output_pdf_path = Path(excel_path).with_name(f'{safe_name}.pdf')
        output_var.set(str(output_pdf_path))

    def update_semana_range_in_dialog():
        excel_path = excel_path_var.get().strip()
        if not excel_path:
            semana_range_var.set('Semanas evaluacion: -')
            return

        raw_sheet = excel_sheet_var.get().strip()
        sheet_name = int(raw_sheet) if raw_sheet.isdigit() else raw_sheet
        semana_min, semana_max = _extract_semana_min_max_from_excel(
            excel_path=excel_path,
            sheet_name=sheet_name,
        )
        if semana_min is None or semana_max is None:
            semana_range_var.set('Semanas evaluacion: no disponible')
        else:
            semana_range_var.set(
                f'Semanas evaluacion: {semana_min} - {semana_max}'
            )

    def choose_excel_file():
        selected = filedialog.askopenfilename(
            title='Seleccionar archivo Excel',
            filetypes=[
                ('Archivos Excel', '*.xlsx *.xls *.xlsm'),
                ('Todos los archivos', '*.*'),
            ],
        )
        if not selected:
            return

        excel_path_var.set(selected)
        sync_dialog_title_with_excel_name()
        update_output_pdf_path()
        try:
            update_semana_range_in_dialog()
        except Exception:
            semana_range_var.set('Semanas evaluacion: no disponible')

    def load_excel_column_to_preview():
        try:
            excel_path = excel_path_var.get().strip()
            if not excel_path:
                raise ValueError('Debes seleccionar un archivo Excel.')

            sync_dialog_title_with_excel_name()
            update_output_pdf_path()
            update_semana_range_in_dialog()

            raw_sheet = excel_sheet_var.get().strip()
            sheet_name = int(raw_sheet) if raw_sheet.isdigit() else raw_sheet
            column_selector_raw = excel_column_var.get().strip()
            if not column_selector_raw:
                column_selector = EVALUATION_COLUMN_NAME
            elif column_selector_raw.isdigit():
                column_selector = int(column_selector_raw)
            else:
                column_selector = column_selector_raw

            values = _extract_values_from_excel(
                excel_path=excel_path,
                column_selector=column_selector,
                sheet_name=sheet_name,
            )
            semana_min, semana_max = _extract_semana_min_max_from_excel(
                excel_path=excel_path,
                sheet_name=sheet_name,
            )

            text_data.delete('1.0', tk.END)
            text_data.insert('1.0', '\n'.join(str(v) for v in values))
            messagebox.showinfo(
                'Datos cargados',
                f'Se cargaron {len(values)} valores desde la columna {column_selector}.',
            )
        except Exception as exc:
            messagebox.showerror('Error', str(exc))

    def render_plot():
        try:
            excel_path = excel_path_var.get().strip()
            if not excel_path:
                raise ValueError('Debes seleccionar un archivo Excel.')

            sync_dialog_title_with_excel_name()
            update_output_pdf_path()
            update_semana_range_in_dialog()

            raw_sheet = excel_sheet_var.get().strip()
            sheet_name = int(raw_sheet) if raw_sheet.isdigit() else raw_sheet
            column_selector_raw = excel_column_var.get().strip()
            if not column_selector_raw:
                column_selector = EVALUATION_COLUMN_NAME
            elif column_selector_raw.isdigit():
                column_selector = int(column_selector_raw)
            else:
                column_selector = column_selector_raw

            report_info = save_buffer_evaluation_report(
                excel_path=excel_path,
                sheet_name=sheet_name,
            )
            if str(column_selector).strip().casefold() in {
                EVALUATION_COLUMN_NAME,
                str(EVALUATION_COLUMN_INDEX),
            }:
                values = report_info["values"]
            else:
                values = _extract_values_from_excel(
                    excel_path=excel_path,
                    column_selector=column_selector,
                    sheet_name=sheet_name,
                )
            semana_min, semana_max = _extract_semana_min_max_from_excel(
                excel_path=excel_path,
                sheet_name=sheet_name,
            )

            interval = int(interval_var.get().strip())
            if interval <= 0:
                raise ValueError('El intervalo debe ser mayor a cero.')

            # Mantiene el titulo objetivo de la grafica solicitada.
            title = default_title
            output_pdf = output_var.get().strip()
            if not output_pdf:
                raise ValueError(
                    'Debes seleccionar un archivo Excel para calcular la ruta de salida.')

            output_prefix = str(Path(output_pdf).with_suffix(''))
            plot_series_evaluation_from_values(
                values=values,
                output_prefix=output_prefix,
                title=title,
                interval=interval,
                semana_min=semana_min,
                semana_max=semana_max,
            )
            messagebox.showinfo(
                'Informe enviado',
                'projection_core actualizo el amortiguador con '
                f'{report_info["evaluated_weeks"]} semanas evaluadas.',
            )
        except Exception as exc:
            messagebox.showerror('Error', str(exc))

    button_row = tk.Frame(root)
    button_row.pack(fill='x', padx=12, pady=(0, 12))

    tk.Button(button_row, text='Elegir Excel',
              command=choose_excel_file).pack(side='left', padx=(0, 8))
    tk.Button(button_row, text='Previsualizar columna',
              command=load_excel_column_to_preview).pack(side='left', padx=(0, 8))
    tk.Button(button_row, text='Generar evaluacion',
              command=render_plot).pack(side='left', padx=(0, 8))
    tk.Button(button_row, text='Cerrar',
              command=root.destroy).pack(side='right')

    form.columnconfigure(1, weight=1)
    root.update_idletasks()
    min_width = root.winfo_reqwidth()
    min_height = root.winfo_reqheight()
    root.geometry(f'{min_width}x{min_height}')
    root.minsize(min_width, min_height)
    root.mainloop()


if __name__ == '__main__':
    launch_data_entry_dialog(
        default_output_prefix=r'C:\Users\Personal\Desktop\Evaluacion de dif modelo vs real ',
    )
