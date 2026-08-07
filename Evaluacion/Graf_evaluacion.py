import argparse
import matplotlib.pyplot as plt
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SERIES_PATH = BASE_DIR / 'temp_series_area2.py'
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / 'ultima_serie.pdf'


def seleccionar_archivo():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        print('No se pudo abrir el selector de archivos. Instala tkinter o usa argumentos por consola.')
        return ''

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        return filedialog.askopenfilename(
            title='Selecciona una serie o un archivo Excel',
            filetypes=[
                ('Archivos de serie', '*.txt;*.csv;*.py'),
                ('Excel', '*.xlsx;*.xls'),
                ('Todos los archivos', '*.*'),
            ],
        )
    except Exception as exc:
        print(f'No se pudo abrir el selector de archivos: {exc}')
        return ''
    finally:
        root.destroy()


def extraer_valores_desde_archivo(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'No se encontró el archivo: {file_path}')

    suffix = path.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        df = pd.read_excel(path)
        if isinstance(df, pd.Series):
            series = df
        else:
            for column in df.columns:
                serie_num = pd.to_numeric(df[column], errors='coerce').dropna()
                if not serie_num.empty:
                    return serie_num.astype(float).tolist()
            flat = pd.to_numeric(df.to_numpy().ravel(),
                                 errors='coerce').dropna()
            return flat.astype(float).tolist()
        return pd.to_numeric(series, errors='coerce').dropna().astype(float).tolist()

    content = path.read_text(encoding='utf-8', errors='ignore')
    matches = re.findall(r'[-+]?\d+(?:[.,]\d+)?', content)
    if not matches:
        raise ValueError(f'No se encontraron valores numéricos en: {path}')
    return [float(match.replace(',', '.')) for match in matches]


def generar_grafica(file_path=None, output_path=None):
    if not file_path:
        file_path = str(DEFAULT_SERIES_PATH)

    values = extraer_valores_desde_archivo(file_path)
    x_values = list(range(1, len(values) + 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_values, values, color='#1f77b4', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)

    ax.set_title('EVALUACION MODELO VS REALES SEMANAS 26,27,28 Y 29')
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
    negative_parts = [max(-v, 0.0) for v in values]
    area_positive = discrete_area(positive_parts)
    area_negative = discrete_area(negative_parts)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    counts = {}
    previous_threshold = 0.0
    for threshold in thresholds:
        counts[f'±{int(threshold * 100)}%'] = sum(
            1 for v in values if previous_threshold < abs(v) <= threshold
        )
        previous_threshold = threshold

    metrics_text = '\n'.join(
        [
            f'Area positiva: {area_positive:.3f}',
            f'Area negativa: {area_negative:.3f}',
            *[f'Datos entre {label}: {count}' for label,
                count in counts.items()],
        ]
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
    print(output_path)
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
    args = parser.parse_args()

    archivo_seleccionado = args.archivo
    if args.dialogo or not archivo_seleccionado:
        archivo_seleccionado = seleccionar_archivo()

    if not archivo_seleccionado:
        archivo_seleccionado = str(DEFAULT_SERIES_PATH)

    output_path = Path(args.salida) if args.salida else DEFAULT_OUTPUT_PATH

    generar_grafica(
        file_path=archivo_seleccionado,
        output_path=output_path,
    )
