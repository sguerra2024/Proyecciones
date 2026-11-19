import pandas as pd
import pytest
import time
import gc

import Graf_evaluacion as graf

try:
    import tkinter as tk
except ImportError:
    tk = None


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ([-0.15, 0, 0.15, -0.5, 0.5], 'CUMPLIDA.'),
        ([-0.151, 0, 0.15, -0.5, 0.5], 'NO CUMPLIDA.'),
        ([0, 0.4, -0.4], 'NO CUMPLIDA.'),
        ([0, 0, 0.4], 'CUMPLIDA.'),
    ],
)
def test_meta_usa_observaciones_y_limites_inclusivos(values, expected):
    agente = graf.AgenteEvaluacionTRIZ(values, '%Dif')

    assert f'Meta: al menos 60%. {expected}' in agente.analizar()


@pytest.mark.parametrize('invalid', [[], [float('nan')], [float('inf')]])
def test_analisis_rechaza_datos_no_finitos(invalid):
    with pytest.raises(ValueError, match='finitos'):
        graf.AgenteEvaluacionTRIZ(invalid, '%Dif')


def test_areas_no_se_confunden_con_meta_ni_compensan_errores():
    agente = graf.AgenteEvaluacionTRIZ([0.3, -0.3], 'error_relativo')
    respuesta = agente.analizar()

    assert '0/2 = 0.0%' in respuesta
    assert 'Área positiva: 0.300 (50.0%)' in respuesta
    assert 'área negativa: -0.300 (50.0%)' in respuesta
    assert 'equilibradas; esto no garantiza precisión' in respuesta
    assert 'Faltan 2 observaciones' in respuesta


@pytest.mark.parametrize(
    ('values', 'bias'),
    [([-0.5, 0.2], 'sobreestimación'), ([-0.1, 0.5], 'subestimación')],
)
def test_interpreta_convencion_de_signo(values, bias):
    assert f'{bias} del modelo.' in graf.AgenteEvaluacionTRIZ(
        values, '%dif'
    ).analizar()


@pytest.mark.parametrize('column', [None, 'Produccion', 'Semana'])
def test_no_interpreta_otra_columna_como_error(column):
    respuesta = graf.AgenteEvaluacionTRIZ([1, 2], column).analizar()
    assert 'selecciona explícitamente %Dif o error_relativo' in respuesta
    assert 'Meta: al menos 60%' not in respuesta


def test_dialogo_triz_solo_delega_el_tema_sin_contexto(monkeypatch):
    calls = []
    monkeypatch.setattr(
        graf, 'consultar_triz_general',
        lambda tema: calls.append(tema) or 'Ideas generales',
    )
    agente = graf.AgenteEvaluacionTRIZ(
        [0.1, -0.2, 0.05], ' %Dif ', ('40', '44')
    )
    assert agente.responder('Peso del patrón') == 'Ideas generales'
    assert calls == ['Peso del patrón']
    assert agente.values == (0.1, -0.2, 0.05)


@pytest.mark.parametrize('text', ['', 'Mi finca privada tiene error 123.456'])
def test_no_envia_preguntas_libres(text):
    with pytest.raises(ValueError, match='temas generales'):
        graf.consultar_triz_general(text)


def test_generar_grafica_entrega_sus_mismos_datos_al_dialogo(tmp_path, monkeypatch):
    source = tmp_path / 'evaluacion.csv'
    output = tmp_path / 'evaluacion.png'
    pd.DataFrame({
        'Semana': [40, 41, 44],
        '%Dif': [-0.3, 0.15, 0],
        'Otra': [100, 200, 300],
    }).to_csv(source, index=False)
    calls = []

    def mostrar(fig, title, agente):
        calls.append((fig, agente))
        assert agente.values == (0.15, 0.0, -0.3)
        assert agente.column == '%Dif'
        assert agente.rango_semanas == ('40', '44')
        assert '2/3 = 66.7%' in agente.analizar()

    monkeypatch.setattr(graf, 'mostrar_grafica_en_dialogo', mostrar)

    assert graf.generar_grafica(
        source, output, column='%Dif', mostrar_dialogo=True
    ) == output
    assert output.is_file()
    assert len(calls) == 1
    assert not graf.plt.fignum_exists(calls[0][0].number)


@pytest.fixture
def window():
    if tk is None:
        pytest.skip('Tkinter no está instalado.')
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'Tkinter sin pantalla disponible: {exc}')
    root.withdraw()
    yield root
    root.destroy()
    gc.collect()


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


def test_panel_muestra_analisis_y_permite_dialogo(window, monkeypatch, tmp_path):
    from tkinter import filedialog, messagebox, ttk

    destino = tmp_path / 'analisis.xlsx'
    monkeypatch.setattr(filedialog, 'asksaveasfilename', lambda **kwargs: str(destino))
    monkeypatch.setattr(messagebox, 'showinfo', lambda *args, **kwargs: None)

    monkeypatch.setattr(graf, 'consultar_triz_general', lambda tema: 'Idea TRIZ simulada')
    panel = graf.crear_panel_dialogo_ia(
        window, graf.AgenteEvaluacionTRIZ([0.1, -0.2], '%dif')
    )
    panel.pack()
    window.update_idletasks()
    widgets = list(_descendants(panel))
    history = next(widget for widget in widgets if isinstance(widget, tk.Text))
    entry = next(widget for widget in widgets if isinstance(widget, ttk.Combobox))
    send = next(
        widget for widget in widgets
        if isinstance(widget, tk.Button) and widget.cget('text') == 'Consultar TRIZ'
    )
    assert '1/2 = 50.0%' in history.get('1.0', 'end')
    assert history.cget('state') == 'disabled'
    exportar = next(widget for widget in widgets if isinstance(widget, ttk.Menubutton))
    assert str(exportar.cget('state')) == 'normal'

    assert str(entry.cget('state')) == 'readonly'
    entry.set('Peso del patrón')
    send.invoke()
    deadline = time.monotonic() + 3
    while 'Idea TRIZ simulada' not in history.get('1.0', 'end'):
        assert time.monotonic() < deadline
        window.update()
        time.sleep(0.01)
    assert 'Consulta general a TRIZ:\nPeso del patrón' in history.get('1.0', 'end')
    assert entry.bind('<Return>')
    assert history.cget('state') == 'disabled'
    assert str(exportar.cget('state')) == 'normal'
    menu = exportar.nametowidget(str(exportar.cget('menu')))
    menu.invoke(2)
    from openpyxl import load_workbook

    with destino.open('rb') as archivo:
        hoja = load_workbook(archivo).active
        assert hoja['A2'].value == 'Idea TRIZ simulada'

    summary = next(
        widget for widget in widgets
        if isinstance(widget, tk.Button) and widget.cget('text') == 'Resumen local'
    )
    summary.invoke()
    assert 'Diagnóstico local:\n' in history.get('1.0', 'end')


def test_panel_informa_error_externo_y_permite_reintentar(window, monkeypatch):
    def fail(_tema):
        raise RuntimeError('Falta clave OpenRouter')

    monkeypatch.setattr(graf, 'consultar_triz_general', fail)
    panel = graf.crear_panel_dialogo_ia(
        window, graf.AgenteEvaluacionTRIZ([0], '%dif')
    )
    widgets = list(_descendants(panel))
    history = next(widget for widget in widgets if isinstance(widget, tk.Text))
    send = next(
        widget for widget in widgets
        if isinstance(widget, tk.Button) and widget.cget('text') == 'Consultar TRIZ'
    )

    send.invoke()

    deadline = time.monotonic() + 3
    while 'Falta clave OpenRouter' not in history.get('1.0', 'end'):
        assert time.monotonic() < deadline
        window.update()
        time.sleep(0.01)
    assert send.cget('state') == 'normal'
    from tkinter import ttk

    exportar = next(widget for widget in widgets if isinstance(widget, ttk.Menubutton))
    assert str(exportar.cget('state')) == 'normal'


@pytest.mark.parametrize(('indice', 'extension'), [(0, '.pdf'), (1, '.docx'), (2, '.xlsx')])
def test_exporta_diagnostico_local_sin_consultar_triz(window, monkeypatch, tmp_path, indice, extension):
    from tkinter import filedialog, messagebox, ttk

    destino = tmp_path / f'diagnóstico semanal{extension}'
    opciones = []
    errores = []

    def elegir_destino(**kwargs):
        opciones.append(kwargs)
        return str(destino)

    monkeypatch.setattr(filedialog, 'asksaveasfilename', elegir_destino)
    monkeypatch.setattr(messagebox, 'showinfo', lambda *args, **kwargs: None)
    monkeypatch.setattr(messagebox, 'showerror', lambda *args, **kwargs: errores.append(args))
    agente = graf.AgenteEvaluacionTRIZ([0.1, -0.2], '%dif')
    panel = graf.crear_panel_dialogo_ia(window, agente)
    exportar = next(w for w in _descendants(panel) if isinstance(w, ttk.Menubutton))
    assert str(exportar.cget('state')) == 'normal'
    menu = exportar.nametowidget(str(exportar.cget('menu')))
    menu.invoke(indice)

    assert not errores
    assert opciones[0]['parent'] is window
    assert opciones[0]['defaultextension'] == extension
    if extension == '.pdf':
        assert destino.read_bytes().startswith(b'%PDF-')
        return
    elif extension == '.docx':
        from docx import Document

        contenido = '\n'.join(p.text for p in Document(str(destino)).paragraphs)
    else:
        from openpyxl import load_workbook

        with destino.open('rb') as archivo:
            hoja = load_workbook(archivo).active
            contenido = '\n'.join(valor for (valor,) in hoja.values if valor)
    assert '1/2 = 50.0%' in contenido


def test_vista_previa_integra_grafica_y_panel_en_misma_ventana(window, monkeypatch):
    agente = graf.AgenteEvaluacionTRIZ([0.1, -0.2], '%dif')
    inspected = []

    def mainloop():
        window.update_idletasks()
        widgets = list(_descendants(window))
        assert any(isinstance(widget, tk.Canvas) for widget in widgets)
        history = next(widget for widget in widgets if isinstance(widget, tk.Text))
        assert agente.analizar() in history.get('1.0', 'end')
        inspected.append(True)

    monkeypatch.setattr(tk, 'Tk', lambda: window)
    monkeypatch.setattr(window, 'mainloop', mainloop)
    fig, _ = graf.plt.subplots()
    try:
        assert graf.mostrar_grafica_en_dialogo(fig, agente=agente) is True
    finally:
        graf.plt.close(fig)
    assert inspected == [True]


def test_vista_previa_mantiene_todas_las_opciones_en_ventana_compacta(
    window, monkeypatch
):
    from tkinter import ttk

    agente = graf.AgenteEvaluacionTRIZ([0.1, -0.2], '%dif')
    esperado = {
        'Consultar TRIZ', 'Resumen local', 'Ajustar peso y parámetros',
        'Exportar análisis', 'Guardar gráfica', 'Cerrar',
    }
    medido = {}

    def mainloop():
        window.geometry('640x460+0+0')
        window.deiconify()
        window.update()
        medido['size'] = (window.winfo_width(), window.winfo_height())
        for widget in _descendants(window):
            if isinstance(widget, (tk.Button, ttk.Menubutton, ttk.Combobox)):
                nombre = 'Selector de temas' if isinstance(widget, ttk.Combobox) else widget.cget('text')
                x = widget.winfo_rootx() - window.winfo_rootx()
                y = widget.winfo_rooty() - window.winfo_rooty()
                medido[nombre] = (
                    bool(widget.winfo_ismapped())
                    and 0 <= x and x + widget.winfo_width() <= window.winfo_width()
                    and 0 <= y and y + widget.winfo_height() <= window.winfo_height()
                    and widget.winfo_height() >= widget.winfo_reqheight()
                    and widget.winfo_width() >= widget.winfo_reqwidth()
                )

    monkeypatch.setattr(tk, 'Tk', lambda: window)
    monkeypatch.setattr(window, 'mainloop', mainloop)
    fig, _ = graf.plt.subplots(figsize=(12, 5))
    try:
        assert graf.mostrar_grafica_en_dialogo(fig, agente=agente)
    finally:
        graf.plt.close(fig)

    assert medido.pop('size') == (640, 460)
    assert medido == dict.fromkeys(esperado | {'Selector de temas'}, True)


def test_formulario_guarda_solo_valores_confirmados(window, monkeypatch):
    from tkinter import messagebox

    saved = []
    monkeypatch.setattr(graf, 'load_model_configuration', lambda: None)
    monkeypatch.setattr(
        graf, 'save_model_configuration',
        lambda *args: saved.append(args),
    )
    monkeypatch.setattr(messagebox, 'askyesno', lambda *args, **kwargs: True)
    monkeypatch.setattr(messagebox, 'showinfo', lambda *args, **kwargs: None)
    graf.mostrar_ajustes_modelo(window)
    form = next(w for w in window.winfo_children() if isinstance(w, tk.Toplevel))
    entries = [w for w in _descendants(form) if isinstance(w, tk.Entry)]
    assert len(entries) == 7
    entries[0].delete(0, 'end')
    entries[0].insert(0, '60')
    entries[1].delete(0, 'end')
    entries[1].insert(0, '200')
    entries[-1].insert(0, 'test-only-password')
    button = next(
        w for w in _descendants(form)
        if isinstance(w, tk.Button) and w.cget('text') == 'Aprobar y aplicar'
    )
    button.invoke()
    assert len(saved) == 1
    assert saved[0][0].pattern_weight == 0.6
    assert saved[0][0].n_estimators == 200
    assert saved[0][1:] == ('Admin', 'test-only-password')
    assert not form.winfo_exists()


def test_formulario_cancelacion_y_error_no_aplican_cambios(window, monkeypatch):
    from tkinter import messagebox

    saved = []
    errors = []
    monkeypatch.setattr(graf, 'load_model_configuration', lambda: None)
    monkeypatch.setattr(graf, 'save_model_configuration', lambda *args: saved.append(args))
    monkeypatch.setattr(messagebox, 'askyesno', lambda *args, **kwargs: False)
    monkeypatch.setattr(
        messagebox, 'showerror', lambda *args, **kwargs: errors.append(args)
    )
    graf.mostrar_ajustes_modelo(window)
    form = next(w for w in window.winfo_children() if isinstance(w, tk.Toplevel))
    entries = [w for w in _descendants(form) if isinstance(w, tk.Entry)]
    button = next(
        w for w in _descendants(form)
        if isinstance(w, tk.Button) and w.cget('text') == 'Aprobar y aplicar'
    )
    entries[-1].insert(0, 'test-only-password')
    button.invoke()
    assert not saved
    assert entries[-1].get() == ''
    entries[0].delete(0, 'end')
    entries[0].insert(0, '81')
    button.invoke()
    assert errors
    assert not saved
    assert form.winfo_exists()
