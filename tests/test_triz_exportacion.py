import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agente_triz" / "src"))
from agente_triz import exportacion


def test_exporta_pdf_paginado_con_acentos(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    ruta = tmp_path / "informe.pdf"
    texto = "Diagnóstico: producción < objetivo & revisión.\n" * 150
    exportacion.exportar_analisis(ruta, texto + "Conclusión final.")
    paginas = pypdf.PdfReader(ruta).pages
    assert len(paginas) > 1
    assert "Diagnóstico: producción < objetivo & revisión." in paginas[0].extract_text()
    assert "Conclusión final." in paginas[-1].extract_text()


def test_exporta_word_con_parrafos(tmp_path):
    from docx import Document

    ruta = tmp_path / "informe.docx"
    exportacion.exportar_analisis(ruta, "Diagnóstico\r\n\r\nRevisión")
    assert [p.text for p in Document(str(ruta)).paragraphs] == [
        "Análisis de Agente TRIZ", "Diagnóstico", "", "Revisión",
    ]


def test_exporta_excel_como_texto_sin_formulas_ni_truncamiento(tmp_path):
    from openpyxl import load_workbook

    ruta = tmp_path / "informe.xlsx"
    larga = "á" * 33000
    exportacion.exportar_analisis(ruta, f"Diagnóstico\n=1+1\n{larga}")
    hoja = load_workbook(ruta).active
    assert hoja["A2"].value == "Diagnóstico"
    assert hoja["A3"].value == "=1+1"
    assert hoja["A3"].data_type == "s"
    assert hoja["A4"].value + hoja["A5"].value == larga


def test_error_de_exportacion_conserva_archivo_previo(tmp_path, monkeypatch):
    ruta = tmp_path / "informe.pdf"
    ruta.write_bytes(b"previo")

    def fallar(temporal, _texto):
        temporal.write_bytes(b"incompleto")
        raise OSError("fallo de prueba")

    monkeypatch.setattr(exportacion, "_guardar_pdf", fallar)
    with pytest.raises(OSError, match="fallo de prueba"):
        exportacion.exportar_analisis(ruta, "Diagnóstico")
    assert ruta.read_bytes() == b"previo"
    assert list(tmp_path.iterdir()) == [ruta]
