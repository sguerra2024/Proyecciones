"""Base SQLite para almacenar y recuperar conocimiento de resolución."""

import re
import sqlite3
from pathlib import Path

ESQUEMA_SQL = """
CREATE TABLE IF NOT EXISTS capitulos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    numero INTEGER NOT NULL,
    titulo TEXT NOT NULL,
    resumen TEXT DEFAULT '',
    contenido TEXT NOT NULL,
    palabras_clave TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_capitulos_numero ON capitulos(numero);
"""

STOPWORDS = frozenset(
    "de la que el en y a los del se las por un para con no una su al lo "
    "como más pero sus le ya o este sí porque esta entre cuando muy sin "
    "sobre también me hasta hay donde quien desde todo nos durante todos "
    "uno les ni contra otros ese eso ante ellos e esto mí antes algunos "
    "qué unos yo otro otras otra él tanto esa estos mucho quienes nada "
    "muchos cual poco ella estar estas algunas algo nosotros mi mis tú "
    "te ti tu tus ellas nosotras vosotros vosotras os mío mía míos mías "
    "tuyo tuya tuyos tuyas suyo suya suyos suyas nuestro nuestra "
    "nuestros nuestras cuyo cuya cuyos cuyas eso esas esos esto estas "
    "estos aquello aquellos aquella aquellas cómo cuándo cuál cuáles "
    "ser es son fue fueron ha han hay está están puede pueden".split()
)

_PATRON_TOKEN = re.compile(r"[a-záéíóúñü0-9]+")


def _tokenizar(texto):
    tokens = _PATRON_TOKEN.findall(str(texto).lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 2}


class BaseConocimiento:
    """Teoría almacenada en SQLite, consultable por relevancia textual."""

    def __init__(self, ruta_bd=None):
        if ruta_bd is None:
            ruta_bd = Path(__file__).resolve().parent / "data" / "teoria.db"
        self.ruta_bd = str(ruta_bd)
        if self.ruta_bd != ":memory:":
            Path(self.ruta_bd).parent.mkdir(parents=True, exist_ok=True)
        self.conexion = sqlite3.connect(self.ruta_bd)
        self.conexion.row_factory = sqlite3.Row
        self.conexion.executescript(ESQUEMA_SQL)

    def cerrar(self):
        self.conexion.close()

    def agregar_capitulo(self, numero, titulo, contenido, resumen="", palabras_clave=""):
        self.conexion.execute(
            "DELETE FROM capitulos WHERE numero = ?", (numero,)
        )
        self.conexion.execute(
            "INSERT INTO capitulos (numero, titulo, resumen, contenido, "
            "palabras_clave) VALUES (?, ?, ?, ?, ?)",
            (numero, titulo, resumen, contenido, palabras_clave),
        )
        self.conexion.commit()

    def listar_capitulos(self):
        filas = self.conexion.execute(
            "SELECT numero, titulo, resumen FROM capitulos ORDER BY numero"
        ).fetchall()
        return [dict(fila) for fila in filas]

    def capitulos_relevantes(self, pregunta, max_capitulos=2):
        tokens = _tokenizar(pregunta)
        if not tokens:
            return []
        filas = self.conexion.execute(
            "SELECT numero, titulo, resumen, contenido, palabras_clave "
            "FROM capitulos"
        ).fetchall()
        puntuados = []
        for fila in filas:
            puntos = 0
            campos = (
                (fila["palabras_clave"], 3),
                (fila["titulo"], 2),
                (fila["resumen"], 1),
                (fila["contenido"], 1),
            )
            for texto, peso in campos:
                puntos += peso * len(tokens & _tokenizar(texto or ""))
            if puntos:
                puntuados.append(
                    {
                        "numero": fila["numero"],
                        "titulo": fila["titulo"],
                        "contenido": fila["contenido"],
                        "puntuacion": puntos,
                    }
                )
        puntuados.sort(key=lambda capitulo: (-capitulo["puntuacion"], capitulo["numero"]))
        return puntuados[:max_capitulos]

    def formatear_contexto(self, pregunta, max_capitulos=2):
        capitulos = self.capitulos_relevantes(pregunta, max_capitulos)
        if not capitulos:
            return ""
        bloques = ["[CONOCIMIENTO DE RESOLUCIÓN DE PROBLEMAS — referencia]"]
        for capitulo in capitulos:
            bloques.append(
                f"Capítulo {capitulo['numero']}: {capitulo['titulo']}\n"
                f"{capitulo['contenido']}"
            )
        return "\n\n".join(bloques)

    def importar_texto(self, ruta_texto, numero_inicial=1):
        """Importa capítulos desde Markdown, separados por encabezados '# '."""
        texto = Path(ruta_texto).read_text(encoding="utf-8")
        return self.importar_contenido(texto, numero_inicial)

    def importar_contenido(self, texto, numero_inicial=1):
        """Importa capítulos desde una cadena con formato Markdown."""
        importados = 0
        for cuerpo in re.split(r"(?m)^#\s+", texto)[1:]:
            lineas = cuerpo.split("\n", 1)
            titulo = lineas[0].strip()
            contenido = lineas[1].strip() if len(lineas) > 1 else ""
            if titulo and contenido:
                importados += 1
                self.agregar_capitulo(
                    numero_inicial + importados - 1, titulo, contenido
                )
        return importados
