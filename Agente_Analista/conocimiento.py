"""
Base de conocimiento en SQLite para el agente.

Almacena teoría de resolución de problemas organizada por capítulos
y permite recuperar los capítulos más relevantes para una consulta.
"""
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

# Palabras vacías frecuentes en español; se excluyen al puntuar relevancia.
STOPWORDS = frozenset(
    'de la que el en y a los del se las por un para con no una su al lo '
    'como más pero sus le ya o este sí porque esta entre cuando muy sin '
    'sobre también me hasta hay donde quien desde todo nos durante todos '
    'uno les ni contra otros ese eso ante ellos e esto mí antes algunos '
    'qué unos yo otro otras otra él tanto esa estos mucho quienes nada '
    'muchos cual poco ella estar estas algunas algo nosotros mi mis tú '
    'te ti tu tus ellas nosotras vosotros vosotras os mío mía míos mías '
    'tuyo tuya tuyos tuyas suyo suya suyos suyas nuestro nuestra '
    'nuestros nuestras cuyo cuya cuyos cuyas eso esas esos esto estas '
    'estos aquello aquellos aquella aquellas cómo cuándo cuál cuáles '
    'ser es son fue fueron ha han hay está están puede pueden'.split()
)

_PATRON_TOKEN = re.compile(r'[a-záéíóúñü0-9]+')


def _tokenizar(texto):
    """Extrae tokens en minúscula, sin stopwords, de un texto."""
    tokens = _PATRON_TOKEN.findall(texto.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


class BaseConocimiento:
    """Teoría de resolución de problemas en SQLite, consultable por relevancia."""

    def __init__(self, ruta_bd=None):
        """
        Abre (o crea) la base de datos de conocimiento.

        Args:
            ruta_bd: ruta al archivo .db. Por defecto,
                     Agente_Analista/data/teoria.db relativo a este módulo.
                     Usa ':memory:' para pruebas.
        """
        if ruta_bd is None:
            ruta_bd = Path(__file__).resolve().parent / 'data' / 'teoria.db'
        self.ruta_bd = str(ruta_bd)
        if self.ruta_bd != ':memory:':
            Path(self.ruta_bd).parent.mkdir(parents=True, exist_ok=True)
        self.conexion = sqlite3.connect(self.ruta_bd)
        self.conexion.row_factory = sqlite3.Row
        self.conexion.executescript(ESQUEMA_SQL)

    def cerrar(self):
        """Cierra la conexión a la base de datos."""
        self.conexion.close()

    def agregar_capitulo(self, numero, titulo, contenido, resumen='',
                         palabras_clave=''):
        """Inserta o reemplaza un capítulo de teoría."""
        self.conexion.execute(
            'DELETE FROM capitulos WHERE numero = ?', (numero,))
        self.conexion.execute(
            'INSERT INTO capitulos (numero, titulo, resumen, contenido, '
            'palabras_clave) VALUES (?, ?, ?, ?, ?)',
            (numero, titulo, resumen, contenido, palabras_clave)
        )
        self.conexion.commit()

    def listar_capitulos(self):
        """Devuelve (numero, titulo, resumen) de todos los capítulos."""
        filas = self.conexion.execute(
            'SELECT numero, titulo, resumen FROM capitulos ORDER BY numero'
        ).fetchall()
        return [dict(f) for f in filas]

    def capitulos_relevantes(self, pregunta, max_capitulos=2):
        """
        Selecciona los capítulos más relevantes para una pregunta.

        Puntuación por token de la pregunta encontrado en:
        palabras_clave x3, titulo x2, resumen/contenido x1.

        Returns:
            list[dict]: capítulos ordenados por relevancia (numero, titulo,
            contenido, puntuacion); vacío si no hay coincidencias.
        """
        tokens = _tokenizar(pregunta)
        if not tokens:
            return []
        filas = self.conexion.execute(
            'SELECT numero, titulo, resumen, contenido, palabras_clave '
            'FROM capitulos'
        ).fetchall()
        puntuados = []
        for fila in filas:
            puntos = 0
            campos = (
                (fila['palabras_clave'], 3),
                (fila['titulo'], 2),
                (fila['resumen'], 1),
                (fila['contenido'], 1),
            )
            for texto, peso in campos:
                tokens_campo = _tokenizar(texto or '')
                puntos += peso * len(tokens & tokens_campo)
            if puntos > 0:
                puntuados.append({
                    'numero': fila['numero'],
                    'titulo': fila['titulo'],
                    'contenido': fila['contenido'],
                    'puntuacion': puntos,
                })
        puntuados.sort(key=lambda c: (-c['puntuacion'], c['numero']))
        return puntuados[:max_capitulos]

    def formatear_contexto(self, pregunta, max_capitulos=2):
        """
        Devuelve texto listo para inyectar como contexto en una consulta LLM.
        """
        capitulos = self.capitulos_relevantes(pregunta, max_capitulos)
        if not capitulos:
            return ''
        bloques = ['[TEORÍA DE RESOLUCIÓN DE PROBLEMAS — referencia]']
        for cap in capitulos:
            bloques.append(
                f"Capítulo {cap['numero']}: {cap['titulo']}\n{cap['contenido']}"
            )
        return '\n\n'.join(bloques)

    def importar_texto(self, ruta_texto):
        """
        Importa capítulos desde un archivo de texto/markdown.

        Formato esperado: cada capítulo inicia con un encabezado markdown
        de nivel 1 ('# Título'). Los encabezados '##' o inferiores se
        conservan como secciones dentro del contenido del capítulo.

        Returns:
            int: número de capítulos importados.
        """
        texto = Path(ruta_texto).read_text(encoding='utf-8')
        partes = re.split(r'(?m)^#\s+', texto)
        # partes[0] es el texto previo al primer '#'; se ignora.
        importados = 0
        for cuerpo in partes[1:]:
            lineas = cuerpo.split('\n', 1)
            titulo = lineas[0].strip()
            contenido = lineas[1].strip() if len(lineas) > 1 else ''
            if not titulo or not contenido:
                continue
            importados += 1
            self.agregar_capitulo(importados, titulo, contenido)
        return importados
