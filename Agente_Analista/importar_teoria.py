"""
Script de línea de comandos para importar teoría de resolución de
problemas a la base de conocimiento del agente.

Uso:
    python importar_teoria.py ruta/al/archivo.txt
    python importar_teoria.py capitulos.md --bd otra_base.db
    python importar_teoria.py capitulos.md --listar

Formato del archivo de texto: cada capítulo inicia con un encabezado
markdown de nivel 1 ('# Título del capítulo'); los '##' son secciones
internas y el contenido va debajo.
"""
from conocimiento import BaseConocimiento
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(
        description='Importa capítulos de teoría a la base de conocimiento.')
    parser.add_argument('archivo', nargs='?',
                        help='Archivo de texto/markdown con los capítulos')
    parser.add_argument('--bd', default=None,
                        help='Ruta de la base SQLite (por defecto, '
                             'Agente_Analista/data/teoria.db)')
    parser.add_argument('--listar', action='store_true',
                        help='Muestra los capítulos ya importados y termina')
    args = parser.parse_args()

    base = BaseConocimiento(args.bd)
    try:
        if args.listar:
            capitulos = base.listar_capitulos()
            if not capitulos:
                print('La base de conocimiento está vacía.')
            for cap in capitulos:
                print(f"Capítulo {cap['numero']}: {cap['titulo']}")
            return

        if not args.archivo:
            parser.error('Indica el archivo de texto a importar '
                         '(o usa --listar).')
        if not Path(args.archivo).exists():
            parser.error(f'No existe el archivo: {args.archivo}')

        importados = base.importar_texto(args.archivo)
        print(f'Se importaron {importados} capítulo(s) a {base.ruta_bd}')
    finally:
        base.cerrar()


if __name__ == '__main__':
    main()
