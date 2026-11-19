"""Lanzador directo para ejecutar desde VS Code o PyCharm."""

import sys
from pathlib import Path

PACKAGE_SOURCE = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(PACKAGE_SOURCE))

from agente_triz.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
