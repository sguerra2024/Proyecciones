Estimaciones de lotes agricolas productivos, en base a patrones historicos y modelo machine learning.

Estructura actual:
- ProyAst.py: interfaz Streamlit.
- api_render.py: API FastAPI para Render.
- projection_core.py: logica compartida de lectura, validacion y proyeccion.

Ejecucion local:
- Streamlit: streamlit run ProyAst.py --server.port 8502
- API: uvicorn api_render:app --reload --port 8000

