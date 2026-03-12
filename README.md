Estimaciones de lotes agricolas productivos, en base a patrones historicos y modelo machine learning.

Estructura actual:
- ProyAst.py: interfaz Streamlit.
- api_render.py: API FastAPI para Render.
- projection_core.py: logica compartida de lectura, validacion y proyeccion.

Ejecucion local:
- Streamlit: streamlit run ProyAst.py --server.port 8502
- API: uvicorn api_render:app --reload --port 8000

Render:
- render.yaml define dos servicios: uno para la API y otro para Streamlit.
- Configura en Render las variables API_KEY, API_URL y FRONTEND_ORIGINS segun el entorno.
- En el servicio Streamlit, API_URL debe apuntar al endpoint /predict o /api/v1/predict de la API desplegada.
