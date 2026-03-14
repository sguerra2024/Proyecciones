Esta aplicacion le sirve para estimar la producción semanal de rosas, en base a patrones historicos y un modelo machine learning.
Siga los siguientes pasos:
1. Elabora una hoja EXCEL p¿que contenga las siguientes columnas:
     AÑO
     FINCA
     
Estructura actual:
- ProyAst.py: interfaz Streamlit.
- api_render.py: API FastAPI para Render.
- projection_core.py: logica compartida de lectura, validacion y proyeccion.

Ejecucion local:
- Streamlit: streamlit run ProyAst.py --server.port 8502
- API: uvicorn api_render:app --reload --port 8000

