Estimaciones de produccion semanal de rosas, en base a patrones historicos y machine learning.

Para iniciar, se requiere un archivo Excel con los campos:
- Anio
- Semana
- Producto
- Finca
- Bloque
- Variedad
- Bloque&Varid
- Produccion
- m2Variedad
- Ciclo
- TMP
- TMP MAX
- TMP MIN
- Tallos/m2

Para crear patrones historicos se recomienda al menos un ano de datos y 7 variedades o mas. El modelo solo puede proyectar variedades incluidas en el archivo de entrada.
Ej.si deseo proyectar la semana 10,11,12,13 del 2016, debo copiar los historicos correspondientes en las semanas a proyectar 2026, para que el modelo proyecte lo indicado 

El siguiente paso es subir el archivo al sistema, seleccionar la variedad y visualizar la proyeccion del modelo.

Estructura actual:
- ProyAst.py: interfaz Streamlit.
- api_render.py: API FastAPI para Render.
- projection_core.py: logica compartida de lectura, validacion y proyeccion.

Ejecucion local:
- Streamlit: streamlit run ProyAst.py --server.port 8502
- API: uvicorn api_render:app --reload --port 8000

