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
- Tallos/m2

Los patrones son producciones anteriores, de similitud de clima, finca, pinch, ciclo. Se recomienda tener al menos datos 7 variedades o mas, de 52 semanas anteriores en un archivo excel. 
El modelo solo puede proyectar variedades incluidas en el archivo de entrada.
Ej. Deseo proyectar la semana 10,11,12,13 del 2026, copio y pego los historicos en la columna de 'Produccion' en las semanas a proyectar 2026, para que el modelo proyecte lo indicado 

El siguiente paso es subir el archivo al sistema, seleccionar la variedad y visualizar la proyeccion del modelo.

Para mayor información contacto: +593 985381052; +1(240)3576750; sguerra@agromejoraecuador.com

Estructura actual:
- ProyAst.py: interfaz Streamlit.
- api_render.py: API FastAPI para Render.
- projection_core.py: logica compartida de lectura, validacion y proyeccion.

Ejecucion local:
- Streamlit: streamlit run ProyAst.py --server.port 8502
- API: uvicorn api_render:app --reload --port 8000

