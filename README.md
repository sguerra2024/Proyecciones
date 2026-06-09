## Proyecciones de Produccion de Rosas

Aplicacion para estimar produccion semanal por variedad (`Bloque&Varid`) usando:

- patrones historicos similares,
- un modelo `RandomForestRegressor`,
- reglas agronomicas de suavizado y control de picos,
- exportacion de resultados a Excel.

Incluye dos modos:

- Proyeccion individual por variedad.
- Proyeccion masiva por finca.

## 1. Requisitos de Entrada (Excel)

La aplicacion espera un archivo `.xlsx` con al menos estas columnas:

- `Anio`
- `Semana`
- `Producto`
- `Finca`
- `Bloque`
- `Variedad`
- `Bloque&Varid`
- `Produccion`
- `m2Variedad`
- `Ciclo`
- `Tallos/m2`

Recomendaciones de calidad de datos:

- Incluir minimo 7 variedades comparables.
- Tener historial de al menos 52 semanas.
- Mantener consistencia en escritura de `Bloque&Varid`.
- Evitar celdas vacias en `Anio`, `Semana`, `Tallos/m2` y `Produccion`.

## 2. Flujo Funcional

### 2.1 Proyeccion Individual

1. Cargar Excel.
2. Seleccionar `Finca`.
3. Seleccionar `Bloque&Variedad`.
4. El sistema selecciona un patron (nunca la misma variedad objetivo).
5. Entrena modelo y genera grafica comparativa:
	- Produccion real
	- Proyeccion por patron
	- Estimado del modelo
6. Permite descargar Excel por navegador (`Proyecto.xlsx`).

### 2.2 Proyeccion Masiva

1. Cargar Excel.
2. Seleccionar `Finca`.
3. Clic en `PROYECTAR FINCA`.
4. Proyecta todas las `Bloque&Varid` de la finca respetando el orden original de entrada.
5. Si alguna variedad no se puede proyectar, se asigna `0` en `Estimado_modelo`.
6. Descarga por navegador del archivo `Proyecto_todas_variedades.xlsx`.

## 3. Reglas de Negocio del Modelo

### 3.1 Seleccion de Patron

- Se calcula distancia por similitud de `Tallos/m2`.
- Nunca se permite usar como patron la misma `Bloque&Varid` proyectada.
- Se prioriza patron de la misma familia de nombre cuando existe (normalizando prefijos numericos).

### 3.2 Variables de Entrenamiento

El entrenamiento usa dos variables:

- `Tallos/m2`
- `Tallos_m2_patron_ponderado`

### 3.3 Ventana de Entrenamiento

Se entrena desde:

- `Anio > 2025`, o
- `Anio == 2025` y `Semana >= 1`.

### 3.4 Ajustes Aplicados

- Mezcla del estimado del modelo con proyeccion del patron (`patron_prediction_weight`).
- Reglas de control de picos (`peak_decay_train`, `peak_decay_pred`).
- Escalado final para alinear media con produccion real observada.

## 4. Exportaciones

### 4.1 Individual

Descarga por navegador con nombre:

- `Proyecto.xlsx`

Hojas generadas:

- `Datos_modelo`
- `Errores_modelo`
- `Promedio_anual`
- `Resumen`

### 4.2 Masiva

Descarga por navegador con nombre:

- `Proyecto_todas_variedades.xlsx`

Contenido:

- Hoja `Estimado_modelo`
- Solo columna `Estimado_modelo`
- Valores sin decimales (enteros)

Nota movil:

- Al descargar desde telefono, el archivo queda en la carpeta de Descargas del navegador/dispositivo.

## 5. Ejecucion Local

### 5.1 Streamlit

```bash
python -m streamlit run ProyAst.py --server.headless true
```

### 5.2 API (FastAPI)

```bash
uvicorn api_render:app --reload --port 8000
```

Endpoints principales:

- `GET /health`
- `POST /api/v1/predict`
- `POST /predict`

## 6. Dependencias

Instalar desde:

```bash
pip install -r requirements.txt
```

Incluye:

- `pandas`, `numpy`, `scikit-learn`, `openpyxl`
- `streamlit`, `matplotlib`
- `fastapi`, `uvicorn`, `python-multipart`

## 7. Estructura del Proyecto

- `ProyAst.py`: interfaz Streamlit y flujo completo de proyeccion.
- `api_render.py`: API para consumo externo/despliegue.
- `projection_core.py`: utilidades de validacion/proyeccion para API.
- `modelos/`: modelos serializados por variedad.

## 8. Observaciones Operativas

- Si en masiva no hay datos suficientes para una variedad, el sistema no detiene el proceso global.
- En esos casos, reporta motivo y exporta `0` para esa fila.
- Para mejores resultados, mantener historial actualizado y consistente por semana.

## 9. Contacto

- +593 985381052
- +1 (240) 3576750
- sguerra@agromejoraecuador.com

