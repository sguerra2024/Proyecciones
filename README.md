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
5. Si alguna variedad no se puede proyectar, se registra el error y el proceso continua con el resto.
6. Descarga por navegador del archivo `Proyecto_todas_variedades.xlsx`.

## 3. Reglas de Negocio del Modelo

### 3.1 Seleccion de Patron

- Se calcula una similitud entre la variedad objetivo y los otros patrones disponibles usando `Tallos/m2`.
- Nunca se permite usar como patron la misma `Bloque&Varid` proyectada.
- Se prioriza un patron de la misma familia de nombre cuando existe (normalizando prefijos numericos).
- El flujo usa un solo patron seleccionado por variedad para construir las variables del modelo.

### 3.2 Variables de Entrenamiento

El modelo de regresion recibe las siguientes variables de entrada:

- `Tallos/m2`
- `Tallos_m2_patron`
- `Produccion_patron`
- `Tallos_m2_patron_ponderado`
- `Produccion_patron_ponderado`
- `Incremento_tallos_patron`
- `Incremento_produccion_patron`
- `sn_alto`

Adicionalmente se agrega una variable temporal derivada para el entrenamiento/prediccion:

- `Semana_orden`

La variable objetivo de entrenamiento es:

- `Produccion_ajustada` (construida a partir de `Produccion` y del patron)

### 3.3 Modelo Utilizado

Se entrena un `RandomForestRegressor` con estos parametros:

- `n_estimators = 100`
- `random_state = 42`
- `max_depth = 16`
- `min_samples_leaf = 1`
- `min_samples_split = 2`
- `max_features = 'sqrt'`

### 3.4 Ventana de Entrenamiento

El entrenamiento se realiza usando datos a partir de:

- `Anio >= 2025`

Antes de calcular metricas y en el set de entrenamiento se excluyen las ultimas 4 semanas por variedad.

Se construye un dataset con historial semanal de la variedad y caracteristicas del patron seleccionado.

### 3.5 Ajustes Aplicados

- Se construye un dataset con variables de nivel, cambios y ciclo semanal.
- Se mezcla la prediccion del modelo con la proyeccion del patron mediante `patron_prediction_weight` y un ajuste residual configurable.
- Se aplican ajustes sobre la serie del patron en semanas con desviaciones altas (z-score) para amortiguar picos.
- Se incorpora una bandera `sn_alto` para ajustar la sensibilidad de mezcla ante escenarios de alta relacion senal/ruido.
- Se ajusta la media final de la prediccion para alinearla con la produccion real observada.

### 3.6 Ajuste por Factor de Diferencia

- Se calcula un factor por `Bloque&Varid` a partir de dos medias de 2025.
- La primera media usa las semanas `1 a 17`.
- La segunda media usa las semanas `22 a 52`.
- El factor se define como `Media_2025_Sem22_52 / Media_2025_Sem1_17`.
- Si no existe factor especifico para una variedad, se usa el factor global ponderado como respaldo.
- El ajuste se aplica solo a la proyeccion de `2026` desde la semana `17` en adelante.
- El valor aplicado queda trazado en la exportacion con el origen del factor (`variedad`, `global` o `neutral`).

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

- Hoja `Estimado_modelo` con columnas base (`Anio`, `Semana`, `Producto`, `Finca`, `Bloque`, `Variedad`, `Bloque&Varid`) mas `Estimado_modelo`.
- Hoja `MSE_por_BloqueVarid` con `Bloque&Varid`, `MSE`, `MSE_proy_patron` y `S/N` (si disponible).
- En la hoja `Estimado_modelo` se exportan solo registros del anio 2026, solo las ultimas 4 semanas por variedad y solo filas con `Estimado_modelo > 0`.
- `Estimado_modelo` se exporta redondeado a entero.
- En la exportacion se incluyen tambien `Factor_diferencia_2025`, `Semanas_factor_2026_desde_17` y `Origen_factor_2025` para auditoria.

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

## 6. Conexiones API con IA

El proyecto mantiene conectividad con proveedores de IA para soporte de análisis y carga de archivos. La lógica está implementada en el archivo `ProyAst.py` y soporta los siguientes proveedores:

- `anthropic`
- `openai`
- `github` (GitHub Models)

### 6.1 Variables de entorno

Se requieren las siguientes variables de entorno para habilitar las conexiones:

- `LLM_PROVIDER`: proveedor activo (`anthropic`, `openai` o `github`)
- `ANTHROPIC_API_KEY` o `ANTHROPIC_KEY`: clave para Anthropic
- `OPENAI_API_KEY`: clave para OpenAI
- `GITHUB_MODELS_TOKEN` o `GITHUB_TOKEN`: token para GitHub Models
- `GITHUB_MODELS_BASE_URL`: URL base opcional para GitHub Models
- `OPENAI_BASE_URL`: URL base opcional para OpenAI compatible
- `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `GITHUB_MODEL`: modelos preferidos por proveedor

### 6.2 Funcionalidad soportada

- Consultas generales a modelos de lenguaje para análisis de contexto.
- Carga de archivos a Anthropic para procesamiento asistido.
- Selección automática de modelos alternativos si el principal no está disponible.

### 6.3 Dependencias

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
- En esos casos, reporta motivo en el resumen de errores y continua con las demas variedades.
- Para mejores resultados, mantener historial actualizado y consistente por semana.

## 9. Contacto

- +593 985381052
- +1 (240) 3576750
- sguerra@agromejoraecuador.com

