## Proyecciones de Produccion de Rosas

Aplicacion para estimar produccion semanal por variedad (`Bloque&Varid`) usando:

- patrones historicos similares,
- un modelo `RandomForestRegressor`,
- exportacion de resultados a Excel.

Incluye dos modos:

- Proyeccion individual por variedad.
- Proyeccion masiva por finca.

Las consultas a la IA utilizan este README como fuente autorizada de reglas y
definiciones, junto con la informacion almacenada en el contexto de la
aplicacion. Las consultas externas pueden usar SerpApi cuando se configura
`SERPAPI_API_KEY` en el entorno.

Cada llamada enviada a un proveedor de IA se registra en SQLite antes de salir
de la aplicacion. La base predeterminada es `data/consultas_ia.db` y contiene el
proveedor, modelo, prompt original y final, respuesta, estado, error y duracion.
La ruta puede cambiarse con `AI_LOG_DB_PATH`. Las credenciales configuradas se
reemplazan por `[REDACTADO]` antes de almacenar cualquier texto.

Streamlit y FastAPI usan el mismo modelo de produccion definido en
`projection_core.py`; no existe un modelo alternativo exclusivo para la API.

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
6. Calcula el escenario informativo con amortiguador para cada variedad.
7. Descarga por navegador del archivo `Proyecto_todas_variedades.xlsx`.

## 3. Reglas de Negocio del Modelo

### 3.1 Seleccion de Patron

- Para la IA, `PATRON` es la mejor opcion historica distinta del mismo `Bloque&Varid` proyectado, cuyo ajuste (`FIT`) respecto a los `Tallos/m2` actuales permite realizar una proyeccion futura.
- Cada serie de `Tallos/m2` se normaliza a media 0 y desviacion estandar 1.
- Se selecciona como patron el candidato con menor MSE entre las series normalizadas.
- Antes del ranking se descartan los candidatos con menos registros validos que
	la variedad objetivo; en ese caso se selecciona el siguiente patron por MSE.
- Nunca se permite usar como patron la misma `Bloque&Varid` proyectada.
- Un `Bloque&Varid` diferente puede usarse como patron aunque corresponda al mismo nombre de variedad.
- El flujo usa un solo patron seleccionado por variedad para construir las variables del modelo.
- Despues de predecir, `S` es la potencia de la proyeccion del patron seleccionado.
- `N` es el MSE entre la estimacion del modelo y la produccion real.
- La relacion se expresa en decibelios como `S/N = 10 * log10(S / N)`.

### 3.2 Variables de Entrenamiento

El modelo de regresion recibe las siguientes variables de entrada:

- `Tallos/m2`
- `Tallos_m2_patron`
- `Produccion_patron`
- `Tallos_m2_patron_ponderado`
- `Produccion_patron_ponderado`
- `Incremento_tallos_patron`
- `Incremento_produccion_patron`

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
- Se ajusta la media final de la prediccion para alinearla con la produccion real observada.

### 3.6 Ajuste por Factor de Diferencia

- Se calcula un factor por `Bloque&Varid` a partir de dos medias de 2025.
- La primera media usa las semanas `1 a 17`.
- La segunda media usa las semanas `22 a 52`.
- El factor se define como `Media_2025_Sem22_52 / Media_2025_Sem1_17`.
- Si no existe factor especifico para una variedad, se usa el factor global ponderado como respaldo.
- El ajuste se aplica solo a las 4 semanas mas recientes de `2026` cuya semana sea mayor que `24`.
- El valor aplicado queda trazado en la exportacion con el origen del factor (`variedad`, `global` o `neutral`).

### 3.7 Amortiguador de Sobreestimacion

- El error real del modelo se registra en la columna `%dif` con la formula
	`(Produccion real - Estimado_modelo) / Produccion real`.
- Un `%dif` positivo indica subestimacion y uno negativo indica
	sobreestimacion; la columna usa proporcion decimal (`0.10` equivale a 10%).
- `projection_core.py` usa este error firmado del Excel procesado para entrenar
	la magnitud y direccion del amortiguador.
- Al generar la evaluacion con `plot_series_evaluation.py`, el informe se guarda
	en `Evaluacion/errores_evaluacion_modelo.csv`; el amortiguador usa exactamente
	el numero de semanas `Anio-Semana` presentes en ese informe, no una ventana
	fija de 16 semanas.
- El amortiguador se calcula con un `RandomForestRegressor` para estimar la magnitud y un `RandomForestClassifier` para determinar el riesgo de sobreestimacion.
- La convención del amortiguador sigue el caso real del error del modelo:
	positivo cuando el modelo subestima y negativo cuando sobreestima.
- Solo los errores relativos negativos de `Evaluacion/errores_evaluacion_modelo.csv` se consideran sobreestimaciones del modelo.
- El error se define como `(real - modelo) / real`; por tanto, los errores positivos no intervienen en la calibracion.
- El limite se obtiene de la mediana historica de la sobreestimacion expresada como proporcion de la prediccion. Con la evaluacion actual es aproximadamente `11%`, en lugar de un limite fijo de `30%`.
- El amortiguador solo se aplica al escenario informativo cuando la probabilidad estimada de sobreestimacion es al menos `60%`.
- `Estimado_modelo` permanece como proyeccion oficial y no es modificado.
- `Proyeccion_con_amortiguador_IA` se calcula como `Estimado_modelo - Amortiguador`, con minimo cero.
- `M2 Amortiguador` representa la proporcion de area requerida por el amortiguador y se redondea hacia arriba.
- En `Analisis Avanzado` la salida se simplifica a `M2 Amortiguador` y `Proyeccion_con_amortiguador_IA`.

## 4. Exportaciones

### 4.1 Individual

Descarga por navegador con nombre:

- `Proyecto.xlsx`

Hojas generadas:

- `Datos_modelo`
- `Errores_modelo`
- `Promedio_anual`
- `Resumen`

Desde `Analisis Avanzado` tambien se puede descargar:

- `Proyecto_amortiguado.xlsx`, con los identificadores de la proyeccion, `M2 Amortiguador` y `Proyeccion_con_amortiguador_IA`.

### 4.2 Masiva

Descarga por navegador con nombre:

- `Proyecto_todas_variedades.xlsx`

Contenido:

- Hoja `Estimado_modelo` con columnas base (`Anio`, `Semana`, `Producto`, `Finca`, `Bloque`, `Variedad`, `Bloque&Varid`), `Estimado_modelo` y `Proyeccion_con_amortiguador_IA`.
- Hoja `MSE_por_BloqueVarid` con `Bloque&Varid`, `MSE`, `MSE_proy_patron` y `S/N` (si disponible).
- En la hoja `Estimado_modelo` se exportan solo registros del anio 2026, solo las ultimas 4 semanas por variedad y solo filas con `Estimado_modelo > 0`.
- `Estimado_modelo` y `Proyeccion_con_amortiguador_IA` se exportan redondeados a enteros.

Desde `Analisis Avanzado` tambien se puede descargar:

- `Proyecto_todas_variedades_amortiguado.xlsx`, con las columnas base, `M2 Amortiguador` y `Proyeccion_con_amortiguador_IA`.

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

Los endpoints de prediccion entrenan el mismo `RandomForestRegressor` de
produccion usado por Streamlit y devuelven la variedad objetivo, el patron
seleccionado, RMSE, promedio de produccion y trazabilidad del factor 2025.

## 6. Conexiones API con IA

El proyecto mantiene conectividad con proveedores de IA para soporte de análisis y carga de archivos. La lógica está implementada en el archivo `ProyAst.py` y soporta los siguientes proveedores:

- `anthropic` con Claude como configuracion predeterminada
- `openai`
- `github` (GitHub Models)

### 6.1 Variables de entorno

Se requieren las siguientes variables de entorno para habilitar las conexiones:

- `LLM_PROVIDER`: proveedor activo; el valor predeterminado es `anthropic`
- `ANTHROPIC_MODEL`: modelo principal; el valor predeterminado es `claude-3-5-sonnet-latest`
- `ANTHROPIC_API_KEY` o `ANTHROPIC_KEY`: clave para Anthropic
- `OPENAI_API_KEY`: clave para OpenAI
- `GITHUB_MODELS_TOKEN` o `GITHUB_TOKEN`: token para GitHub Models
- `GITHUB_MODELS_BASE_URL`: URL base opcional para GitHub Models
- `OPENAI_BASE_URL`: URL base opcional para OpenAI compatible
- `OPENAI_MODEL` y `GITHUB_MODEL`: modelos alternativos por proveedor

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

- `ProyAst.py`: interfaz Streamlit, seleccion de patron y exportaciones.
- `api_render.py`: API para consumo externo/despliegue.
- `projection_core.py`: modelo de produccion compartido por Streamlit y API.
- `Evaluacion/errores_evaluacion_modelo.csv`: errores relativos usados para calibrar el amortiguador.
- `modelos/`: modelos serializados por variedad.

## 8. Observaciones Operativas

- Si en masiva no hay datos suficientes para una variedad, el sistema no detiene el proceso global.
- En esos casos, reporta motivo en el resumen de errores y continua con las demas variedades.
- La proyeccion amortiguada es un escenario operativo informativo; las graficas, metricas y proyeccion oficial conservan `Estimado_modelo`.
- Para mejores resultados, mantener historial actualizado y consistente por semana.

## 9. Contacto

- +593 985381052
- +1 (240) 3576750
- sguerra@agromejoraecuador.com

