# Agente TRIZ

Paquete Python independiente para consultar un agente de resolución de problemas
por medio de OpenRouter. No depende de `ProyAst`, Streamlit ni de los datos de
AGROMEJORA_DATA. Puede usarse como librería desde otra aplicación.

## Instalación

Desde este directorio:

```powershell
python -m pip install .
```

Para desarrollo local, instala en modo editable:

```powershell
python -m pip install -e .
```

## Configuración

Define `OPENROUTER_API_KEY` antes de iniciar la aplicación. `OPENROUTER_MODEL`
es opcional; si no se define, se usa `anthropic/claude-sonnet-4.5`.

```powershell
$env:OPENROUTER_API_KEY = "tu-clave"
$env:OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
```

También se puede pasar la clave directamente al constructor. Evita guardar claves
en el código fuente.

## Ejecutar el diálogo

Desde el directorio `agente_triz`, ejecuta:

```powershell
python launch.py
```

También puedes instalar el paquete y ejecutarlo con:

```powershell
python -m pip install -e .
python -m agente_triz
```

La instalación crea además el comando `agente-triz`. En VS Code o PyCharm,
abre `agente_triz/launch.py` y pulsa **Run**. El lanzador abre directamente el
diálogo en una ventana independiente; no abre automáticamente al ejecutar
`ProyAst.py`. La ventana se ejecuta localmente, pero **Explorar soluciones**
envía la consulta a OpenRouter.
En pantallas pequeñas, usa la barra vertical del formulario. Los botones
**Explorar soluciones**, **Agregar enlace** y **Cerrar** permanecen al pie.
Tras obtener una respuesta, usa **Exportar análisis** para elegir **PDF
(.pdf)**, **Word (.docx)** o **Excel (.xlsx)** y seleccionar dónde guardarla.
Se exporta el texto del análisis mostrado, sin volver a consultar OpenRouter.

## Integración desde Python

```python
from agente_triz import AgenteTRIZ

agente = AgenteTRIZ()
respuesta = agente.consultar(
    "¿Cómo puedo investigar la causa raíz de este problema?",
    contexto_adicional="La falla aparece después del cambio de proceso.",
)
print(respuesta)
```

Hay métodos de conveniencia para errores de estimación, series históricas,
cambios de sistema, mercados y análisis de rentabilidad. Para una serie numérica,
el paquete calcula estadísticas antes de enviarlas al modelo:

```python
respuesta = agente.analizar_serie_historica(
    "Producción semanal",
    [120, 135, 128, 142],
)
```

La guía TRIZ incluye métodos de solución. El primero eleva el problema
específico a uno genérico, genera alternativas en ese nivel y las adapta de
vuelta al sistema concreto, preservando las restricciones y validando con
evidencia. El segundo explora si una acción perjudicial puede transferirse,
realizarse dentro del propio sistema con pequeños cambios o eliminarse por no
ser necesaria; cada opción se trata como hipótesis que debe validarse. El
tercero identifica contradicciones técnicas o físicas que bloquean una función
y las usa para orientar alternativas sin asumir compromisos prematuros. El
cuarto aplica principios inventivos solo si el problema ya está definido y
requiere una búsqueda inventiva; cada principio debe relacionarse con el
conflicto y conducir a alternativas comprobables.

La clase `AgenteAnalista` se conserva como alias de compatibilidad. Para nuevas
integraciones se recomienda importar `AgenteTRIZ`.

### Recomendaciones Lean Production

Pide recomendaciones Lean al describir el problema, o selecciona **Lean
Production** en `Graf_evaluacion.py` para ideas generales sobre el ciclo semanal
de proyección y evaluación. El agente también incorpora Lean cuando la consulta
trata sobre flujo, desperdicio o variabilidad operativa. Lean complementa las
vías TRIZ existentes: propone herramientas según el problema, distingue lo
observado de las hipótesis y sugiere un piloto con un indicador antes y después.
No estima ahorros sin evidencia ni aplica cambios al modelo. El tema
predefinido de la gráfica no envía datos de evaluación.

## Diálogo de exploración de soluciones

El paquete incluye un diálogo modal de escritorio Tkinter para enfocar el
problema, seleccionar una o más de las cuatro vías y ver enlaces del banco
local filtrados por las vías seleccionadas y por los principios inventivos
activos. El diálogo también permite cargar un enlace, añadir una nota propia y
asociarlo con uno o varios de los 40 principios clásicos. Los enlaces cargados
se guardan en `%APPDATA%/AgenteTRIZ/banco_inspiracion.json` en Windows; se usa
la carpeta de usuario si `APPDATA` no está definida. Copia y completa
`src/agente_triz/data/plantilla_banco_inspiracion.json` para preparar entradas
por adelantado, o proporciona la ruta a un banco JSON existente:

```python
from agente_triz import AgenteTRIZ, mostrar_dialogo_solucion

agente = AgenteTRIZ()
respuesta = mostrar_dialogo_solucion(
    agente,
    ruta_banco="data/mi_banco_inspiracion.json",
)
agente.cerrar()
```

Cada recurso debe indicar sus `herramientas_triz`, usando uno o más de estos
identificadores: `problema_generico`, `transferir_accion`, `contradicciones` y
`principios_inventivos`. Para esta última vía, añade `principios_inventivos`
como lista de identificadores en cada entrada. Los identificadores del
catálogo se corresponden con los nombres mostrados en la ventana. Al filtrar,
los principios activos restringen los recursos de la vía 4; las referencias
de otras vías seleccionadas siguen apareciendo. Los recursos filtrados y sus
notas se incluyen como contexto al pedir el análisis. El diálogo prepara
el contexto SQLite en el hilo de la ventana y hace la consulta a OpenRouter
en segundo plano. No abre, descarga ni verifica los enlaces. Para
integrarlo en una ventana Tkinter existente, pasa la raíz mediante `parent`;
la función devuelve el análisis o `None` si se cierra antes de generarlo.

## Base de conocimiento

El paquete carga en memoria el material Markdown incluido en `data/`, entre
ello una síntesis de conceptos fundamentales TRIZ. Su atribución y condiciones
de licencia se indican en `data/NOTICE.txt`. También se puede proporcionar una
base SQLite existente:

```python
from agente_triz import AgenteTRIZ, BaseConocimiento

base = BaseConocimiento("conocimiento/teoria.db")
base.importar_texto("teoria_triz.md")
base.cerrar()

agente = AgenteTRIZ(ruta_conocimiento="conocimiento/teoria.db")
```

Los archivos Markdown deben separar capítulos con encabezados de nivel 1 (`#`).
Los capítulos relevantes se añaden como contexto a cada consulta. Importa tus
propios capítulos para ampliar el conocimiento especializado de TRIZ.
`agente.cerrar()` cierra la conexión local cuando ya no se necesite.

El contenido adaptado de Wikipedia se distribuye bajo CC BY-SA 4.0; esta
licencia de contenido no cambia la licencia del código del paquete. Si
redistribuyes o adaptas ese contenido, conserva la atribución y aplica los
términos de compartir igual indicados en el aviso.

## Plantilla del banco de inspiración

Usa `src/agente_triz/data/plantilla_banco_inspiracion.json` como formato de captura para un
banco de preguntas, enlaces y videos que ayude a generar ideas después de
formular el problema. Duplica la estructura de `entradas` por cada recurso y
reemplaza el registro de ejemplo.

Cada entrada guarda su tipo, enlace, idioma, pregunta para estimular la
transferencia de ideas, notas originales, etiquetas y relación con herramientas
TRIZ. Incluye los datos de autoría, organización y licencia cuando se conozcan;
no asumas que un recurso es reutilizable solo porque puede verse en línea.
Para videos, registra título y duración como metadatos y resume la idea con
palabras propias. No incluyas el archivo, una transcripción o extractos extensos
sin tener autorización o una licencia que permita ese uso.

Esta plantilla define el formato de captura; el agente todavía no importa ni
consulta automáticamente este banco. La integración de búsqueda y recuperación
del banco se realiza por separado.

## Privacidad y límites

Las preguntas y el contexto adicional se envían al proveedor de OpenRouter
configurado. No incluyas datos personales, credenciales ni información
confidencial sin autorización. El agente no consulta automáticamente fuentes web:
las respuestas sobre precios o tendencias actuales requieren datos externos
proporcionados por la aplicación que lo integra.

## Relación con ProyAst

`ProyAst` conserva su flujo de IA agrícola independiente y ofrece este agente
en `Análisis Avanzado > Solución de problemas inventivos TRIZ`. La interfaz
reutiliza las vías TRIZ del paquete, muestra las respuestas y registra las
consultas en la base SQLite de IA de ProyAst; las llamadas usan
`OPENROUTER_API_KEY` y el modelo opcional `OPENROUTER_MODEL`.
