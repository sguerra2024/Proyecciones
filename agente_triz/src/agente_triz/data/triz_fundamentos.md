# Fundamentos TRIZ: enfoque sistemático para resolver problemas inventivos

Esta síntesis introductoria presenta conceptos TRIZ en palabras propias. TRIZ
es una metodología sistemática de resolución de problemas inventivos: parte de
describir el sistema y el resultado buscado, identifica conflictos entre
requisitos y ayuda a explorar soluciones usando patrones de evolución y
principios de invención. No reemplaza la validación técnica ni los datos del
dominio.

## Definir el problema y el resultado ideal

Describe el sistema, su función principal, el contexto de uso y la brecha entre
el estado actual y el resultado deseado. Separa hechos observados de supuestos.
Formula un resultado ideal que resuelva la necesidad con el menor costo,
complejidad, consumo de recursos y efectos perjudiciales posibles. Úsalo como
dirección de búsqueda, no como promesa de que se alcanzará sin costo.

### Guía de abordaje: eliminar un efecto no deseado

Cuando el problema consista en un efecto negativo específico o en una función
insuficiente, define la necesidad en tres partes:

1. **Efecto que se quiere eliminar o corregir:** describe qué efecto negativo
   ocurre, o qué función del sistema resulta insuficiente. Indica dónde y
   cuándo aparece y cómo se observa; no presupongas aún su causa.
2. **Abordaje del problema:** describe el problema en términos de la función
   afectada y busca cómo mejorar la funcionalidad del sistema. Separa el efecto
   observado de las posibles causas y restricciones.
3. **Cambio descriptivo del sistema:** expresa el cambio buscado como una
   descripción verificable de cómo debería funcionar el sistema después de
   intervenir. Describe el resultado esperado, no una solución técnica elegida
   de antemano.

Plantilla inicial:

> El sistema **[contexto]** presenta **[efecto negativo o función insuficiente]**
> en **[condición observable]**. Se busca mejorar **[función afectada]** para
> obtener **[cambio observable esperado]**, respetando **[restricciones
> verificadas]**.

Esta formulación orienta el análisis; las causas y la solución deben validarse
con evidencia antes de considerarlas confirmadas.

## Contradicciones técnicas y físicas

Una contradicción técnica aparece cuando mejorar un aspecto del sistema empeora
otro: por ejemplo, aumentar la velocidad puede elevar el consumo de energía.
Nombra explícitamente ambos aspectos y verifica que el conflicto ocurra en el
caso real.

Una contradicción física exige que una misma característica tenga requisitos
opuestos, a menudo en momentos, lugares o condiciones distintos. Antes de
aceptar el compromiso, busca si se pueden separar esos requisitos por tiempo,
espacio, condición o nivel del sistema.

El objetivo de formular contradicciones es evitar limitarse prematuramente a
una solución de compromiso. No elijas un principio de invención sin comprobar
que responde al conflicto concreto.

## Recursos disponibles

Antes de añadir componentes, identifica recursos existentes o poco utilizados:
materiales, energía, espacio, tiempo, información, propiedades del entorno,
subproductos y capacidades de componentes cercanos. Evalúa si alguno puede
cumplir una función adicional o reducir un efecto no deseado. Toda propuesta
debe comprobarse frente a seguridad, calidad, costo y restricciones operativas.

## Principios de invención como generadores de alternativas

Los principios de invención son heurísticas para ampliar el espacio de
soluciones, no recetas que garanticen éxito. Algunos patrones de búsqueda son:

- Dividir un elemento en partes independientes para facilitar ajuste o
  mantenimiento.
- Separar en el espacio, el tiempo o las condiciones los requisitos que
  entran en conflicto.
- Hacer que un componente cumpla funciones adicionales cuando ello reduzca el
  sistema total.
- Preparar una acción antes de que se necesite para evitar demoras o fallas.
- Sustituir una interacción directa por otra mediada cuando se reduzca el daño o
  la variabilidad.

Genera varias opciones con estos patrones; explica qué contradicción aborda
cada una y qué nueva desventaja podría introducir. Después filtra con pruebas,
mediciones y restricciones del caso.

## Métodos de solución

### Herramienta 1: elevar el problema específico a un problema genérico

Reformula el problema concreto en términos de la función que falla, el efecto
que se quiere eliminar o la contradicción que se debe resolver. Conserva las
relaciones esenciales entre causa observada, función y efecto, pero aparta
temporalmente nombres propios, detalles accidentales y una solución asumida.
La abstracción debe preservar las restricciones que realmente definen el
problema; no debe borrar los datos que podrían cambiar la solución.

Busca entonces una estrategia general aplicable a ese tipo de problema usando
las herramientas TRIZ pertinentes. Finalmente, traduce la estrategia al caso
original: identifica qué componentes o condiciones concretas corresponden a
cada elemento de la solución genérica, comprueba las restricciones y diseña
una prueba observable. La solución genérica orienta la búsqueda; no demuestra
por sí sola que la adaptación específica vaya a funcionar.

Flujo:

1. Registrar el problema específico y la evidencia disponible.
2. Expresarlo como un problema genérico sin adelantar una solución.
3. Resolver el problema en ese nivel de abstracción y generar alternativas.
4. Mapear cada alternativa de vuelta a los componentes del caso concreto.
5. Verificar restricciones, riesgos y evidencia mediante una prueba.

### Herramienta 2: concebir soluciones mediante transferencia o eliminación de la acción

Ante una acción perjudicial, costosa o insuficiente, genera y evalúa soluciones
con estas preguntas:

1. **Transferir la acción:** ¿puede realizarla otra persona, sistema o proceso
   que la ejecute con menor perjuicio o mayor eficacia? Comprueba que la
   transferencia no traslade el riesgo a otra parte sin resolverlo.
2. **Modificar el sistema:** ¿puede el propio sistema realizar la acción con
   pequeñas modificaciones? Identifica el cambio mínimo, sus efectos
   secundarios y cómo verificar su funcionamiento.
3. **Cuestionar la necesidad:** ¿es realmente necesaria la acción que realiza
   el sistema o proceso? Comprueba qué función cumple y qué consecuencias
   tendría reducirla o eliminarla.

Estas preguntas generan hipótesis de solución, no conclusiones. Compara las
alternativas con el resultado deseado y las restricciones del problema; valida
con evidencia antes de recomendar una implementación.

### Herramienta 3: identificar la contradicción técnica o física

Cuando una función del sistema no alcanza el resultado deseado, identifica si
hay requisitos en conflicto que limitan su mejora:

- **Contradicción técnica:** al mejorar una característica o función, otra
  característica empeora. Describe qué se quiere mejorar y qué efecto adverso
  aparece como consecuencia; comprueba que ambos estén respaldados por el caso.
- **Contradicción física:** una misma característica debe tener valores o
  estados opuestos para satisfacer necesidades distintas. Nombra la
  característica y los dos requisitos, y determina si corresponden a
  componentes, lugares, momentos o condiciones diferentes.

Formula el conflicto con precisión antes de proponer soluciones. Busca cómo
mejorar la función sin aceptar automáticamente un compromiso: comprueba si los
requisitos pueden separarse en el tiempo, el espacio, las condiciones o entre
componentes. Usa la contradicción como guía para generar alternativas y
verifica que estas no deterioren otras funciones importantes.

### Herramienta 4: aplicar principios inventivos cuando el problema lo requiera

Usa principios inventivos solo después de definir el problema y comprobar que
la necesidad requiere una búsqueda inventiva, por ejemplo, porque existe una
contradicción técnica o física que no se resuelve con una corrección directa.
No los apliques por rutina a problemas que se resuelven con acciones conocidas
o datos que aún faltan.

Una vez formulado el problema:

1. Resume la contradicción, el efecto que se quiere eliminar y la función que
   se debe mejorar.
2. Selecciona principios que puedan responder a ese conflicto; explica la
   relación entre cada principio y el problema, sin presentar la selección
   como garantía de éxito.
3. Usa los principios para generar más de una alternativa concreta.
4. Adapta las alternativas al sistema real y revisa restricciones, recursos,
   efectos secundarios y riesgos.
5. Define cómo probar la opción más prometedora y qué resultado observable
   indicaría si funciona.

Si no se puede explicar por qué un principio aplica al problema definido, vuelve
a revisar la formulación o elige otro método de solución.

## Procedimiento práctico

1. Delimita el sistema y redacta una necesidad medible.
2. Identifica función útil, efectos perjudiciales y recursos existentes.
3. Expresa el conflicto como una contradicción verificable.
4. Describe el resultado ideal y las restricciones que no pueden violarse.
5. Genera alternativas usando separación de requisitos y principios inventivos.
6. Compara las opciones por beneficio, costo, riesgo y evidencia disponible.
7. Prueba la alternativa de menor riesgo y mide si resuelve la contradicción.
8. Actualiza el modelo del problema con los resultados; no presentes una
   hipótesis como causa confirmada.

## Fuentes y alcance

Esta guía es una adaptación resumida y reescrita; no reproduce diapositivas ni
listas extensas de principios. Está destinada a orientar el análisis y no es
sustituto de formación especializada ni de la consulta de fuentes técnicas
primarias.
