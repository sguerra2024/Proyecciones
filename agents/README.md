# Agente Especializado: Analista de Datos y Mercados

## Descripción

**Nombre:** Analista de Datos y Mercados  
**Modelo:** Claude Sonnet 4.6  
**Propósito:** Análisis especializado en rentabilidad, mercados, gestión de cambios y errores de estimados

## Capacidades

### 1. **Análisis de Cambios del Cliente**
- Cálculo del costo real de cambios solicitados
- Presentación de opciones alternativas con impacto económico
- Impacto en beneficios y plazos

### 2. **Análisis de Errores en Estimados**
- Cálculo de pérdida financiera por error
- Identificación de causas raíz
- Detección de patrones recurrentes
- Medidas preventivas

### 3. **Investigación de Mercado**
- Análisis de precios y elasticidad de demanda
- Tendencias emergentes
- Segmentación de mercados
- Oportunidades no explotadas

### 4. **Optimización de Beneficios**
- Análisis de rentabilidad y KPIs
- Recomendaciones priorizadas por impacto
- Proyecciones de flujo de caja

## Uso en la Aplicación Streamlit

### Desde la UI
1. Abre la app ProyAst.py
2. Ubica la sección **"Analista de Datos y Mercados"** en el panel derecho
3. Selecciona el tipo de análisis requerido:
   - **Cambio del Cliente:** Analiza impacto de solicitudes de cambio
   - **Error en Estimado:** Evalúa errores de proyección
   - **Investigación de Mercado:** Analiza productos y regiones
   - **Optimizar Beneficios:** Identifica oportunidades de mejora

### Uso Programático

```python
from agent_manager import AgenteAnalistasMercados

# Inicializar agente
agente = AgenteAnalistasMercados()

# Análisis de cambio
respuesta = agente.analizar_cambio_cliente(
    "Aumentar volumen en 20%",
    impacto_estimado={'costo': 5000, 'tiempo_semanas': 2}
)

# Análisis de error
respuesta = agente.analizar_error_estimado(
    "Proyección de producción fue 15% menor",
    desviacion=-15.0
)

# Investigación de mercado
respuesta = agente.investigar_mercado(
    "Flores de corte premium",
    region="Región Andina"
)

# Optimización
datos = {
    'ingresos_mensuales': 50000,
    'costos_fijos': 15000,
    'costos_variables': 20000
}
respuesta = agente.optimizar_beneficios(str(datos))
```

## Configuración

### Variables de Entorno Requeridas

```env
ANTHROPIC_API_KEY=tu_clave_aqui
ANTHROPIC_MODEL=claude-sonnet-4-6  # Opcional
```

## Archivos

- **analista_mercados.yaml** - Definición YAML del agente
- **agent_manager.py** - Clase Python `AgenteAnalistasMercados` para integración

## Características Clave

✓ Respuestas cuantificadas financieramente  
✓ Análisis de impacto comparativo  
✓ Recomendaciones priorizadas por ROI  
✓ Detección de patrones y tendencias  
✓ Integración directa con Streamlit  
✓ Manejo robusto de errores  

## Ejemplo de Salida

### Análisis de Cambio del Cliente
```
IMPACTO FINANCIERO:
- Costo directo: $5,000
- Impacto en tiempo: +2 semanas
- Pérdida de oportunidades: $2,000
- Costo total estimado: $7,000

OPCIONES ALTERNATIVAS:
1. Implementación gradual: Reduce costo a $3,500, extiende 6 semanas
2. Tercerizar: Costo $4,000, mantiene plazo

RECOMENDACIÓN:
Opción 2 (tercerizar) maximiza beneficios con menor exposición al riesgo.
```

## Soporte

Para reportar problemas o sugerencias:
1. Verifica que `ANTHROPIC_API_KEY` esté configurada
2. Confirma que la librería `anthropic` esté instalada
3. Revisa los logs en la terminal de Streamlit
