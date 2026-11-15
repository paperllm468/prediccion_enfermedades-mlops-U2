# Proceso de Machine Learning: qué se hizo en cada etapa

Este documento cuenta, etapa por etapa, **qué hace el código**, **por qué se tomó cada decisión** y **qué produce**. Cada etapa corresponde a un archivo de `src/` o `servicio/` y a una fase del pipeline de MLOps de la Parte 1.

```
 DATOS                      MODELO                                   PRODUCCIÓN
┌───────────┐  ┌──────────┐  ┌─────┐  ┌───────────────┐  ┌─────────┐  ┌───────────┐  ┌───────────┐
│1. Ingesta │─▶│2. Limpieza│─▶│3.EDA│─▶│4. Entrenam. + │─▶│5.Registro│─▶│6-7. Servicio│─▶│8. Monitoreo│
│           │  │           │  │     │  │ validación    │  │ modelo  │  │ predicción │  │  (logs)   │
└───────────┘  └──────────┘  └─────┘  └───────┬───────┘  └─────────┘  └───────────┘  └─────┬─────┘
      ▲                                       │ falla puerta de calidad                    │
      │                                       ▼                                            │
      │                                 no se publica                                      │
      └──────────────────────── re-entrenamiento con datos nuevos ◀────────────────────────┘
```

---

## Etapa 0: Configuración (`src/config.py`)

Todas las rutas, parámetros e hiperparámetros generales están en un solo archivo:
- **Semilla fija (42):** la misma ejecución produce los mismos datos y el mismo modelo (**reproducibilidad**).
- **Variables del modelo:** `temperatura`, `dias_sintomas`, `intensidad_dolor`; objetivo: `diagnostico`.
- **Rangos clínicos válidos:** se usan tanto en la limpieza como al validar la entrada del médico, así que la regla es la misma en entrenamiento y en producción.
- **Umbral de calidad:** F1 macro ≥ 0.80.
- **Casos de referencia:** 4 pacientes de ejemplo, uno por estado.

---

## Etapa 1: Ingesta de datos (`src/ingesta.py`)

**Objetivo:** obtener los registros de pacientes con su diagnóstico.

**Qué hace:**
1. Genera 5 000 pacientes sintéticos con `paciente_id`, `edad`, `sexo`, `temperatura`, `dias_sintomas` e `intensidad_dolor`.
2. Asigna el diagnóstico con `diagnostico_medico()`, que simula el criterio de un médico experto:

| Regla (en orden) | Diagnóstico |
|---|---|
| temperatura < 37.5 y dolor ≤ 2 | NO ENFERMO |
| días > 180 y dolor ≥ 8 | ENFERMEDAD TERMINAL |
| días > 30 | ENFERMEDAD CRÓNICA |
| temperatura ≥ 39 o dolor ≥ 7 | ENFERMEDAD AGUDA |
| resto | ENFERMEDAD LEVE |

3. **Ensucia los datos a propósito**, como pasa con los datos clínicos reales:

| Problema introducido | Proporción | Simula |
|---|---|---|
| Etiquetas con ruido | 3 % | Errores de diagnóstico o digitación |
| Temperatura en Fahrenheit | 1 % | Equipos o sistemas con otra unidad |
| Dolor = 15, días = −5 | 0.5 % c/u | Errores de captura |
| Valores faltantes | 2 % por variable | Campos no diligenciados |
| Sin diagnóstico | 0.5 % | Casos sin cerrar |
| Duplicados | 1 % | Cargas repetidas |

**Por qué se simula:** los datos de salud son sensibles y no hay un dataset real disponible para el ejercicio. El modelo **no conoce** las reglas: las debe **aprender** de los datos, como haría con datos reales. En producción, solo esta función cambiaría para leer de las historias clínicas.

**Salida:** `data/raw/pacientes_raw.csv` (~5 050 filas).

---

## Etapa 2: Validación y limpieza (`src/limpieza.py`)

| Paso | Acción | Justificación |
|---|---|---|
| 1 | Elimina duplicados por `paciente_id` | Un paciente repetido pesaría doble en el entrenamiento |
| 2 | Elimina registros sin diagnóstico o con un diagnóstico desconocido | Sin etiqueta no sirven para aprendizaje supervisado |
| 3 | Convierte temperaturas > 45 de °F a °C | Ningún humano vivo tiene > 45 °C: es un error de unidad recuperable |
| 4 | Elimina valores fuera de los rangos clínicos | Son errores no recuperables |
| 5 | **Conserva** los faltantes | Se imputan con la mediana **dentro del pipeline del modelo**, para que entrenamiento y producción traten los faltantes igual |

**Salidas:** `data/processed/pacientes_limpios.csv` y `reports/limpieza_reporte.json`, con cuántos registros se eliminaron o corrigieron en cada paso (trazabilidad).

---

## Etapa 3: Análisis exploratorio, EDA (`src/eda.py`)

**Qué analiza:**
1. Faltantes por variable en los datos crudos.
2. Estadística descriptiva (media, desviación, cuartiles).
3. **Distribución de clases:** muestra el desbalance. ENFERMEDAD CRÓNICA es la clase minoritaria, como lo serían las enfermedades huérfanas.
4. Promedio de cada variable por clase: qué variable separa cada estado (por ejemplo, `dias_sintomas` separa CRÓNICA).
5. Correlación entre variables.
6. Gráficas: barras de clases, histogramas por clase y diagramas de caja por clase.

**Conclusiones que guían el modelado:**
- Hay **desbalance**, por eso se usa `class_weight="balanced"`, la partición estratificada y la métrica **F1 macro** (no solo accuracy).
- Las fronteras entre clases son **umbrales** sobre las variables, lo que favorece a los modelos basados en árboles.
- Hay faltantes, así que se necesita imputación.

**Salidas:** `reports/eda_resumen.txt` y `reports/figuras/` (`distribucion_clases.png`, `histogramas_por_clase.png`, `boxplots_por_clase.png`).

---

## Etapa 4: Entrenamiento y validación (`src/entrenamiento.py`)

1. **Partición estratificada:** 80 % entrenamiento, 20 % prueba. El conjunto de prueba no se toca hasta la evaluación final.
2. **Modelos candidatos.** Cada uno es un `Pipeline` de scikit-learn con su imputador:

| Modelo | Rol |
|---|---|
| Regresión logística (+ escalado) | Línea base simple e interpretable |
| Árbol de decisión (profundidad 8) | Interpretable, captura umbrales |
| Random Forest (150 árboles) | Más robusto ante ruido en las etiquetas |

3. **Validación cruzada estratificada de 5 particiones** con F1 macro. Se elige el de mayor F1 promedio. Lo esperado es un modelo de árboles (Random Forest o árbol), porque la relación es no lineal y basada en umbrales.
4. Se re-entrena el mejor con todo el conjunto de entrenamiento y se **evalúa en prueba**: accuracy, F1 macro, precisión/recall/F1 por clase y matriz de confusión.
5. **Puertas de calidad** (`verificar_calidad`):
   - F1 macro en prueba ≥ 0.80.
   - **Prueba de humo:** los 5 casos de referencia deben dar su estado esperado. Esto garantiza que el servicio puede devolver **los 5 estados**.
   - Si algo falla, se lanza un error y el pipeline (y el `docker build`) se detiene: **un modelo malo nunca llega al médico**.

**Salida:** `reports/metricas.json`.

---

## Etapa 5: Registro del modelo (`src/entrenamiento.py → registrar_modelo`)

- `models/modelo.joblib`: el pipeline completo (imputador + modelo) serializado.
- `models/metadatos.json`: un **registro de modelos** simple con:
  - `version` (fecha-hora del entrenamiento)
  - algoritmo elegido y comparación de la validación cruzada
  - métricas en prueba
  - **hash MD5 de los datos** de entrenamiento (versionado de datos: se sabe con qué datos exactos se entrenó)
  - versión de scikit-learn y semilla

---

## Etapa 6: Predicción (`src/prediccion.py`)

1. `validar_entradas()`: convierte y verifica los 3 valores con los mismos rangos de la limpieza.
2. `cargar_modelo()`: carga el `.joblib` **una sola vez** (caché en memoria).
3. `predecir()`: devuelve el estado. `predecir_con_detalle()` agrega la probabilidad de cada estado y la versión del modelo.
4. `registrar_prediccion()`: escribe cada consulta en `logs/predicciones.csv` (etapa 8).

---

## Etapa 7: Despliegue (`servicio/app.py` + `Dockerfile`)

- **Flask** expone `/` (formulario web), `/predecir` (API JSON), `/modelo/info` y `/salud`.
- **Dockerfile multi-etapa:**
  - Etapa *entrenamiento*: ejecuta todo el pipeline.
  - Etapa *servicio*: imagen final sin datos ni librerías de EDA, solo con el modelo y el código de inferencia. Así queda más liviana y segura.
- **gunicorn** como servidor de producción, **usuario no-root** y **HEALTHCHECK**.
- **Versiones fijas** en `requirements.txt`, para que el modelo se cargue con la misma versión de scikit-learn con la que se entrenó.

---

## Etapa 8: Monitoreo y re-entrenamiento

- `logs/predicciones.csv` guarda fecha, entradas, estado predicho y versión del modelo. Sirve para:
  - **Data drift:** comparar la distribución de las entradas reales con la de entrenamiento (ver `reports/eda_resumen.txt`).
  - **Desempeño real:** cruzar con el diagnóstico confirmado por el médico.
- `/salud` + `HEALTHCHECK` vigilan la disponibilidad del servicio.
- **Re-entrenamiento:** al reconstruir la imagen (`docker compose up -d --build`) se repite el pipeline. En un caso real, la ingesta leería los nuevos casos confirmados. El modelo nuevo vuelve a pasar las puertas de calidad y recibe una nueva versión.

---

## Relación con los principios MLOps (Parte 1)

| Principio | Dónde se aplica en el código |
|---|---|
| Automatización | `src/pipeline.py` + `RUN python -m src.pipeline` en el Dockerfile |
| Reproducibilidad | Semilla fija, versiones fijas, configuración central |
| Versionado de datos y modelos | Hash de datos + versión en `metadatos.json` |
| Pruebas | Validación de datos, validación cruzada, puertas de calidad, prueba de humo |
| Despliegue | Docker multi-etapa, API REST, health check |
| Monitoreo | `logs/predicciones.csv`, `/salud`, `/modelo/info` |
| Entrenamiento continuo | Reconstruir la imagen re-ejecuta el pipeline completo |
