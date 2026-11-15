# Verificación de la solución

Cómo comprobar que la solución funciona y qué resultado se espera en cada caso.

---

## 1. Durante la construcción de la imagen

Al ejecutar `docker build -t modelo-medico .` la consola debe mostrar el pipeline en orden:

| # | Mensaje esperado | Qué confirma |
|---|---|---|
| 1 | `[ingesta] ~5050 registros guardados...` | Los datos se obtuvieron |
| 2 | `[limpieza] ~5050 -> ~4900 registros. Detalle: {...}` | Se eliminaron duplicados, registros sin diagnóstico y valores imposibles |
| 3 | `[eda] Resumen y figuras guardados...` | El EDA se generó |
| 4 | `[entrenamiento] CV regresion_logistica / arbol_decision / random_forest: F1 macro = ...` | Se compararon 3 modelos con validación cruzada |
| 5 | `[entrenamiento] Mejor modelo en validación cruzada: ...` | Selección automática del modelo |
| 6 | `[entrenamiento] Prueba: accuracy=... F1 macro=...` | Evaluación en datos no vistos |
| 7 | `[entrenamiento] Puertas de calidad superadas...` | F1 ≥ 0.80 y los 4 casos de referencia correctos |
| 8 | `Pipeline completado con éxito.` | El modelo quedó registrado |

> Si el modelo no pasa las puertas de calidad, aparece `RuntimeError: Modelo rechazado...` o `Prueba de humo fallida...` y la imagen **no se construye**. Ese es el comportamiento deseado.

---

## 2. Con el contenedor corriendo

| # | Prueba | Cómo | Resultado esperado |
|---|---|---|---|
| 9 | Contenedor activo | `docker ps` | Estado `Up ... (healthy)` |
| 10 | Salud | `curl http://localhost:5000/salud` | `{"estado": "ok", "version_modelo": "...", ...}` |
| 11 | Info del modelo | `curl http://localhost:5000/modelo/info` | Algoritmo, métricas, versión, hash de datos |
| 12 | Página web | Abrir `http://localhost:5000` | Formulario con 3 campos y botón *Predecir* |

---

## 3. Predicciones: los 4 estados

Plantilla de petición (cambiar los valores):

```bash
curl -X POST http://localhost:5000/predecir -H "Content-Type: application/json" -d "{\"temperatura\": 36.5, \"dias_sintomas\": 0, \"intensidad_dolor\": 0}"
```

| # | temperatura | dias_sintomas | intensidad_dolor | Resultado esperado |
|---|---|---|---|---|
| 13 | 36.5 | 0 | 0 | **NO ENFERMO** |
| 14 | 37.8 | 3 | 4 | **ENFERMEDAD LEVE** |
| 15 | 39.5 | 2 | 8 | **ENFERMEDAD AGUDA** |
| 16 | 37.6 | 90 | 5 | **ENFERMEDAD CRÓNICA** |

Estos casos están garantizados: el pipeline los verifica antes de publicar el modelo.

Otros ejemplos, lejos de las fronteras entre clases:

| # | temperatura | dias_sintomas | intensidad_dolor | Resultado esperado |
|---|---|---|---|---|
| 17 | 36.8 | 1 | 1 | NO ENFERMO |
| 18 | 38.2 | 5 | 5 | ENFERMEDAD LEVE |
| 19 | 40.5 | 3 | 9 | ENFERMEDAD AGUDA |
| 20 | 38.0 | 200 | 5 | ENFERMEDAD CRÓNICA |

> Como es un modelo de ML (no reglas fijas), **cerca de las fronteras** (por ejemplo, temperatura 37.5 o 30–31 días) puede equivocarse ocasionalmente. Es normal y se refleja en las métricas de `reports/metricas.json`.

---

## 4. Validación de entradas (errores)

| # | Entrada enviada a `/predecir` | Resultado esperado |
|---|---|---|
| 21 | `{"temperatura": 50, "dias_sintomas": 2, "intensidad_dolor": 3}` | 400: `temperatura` fuera de rango [34.0, 43.0] |
| 22 | `{"temperatura": 37, "dias_sintomas": -1, "intensidad_dolor": 3}` | 400: `dias_sintomas` fuera de rango [0, 3650] |
| 23 | `{"temperatura": 37, "dias_sintomas": 2, "intensidad_dolor": 11}` | 400: `intensidad_dolor` fuera de rango [0, 10] |
| 24 | `{"temperatura": "alta", "dias_sintomas": 2, "intensidad_dolor": 3}` | 400: los valores deben ser numéricos |
| 25 | `{"temperatura": 37, "dias_sintomas": 2}` | 400: falta el campo `intensidad_dolor` |
| 26 | Cuerpo vacío o que no es JSON | 400: el cuerpo debe ser un JSON válido |

En la página web, el navegador bloquea primero los valores fuera de rango (atributos `min`/`max`).

---

## 5. Monitoreo

| # | Prueba | Cómo | Resultado esperado |
|---|---|---|---|
| 27 | Log de predicciones | Hacer varias consultas y luego `docker exec modelo-medico cat logs/predicciones.csv` | Una fila por consulta: fecha, 3 entradas, estado y versión del modelo |
| 28 | Reportes del entrenamiento | `docker exec modelo-medico cat reports/metricas.json` | Accuracy, F1 macro, reporte por clase y matriz de confusión |
