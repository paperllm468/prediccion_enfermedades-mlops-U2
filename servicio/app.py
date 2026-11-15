"""
app.py — Etapa 7: Despliegue del modelo como servicio web (Flask)
-----------------------------------------------------------------
Rutas disponibles:

    GET/POST  /              Página web con formulario para el médico
    POST      /predecir      API JSON: recibe los 3 valores y retorna el estado
    GET       /modelo/info   Metadatos del modelo desplegado (versión, métricas)
    GET       /estadisticas  Reporte JSON: total por categoría, últimas 5, fecha de la última
    GET       /reporte       El mismo reporte en una página web para el médico
    GET       /salud         Health check (lo usa Docker)

Uso local (desde la raíz del proyecto, después de entrenar):
    python -m servicio.app
En Docker se ejecuta con gunicorn (ver Dockerfile).
"""

from flask import Flask, jsonify, render_template, request

from src import config
from src.estadisticas import generar_reporte
from src.prediccion import DatosInvalidosError, cargar_metadatos, cargar_modelo, predecir_con_detalle

app = Flask(__name__)
app.json.ensure_ascii = False  # Devolver tildes (ej. "CRÓNICA") sin escapar
app.json.sort_keys = False     # Respetar el orden de los estados en las respuestas


@app.route("/", methods=["GET", "POST"])
def inicio():
    """Formulario web. Con POST procesa los datos y muestra el resultado."""
    detalle, error, valores = None, None, {}

    if request.method == "POST":
        valores = {campo: request.form.get(campo, "") for campo in config.FEATURES}
        try:
            detalle = predecir_con_detalle(**valores)
        except DatosInvalidosError as e:
            error = str(e)

    return render_template("index.html", detalle=detalle, error=error,
                           valores=valores, rangos=config.RANGOS)


@app.route("/predecir", methods=["POST"])
def predecir_api():
    """API JSON.

    Petición:  {"temperatura": 39.5, "dias_sintomas": 2, "intensidad_dolor": 8}
    Respuesta: {"estado": "ENFERMEDAD AGUDA", "probabilidades": {...},
                "entrada": {...}, "version_modelo": "..."}
    Error 400: {"error": "..."}
    """
    datos = request.get_json(silent=True)
    if not isinstance(datos, dict):
        return jsonify(error="El cuerpo de la petición debe ser un JSON válido."), 400

    faltantes = [c for c in config.FEATURES if c not in datos]
    if faltantes:
        return jsonify(error=f"Faltan los campos: {', '.join(faltantes)}."), 400

    try:
        return jsonify(predecir_con_detalle(**{c: datos[c] for c in config.FEATURES}))
    except DatosInvalidosError as e:
        return jsonify(error=str(e)), 400


@app.route("/modelo/info", methods=["GET"])
def info_modelo():
    """Información del modelo en producción (trazabilidad / versionado)."""
    return jsonify(cargar_metadatos())


@app.route("/estadisticas", methods=["GET"])
def estadisticas():
    """Estadísticas de las predicciones realizadas (leídas de logs/predicciones.csv).

    Respuesta: {"total_predicciones": 7,
                "total_por_categoria": {"NO ENFERMO": 2, ...},
                "ultimas_predicciones": [... 5 más recientes, la última primero ...],
                "fecha_ultima_prediccion": "2026-11-15T01:10:00+00:00"}
    """
    return jsonify(generar_reporte())


@app.route("/reporte", methods=["GET"])
def reporte():
    """Página web con las estadísticas de las predicciones."""
    return render_template("reporte.html", reporte=generar_reporte())


@app.route("/salud", methods=["GET"])
def salud():
    """Health check: verifica que el modelo esté cargado."""
    try:
        cargar_modelo()
    except FileNotFoundError as e:
        return jsonify(estado="error", detalle=str(e)), 503
    return jsonify(estado="ok", version_modelo=cargar_metadatos().get("version"),
                   estados_posibles=config.ESTADOS)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
