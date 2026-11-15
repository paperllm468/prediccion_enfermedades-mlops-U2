"""
estadisticas.py — Reporte de las predicciones realizadas
--------------------------------------------------------
Lee el log de monitoreo (logs/predicciones.csv), donde se exporta cada
predicción hecha por el servicio, y construye el reporte que piden los médicos:

    - Número total de predicciones por cada categoría (los 5 estados).
    - Últimas 5 predicciones realizadas (la más reciente primero).
    - Fecha de la última predicción.

Si todavía no se ha hecho ninguna predicción, el reporte sale con los valores
por defecto: conteos en 0, lista vacía y fecha None.

Uso directo (desde la raíz del proyecto):
    python -m src.estadisticas
"""

import csv
import json
from collections import Counter

from src import config

N_ULTIMAS = 5


def leer_predicciones(ruta=None):
    """Retorna la lista de predicciones registradas (una por fila del log)."""
    ruta = ruta or config.RUTA_LOG_PREDICCIONES
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _formatear(fila):
    """Convierte una fila del CSV (todo texto) a un diccionario con tipos correctos."""
    return {
        "fecha_utc": fila["fecha_utc"],
        "temperatura": float(fila["temperatura"]),
        "dias_sintomas": int(fila["dias_sintomas"]),
        "intensidad_dolor": int(fila["intensidad_dolor"]),
        "estado": fila["estado_predicho"],
        "version_modelo": fila.get("version_modelo") or None,
    }


def generar_reporte(ruta=None):
    """Construye el reporte de estadísticas de las predicciones realizadas."""
    filas = leer_predicciones(ruta)
    conteo = Counter(f["estado_predicho"] for f in filas)

    return {
        "total_predicciones": len(filas),
        "total_por_categoria": {estado: conteo.get(estado, 0) for estado in config.ESTADOS},
        "ultimas_predicciones": [_formatear(f) for f in reversed(filas[-N_ULTIMAS:])],
        "fecha_ultima_prediccion": filas[-1]["fecha_utc"] if filas else None,
    }


if __name__ == "__main__":
    print(json.dumps(generar_reporte(), indent=2, ensure_ascii=False))
