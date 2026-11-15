"""
prediccion.py — Etapa 6: Predicción (inferencia en producción)
--------------------------------------------------------------
Carga el modelo registrado (models/modelo.joblib) y expone la función que usa
el servicio web para responder al médico.

Función principal:
    predecir(temperatura, dias_sintomas, intensidad_dolor) -> str

Retorna uno de los 5 estados:
    NO ENFERMO | ENFERMEDAD LEVE | ENFERMEDAD AGUDA | ENFERMEDAD CRÓNICA | ENFERMEDAD TERMINAL

Cada predicción se registra en logs/predicciones.csv. Ese registro es la base
del MONITOREO: permite detectar cambios en los datos de entrada (data drift) y,
cuando se confirme el diagnóstico real, medir el desempeño en producción y
usar esos casos para re-entrenar.
"""

import csv
import json
from datetime import datetime, timezone
from functools import lru_cache

import joblib
import pandas as pd

from src import config


class DatosInvalidosError(ValueError):
    """Se lanza cuando alguno de los valores de entrada no es válido."""


def validar_entradas(temperatura, dias_sintomas, intensidad_dolor):
    """Convierte los valores al tipo correcto y verifica los rangos clínicos.

    Retorna un diccionario {variable: valor}. Lanza DatosInvalidosError si
    algún valor no es numérico o está fuera de rango.
    """
    try:
        entrada = {
            "temperatura": float(temperatura),
            "dias_sintomas": int(float(dias_sintomas)),
            "intensidad_dolor": int(float(intensidad_dolor)),
        }
    except (TypeError, ValueError, OverflowError):
        raise DatosInvalidosError(
            "Todos los valores deben ser numéricos: temperatura, dias_sintomas e intensidad_dolor."
        )

    for nombre, valor in entrada.items():
        minimo, maximo = config.RANGOS[nombre]
        if not (minimo <= valor <= maximo):
            raise DatosInvalidosError(
                f"El valor de '{nombre}' ({valor}) está fuera del rango permitido [{minimo}, {maximo}]."
            )
    return entrada


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el modelo una sola vez y lo mantiene en memoria."""
    if not config.RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No existe el modelo en {config.RUTA_MODELO}. Ejecute primero: python -m src.pipeline"
        )
    return joblib.load(config.RUTA_MODELO)


@lru_cache(maxsize=1)
def cargar_metadatos():
    """Retorna los metadatos del modelo (versión, métricas, fecha...)."""
    if not config.RUTA_METADATOS.exists():
        return {}
    with open(config.RUTA_METADATOS, encoding="utf-8") as f:
        return json.load(f)


def _a_dataframe(entrada):
    return pd.DataFrame([entrada], columns=config.FEATURES)


def predecir(temperatura, dias_sintomas, intensidad_dolor):
    """Retorna el estado del paciente predicho por el modelo."""
    entrada = validar_entradas(temperatura, dias_sintomas, intensidad_dolor)
    return str(cargar_modelo().predict(_a_dataframe(entrada))[0])


def predecir_con_detalle(temperatura, dias_sintomas, intensidad_dolor):
    """Predicción + probabilidad de cada estado + versión del modelo.

    También registra la consulta en el log de monitoreo.
    """
    entrada = validar_entradas(temperatura, dias_sintomas, intensidad_dolor)
    modelo = cargar_modelo()
    X = _a_dataframe(entrada)

    estado = str(modelo.predict(X)[0])
    probabilidades = {
        str(clase): round(float(p), 3)
        for clase, p in zip(modelo.classes_, modelo.predict_proba(X)[0])
    }
    registrar_prediccion(entrada, estado)

    return {
        "estado": estado,
        "probabilidades": probabilidades,
        "entrada": entrada,
        "version_modelo": cargar_metadatos().get("version"),
    }


def registrar_prediccion(entrada, estado):
    """Agrega la consulta a logs/predicciones.csv (insumo del monitoreo).

    Si no se puede escribir el log, la predicción no se interrumpe.
    """
    try:
        config.DIR_LOGS.mkdir(parents=True, exist_ok=True)
        es_nuevo = not config.RUTA_LOG_PREDICCIONES.exists()
        with open(config.RUTA_LOG_PREDICCIONES, "a", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            if es_nuevo:
                escritor.writerow(["fecha_utc", *config.FEATURES, "estado_predicho", "version_modelo"])
            escritor.writerow([
                datetime.now(timezone.utc).isoformat(),
                *(entrada[c] for c in config.FEATURES),
                estado,
                cargar_metadatos().get("version"),
            ])
    except OSError:
        pass


if __name__ == "__main__":
    # Uso directo: python -m src.prediccion  -> predice los casos de referencia
    for entrada, esperado in config.CASOS_REFERENCIA:
        print(f"{entrada} -> {predecir(**entrada)} (esperado: {esperado})")
