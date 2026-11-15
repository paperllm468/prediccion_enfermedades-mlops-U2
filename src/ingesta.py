"""
ingesta.py — Etapa 1: Ingesta de datos
--------------------------------------
En un proyecto real, esta etapa extraería los registros de pacientes desde las
historias clínicas electrónicas (EHR), los laboratorios y los registros de
enfermedades raras.

Como no se dispone de datos reales (son datos sensibles), aquí se SIMULA esa
fuente: se generan pacientes sintéticos y su diagnóstico lo asigna una función
que imita el criterio de un médico experto. Además se introducen, a propósito,
los problemas típicos de los datos clínicos reales (faltantes, duplicados,
errores de unidades, valores imposibles y etiquetas con ruido) para que la
etapa de limpieza tenga sentido.

Salida: data/raw/pacientes_raw.csv
"""

import numpy as np
import pandas as pd

from src import config


def diagnostico_medico(temperatura, dias_sintomas, intensidad_dolor):
    """Simula el diagnóstico de un médico experto (la "etiqueta verdadera").

    Reglas clínicas usadas para etiquetar los datos de entrenamiento:
        1. Sin fiebre (< 37.5 °C) y dolor mínimo (<= 2)       -> NO ENFERMO
        2. Síntomas por más de 30 días                         -> ENFERMEDAD CRÓNICA
        3. Fiebre alta (>= 39 °C) o dolor intenso (>= 7)       -> ENFERMEDAD AGUDA
        4. Cualquier otro caso                                 -> ENFERMEDAD LEVE

    El modelo de ML NO conoce estas reglas: debe aprenderlas a partir de los datos.
    """
    if temperatura < 37.5 and intensidad_dolor <= 2:
        return config.NO_ENFERMO
    if dias_sintomas > 30:
        return config.ENFERMEDAD_CRONICA
    if temperatura >= 39.0 or intensidad_dolor >= 7:
        return config.ENFERMEDAD_AGUDA
    return config.ENFERMEDAD_LEVE


def _generar_pacientes(n, rng):
    """Genera n pacientes "limpios" con su diagnóstico."""
    temperatura = np.clip(rng.normal(37.4, 1.0, n), 35.0, 42.0).round(1)

    # ~15 % de pacientes con síntomas prolongados (casos crónicos: clase minoritaria)
    es_prolongado = rng.random(n) < 0.15
    dias_sintomas = np.where(es_prolongado, rng.integers(31, 366, n), rng.integers(0, 31, n))

    intensidad_dolor = rng.integers(0, 11, n)
    edad = rng.integers(0, 96, n)
    sexo = rng.choice(["F", "M"], n)

    diagnostico = [
        diagnostico_medico(t, d, p)
        for t, d, p in zip(temperatura, dias_sintomas, intensidad_dolor)
    ]

    return pd.DataFrame({
        "paciente_id": [f"P{i:05d}" for i in range(n)],
        "edad": edad,
        "sexo": sexo,
        "temperatura": temperatura,
        "dias_sintomas": dias_sintomas,
        "intensidad_dolor": intensidad_dolor,
        config.OBJETIVO: diagnostico,
    })


def _ensuciar(df, rng):
    """Introduce problemas de calidad típicos de los datos clínicos reales."""
    n = len(df)
    # Columnas numéricas como float para poder contener valores faltantes (NaN)
    df[config.FEATURES] = df[config.FEATURES].astype(float)

    def indices(proporcion):
        return rng.choice(n, size=int(n * proporcion), replace=False)

    # 1. Ruido en las etiquetas (3 %): errores de diagnóstico o de digitación
    idx = indices(0.03)
    df.loc[idx, config.OBJETIVO] = rng.choice(config.ESTADOS, size=len(idx))

    # 2. Temperatura registrada en Fahrenheit en vez de Celsius (1 %)
    idx = indices(0.01)
    df.loc[idx, "temperatura"] = (df.loc[idx, "temperatura"] * 9 / 5 + 32).round(1)

    # 3. Valores imposibles (0.5 % cada uno): dolor fuera de escala y días negativos
    df.loc[indices(0.005), "intensidad_dolor"] = 15
    df.loc[indices(0.005), "dias_sintomas"] = -5

    # 4. Valores faltantes (2 % en cada variable)
    for columna in config.FEATURES:
        df.loc[indices(0.02), columna] = np.nan

    # 5. Pacientes sin diagnóstico registrado (0.5 %)
    df.loc[indices(0.005), config.OBJETIVO] = None

    # 6. Registros duplicados (1 %), por ejemplo por cargas repetidas
    duplicados = df.sample(frac=0.01, random_state=config.SEMILLA)
    df = pd.concat([df, duplicados], ignore_index=True)

    # Mezclar el orden de las filas
    return df.sample(frac=1, random_state=config.SEMILLA).reset_index(drop=True)


def generar_datos(n=config.N_PACIENTES, semilla=config.SEMILLA):
    """Genera el dataset crudo simulado."""
    rng = np.random.default_rng(semilla)
    df = _generar_pacientes(n, rng)
    return _ensuciar(df, rng)


def ejecutar():
    """Ejecuta la etapa de ingesta y guarda los datos crudos."""
    df = generar_datos()
    config.DATOS_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATOS_RAW, index=False)
    print(f"[ingesta] {len(df)} registros guardados en {config.DATOS_RAW}")
    return df


if __name__ == "__main__":
    ejecutar()
