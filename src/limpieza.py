"""
limpieza.py — Etapa 2: Validación y limpieza de datos
-----------------------------------------------------
Corrige o descarta los problemas de calidad de los datos crudos:

    1. Elimina registros duplicados (mismo paciente_id).
    2. Elimina pacientes sin diagnóstico o con un diagnóstico desconocido.
    3. Corrige temperaturas registradas en Fahrenheit (valores > 45 °C).
    4. Elimina valores imposibles (fuera de los rangos clínicos de config.RANGOS).
    5. Los valores FALTANTES se conservan: se imputan dentro del pipeline del
       modelo (SimpleImputer). Así el mismo tratamiento se aplica en el
       entrenamiento y en producción, cuando el médico consulta el modelo.

Entrada: DataFrame crudo (data/raw/pacientes_raw.csv)
Salida:  data/processed/pacientes_limpios.csv y reports/limpieza_reporte.json
"""

import json

import pandas as pd

from src import config


def limpiar(df):
    """Aplica las reglas de limpieza y retorna (df_limpio, reporte)."""
    reporte = {"filas_iniciales": int(len(df))}

    # 1. Duplicados
    antes = len(df)
    df = df.drop_duplicates(subset="paciente_id").copy()
    reporte["duplicados_eliminados"] = int(antes - len(df))

    # 2. Etiquetas faltantes o inválidas
    antes = len(df)
    df = df[df[config.OBJETIVO].isin(config.ESTADOS)].copy()
    reporte["sin_diagnostico_eliminados"] = int(antes - len(df))

    # 3. Fahrenheit -> Celsius
    es_fahrenheit = df["temperatura"] > 45
    df.loc[es_fahrenheit, "temperatura"] = ((df.loc[es_fahrenheit, "temperatura"] - 32) * 5 / 9).round(1)
    reporte["temperaturas_convertidas_de_fahrenheit"] = int(es_fahrenheit.sum())

    # 4. Valores fuera de rango (los NaN se conservan para imputarlos después)
    antes = len(df)
    valido = pd.Series(True, index=df.index)
    for columna, (minimo, maximo) in config.RANGOS.items():
        valido &= df[columna].isna() | df[columna].between(minimo, maximo)
    df = df[valido].reset_index(drop=True)
    reporte["fuera_de_rango_eliminados"] = int(antes - len(df))

    # 5. Resumen de faltantes que quedan para imputar
    reporte["faltantes_a_imputar"] = {c: int(df[c].isna().sum()) for c in config.FEATURES}
    reporte["filas_finales"] = int(len(df))
    return df, reporte


def ejecutar(df_raw):
    """Ejecuta la etapa de limpieza y guarda los resultados."""
    df, reporte = limpiar(df_raw)

    config.DATOS_PROCESADOS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATOS_PROCESADOS, index=False)

    config.DIR_REPORTES.mkdir(parents=True, exist_ok=True)
    with open(config.DIR_REPORTES / "limpieza_reporte.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"[limpieza] {reporte['filas_iniciales']} -> {reporte['filas_finales']} registros. Detalle: {reporte}")
    return df


if __name__ == "__main__":
    ejecutar(pd.read_csv(config.DATOS_RAW))
