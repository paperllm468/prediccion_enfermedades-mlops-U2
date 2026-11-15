"""
eda.py — Etapa 3: Análisis exploratorio de datos (EDA)
------------------------------------------------------
Genera un resumen estadístico y gráficas para entender los datos antes de
entrenar:

    - Calidad de los datos crudos (faltantes por variable).
    - Estadística descriptiva de las variables.
    - Distribución de las clases (detecta el desbalance).
    - Promedio de cada variable por clase (qué variables separan las clases).
    - Correlación entre variables.
    - Histogramas y diagramas de caja por clase.

Salidas:
    reports/eda_resumen.txt
    reports/figuras/distribucion_clases.png
    reports/figuras/histogramas_por_clase.png
    reports/figuras/boxplots_por_clase.png
"""

import matplotlib

matplotlib.use("Agg")  # Backend sin pantalla (necesario dentro de Docker)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402


def resumen_texto(df_raw, df):
    """Construye el resumen estadístico en texto."""
    secciones = [
        "RESUMEN DEL ANÁLISIS EXPLORATORIO (EDA)",
        "=" * 60,
        f"Registros crudos: {len(df_raw)} | Registros limpios: {len(df)}",
        "",
        "1. Valores faltantes en los datos crudos:",
        df_raw[config.FEATURES + [config.OBJETIVO]].isna().sum().to_string(),
        "",
        "2. Estadística descriptiva (datos limpios):",
        df[config.FEATURES].describe().round(2).to_string(),
        "",
        "3. Distribución de clases:",
        df[config.OBJETIVO].value_counts().reindex(config.ESTADOS).to_string(),
        "",
        "   Proporción (%):",
        (df[config.OBJETIVO].value_counts(normalize=True).reindex(config.ESTADOS) * 100).round(1).to_string(),
        "",
        "4. Promedio de cada variable por clase:",
        df.groupby(config.OBJETIVO)[config.FEATURES].mean().reindex(config.ESTADOS).round(2).to_string(),
        "",
        "5. Correlación entre variables:",
        df[config.FEATURES].corr().round(2).to_string(),
    ]
    return "\n".join(secciones)


def grafica_distribucion_clases(df):
    conteo = df[config.OBJETIVO].value_counts().reindex(config.ESTADOS)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(conteo.index, conteo.values, color=["#16a34a", "#ca8a04", "#dc2626", "#7c3aed", "#374151"])
    ax.set_title("Distribución de clases")
    ax.set_ylabel("Número de pacientes")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "distribucion_clases.png", dpi=100)
    plt.close(fig)


def grafica_histogramas(df):
    fig, ejes = plt.subplots(1, len(config.FEATURES), figsize=(15, 4))
    for ax, columna in zip(ejes, config.FEATURES):
        for estado in config.ESTADOS:
            valores = df.loc[df[config.OBJETIVO] == estado, columna].dropna()
            ax.hist(valores, bins=30, alpha=0.5, label=estado)
        ax.set_title(columna)
    ejes[0].legend(fontsize=8)
    fig.suptitle("Distribución de cada variable por clase")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "histogramas_por_clase.png", dpi=100)
    plt.close(fig)


def grafica_boxplots(df):
    fig, ejes = plt.subplots(1, len(config.FEATURES), figsize=(15, 4))
    for ax, columna in zip(ejes, config.FEATURES):
        datos = [df.loc[df[config.OBJETIVO] == e, columna].dropna() for e in config.ESTADOS]
        ax.boxplot(datos)
        ax.set_xticks(range(1, len(config.ESTADOS) + 1))
        ax.set_xticklabels([e.replace("ENFERMEDAD ", "") for e in config.ESTADOS], fontsize=8)
        ax.set_title(columna)
    fig.suptitle("Diagramas de caja por clase")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "boxplots_por_clase.png", dpi=100)
    plt.close(fig)


def ejecutar(df_raw, df):
    """Ejecuta el EDA y guarda el resumen y las figuras."""
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    texto = resumen_texto(df_raw, df)
    with open(config.DIR_REPORTES / "eda_resumen.txt", "w", encoding="utf-8") as f:
        f.write(texto)

    grafica_distribucion_clases(df)
    grafica_histogramas(df)
    grafica_boxplots(df)
    print(f"[eda] Resumen y figuras guardados en {config.DIR_REPORTES}")


if __name__ == "__main__":
    ejecutar(pd.read_csv(config.DATOS_RAW), pd.read_csv(config.DATOS_PROCESADOS))
