"""
pipeline.py — Orquestador del pipeline de entrenamiento
-------------------------------------------------------
Ejecuta en orden todas las etapas, de los datos al modelo publicado:

    1. Ingesta        -> data/raw/pacientes_raw.csv
    2. Limpieza       -> data/processed/pacientes_limpios.csv
    3. EDA            -> reports/eda_resumen.txt + reports/figuras/
    4. Entrenamiento  -> comparación de modelos + evaluación + puertas de calidad
    5. Registro       -> models/modelo.joblib + models/metadatos.json

Uso (desde la raíz del proyecto):
    python -m src.pipeline

Este mismo comando sirve para RE-ENTRENAR cuando llegan datos nuevos.
Dentro de Docker se ejecuta automáticamente al construir la imagen.
"""

from src import eda, entrenamiento, ingesta, limpieza


def main():
    print("=" * 60)
    print("PIPELINE DE ENTRENAMIENTO")
    print("=" * 60)

    print("\n--- 1. Ingesta de datos ---")
    df_raw = ingesta.ejecutar()

    print("\n--- 2. Validación y limpieza ---")
    df_limpio = limpieza.ejecutar(df_raw)

    print("\n--- 3. Análisis exploratorio (EDA) ---")
    eda.ejecutar(df_raw, df_limpio)

    print("\n--- 4-5. Entrenamiento, validación y registro del modelo ---")
    entrenamiento.ejecutar(df_limpio)

    print("\nPipeline completado con éxito.")


if __name__ == "__main__":
    main()
