"""
entrenamiento.py — Etapas 4 y 5: Entrenamiento, validación y registro del modelo
--------------------------------------------------------------------------------
    1. Separa los datos en entrenamiento (80 %) y prueba (20 %), estratificado
       por clase para conservar la proporción de cada estado.
    2. Compara 3 modelos candidatos con validación cruzada estratificada de 5
       particiones, usando F1 macro (pondera igual todas las clases, incluida
       la minoritaria):
           - Regresión logística (línea base, interpretable)
           - Árbol de decisión
           - Random Forest
       Cada candidato es un Pipeline de scikit-learn que incluye la imputación
       de faltantes, así el preprocesamiento viaja junto con el modelo.
    3. Re-entrena el mejor modelo y lo evalúa en el conjunto de prueba.
    4. Puertas de calidad (si alguna falla, el modelo NO se publica):
           a) F1 macro en prueba >= config.UMBRAL_F1_MINIMO
           b) Prueba de humo: los casos de referencia producen los 4 estados.
    5. Registra el modelo: models/modelo.joblib + models/metadatos.json
       (versión, fecha, métricas, hash de los datos, versión de scikit-learn).

Salidas: models/modelo.joblib, models/metadatos.json, reports/metricas.json
"""

import hashlib
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config


def modelos_candidatos():
    """Define los modelos a comparar. Todos manejan el desbalance con class_weight."""
    return {
        "regresion_logistica": Pipeline([
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
            ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "arbol_decision": Pipeline([
            ("imputador", SimpleImputer(strategy="median")),
            ("modelo", DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                              random_state=config.SEMILLA)),
        ]),
        "random_forest": Pipeline([
            ("imputador", SimpleImputer(strategy="median")),
            ("modelo", RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced",
                                              random_state=config.SEMILLA, n_jobs=-1)),
        ]),
    }


def _hash_archivo(ruta):
    """Huella del dataset usado: permite saber con qué datos se entrenó cada versión."""
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def _a_json(valor):
    """Convierte tipos de numpy a tipos nativos de Python para guardarlos en JSON."""
    return valor.item() if hasattr(valor, "item") else str(valor)


def comparar_modelos(X_train, y_train):
    """Validación cruzada de cada candidato. Retorna (nombre_del_mejor, resultados)."""
    cv = StratifiedKFold(n_splits=config.FOLDS_CV, shuffle=True, random_state=config.SEMILLA)
    resultados = {}
    for nombre, modelo in modelos_candidatos().items():
        puntajes = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1_macro")
        resultados[nombre] = {
            "f1_macro_media": round(float(puntajes.mean()), 4),
            "f1_macro_std": round(float(puntajes.std()), 4),
        }
        print(f"[entrenamiento] CV {nombre}: F1 macro = {puntajes.mean():.4f} ± {puntajes.std():.4f}")
    mejor = max(resultados, key=lambda k: resultados[k]["f1_macro_media"])
    return mejor, resultados


def evaluar(modelo, X_test, y_test):
    """Métricas del modelo final sobre el conjunto de prueba (datos nunca vistos)."""
    y_pred = modelo.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "reporte_por_clase": classification_report(y_test, y_pred, labels=config.ESTADOS,
                                                   output_dict=True, zero_division=0),
        "matriz_confusion": {
            "etiquetas": config.ESTADOS,
            "valores": confusion_matrix(y_test, y_pred, labels=config.ESTADOS).tolist(),
        },
    }


def verificar_calidad(modelo, metricas):
    """Puertas de calidad. Lanza un error si el modelo no es apto para producción."""
    if metricas["f1_macro"] < config.UMBRAL_F1_MINIMO:
        raise RuntimeError(
            f"Modelo rechazado: F1 macro {metricas['f1_macro']} < umbral {config.UMBRAL_F1_MINIMO}"
        )

    for entrada, esperado in config.CASOS_REFERENCIA:
        X = pd.DataFrame([entrada], columns=config.FEATURES)
        obtenido = str(modelo.predict(X)[0])
        if obtenido != esperado:
            raise RuntimeError(f"Prueba de humo fallida: {entrada} -> {obtenido} (esperado {esperado})")
    print("[entrenamiento] Puertas de calidad superadas (F1 mínimo + casos de referencia).")


def registrar_modelo(modelo, nombre, metricas, resultados_cv, n_train, n_test):
    """Guarda el modelo y sus metadatos (registro de modelos simple)."""
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, config.RUTA_MODELO)

    ahora = datetime.now(timezone.utc)
    metadatos = {
        "version": ahora.strftime("%Y%m%d%H%M%S"),
        "fecha_entrenamiento": ahora.isoformat(),
        "modelo": nombre,
        "features": config.FEATURES,
        "clases": [str(c) for c in modelo.classes_],
        "hash_datos_entrenamiento": _hash_archivo(config.DATOS_PROCESADOS),
        "n_entrenamiento": n_train,
        "n_prueba": n_test,
        "metricas_prueba": {"accuracy": metricas["accuracy"], "f1_macro": metricas["f1_macro"]},
        "comparacion_validacion_cruzada": resultados_cv,
        "version_sklearn": sklearn.__version__,
        "semilla": config.SEMILLA,
    }
    with open(config.RUTA_METADATOS, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=2, ensure_ascii=False, default=_a_json)

    config.DIR_REPORTES.mkdir(parents=True, exist_ok=True)
    with open(config.DIR_REPORTES / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False, default=_a_json)

    print(f"[entrenamiento] Modelo '{nombre}' v{metadatos['version']} guardado en {config.RUTA_MODELO}")


def ejecutar(df):
    """Ejecuta entrenamiento, evaluación, puertas de calidad y registro."""
    X = df[config.FEATURES]
    y = df[config.OBJETIVO]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TAMANO_TEST, stratify=y, random_state=config.SEMILLA
    )

    mejor, resultados_cv = comparar_modelos(X_train, y_train)
    print(f"[entrenamiento] Mejor modelo en validación cruzada: {mejor}")

    modelo = modelos_candidatos()[mejor]
    modelo.fit(X_train, y_train)

    metricas = evaluar(modelo, X_test, y_test)
    print(f"[entrenamiento] Prueba: accuracy={metricas['accuracy']} | F1 macro={metricas['f1_macro']}")

    verificar_calidad(modelo, metricas)
    registrar_modelo(modelo, mejor, metricas, resultados_cv, len(X_train), len(X_test))
    return modelo


if __name__ == "__main__":
    ejecutar(pd.read_csv(config.DATOS_PROCESADOS))
