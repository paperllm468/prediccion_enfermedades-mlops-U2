"""
config.py
---------
Configuración central del proyecto: rutas, parámetros, variables del modelo y
umbrales de calidad. Todas las etapas leen de aquí, para que el pipeline sea
reproducible y fácil de modificar en un solo lugar.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas (relativas a la raíz del proyecto)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent

DATOS_RAW = RAIZ / "data" / "raw" / "pacientes_raw.csv"
DATOS_PROCESADOS = RAIZ / "data" / "processed" / "pacientes_limpios.csv"

DIR_MODELOS = RAIZ / "models"
RUTA_MODELO = DIR_MODELOS / "modelo.joblib"
RUTA_METADATOS = DIR_MODELOS / "metadatos.json"

DIR_REPORTES = RAIZ / "reports"
DIR_FIGURAS = DIR_REPORTES / "figuras"

DIR_LOGS = RAIZ / "logs"
RUTA_LOG_PREDICCIONES = DIR_LOGS / "predicciones.csv"

# ---------------------------------------------------------------------------
# Parámetros generales
# ---------------------------------------------------------------------------
SEMILLA = 42            # Reproducibilidad: misma semilla = mismos datos y mismo modelo
N_PACIENTES = 5000      # Tamaño del dataset simulado
TAMANO_TEST = 0.2       # 20 % de los datos se reserva para la evaluación final
FOLDS_CV = 5            # Validación cruzada estratificada de 5 particiones

# Puerta de calidad: si el mejor modelo no alcanza este F1 macro en test,
# el pipeline falla y el modelo NO se publica.
UMBRAL_F1_MINIMO = 0.80

# ---------------------------------------------------------------------------
# Variables del problema
# ---------------------------------------------------------------------------
FEATURES = ["temperatura", "dias_sintomas", "intensidad_dolor"]
OBJETIVO = "diagnostico"

NO_ENFERMO = "NO ENFERMO"
ENFERMEDAD_LEVE = "ENFERMEDAD LEVE"
ENFERMEDAD_AGUDA = "ENFERMEDAD AGUDA"
ENFERMEDAD_CRONICA = "ENFERMEDAD CRÓNICA"
ENFERMEDAD_TERMINAL = "ENFERMEDAD TERMINAL"
ESTADOS = [NO_ENFERMO, ENFERMEDAD_LEVE, ENFERMEDAD_AGUDA, ENFERMEDAD_CRONICA, ENFERMEDAD_TERMINAL]

# Rangos clínicamente válidos de cada variable (mínimo, máximo)
RANGOS = {
    "temperatura": (34.0, 43.0),      # °C
    "dias_sintomas": (0, 3650),       # días
    "intensidad_dolor": (0, 10),      # escala 0-10
}

# Casos de referencia: el modelo publicado DEBE predecir correctamente cada uno.
# Garantizan que el servicio puede retornar los 5 estados (prueba de humo).
CASOS_REFERENCIA = [
    ({"temperatura": 36.5, "dias_sintomas": 0, "intensidad_dolor": 0}, NO_ENFERMO),
    ({"temperatura": 37.8, "dias_sintomas": 3, "intensidad_dolor": 4}, ENFERMEDAD_LEVE),
    ({"temperatura": 39.5, "dias_sintomas": 2, "intensidad_dolor": 8}, ENFERMEDAD_AGUDA),
    ({"temperatura": 37.6, "dias_sintomas": 90, "intensidad_dolor": 5}, ENFERMEDAD_CRONICA),
    ({"temperatura": 37.0, "dias_sintomas": 365, "intensidad_dolor": 9}, ENFERMEDAD_TERMINAL),
]
