"""
Pruebas unitarias del modelo y del reporte de estadísticas.

Correr desde la raíz del proyecto:
    pytest -v
"""

import pytest

from src import config
from src.prediccion import predecir

# Un caso claro (lejos de las fronteras entre clases) por cada estado
CASOS = [
    ({"temperatura": 36.5, "dias_sintomas": 0, "intensidad_dolor": 0}, config.NO_ENFERMO),
    ({"temperatura": 37.8, "dias_sintomas": 3, "intensidad_dolor": 4}, config.ENFERMEDAD_LEVE),
    ({"temperatura": 39.5, "dias_sintomas": 2, "intensidad_dolor": 8}, config.ENFERMEDAD_AGUDA),
    ({"temperatura": 37.6, "dias_sintomas": 90, "intensidad_dolor": 5}, config.ENFERMEDAD_CRONICA),
    ({"temperatura": 37.0, "dias_sintomas": 365, "intensidad_dolor": 9}, config.ENFERMEDAD_TERMINAL),
]


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("entrada, esperado", CASOS, ids=[e for _, e in CASOS])
def test_prediccion_retorna_estado_esperado(entrada, esperado):
    """Dados unos parámetros de entrada, el modelo retorna el estado esperado."""
    assert predecir(**entrada) == esperado


def test_modelo_obtiene_las_cinco_categorias():
    """Con distintos grupos de parámetros se obtienen las 5 categorías."""
    obtenidos = {predecir(**entrada) for entrada, _ in CASOS}
    assert obtenidos == set(config.ESTADOS)
    assert len(config.ESTADOS) == 5


# ---------------------------------------------------------------------------
# Servicio y estadísticas
# ---------------------------------------------------------------------------
def test_estadisticas_vacias_antes_de_predecir(cliente):
    """Antes de cualquier predicción, las estadísticas tienen los valores por defecto."""
    respuesta = cliente.get("/estadisticas")
    assert respuesta.status_code == 200

    datos = respuesta.get_json()
    assert datos["total_predicciones"] == 0
    assert datos["total_por_categoria"] == {estado: 0 for estado in config.ESTADOS}
    assert datos["ultimas_predicciones"] == []
    assert datos["fecha_ultima_prediccion"] is None


def test_ultima_prediccion_aparece_en_estadisticas(cliente):
    """Tras una predicción, las estadísticas la reportan como la última realizada."""
    entrada, esperado = CASOS[2]  # ENFERMEDAD AGUDA
    respuesta = cliente.post("/predecir", json=entrada)
    assert respuesta.status_code == 200
    assert respuesta.get_json()["estado"] == esperado

    datos = cliente.get("/estadisticas").get_json()
    assert datos["total_predicciones"] == 1
    assert datos["total_por_categoria"][esperado] == 1
    assert datos["ultimas_predicciones"][0]["estado"] == esperado
    assert datos["fecha_ultima_prediccion"] == datos["ultimas_predicciones"][0]["fecha_utc"]


def test_estadisticas_guardan_solo_las_ultimas_cinco(cliente):
    """Con más de 5 predicciones, se cuentan todas pero solo se listan las 5 últimas."""
    secuencia = CASOS + [CASOS[0]]  # 6 predicciones; la última es NO ENFERMO
    for entrada, _ in secuencia:
        cliente.post("/predecir", json=entrada)

    datos = cliente.get("/estadisticas").get_json()
    assert datos["total_predicciones"] == 6
    assert datos["total_por_categoria"][config.NO_ENFERMO] == 2
    assert len(datos["ultimas_predicciones"]) == 5
    assert [p["estado"] for p in datos["ultimas_predicciones"]] == [e for _, e in reversed(secuencia[-5:])]


def test_entrada_fuera_de_rango_retorna_error(cliente):
    """Un valor fuera de rango no produce predicción: responde 400."""
    respuesta = cliente.post("/predecir", json={"temperatura": 50, "dias_sintomas": 2, "intensidad_dolor": 3})
    assert respuesta.status_code == 400
    assert "error" in respuesta.get_json()
