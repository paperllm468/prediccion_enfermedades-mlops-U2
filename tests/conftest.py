"""
Configuración compartida de las pruebas (pytest).

- Si el modelo no existe, ejecuta el pipeline de entrenamiento una vez.
- Redirige el log de predicciones a una carpeta temporal en cada prueba, para
  que las estadísticas partan siempre vacías y no se mezclen entre pruebas.
"""

import pytest

from src import config


@pytest.fixture(scope="session", autouse=True)
def modelo_entrenado():
    """Garantiza que exista models/modelo.joblib antes de correr las pruebas."""
    if not config.RUTA_MODELO.exists():
        from src import pipeline
        pipeline.main()


@pytest.fixture(autouse=True)
def log_temporal(tmp_path, monkeypatch):
    """Cada prueba escribe sus predicciones en un log propio y vacío."""
    monkeypatch.setattr(config, "DIR_LOGS", tmp_path / "logs")
    monkeypatch.setattr(config, "RUTA_LOG_PREDICCIONES", tmp_path / "logs" / "predicciones.csv")


@pytest.fixture
def cliente():
    """Cliente HTTP de prueba del servicio Flask (no levanta un servidor real)."""
    from servicio.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
