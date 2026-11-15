# =====================================================================
# Imagen Docker del servicio de predicción de estado del paciente
#
# Construcción MULTI-ETAPA (separa entrenamiento y producción):
#   Etapa 1 "entrenamiento": ejecuta el pipeline completo de ML
#            (ingesta -> limpieza -> EDA -> entrenamiento -> registro)
#            y produce el modelo entrenado.
#   Etapa 2 "servicio": imagen final y liviana que solo contiene el modelo
#            entrenado, el código de predicción y el servicio web.
#
# Construir:  docker build -t modelo-medico .
# Ejecutar:   docker run -d -p 5000:5000 --name modelo-medico modelo-medico
# =====================================================================


# ---------------------------------------------------------------------
# ETAPA 1: ENTRENAMIENTO
# ---------------------------------------------------------------------
FROM python:3.12-slim AS entrenamiento

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /proyecto

# Dependencias primero para aprovechar la caché de capas de Docker
COPY requirements.txt requirements-entrenamiento.txt ./
RUN pip install --no-cache-dir -r requirements-entrenamiento.txt

# Código del pipeline de ML
COPY src/ src/

# Ejecuta el pipeline. Si el modelo no supera las puertas de calidad
# (F1 mínimo y casos de referencia), este paso falla y la imagen NO se construye.
RUN python -m src.pipeline


# ---------------------------------------------------------------------
# ETAPA 2: SERVICIO (imagen final de producción)
# ---------------------------------------------------------------------
FROM python:3.12-slim AS servicio

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /proyecto

# Solo dependencias de producción (sin matplotlib)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Solo el código necesario para predecir y servir
COPY src/__init__.py src/config.py src/prediccion.py src/estadisticas.py src/
COPY servicio/ servicio/

# Artefactos producidos en la etapa de entrenamiento:
# modelo + metadatos (registro de modelos) y reportes (EDA, métricas)
COPY --from=entrenamiento /proyecto/models/ models/
COPY --from=entrenamiento /proyecto/reports/ reports/

# Usuario sin privilegios (seguridad) y carpeta para el log de monitoreo
RUN useradd --create-home appuser \
    && mkdir -p logs \
    && chown -R appuser /proyecto
USER appuser

EXPOSE 5000

# Docker consulta /salud cada 30 s para saber si el servicio y el modelo están bien
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/salud')" || exit 1

# Servidor de producción gunicorn con 2 workers
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "servicio.app:app"]
