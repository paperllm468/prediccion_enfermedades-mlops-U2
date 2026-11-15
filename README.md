# Predicción de enfermedades: MLOps (Unidad 2)

## El problema

En medicina existe mucha información de los pacientes, pero para las enfermedades **huérfanas** (poco comunes) los datos escasean. Se necesita un modelo que, a partir de los síntomas de un paciente, estime si puede sufrir alguna enfermedad, tanto para enfermedades comunes (muchos datos) como huérfanas (pocos datos).

En la unidad 1 se construyó una solución local para el médico: un modelo que recibe tres valores clínicos (temperatura, días con síntomas e intensidad del dolor) y retorna el estado del paciente, servido como página web y API dentro de una imagen de **Docker**.

## Propósito del repositorio

Llevar un control de versiones ordenado de esa solución y de su evolución:

- Guardar el código de la solución inicial tal como se presentó en la unidad 1.
- Incorporar nuevos requerimientos de los médicos mediante **ramas** y **pull requests**, de modo que cada cambio quede trazable.
- Más adelante, automatizar pruebas y publicación de la imagen con **GitHub Actions** (CI/CD).

## Estructura (prevista)

| Rama | Contenido |
|---|---|
| `main` | Rama principal; recibe los cambios solo a través de pull requests |
| `solución-inicial` | Archivos de la unidad 1 (código, Dockerfile, documentación) |
| Ramas de requerimientos | Un cambio por rama, integrado a `main` con un PR |

El código de la solución (carpetas `src/` y `servicio/`, `Dockerfile`, `docker-compose.yml`) se agrega en la rama `solución-inicial`, y este archivo se reemplaza por el README completo de la solución.

> Herramienta de apoyo: la decisión clínica final siempre es del médico.
