# Regresión Lineal — Predicción de Retrasos en Vuelos

**Autor:** Mauricio Olguín Sánchez
**Matrícula:** A01711522

## Modelos implementados

El proyecto implementa dos modelos de regresión para predecir el retraso en la salida de vuelos (`DepDelay`) a partir de variables relacionadas con los horarios y la distancia:

- **Modelo manual:** regresión lineal implementada desde cero. Aprende los pesos y el sesgo minimizando el error cuadrático medio (MSE) mediante descenso de gradiente por mini-batches.
- **Modelo XGBoost:** modelo de ensamble basado en árboles de decisión construidos secuencialmente mediante boosting. Captura relaciones no lineales e interacciones entre las variables, y se entrena con `XGBRegressor` usando el algoritmo histograma para acelerar el procesamiento de un dataset grande.

## Reporte

📄 [Reporte completo (PDF)](reporte/reporte.pdf)

## Dataset

El proyecto utiliza el dataset **Airlines_DepDelay_10M** de OpenML (10,000,000 de registros, 1987–2013), basado en datos del *Bureau of Transportation Statistics de EE. UU.*

⬇️ [Descargar dataset (.arff)](https://www.openml.org/search?type=data&status=active&id=42728&sort=runs)
