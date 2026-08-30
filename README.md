# Regresión Lineal sin Frameworks — Predicción de Retrasos en Vuelos

**Autor:** Mauricio Olguín Sánchez
**Matrícula:** A01711522

## Alcance actual

Implementación manual de un modelo de regresión lineal entrenado con Descenso de Gradiente por Mini-Batch, para predecir el retraso en la salida de vuelos (`DepDelay`) a partir de variables de horarios y distancia.

Incluye el pipeline completo de ETL, un análisis exploratorio de datos (EDA) y la evaluación del modelo mediante MSE y $R^2$ sobre conjuntos de train, validation y test.

> Esta es la primera entrega del proyecto (modelo implementado "a mano"). La segunda parte,
> usando frameworks de Machine Learning, está pendiente.

## Reporte

📄 [Reporte completo (PDF)](reporte/reporte.pdf)

## Dataset

El proyecto utiliza el dataset **Airlines_DepDelay_10M** de OpenML (10,000,000 de registros, 1987–2013), basado en datos del *Bureau of Transportation Statistics de EE. UU.*

⬇️ [Descargar dataset (.arff)](https://www.openml.org/search?type=data&status=active&id=42728&sort=runs)