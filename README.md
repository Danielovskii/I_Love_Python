# I Love Python
I Love Python, screw MATLAB (nah, but python better)

# Credit Card Default Classification Project

## Overview

Este proyecto tiene como objetivo analizar y predecir el incumplimiento de tarjetas de crédito utilizando datos proporcionados en los archivos `creditCardDefault_train.csv` y `creditCardDefault_test.csv`. Se implementaron dos modelos de aprendizaje automático: **Regresión Logística** y **Random Forest**, siguiendo una metodología que incluye preprocesamiento de datos, ajuste de hiperparámetros, evaluación de modelos y visualización de resultados. Este trabajo se desarrolló en Python utilizando bibliotecas como `pandas`, `scikit-learn`, `matplotlib` y `seaborn`.

El proyecto cumple con los requisitos de la actividad:
- Uso de Regresión Logística y otro método de clasificación (Random Forest).
- Generación de tres gráficos de interacción entre variables predictorias.
- Creación de una curva AUC-ROC con el área bajo la curva.
- Presentación de matrices de confusión y precisión de los modelos.

## Estructura del Proyecto

- **`creditCardDefault_train.csv`**: Conjunto de datos de entrenamiento.
- **`creditCardDefault_test.csv`**: Conjunto de datos de prueba.
- **`credit_classifier.py`**: Script principal que contiene el código para cargar datos, entrenar modelos, generar visualizaciones y evaluar resultados.
- **`README.md`**: Este archivo, que documenta el proyecto.

## Data Description

Los conjuntos de datos contienen información sobre clientes de tarjetas de crédito, con las siguientes variables predictoras:
- `creditLimit`: Límite de crédito.
- `gender`: Género del cliente.
- `edu`: Nivel educativo.
- `age`: Edad.
- `nDelay`: Número de retrasos en pagos.
- `billAmt1` a `billAmt6`: Montos de facturación de los últimos 6 meses.
- `default`: Variable objetivo (0 = sin incumplimiento, 1 = incumplimiento).

Los datos se dividieron en entrenamiento (`train`) y prueba (`test`) para entrenar y evaluar los modelos, respectivamente.

## Metodología

### 1. Preprocesamiento de Datos
- **Carga de datos**: Se usó `pandas` para leer los archivos CSV.
- **Escalado**: Las variables predictoras se escalaron con `StandardScaler` para la Regresión Logística, asegurando convergencia del modelo. Random Forest no requiere escalado.
- **Conversión**: La variable objetivo `default` se aseguró que estuviera en formato binario (0 y 1).

### 2. Modelos de Clasificación
#### Regresión Logística
- Implementada con `LogisticRegression` de `scikit-learn`.
- Configuración: `max_iter=2000` para garantizar convergencia tras escalar los datos.
- Entrenada con el conjunto de entrenamiento escalado.

#### Random Forest
- Implementada con `RandomForestClassifier`.
- **Ajuste de hiperparámetros**: Se probó `max_depth` de 1 a 25, seleccionando el valor óptimo basado en la precisión en el conjunto de prueba.
- Se graficó la precisión de entrenamiento y prueba para analizar el tradeoff entre sesgo y varianza.
- Se calcularon las importancias de las características para identificar las variables más influyentes.

### 3. Visualizaciones
- **Gráficos de interacción**:
  1. `creditLimit` vs `age`.
  2. `billAmt1` vs `billAmt2`.
  3. `nDelay` vs `creditLimit`.
  - Estos gráficos de dispersión muestran la relación entre pares de variables, coloreados por la variable `default`.
- **Curva AUC-ROC**:
  - Se generó una curva ROC para ambos modelos, incluyendo el área bajo la curva (AUC).
  - Se calcularon los mejores umbrales basados en el G-mean (`sqrt(TPR * (1 - FPR))`), marcados en la gráfica.
- **Matriz de confusión vs `max_depth`**:
  - Se graficó la precisión del Random Forest en función de `max_depth` para el conjunto de entrenamiento y prueba.

### 4. Evaluación
- **Métricas**:
  - **Precisión**: Proporción de predicciones correctas (`(TN + TP) / (TN + FP + FN + TP)`).
  - **AUC**: Área bajo la curva ROC, que mide la capacidad de discriminación del modelo.
- **Matrices de confusión**: Presentadas para ambos modelos con valores de Verdaderos Negativos (TN), Falsos Positivos (FP), Falsos Negativos (FN) y Verdaderos Positivos (TP).
- **Importancia de características**: Para Random Forest, se listaron las 5 variables más influyentes.

## Resultados

El script `credit_classifier.py` genera los siguientes resultados:
- **Gráficos**:
  - Precisión vs `max_depth` para Random Forest.
  - Tres gráficos de interacción.
  - Curva ROC con AUC y mejores umbrales.
  - Matrices de confusión con precisión en los títulos.
- **Salida impresa**:
  - Mejor `max_depth` y precisión para Random Forest.
  - Top 5 características de Random Forest.
  - Mejores umbrales basados en G-mean.
  - Evaluación detallada con precisión, AUC y matrices de confusión.

Ejemplo de salida (los valores dependerán de los datos):
