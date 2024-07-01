# Biofeedback Estrés

## Descripción

Este proyecto genera datos simulados para estudios de biofeedback y estrés, entrenando modelos de clasificación y regresión para predecir tratamientos y tiempos de recuperación.


## Instalación

1. Clona el repositorio:
    ```
    git clone https://github.com/santigarciamarzano/biofeedback_estres.git
    ```

2. Navega al directorio del proyecto:
    ```
    cd nombre_del_proyecto
    ```

3. Instala las dependencias:
    ```
    pip install -r requirements.txt
    ```

## Uso

Para uso completo:
    ```
    python train_custom.ipynb
    ```

Para uso por secciones:

1. Genera datos y tratamientos:
    ```
    python data_generator.py
    ```

2. Realiza visualizaciones de los datos generados:
    ```
    python data_visualization.py
    ```

3. Entrena los modelos:
    ```
    python train.py
    ```

4. Realiza predicciones con los modelos entrenados:
    ```
    python predict.py
    ```

## Estructura del Proyecto

biofeedback_estres/
│
├── data_generator.py # Generación de datos simulados
├── data_visualization.py # Visualización de los datos generados
├── train.py # Entrenamiento de modelos
├── predict.py # Predicciones con modelos entrenados
├── requirements.txt # Dependencias del proyecto
├── README.md # Documentación del proyecto
└── datos_generados.csv # Archivo de datos generados (opcional)