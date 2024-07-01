import pandas as pd
import pickle

def predict(models, classification_columns, regression_columns, nueva_persona):
    # Convertir nueva_persona a DataFrame
    datos_nueva_persona = pd.DataFrame([nueva_persona])

    # Convertir las columnas categóricas en variables dummy
    datos_nueva_persona = pd.get_dummies(datos_nueva_persona, columns=['sexo', 'fumador', 'deportista'], drop_first=True)

    # Asegurarse de que las nuevas características coincidan con las de entrenamiento
    for col in classification_columns:
        if col not in datos_nueva_persona.columns:
            datos_nueva_persona[col] = 0

    # Reordenar las columnas para que coincidan con las de entrenamiento
    datos_nueva_persona = datos_nueva_persona[classification_columns]

    # Predicción de tiempo de recuperación de conductancia
    datos_nueva_persona_regression = datos_nueva_persona.copy()
    datos_nueva_persona_regression['tiempo_hasta_conductancia_max'] = nueva_persona['tiempo_hasta_conductancia_max']
    tiempo_recuperacion_conductancia_predicho = models['regression'].predict(datos_nueva_persona_regression[regression_columns])

    print("Tiempo de recuperación de conductancia predicho para la nueva persona:", tiempo_recuperacion_conductancia_predicho[0], "segundos")

    # Predicción del mejor tratamiento
    mejor_tratamiento_predicho_dtc = models['dtc'].predict(datos_nueva_persona)
    mejor_tratamiento_predicho_knn = models['knn'].predict(datos_nueva_persona)

    print("Mejor tratamiento predicho por el árbol de decisión para la nueva persona:", mejor_tratamiento_predicho_dtc[0])
    print("Mejor tratamiento predicho por el algoritmo de vecinos cercanos para la nueva persona:", mejor_tratamiento_predicho_knn[0])

if __name__ == "__main__":
    # Cargar los modelos y las columnas desde los archivos pickle
    with open('models/dtc_model.pkl', 'rb') as file:
        dtc = pickle.load(file)

    with open('models/knn_model.pkl', 'rb') as file:
        knn = pickle.load(file)

    with open('models/regression_model.pkl', 'rb') as file:
        regression = pickle.load(file)

    with open('models/classification_columns.pkl', 'rb') as file:
        classification_columns = pickle.load(file)

    with open('models/regression_columns.pkl', 'rb') as file:
        regression_columns = pickle.load(file)

    models = {
        'dtc': dtc,
        'knn': knn,
        'regression': regression
    }

    # Definir un ejemplo de nueva persona para predicción
    nueva_persona = {
        'frecuencia_cardiaca_inicial': 60,
        'conductancia_inicial': 80,
        'frecuencia_cardiaca_maxima': 75,
        'fumador': 0,  # 0 para No, 1 para Sí
        'conductancia_maxima': 90,
        'deportista': 1,  # 0 para No, 1 para Sí
        'sexo': 0,  # 0 para Hombre, 1 para Mujer
        'tiempo_hasta_conductancia_max': 150  # Tiempo hasta conductancia máxima
    }

    # Realizar predicción
    predict(models, classification_columns, regression_columns, nueva_persona)
