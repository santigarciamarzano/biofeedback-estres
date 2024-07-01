import pandas as pd
import numpy as np
import pickle
import streamlit as st

REGRESSION_MODEL_PATH = 'models/regression_model.pkl'
DTC_MODEL_PATH = 'dtc_model.pkl'
CLASSIFICATION_COLUMNS_PATH = 'classification_columns.pkl'

def load_models():
    with open(REGRESSION_MODEL_PATH, 'rb') as file:
        regression_model = pickle.load(file)

    with open(DTC_MODEL_PATH, 'rb') as file:
        dtc_model = pickle.load(file)

    with open(CLASSIFICATION_COLUMNS_PATH, 'rb') as file:
        classification_columns = pickle.load(file)
    
    return regression_model, dtc_model, classification_columns

def model_prediction_regression(data, model):
    data = np.asarray(data).reshape(1, -1)
    prediction = model.predict(data)
    return prediction

def model_prediction_classification(data, model, classification_columns):
    data_df = pd.DataFrame([data], columns=['frecuencia_cardiaca_inicial', 'conductancia_inicial', 'frecuencia_cardiaca_maxima',
                                            'fumador', 'conductancia_maxima', 'deportista', 'sexo', 'tiempo_hasta_conductancia_max'])
    data_df = pd.get_dummies(data_df, columns=['sexo', 'fumador', 'deportista'], drop_first=True)
    
    for col in classification_columns:
        if col not in data_df.columns:
            data_df[col] = 0
    
    data_df = data_df[classification_columns]
    prediction = model.predict(data_df)
    return prediction

def main():
    st.title("Sistema de Predicción de Recuperación de Conductancia y Tratamiento")

    regression_model, dtc_model, classification_columns = load_models()
    
    # Título de la página
    html_temp = """
    <h1 style="color:#181082;text-align:center;">Sistema de Predicción de Recuperación de Conductancia y Tratamiento</h1>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Lectura de datos de entrada
    st.header("Datos de Entrada")
    frecuencia_cardiaca_inicial = st.number_input("Frecuencia Cardiaca Inicial:", min_value=0, step=1)
    conductancia_inicial = st.number_input("Conductancia Inicial:", min_value=0, step=1)
    frecuencia_cardiaca_maxima = st.number_input("Frecuencia Cardiaca Máxima:", min_value=0, step=1)
    fumador = st.radio("¿Es Fumador?", ('No', 'Sí'))
    conductancia_maxima = st.number_input("Conductancia Máxima:", min_value=0, step=1)
    deportista = st.radio("¿Es Deportista?", ('No', 'Sí'))
    sexo = st.radio("Sexo:", ('Mujer', 'Varón'))
    tiempo_hasta_conductancia_max = st.number_input("Tiempo hasta Conductancia Máxima:", min_value=0, step=1)

    # El botón de predicción
    if st.button("Realizar Predicciones"):
        data_regression = [frecuencia_cardiaca_inicial, conductancia_inicial, frecuencia_cardiaca_maxima, 
                           1 if fumador == 'Sí' else 0, conductancia_maxima, 1 if deportista == 'Sí' else 0, 
                           1 if sexo == 'Mujer' else 0, tiempo_hasta_conductancia_max]
        data_classification = {'frecuencia_cardiaca_inicial': frecuencia_cardiaca_inicial,
                               'conductancia_inicial': conductancia_inicial,
                               'frecuencia_cardiaca_maxima': frecuencia_cardiaca_maxima,
                               'fumador': 1 if fumador == 'Sí' else 0,
                               'conductancia_maxima': conductancia_maxima,
                               'deportista': 1 if deportista == 'Sí' else 0,
                               'sexo': 1 if sexo == 'Mujer' else 0,
                               'tiempo_hasta_conductancia_max': tiempo_hasta_conductancia_max}

        prediction_regression = model_prediction_regression(data_regression, regression_model)
        prediction_classification = model_prediction_classification(data_classification, dtc_model, classification_columns)
        
        st.success(f'Tiempo de recuperación de conductancia predicho: {prediction_regression[0]:.2f} segundos')
        st.success(f'Mejor tratamiento predicho: {prediction_classification[0]}')

if __name__ == "__main__":
    main()

