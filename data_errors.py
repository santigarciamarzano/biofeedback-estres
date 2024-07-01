import numpy as np
import pandas as pd

def meter_nan(value, prob_nan=0.01):
    if np.random.rand() < prob_nan:
        return np.nan
    else:
        return value

def meter_outlier(valor, outlier_prob=0.08, outlier_rango=(150, 200)):
    if np.random.rand() < outlier_prob:
        return np.random.randint(outlier_rango[0], outlier_rango[1] + 1)
    else:
        return valor

def introduce_errores(df):
    columnas_df = ['frecuencia_cardiaca_inicial', 'conductancia_inicial',
                   'temperatura_ambiente', 'frecuencia_cardiaca_maxima',
                   'tiempo_hasta_conductancia_max', 'tiempo_hasta_frecuencia_cardiaca_max',
                   'tiempo_conductancia_recuperacion', 'tiempo_frecuencia_recuperacion']

    columnas_outliers = ['frecuencia_cardiaca_inicial', 'conductancia_inicial',
                         'temperatura_ambiente', 'frecuencia_cardiaca_maxima']

    for columna in columnas_df:
        df[columna] = df[columna].apply(lambda x: meter_nan(x))

    for columna in columnas_outliers:
        df[columna] = df[columna].apply(lambda x: meter_outlier(x))

    return df

if __name__ == "__main__":
    df = pd.read_csv('datos_recuperacion.csv')
    df_con_errores = introduce_errores(df)
    df_con_errores.to_csv('datos_con_errores.csv', index=False)
    print("Datos con errores generados y guardados en datos_con_errores.csv")
