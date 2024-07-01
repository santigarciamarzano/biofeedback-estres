import pandas as pd

def clean_data(data_path='datos_recuperacion.csv'):
    
    df = pd.read_csv(data_path)

    # Eliminar filas con valores faltantes (NaN)
    df = df.dropna()

    # Convertir columnas numéricas a enteros
    columnas_numericas = ['conductancia_inicial', 'frecuencia_cardiaca_inicial', 'temperatura_ambiente',
                          'frecuencia_cardiaca_maxima', 'tiempo_hasta_conductancia_max',
                          'tiempo_hasta_frecuencia_cardiaca_max', 'tiempo_conductancia_recuperacion',
                          'tiempo_frecuencia_recuperacion', 'conductancia_maxima']
    df[columnas_numericas] = df[columnas_numericas].astype(int)

    # Función para identificar outliers y reemplazarlos por la media
    def reemplazar_outliers_por_media(columna):
        q25, q75 = columna.quantile(q=[0.25, 0.75])
        rango_inter = q75 - q25
        min_val = q25 - 1.5 * rango_inter
        max_val = q75 + 1.5 * rango_inter
        mascara = (columna >= min_val) & (columna <= max_val)
        media_columna = columna[mascara].mean()
        columna.loc[~mascara] = media_columna
        return columna

    
    for columna in columnas_numericas:
        df[columna] = reemplazar_outliers_por_media(df[columna])

   
    df.to_csv('datos_limpios.csv', index=False)
    print("Datos limpios guardados en datos_recuperacion_limpio.csv")

if __name__ == "__main__":
    clean_data()
