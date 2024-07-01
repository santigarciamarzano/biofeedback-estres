import numpy as np
import pandas as pd

def generate_data(num_personas=40):
    data = {
        'sexo': np.random.choice(['Hombre', 'Mujer'], num_personas),
        'edad': np.random.randint(20, 61, num_personas),
        'temperatura_ambiente': np.random.uniform(20, 31, num_personas),
        'fumador': np.random.choice([True, False], num_personas),
        'deportista': np.random.choice([True, False], num_personas),
        'tiempo_hasta_conductancia_max': np.random.randint(100, 181, num_personas),
        'tiempo_hasta_frecuencia_cardiaca_max': np.random.randint(35, 61, num_personas),
        'tiempo_conductancia_recuperacion': np.random.randint(120, 241, num_personas),
        'tiempo_frecuencia_recuperacion': np.random.randint(20, 41, num_personas),
        'momento_dia': np.random.choice(['Mañana', 'Tarde', 'Noche'], num_personas),
        'frecuencia_cardiaca_inicial': np.concatenate([
            np.random.randint(45, 61, int(0.65 * num_personas)),
            np.random.choice([np.random.randint(20, 46), np.random.randint(60, 66)], int(0.35 * num_personas))
        ]),
        'conductancia_inicial': np.random.randint(60, 91, num_personas),
    }
    df = pd.DataFrame(data)

    # Agregamos la columna de conductancia maxima y condición que sea mayor que inicial
    df['conductancia_maxima'] = np.random.randint(90, 121, num_personas)
    df['conductancia_maxima'] = np.maximum(df['conductancia_maxima'], df['conductancia_inicial'])

    # Agregamos la columna de frecuencia cardiaca maxima y condición que sea mayor que inicial
    df['frecuencia_cardiaca_maxima'] = np.random.randint(90, 151, num_personas)
    df['frecuencia_cardiaca_maxima'] = np.maximum(df['frecuencia_cardiaca_maxima'], df['frecuencia_cardiaca_inicial'])
    
    # Correlación entre conductancia y temperatura
    df['conductancia_inicial'] += df['temperatura_ambiente'] * 2

    # Correlación entre frecuencia cardíaca y persona que fuma
    df.loc[df['fumador'], 'frecuencia_cardiaca_inicial'] += 20
    df.loc[df['fumador'], 'frecuencia_cardiaca_maxima'] += 10

    # Correlación entre deportista y frecuencia cardíaca
    df.loc[df['deportista'], 'frecuencia_cardiaca_inicial'] -= 10
    df.loc[df['deportista'], 'frecuencia_cardiaca_maxima'] -= 20

    # Correlación temperatura y frecuencia cardíaca
    df['frecuencia_cardiaca_inicial'] += df['temperatura_ambiente'] * 0.5
    df['frecuencia_cardiaca_maxima'] += df['temperatura_ambiente'] * 0.3

    # Correlación temperatura y frecuencia cardíaca
    df['conductancia_inicial'] += df['temperatura_ambiente'] * 0.5
    df['conductancia_maxima'] += df['temperatura_ambiente'] * 0.3

    # Correlación entre frecuencia cardíaca y género
    df['frecuencia_cardiaca_inicial'] += np.where(df['sexo'] == 'Mujer', 3, -3)
    df['frecuencia_cardiaca_maxima'] += np.where(df['sexo'] == 'Mujer', 3, -3)

    # Correlación entre conductancia y género
    df['conductancia_inicial'] += np.where(df['sexo'] == 'Mujer', 20, 0)
    df['conductancia_maxima'] += np.where(df['sexo'] == 'Mujer', 20, 0)

    # Correlación entre tiempo y fumador
    df.loc[df['fumador'], 'tiempo_hasta_frecuencia_cardiaca_max'] -= 15

    # Correlación entre tiempo y deportista
    df.loc[df['deportista'], 'tiempo_frecuencia_recuperacion'] -= 10

    # Correlación leve entre tiempo_frecuencia_recuperacion y ser fumador
    df.loc[df['fumador'], 'tiempo_frecuencia_recuperacion'] += 15

    for i in range(num_personas, 4000):
        persona_anterior = df.iloc[np.random.randint(num_personas)]
        nueva_persona = persona_anterior.copy()
        ruido_conductancia = np.random.uniform(-2, 2)
        ruido_frecuencia = np.random.uniform(-5, 5)
        ruido_temperatura = np.random.uniform(-4, 4)
        ruido_conductancia_maxima = np.random.uniform(-3, 3)
        ruido_frecuencia_maxima = np.random.uniform(-3, 3)
        ruido_tiempo_conductancia_max = np.random.uniform(-5, 5)
        ruido_tiempo_frecuencia_max = np.random.uniform(-2, 2)
        ruido_tiempo_conductancia_recuperacion = np.random.uniform(-5, 5)
        ruido_tiempo_frecuencia_recuperacion = np.random.uniform(-2, 2)

        if persona_anterior['fumador']:
            ruido_conductancia = abs(ruido_conductancia)
            ruido_frecuencia = abs(ruido_frecuencia)
            ruido_conductancia_maxima = abs(ruido_conductancia_maxima)
            ruido_frecuencia_maxima = abs(ruido_frecuencia_maxima)
            ruido_tiempo_conductancia_max = abs(ruido_tiempo_conductancia_max)
            ruido_tiempo_frecuencia_max = abs(ruido_tiempo_frecuencia_max)
            ruido_tiempo_conductancia_recuperacion = abs(ruido_tiempo_conductancia_recuperacion)
            ruido_tiempo_frecuencia_recuperacion = abs(ruido_tiempo_frecuencia_recuperacion)

        nueva_persona['conductancia_inicial'] += ruido_conductancia
        nueva_persona['frecuencia_cardiaca_inicial'] += ruido_frecuencia
        nueva_persona['temperatura_ambiente'] += ruido_temperatura
        nueva_persona['conductancia_maxima'] += ruido_conductancia_maxima
        nueva_persona['frecuencia_cardiaca_maxima'] += ruido_frecuencia_maxima
        nueva_persona['tiempo_hasta_conductancia_max'] += ruido_tiempo_conductancia_max
        nueva_persona['tiempo_hasta_frecuencia_cardiaca_max'] += ruido_tiempo_frecuencia_max
        nueva_persona['tiempo_conductancia_recuperacion'] += ruido_tiempo_conductancia_recuperacion
        nueva_persona['tiempo_frecuencia_recuperacion'] += ruido_tiempo_frecuencia_recuperacion

        df = pd.concat([df, nueva_persona.to_frame().T], ignore_index=True)

    df.index = ['persona{}'.format(i) for i in range(1, len(df) + 1)]

    return df

def generate_tratamientos(df):
    prob_respirar = 0.5
    prob_musica = 0.2
    prob_caminar = 0.3

    tratamientos_deportistas = ['respirar'] * int(prob_respirar * len(df))
    tratamientos_no_deportistas = np.random.choice(['respirar', 'musica', 'caminar'],
                                                    size=len(df) - len(tratamientos_deportistas),
                                                    p=[prob_musica / (prob_musica + prob_caminar + prob_respirar),
                                                       prob_caminar / (prob_musica + prob_caminar + prob_respirar),
                                                       prob_respirar / (prob_musica + prob_caminar + prob_respirar)])

    tratamientos = np.concatenate((tratamientos_deportistas, tratamientos_no_deportistas))
    np.random.shuffle(tratamientos)

    df['tratamiento'] = tratamientos

    return df

if __name__ == "__main__":
    df = generate_data()
    df = generate_tratamientos(df)
    df.to_csv('datos_generados_y_tratamientos.csv', index=False)
