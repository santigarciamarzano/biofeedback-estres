import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
    plt.title('Matriz de Correlación')
    plt.show()

def plot_gender_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='sexo', palette='pastel')
    plt.xlabel('Género')
    plt.ylabel('Cantidad de Personas')
    plt.title('Distribución de Género')
    plt.show()

def plot_frequency_histogram(df):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='frecuencia_cardiaca_inicial', hue='sexo', bins=20, kde=True, palette='muted')
    plt.xlabel('Frecuencia Cardíaca Inicial')
    plt.ylabel('Personas')
    plt.title('Histograma de Frecuencias Cardiacas por Género')
    plt.legend(title='Género', labels=['Mujer', 'Hombre'])
    plt.show()

def plot_athlete_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='deportista', y='frecuencia_cardiaca_inicial', palette='Set3')
    plt.xlabel('Deportista')
    plt.ylabel('Frecuencia Cardíaca Inicial')
    plt.title('Distribución de Frecuencias Cardíacas Iniciales para Deportistas y No Deportistas')
    plt.xticks(ticks=[0, 1], labels=['No Deportista', 'Deportista'])
    plt.show()

def plot_scatter_frequencies(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='frecuencia_cardiaca_inicial', y='frecuencia_cardiaca_maxima', hue='fumador', palette='plasma')
    plt.xlabel('Frecuencia Cardiaca Inicial')
    plt.ylabel('Frecuencia Cardiaca Final')
    plt.title('Relación entre Frecuencia Cardiaca Inicial y Final')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('datos_generados.csv')
    plot_correlation_matrix(df)
    plot_gender_distribution(df)
    plot_frequency_histogram(df)
    plot_athlete_distribution(df)
    plot_scatter_frequencies(df)

