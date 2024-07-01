import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pickle

def train_models():
    df = pd.read_csv('datasets/datos_generados.csv')

    X_classification = df[['frecuencia_cardiaca_inicial', 'frecuencia_cardiaca_maxima', 'conductancia_inicial', 'conductancia_maxima',
                           'fumador', 'deportista', 'sexo', 'tiempo_conductancia_recuperacion', 'tiempo_frecuencia_recuperacion']]
    y_classification = df['tratamiento']

    X_classification = pd.get_dummies(X_classification, columns=['fumador', 'deportista', 'sexo'], drop_first=True)

    X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train_classification, y_train_classification)

    knn = KNeighborsClassifier()
    knn.fit(X_train_classification, y_train_classification)

    y_pred_dtc = dtc.predict(X_test_classification)
    y_pred_knn = knn.predict(X_test_classification)

    accuracy_dtc = accuracy_score(y_test_classification, y_pred_dtc)
    accuracy_knn = accuracy_score(y_test_classification, y_pred_knn)

    print("Exactitud del árbol de decisión:", accuracy_dtc)
    print("Exactitud del modelo KNN:", accuracy_knn)

    X_regression = df[['frecuencia_cardiaca_inicial', 'conductancia_inicial', 'frecuencia_cardiaca_maxima', 'fumador',
                       'conductancia_maxima', 'deportista', 'sexo', 'tiempo_hasta_conductancia_max']]
    y_regression = df['tiempo_conductancia_recuperacion']

    X_regression = pd.get_dummies(X_regression, columns=['sexo', 'fumador', 'deportista'], drop_first=True)

    X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

    regression = LinearRegression()
    regression.fit(X_train_regression, y_train_regression)

    y_pred_regression = regression.predict(X_test_regression)

    mse = mean_squared_error(y_test_regression, y_pred_regression)
    r2 = r2_score(y_test_regression, y_pred_regression)

    print("Métricas para el tiempo de recuperación de conductancia:")
    print("MSE:", mse)
    print("R^2:", r2)

    # Guardar los modelos en archivos pickle
    with open('models/dtc_model.pkl', 'wb') as file:
        pickle.dump(dtc, file)

    with open('models/knn_model.pkl', 'wb') as file:
        pickle.dump(knn, file)

    with open('models/regression_model.pkl', 'wb') as file:
        pickle.dump(regression, file)

    # Guardar las columnas utilizadas en clasificación y regresión
    with open('models/classification_columns.pkl', 'wb') as file:
        pickle.dump(X_classification.columns.tolist(), file)

    with open('models/regression_columns.pkl', 'wb') as file:
        pickle.dump(X_regression.columns.tolist(), file)

    models = {
        'dtc': dtc,
        'knn': knn,
        'regression': regression
    }

    test_data = {
        'X_test_classification': X_test_classification,
        'y_test_classification': y_test_classification,
        'y_pred_dtc': y_pred_dtc,
        'y_pred_knn': y_pred_knn,
        'X_test_regression': X_test_regression,
        'y_test_regression': y_test_regression,
        'y_pred_regression': y_pred_regression
    }

    return models, test_data

if __name__ == "__main__":
    train_models()
