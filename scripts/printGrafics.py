import matplotlib.pyplot as plt
import numpy as np

def plot_results(y_test, y_test_pred, modelName, X_train=None, y_train=None, model=None):
    if 1:    
        plt.figure(figsize=(10,5))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', linewidth=2)  # Línea ideal
        plt.xlabel("Valores Reales")
        plt.ylabel("Predicciones")
        plt.title(modelName+": Predicciones vs. Valores Reales")
        plt.show()
    if 1:
        residuos = y_test - y_test_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test_pred, residuos, color="purple", alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel("Predicciones")
        plt.ylabel("Residuos (Error)")
        plt.title("Gráfico de Residuos - "+ modelName)
        plt.show()
    if 0:
        # Si X_train tiene solo una variable
        X_range = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)  # Rango de valores
        y_range_pred = model.predict(X_range)  # Predicciones para esos valores

        plt.figure(figsize=(8, 5))
        plt.scatter(X_train, y_train, color='blue', alpha=0.5, label="Datos Reales")
        plt.plot(X_range, y_range_pred, color='red', label="Modelo SVR")
        plt.xlabel("Variable Independiente")
        plt.ylabel("Consumo (kWh)")
        plt.title("Regresión con SVR")
        plt.legend()
        plt.show()

