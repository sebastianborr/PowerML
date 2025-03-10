import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os

# Modelos
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# Preprocesado y métricas
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import time

# Guardar y cargar modelos
import joblib

# Cargar otros scripts
from scripts.preprocess_data_steel import preprocess_data_steel
from scripts.printGrafics import plot_results

def main(saveModel, saveResults, pretrainedModel, pretrainedName):
    # Carga de datos
    df = pd.read_csv('data/00_Steel_industry_data.csv')

    # Preprocesado de los datos
    X, y = preprocess_data_steel(df)
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar las características
    scale=1
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # Cargar el modelo sin entrenar
    #untrainedModel = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    untrainedModel = LinearRegression()
    #untrainedModel = RandomForestRegressor(n_estimators=100, random_state=42)
    #untrainedModel = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    #untrainedModel = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    modelName = untrainedModel.__class__.__name__

    # Entrenar el modelo
    trainedmodel, train_time = train_model(untrainedModel, X_train_scaled, y_train)

    # Evaluar el modelo
    [mae, mse, r2] = evaluate_model(trainedmodel, X_train_scaled, X_test_scaled, y_train, y_test, modelName)

    # Guardar los resultados
    if saveModel and pretrainedModel==0:
        save_model(trainedmodel, modelName)

    if saveResults and pretrainedModel==0:
        save_results(mae, mse, r2, train_time, modelName, trainedmodel)

#- 
def train_model(untrainedModel,X_train, y_train):
    # Entrenar el modelo
    start_time = time.time()    
    untrainedModel.fit(X_train, y_train)
    training_time = time.time() - start_time
    return untrainedModel, training_time

def evaluate_model(model, X_train, X_test, y_train, y_test,modelName=""):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluación en el conjunto de prueba
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
#    accu = accuracy_score(y_test, y_test_pred)
    if  1:
        print(f'MAE: {mae:.2f}')
        print(f'MSE: {mse:.2f}')
        print(f'R²: {r2:.2f}')
        #print(f'Accuracy: {accu:.2f}')  
        plot_results(y_test, y_test_pred, modelName)

    return mae, mse, r2

def save_model(model, modelName):
    # Guardar el modelo
    joblib.dump(model, ("./models/" + modelName + ".pkl"))

def save_results(mae,mse, r2, training_time, modelName, model):
    # Guardar los resultados

    model_params = model.get_params()
    results_dict = {
        'ModelName': modelName,
        'Kernel': model_params.get('kernel','N/A'),
        'Params': str(model_params),
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'TrainTime': training_time
    }
    results_df = pd.DataFrame([results_dict])
    results_df.to_csv(f'./output/ResultsML/{modelName}_results.csv', index=False)


if __name__ == '__main__':
    # parámetro
    '''
   saveModel(boolean): Guardar el modelo entrenado
   saveResults(boolean): Guardar los resultados de la evaluación
   pretrainedModel(boolean): Cargar un modelo preentrenado
   nameModel(string): Nombre del modelo a cargar
    '''
    main(saveModel=False, saveResults=False, pretrainedModel=False, pretrainedName="SVR")
