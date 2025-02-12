import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar datos
df = pd.read_csv('data/Steel_industry_data.csv')

# Convertir 'date' a datetime
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')

# Extraer características de fecha
df['Hour'] = df['date'].dt.hour
df['Minute'] = df['date'].dt.minute
df['Week'] = df['date'].dt.isocalendar().week
df['Day'] = df['date'].dt.day
df['Day_of_Week_Num'] = df['date'].dt.weekday  # 0=Lunes, 6=Domingo

# Codificar variables categóricas
label_encoder = LabelEncoder()
df['WeekStatus'] = label_encoder.fit_transform(df['WeekStatus'])  # Weekday=0, Weekend=1
df['Load_Type'] = label_encoder.fit_transform(df['Load_Type'])  # Light=0, Medium=1, High=2
df['Day_of_week'] = label_encoder.fit_transform(df['Day_of_week'])  # Monday=0, ..., Sunday=6

# Selección de características y target
X = df[['Hour', 'Minute', 'Week', 'Day', 'Day_of_Week_Num', 'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor',
        'NSM', 'WeekStatus', 'Load_Type']]
y = df['Usage_kWh']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Crear el modelo SVM
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# Entrenar el modelo
svm_model.fit(X_train_scaled, y_train)

# Predicciones
y_train_pred = svm_model.predict(X_train_scaled)
y_test_pred = svm_model.predict(X_test_scaled)


# Evaluación en el conjunto de prueba
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'R²: {r2:.2f}')

#MAE: 2.37
#MSE: 28.19
#R²: 0.97
