import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data_electricity(df):
    """
    Preprocesa el dataset de consumo eléctrico y meteorología.
    - Convierte la columna DateTime a formato datetime.
    - Extrae características temporales como hora, minuto, día y semana.
    - Escala las variables numéricas para mejorar el rendimiento en modelos LSTM.
    - Incluye las variables objetivo dentro de las secuencias de entrada.
    - Agrega la columna WeekStatus para indicar si es día de semana o fin de semana.
    """
    # Convertir DateTime a formato datetime
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Extraer características temporales
    df['Hour'] = df['DateTime'].dt.hour
    df['Minute'] = df['DateTime'].dt.minute
    df['Week'] = df['DateTime'].dt.isocalendar().week
    df['Day'] = df['DateTime'].dt.day
    df['Day_of_Week_Num'] = df['DateTime'].dt.weekday
    
    # Agregar columna WeekStatus (Weekday o Weekend)
    df['WeekStatus'] = df['Day_of_Week_Num'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    
    # Convertir WeekStatus a numérico (0 para Weekday, 1 para Weekend)
    df['WeekStatus'] = df['WeekStatus'].map({'Weekday': 0, 'Weekend': 1})

    # Limpieza de columnas numéricas
    cols_numeric = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
    for col in cols_numeric:
        df[col] = df[col].astype(str).str.replace('º', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')


    # Seleccionar características relevantes para X
    features = ['Hour', 'Minute', 'Week', 'Day', 'Day_of_Week_Num', 'WeekStatus',
                'Temperature', 'Humidity', 'Wind Speed',
                'general diffuse flows', 'diffuse flows']
    
    # Variables objetivo (consumo por zonas)
    targets = ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']
    
    X = df[features]
    y = df[targets]
    
    # Escalar los datos con MinMaxScaler para LSTM
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Convertir X de nuevo a DataFrame para insertar las variables objetivo en la secuencia
    X_exog = pd.DataFrame(X_scaled, columns=features)
    X_exog.insert(0, 'Zone 1 Power Consumption', y_scaled[:, 0])
    X_exog.insert(1, 'Zone 2 Power Consumption', y_scaled[:, 1])
    X_exog.insert(2, 'Zone 3 Power Consumption', y_scaled[:, 2])
    
    return X_exog.values, y_scaled, scaler_X, scaler_y
