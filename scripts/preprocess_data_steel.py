import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def preprocess_data_steel(df):
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
        df['Hour'] = df['date'].dt.hour
        df['Minute'] = df['date'].dt.minute
        df['Week'] = df['date'].dt.isocalendar().week
        df['Day'] = df['date'].dt.day
        df['Day_of_Week_Num'] = df['date'].dt.weekday

        label_encoder = LabelEncoder()
        df['WeekStatus'] = label_encoder.fit_transform(df['WeekStatus'])
        df['Load_Type'] = label_encoder.fit_transform(df['Load_Type'])
        df['Day_of_week'] = label_encoder.fit_transform(df['Day_of_week'])

        features = ['Hour', 'Minute', 'Week', 'Day', 'Day_of_Week_Num',
                'Lagging_Current_Reactive.Power_kVarh',
                'Leading_Current_Reactive_Power_kVarh',
                'Lagging_Current_Power_Factor',
                'Leading_Current_Power_Factor','NSM',
                'WeekStatus','Load_Type']
        targets = ['Usage_kWh']   

        X = df[features]
        y = df[targets]  
        
        # Escalar los datos con MinMaxScaler para LSTM
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()       
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        X_exog = pd.DataFrame(X_scaled, columns=features)

        # Para el modelo LSTM queremos incluir la propia variable target (Usage_kWh)
        # dentro de la secuencia, así la usamos como referencia en la secuencia.
        # Insertamos 'Usage_kWh' como primera columna en X_exog.
        X_exog.insert(0, 'Usage_kWh', y_scaled)

        return X_exog.values, y_scaled, scaler_X, scaler_y