import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    
    X = df[['Hour', 'Minute', 'Week', 'Day', 'Day_of_Week_Num',
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh',
            'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor','NSM',
            'WeekStatus','Load_Type']]
    y = df['Usage_kWh']
    
    return X, y