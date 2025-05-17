import requests
import pandas as pd
from datetime import datetime, timedelta

# Función para realizar la consulta
def get_data_for_time_interval(start_time, end_time, uid, time_interval):
    url = f"https://openapi.kunna.es/data/operation/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3Mzk1MTkzNzJ9.crW_mRPbpF7qJnuUtEdsSs4Rf-grJOKnoOVqpndLOwE/time_start/{start_time}/time_end/{end_time}/last/uid/{uid}/name/{time_interval}"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error en la solicitud: {response.status_code}")
        return None
    data = response.json()
    
    if data["result"]["values"] == [[]]:
        print(f"No hay datos para el intervalo de tiempo {start_time} - {end_time}")
        return None
    print(f"Datos obtenidos para el intervalo de tiempo {start_time} - {end_time}")
   
    if "result" in data and data["result"] and "values" in data["result"]:
        return data["result"]["values"]
    else:
        return None

# Función para obtener el salto temporal dependiendo del intervalo seleccionado
def get_time_delta(time_interval, current_time):
    if time_interval == "15m":
        return timedelta(minutes=15)
    elif time_interval == "1h":
        return timedelta(hours=1)
    elif time_interval == "1d":
        return timedelta(days=1)
    elif time_interval == "1M":
        # Para "1M", avanzamos al siguiente día 1 del mes siguiente
        next_month = current_time.replace(day=28) + timedelta(days=4)  # Garantiza pasar al siguiente mes
        return timedelta(days=(next_month.replace(day=1) - current_time).days)
    else:
        raise ValueError("Intervalo de tiempo no válido")

# Función para recorrer el calendario y obtener los datos
def fetch_and_store_data(start_date, end_date, uid, time_interval, filename="output.csv"):
    # Convertir las fechas a objetos datetime para manipulación
    start_datetime = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
    
    # Lista para almacenar los resultados
    all_data = []

    # Obtener el salto temporal basado en el intervalo seleccionado
    current_time = start_datetime
    while current_time <= end_datetime:
        # Convertir la fecha a formato de string que requiere la API
        time_start = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        time_end = (current_time + get_time_delta(time_interval, current_time)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Obtener los datos para el intervalo de tiempo actual
        data = get_data_for_time_interval(time_start, time_end, uid, time_interval)
        
        # Si los datos existen, añadirlos a la lista
        if data:
            for entry in data:
                # Obtener el día de la semana en inglés (adecuandome a otros datasets)
                date_object = datetime.strptime(entry[0], "%Y-%m-%dT%H:%M:%SZ")
                weekday = date_object.strftime("%A")  # Día de la semana en inglés
                
                all_data.append({
                    "time": entry[0],  # Fecha y hora
                    "weekday": weekday,  # Día de la semana en inglés
                    "uid": entry[1],  # UID del sensor
                    "last": entry[2]  # Último valor registrado
                })
        
        # Avanzar al siguiente intervalo de tiempo según el tipo seleccionado
        current_time += get_time_delta(time_interval, current_time)
    
    # Crear un DataFrame y guardarlo como CSV
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"Datos guardados en {filename}")

# Parámetros de fecha, UID y tipo de intervalo
start_date = "2022-03-01T00:00:00Z"
end_date = "2025-03-01T23:59:00Z"
uid = "MLU00390001"
time_interval = "15m"  # "15m", "1h", "1d", "1M"
fname ="/mnt/c/Users/Sebas/Desktop/PowerML/output/Politecnica.csv"

# Ejecutar el proceso y almacenar los datos en CSV
fetch_and_store_data(start_date, end_date, uid, time_interval, fname)
