import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Cambia el backend para evitar errores
import plotly.express as px
import seaborn as sns # Importar seaborn para gráficos más atractivos

# Cargo los datos desde el archivo CSV
data = pd.read_csv('/mnt/c/Users/Sebas/Desktop/PowerML/data/00_Steel_industry_data.csv')

# Primeras filas de los datos
#  date  Usage_kWh  Lagging_Current_Reactive.Power_kVarh  Leading_Current_Reactive_Power_kVarh  CO2(tCO2)  Lagging_Current_Power_Factor  Leading_Current_Power_Factor   NSM WeekStatus Day_of_week   Load_Type
print(data.head())
# Voy a extraer el número de días de la semana ese año
dias_semana = data['Day_of_week'].value_counts()
# Ordenar los días de la semana
weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_ordenados = dias_semana.reindex(weekDays, fill_value=0)
# Mostrar resultados
print("Número de días de la semana:")
print(dias_ordenados/4/24) # Divido por 4 porque hay 4 registros por hora y por 24 porque hay 24 horas en un día

# -------- o -------- #

# Ploteo de los datos promediados

# Convierto la columna de fechas a datetime. El formato de la fecha es Día / mes / año
data['date'] = pd.to_datetime(data['date'],dayfirst=True)

# Cojo todos los meses
months = np.arange(12)+1
data_jan_to_dec = data[data['date'].dt.month.isin(months)]
# Columnas auxiliares
data_jan_to_dec['Month'] = data_jan_to_dec['date'].dt.month
data_jan_to_dec['Weekday'] = data_jan_to_dec['date'].dt.day_name()



# Calcular consumo medio por día de la semana
dailyConsumption = data_jan_to_dec.groupby('Weekday')['Usage_kWh'].mean().reindex(weekDays)

# Calculo del consumo medio por mes
consumo_por_mes = data_jan_to_dec.groupby('Month')['Usage_kWh'].mean()

# Graficar consumo medio por día de la semana
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
dailyConsumption.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Consumo Medio por Día de la Semana (Enero-Diciembre)')
plt.ylabel('Consumo Medio (kWh)')
plt.xlabel('Día de la Semana')
plt.xticks(rotation=45)

# Graficar consumo medio por mes
plt.subplot(1, 2, 2)
consumo_por_mes.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Consumo Medio por Mes')
plt.ylabel('Consumo Medio (kWh)')
plt.xlabel('Mes')
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
plt.xticks(ticks=range(len(meses)), labels =meses, rotation=0)

# Ajustar el diseño y mostrar
plt.tight_layout()
plt.show(block=True)
plt.show()
  

#-
# Ahora saco la info por semana y hora
data['Week'] = data['date'].dt.isocalendar().week  # Agregar columna de semana
data['Hour'] = data['date'].dt.hour  # Agregar columna de hora
data['Day'] = data['date'].dt.day
data['Minute'] = data['date'].dt.minute

# Filtrar datos de las semanas de interés (ejemplo: semanas 10 a 15)
semanas_interes = range(10, 16)
datosXsemana = data[data['Week'].isin(semanas_interes)]

# Crear la gráfica interactiva
fig = px.scatter(
    datosXsemana, 
    x='date', 
    y='Usage_kWh', 
    color='Week',  # Diferenciar por semana
    hover_data={'date': True, 'Usage_kWh': True, 'Hour': True},  # Información al pasar el cursor
    title='Consumo de energía a lo largo de semanas seleccionadas'
)

# Personalización opcional
fig.update_traces(marker=dict(size=8, opacity=0.7))  # Tamaño y transparencia de los puntos
fig.update_layout(xaxis_title='Fecha y Hora', yaxis_title='Consumo (kWh)')
# Mostrar la gráfica
fig.show()



#-
# Filtrar las semanas de interés

# 'Hour' y 'Minute' enteras o strings
datosXsemana['Hour_Minute'] = datosXsemana['Hour'].astype(str).str.zfill(2) + ':' + datosXsemana['Minute'].astype(str).str.zfill(2)
datosXsemana['Day'] = datosXsemana['Day'].astype(str).str.zfill(2)
datosXsemana['Week_Day'] = datosXsemana['Week'].astype(str) + '-' + datosXsemana['Day']
#datosXsemana['Week_Day'] = datosXsemana['Week'].astype(str) + '-' + datosXsemana['Day'].astype(str)
print(datosXsemana.head())

# VISUALIAZACIÓN
# Pivotear los datos crudos
heatmap_data = datosXsemana.pivot(index='Hour_Minute', columns='Week_Day', values='Usage_kWh')

plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    linewidths=0.5,
    annot=False,
    cbar_kws={'label': 'Consumo Energético (kWh)'}
)
plt.title('Heatmap del Consumo Energético por 15 minutos')
plt.xlabel('Semana-Día')
plt.ylabel('Hora y Minuto del Día')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show(block=True)
plt.savefig("output_plot.png") 