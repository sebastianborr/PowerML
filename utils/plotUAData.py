import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from datetime import datetime

# Cargar los datos
data = pd.read_csv("/mnt/c/Users/Sebas/Desktop/PowerML/data/03_Politecnica4.csv")

# Convertir la columna de fechas a formato datetime
data['time'] = pd.to_datetime(data['time'])

# Pedir al usuario el mes y año que desea visualizar
year_selected = int(input("Ingrese el año que desea visualizar (ej. 2023): "))
month_selected = int(input("Ingrese el mes que desea visualizar (1-12): "))

# Filtrar los datos para el mes y año seleccionados
data_selected = data[(data['time'].dt.year == year_selected) & (data['time'].dt.month == month_selected)]

# Si no hay datos en ese mes/año, mostrar un mensaje y salir
if data_selected.empty:
    print("No hay datos disponibles para el mes y año seleccionados.")
else:
    # Extraer el día de la semana
    data_selected['Weekday'] = data_selected['time'].dt.day_name()
    weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Calcular consumo medio por día de la semana
    dailyConsumption = data_selected.groupby('Weekday')['last'].mean().reindex(weekDays)


    # Extraer mes y año como columnas
    data['Year'] = data['time'].dt.year
    data['Month'] = data['time'].dt.month

    # Calcular consumo medio por mes
    monthlyConsumption = data.groupby('Month')['last'].mean()

    # Graficar consumo medio por día de la semana
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    dailyConsumption.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Consumo Medio por Día de la Semana ({month_selected}/{year_selected})')
    plt.ylabel('Consumo Medio (kWh)')
    plt.xlabel('Día de la Semana')
    plt.xticks(rotation=45)

    # Graficar consumo medio por mes
    plt.subplot(1, 2, 2)
    monthlyConsumption.plot(kind='bar', color='orange', edgecolor='black')
    plt.title('Consumo Medio por Mes')
    plt.ylabel('Consumo Medio (kWh)')
    plt.xlabel('Mes')
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    plt.xticks(ticks=range(1, 13), labels=meses, rotation=0)

    plt.tight_layout()
    plt.show()

    # Extraer información por semana y hora
    data_selected['Week'] = data_selected['time'].dt.isocalendar().week
    data_selected['Hour'] = data_selected['time'].dt.hour
    data_selected['Day'] = data_selected['time'].dt.day
    data_selected['Minute'] = data_selected['time'].dt.minute

    # Filtrar datos de las semanas del mes seleccionado
    weeks_in_selected_month = data_selected['Week'].unique()
    datosXsemana = data_selected[data_selected['Week'].isin(weeks_in_selected_month)]

    # Crear gráfica interactiva de consumo por semana
    fig = px.scatter(
        datosXsemana,
        x='time',
        y='last',
        color='Week',
        hover_data={'time': True, 'last': True, 'Hour': True},
        title=f'Consumo de energía a lo largo de las semanas ({month_selected}/{year_selected})'
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(xaxis_title='Fecha y Hora', yaxis_title='Consumo (kWh)')
    fig.show()

    # Crear heatmap de consumo energético
    datosXsemana['Hour_Minute'] = datosXsemana['Hour'].astype(str).str.zfill(2) + ':' + datosXsemana['Minute'].astype(str).str.zfill(2)
    datosXsemana['Week_Day'] = datosXsemana['Week'].astype(str) + '-' + datosXsemana['Day'].astype(str)
    datosXsemana = datosXsemana.drop_duplicates(subset=['Hour_Minute', 'Week_Day'])

    heatmap_data = datosXsemana.pivot_table(index='Hour_Minute', columns='Week_Day', values='last', aggfunc='mean')

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        linewidths=0.5,
        annot=False,
        cbar_kws={'label': 'Consumo Energético (kWh)'}
    )
    plt.title(f'Heatmap del Consumo Energético ({month_selected}/{year_selected})')
    plt.xlabel('Semana-Día')
    plt.ylabel('Hora y Minuto del Día')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"output_plot_{year_selected}_{month_selected}.png")
