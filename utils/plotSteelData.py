#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steel Industry – Consumo energético
----------------------------------
• Barras anuales (día-semana y mes completo)
• Serie continua + Heat-map para un rango de fechas
  – Tramos de sáb-dom en rojo dentro de la MISMA línea
"""

# ───────────────── BACKEND ────────────────── #
import matplotlib
matplotlib.use('TkAgg')                 # pon 'Agg' si no hay GUI

# ─────────────── LIBRERÍAS ────────────────── #
import os, sys, warnings, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
import numpy as np

sns.set_style('whitegrid')
warnings.simplefilter('ignore', category=pd.errors.SettingWithCopyWarning)

# ───────────── FUNCIONES ───────────── #
def plot_series(df, date_col, kwh_col, t0, t1, title=None):
    """
    Serie continua (una sola línea) con segmentos:
       • Laborable  → azul
       • Fin de semana → rojo
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    df['is_weekend'] = df[date_col].dt.weekday >= 5

    # preparar segmentos para LineCollection
    x = mdates.date2num(df[date_col])
    y = df[kwh_col].values
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # color de cada segmento según si el origen O el destino es weekend
    color_seg = np.where(
        df['is_weekend'][:-1].values | df['is_weekend'][1:].values,
        'red', 'tab:blue'
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    lc = LineCollection(segments, colors=color_seg, linewidths=.9)
    ax.add_collection(lc)
    ax.autoscale()                      # ajusta límites

    # sombrear fines de semana (opcional)
    for day in pd.date_range(t0.normalize(), t1.normalize(), freq='D'):
        if day.weekday() >= 5:
            ax.axvspan(day, day + pd.Timedelta(days=1),
                       color='red', alpha=0.08, zorder=0)

    ax.set_xlabel('Date - Time')
    ax.set_title(title or '')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))

    # leyenda manual
    h_lab = mlines.Line2D([], [], color='tab:blue', label='Weekday')
    h_wkd = mlines.Line2D([], [], color='red',      label='Weekend')
    ax.legend(handles=[h_lab, h_wkd])
    plt.tight_layout(); plt.show()

def plot_heatmap(df, hcol, dcol, vcol, title):
    """
    Heat-map Hora × (Semana-día) con etiquetas de sáb-dom en rojo.
    """
    horas = pd.date_range('00:00', '23:45', freq='15min').strftime('%H:%M')
    dias  = sorted(df[dcol].unique(),
                   key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    tabla = (df.pivot_table(index=hcol, columns=dcol, values=vcol, aggfunc='mean')
               .reindex(index=horas, columns=dias))

    plt.figure(figsize=(15, 7))
    ax = sns.heatmap(tabla, cmap='YlOrRd', linewidths=.4,
                     cbar_kws={'label': 'kWh'})
    ax.set_title(title); ax.set_xlabel('Week-Day'); ax.set_ylabel('Hour-Minute')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.invert_yaxis()

    # colorear etiquetas de fin de semana
    for lab in ax.get_xticklabels():
        col = lab.get_text()
        fecha = df.loc[df[dcol] == col, 'date'].iloc[0]
        if pd.to_datetime(fecha).weekday() >= 5:
            lab.set_color('red')

    plt.tight_layout(); plt.show()


# ───────────── CARGA CSV ───────────── #
FILE = 'data/00_Steel_industry_data.csv'
if not os.path.exists(FILE):
    sys.exit('CSV no encontrado')

steel = pd.read_csv(FILE)
steel.columns = steel.columns.str.strip().str.lower()
kwh = 'usage_kwh'
steel['date'] = pd.to_datetime(steel['date'], dayfirst=True)

# ── 1. BARRAS ANUALES ── #
week_order = ['Monday','Tuesday','Wednesday','Thursday',
              'Friday','Saturday','Sunday']
month_es   = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']

steel['weekday'] = steel['date'].dt.day_name()
steel['month']   = steel['date'].dt.month

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
steel.groupby('weekday')[kwh].mean().reindex(week_order)\
     .plot.bar(ax=ax1, color='skyblue', edgecolor='black')
ax1.set_title('Average Daily Consumption'); ax1.set_xlabel(''); ax1.set_ylabel('kWh')
ax1.tick_params(axis='x', rotation=45)

ax2.bar(steel.groupby('month')[kwh].mean().index,
        steel.groupby('month')[kwh].mean().values,
        color='orange', edgecolor='black')
ax2.set_xticks(range(1, 13)); ax2.set_xticklabels(month_es)
ax2.set_title('Average consumption per month'); ax2.set_xlabel('Month'); ax2.set_ylabel('kWh')
plt.tight_layout(); plt.show()

# ── 2. RANGO DE FECHAS ── #
start_date = pd.to_datetime('2018-04-01')
end_date   = pd.to_datetime('2018-04-30')        # inclusive
end_limit  = end_date + pd.Timedelta(days=1)

rng = steel[(steel['date'] >= start_date) & (steel['date'] < end_limit)]
if rng.empty:
    sys.exit('Sin datos en ese rango')

# ── 3. SERIE CONTINUA ── #
plot_series(rng, 'date', kwh,
            start_date, end_date,
            title=f'Continuous Series {start_date.date()} → {end_date.date()}')

# ── 4. HEAT-MAP ── #
rng['hour_min'] = rng['date'].dt.strftime('%H:%M')
rng['week_day'] = (
        rng['date'].dt.isocalendar().week.astype(int).astype(str)
        + '-' +
        rng['date'].dt.day.astype(str)
)
rng = rng.drop_duplicates(subset=['hour_min', 'week_day'])

plot_heatmap(rng, 'hour_min', 'week_day', kwh,
             title=f'Heat-map {start_date.date()} → {end_date.date()}')
