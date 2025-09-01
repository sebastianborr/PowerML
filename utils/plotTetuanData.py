#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tetúan City – Consumo energético por zonas (10-min)
--------------------------------------------------
• Barras semanales y mensuales (3 zonas → 3 subgráficas verticales
  en cada figura, mismo color)
• Serie continua con las 3 zonas superpuestas
• Heat-map de la zona elegida
"""

# ───────────────── BACKEND ───────────────── #
import matplotlib
matplotlib.use("TkAgg")            # 'Agg' si no hay GUI

# ─────────────── LIBRERÍAS ──────────────── #
import os, sys, warnings, pandas as pd, seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import matplotlib.lines as mlines

sns.set_style("whitegrid")
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

# ───────── CONFIGURACIÓN ───────── #
DATA_FILE = "data/02_Tetuan_City_power_consumption.csv"
ZONE_COLS = {
    "zone1": "zone 1 power consumption",
    "zone2": "zone 2  power consumption",
    "zone3": "zone 3  power consumption",
}
ZONE_FOR_HEATMAP = "zone3"     # 'zone1' | 'zone2' | 'zone3'
START_DATE = "2017-02-01"
END_DATE   = "2017-02-28"      # inclusive

# ─────── FUNCIONES AUXILIARES ─────── #
def linecollection_fixed(df, xcol, ycol, color):
    x = mdates.date2num(df[xcol]); y = df[ycol].values
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    return LineCollection(segs, colors=color, linewidths=.9)

def plot_series_multi(df, date_col, zone_cols, t0, t1):
    cmap_zone = {"zone1": "tab:blue", "zone2": "tab:green", "zone3": "tab:orange"}
    fig, ax = plt.subplots(figsize=(14, 5))
    for key, zcol in zone_cols.items():
        ax.add_collection(linecollection_fixed(df, date_col, zcol, cmap_zone[key]))
    ax.autoscale()
    for d in pd.date_range(t0, t1, freq="D"):
        if d.weekday() >= 5:
            ax.axvspan(d, d + pd.Timedelta(days=1), color="red", alpha=.08, zorder=0)
    ax.set_ylabel("kWh"); ax.set_xlabel("Date – Time")
    ax.set_title(f"Continuous Series {t0.date()} → {t1.date()}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    handles = [
        mlines.Line2D([], [], color="tab:blue",   label="Zone 1"),
        mlines.Line2D([], [], color="tab:green",  label="Zone 2"),
        mlines.Line2D([], [], color="tab:orange", label="Zone 3"),
        Patch(facecolor="red", alpha=.15, label="Weekend")
    ]
    ax.legend(handles=handles, ncol=4)   # ← añade handles=
    plt.tight_layout(); plt.show()

def heatmap_zone(df, date_col, value_col, title):
    df = df.copy()
    df["hour_min"] = df[date_col].dt.strftime("%H:%M")
    df["week_day"] = (df[date_col].dt.isocalendar().week.astype(int).astype(str)
                      + "-" + df[date_col].dt.day.astype(str))
    df = df.drop_duplicates(subset=["hour_min", "week_day"])
    horas = pd.date_range("00:00", "23:50", freq="10min").strftime("%H:%M")
    dias  = sorted(df["week_day"].unique(),
                   key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1])))
    tabla = (df.pivot_table(index="hour_min", columns="week_day",
                            values=value_col, aggfunc="mean")
               .reindex(index=horas, columns=dias))
    plt.figure(figsize=(15, 7))
    ax = sns.heatmap(tabla, cmap="YlOrRd", linewidths=.4,
                     cbar_kws={"label": "kWh"})
    ax.set_title(title); ax.set_xlabel("Week-Day"); ax.set_ylabel("Hour-Minute")
    ax.invert_yaxis()
    for lab in ax.get_xticklabels():
        col = lab.get_text()
        fecha = df.loc[df["week_day"] == col, date_col].iloc[0]
        if fecha.weekday() >= 5:
            lab.set_color("red")
    plt.tight_layout(); plt.show()

def bar_subplots(df, group_col, title, xlabels, ylabel,y):
    """Tres subplots verticales (uno por zona) con barras azules."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    color = ["tab:blue","tab:green","tab:orange"]
    if y:
        xlabels_2 = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
    else:
        xlabels_2 = xlabels
    ii=0
    for ax, (zkey, zcol) in zip(axes, ZONE_COLS.items()):
        (df.groupby(group_col)[zcol].mean()
           .reindex(xlabels)
           .plot(kind="bar", ax=ax, color=color[ii], edgecolor="black", width=0.9))
        ax.set_xticklabels(xlabels_2, rotation=45)
        ax.set_title(f"{zkey.title()}"); ax.set_xlabel(""); ax.set_ylabel(ylabel)
        if group_col == "weekday": ax.tick_params(axis="x", rotation=45)
        ii += 1
    fig.suptitle(title, fontsize=14, y=.93)
    plt.tight_layout(); plt.show()

# ──────────── CARGA CSV ─────────── #
if not os.path.exists(DATA_FILE):
    sys.exit("❌ CSV no encontrado")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()
df["datetime"] = pd.to_datetime(df["datetime"])

# ── 0. SUBFIGURAS DE PROMEDIOS ── #
week_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
month_order = list(range(1, 13))
df["weekday"] = df["datetime"].dt.day_name()
df["month"]   = df["datetime"].dt.month

bar_subplots(df, "weekday",
             title="Average Daily Consumption (All Zones)",
             xlabels=week_order, ylabel="kWh", y=False)

bar_subplots(df, "month",
             title="",
             xlabels=month_order, ylabel="kWh", y=True)

# ── RANGO DE FECHAS PARA SERIE Y HEATMAP ── #
t0 = pd.to_datetime(START_DATE); t1 = pd.to_datetime(END_DATE)
mask = (df["datetime"] >= t0) & (df["datetime"] < (t1 + pd.Timedelta(days=1)))
df_rng = df.loc[mask].copy()
if df_rng.empty: sys.exit("❗ Sin datos en ese rango.")

# ── SERIE CONTINUA (3 zonas) ── #
plot_series_multi(df_rng, "datetime", ZONE_COLS, t0, t1)

# ── HEAT-MAP ZONA ELEGIDA ── #
heatmap_zone(
    df_rng[["datetime", ZONE_COLS[ZONE_FOR_HEATMAP]]]
        .rename(columns={ZONE_COLS[ZONE_FOR_HEATMAP]: "value"}),
    date_col="datetime",
    value_col="value",
    title=f"Heat-map {ZONE_FOR_HEATMAP.title()} {t0.date()} → {t1.date()}"
)
