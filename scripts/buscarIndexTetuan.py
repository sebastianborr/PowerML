# -------------------------------------------
# util_fecha_a_indice.py
# -------------------------------------------
import datetime as dt

# nº de filas por un día completo (24 h · 6 muestras/h)
PASOS_POR_DIA = 24 * 6          # = 144

INICIO_DATASET = dt.datetime(2017, 1, 1, 0, 0)   # 01-01-2017 00:00
FIN_DATASET    = dt.datetime(2017,12,30,23,50)   # 30-12-2017 23:50


def fila_para_fecha(dia: int, mes: int, anio: int) -> int:
    """
    Devuelve el índice (0-based) de la fila que corresponde a las 00:00
    de la fecha indicada dentro del CSV de Tetuan 2017.

    Si la fecha está fuera del rango cubierto, lanza ValueError.
    """
    fecha = dt.datetime(anio, mes, dia)  # 00:00 por defecto

    if fecha < INICIO_DATASET or fecha > FIN_DATASET.replace(hour=0, minute=0):
        raise ValueError("La fecha no existe en el dataset.")

    # días transcurridos desde el 01-01-2017
    dias_transcurridos = (fecha - INICIO_DATASET).days

    # índice de la primera fila de ese día
    indice = dias_transcurridos * PASOS_POR_DIA
    return indice   # 0-based (sin contar la cabecera)

# ------------------ ejemplos ------------------