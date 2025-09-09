#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# ---------- Configuración de alias de columnas ----------

ALIASES = {
    # Identidad / descripción
    "model": ["model", "Model", "estimator", "Estimator", "algorithm", "Algorithm"],
    "params": ["Params", "parameters", "Parameters", "param_str", "Param_String"],

    # Hiperparámetros frecuentes en NN
    "num_layers": ["num_layers", "layers", "n_layers"],
    "dropout": ["dropout", "drop_out"],
    "epochs": ["epochs", "n_epochs"],
    "batch_size": ["batch_size", "bs"],

    # BiLSTM específicos
    "units_lstm1": ["units_lstm1"],
    "units_lstm2": ["units_lstm2"],
    "dropout_rate": ["dropout_rate", "dropoutRate"],
    "feature_att": ["feature_att"],
    "temporal_att": ["temporal_att"],

    # Transformers específicos
    "architecture": ["architecture", "arch"],
    "d_model": ["d_model"],
    "num_heads": ["num_heads", "heads"],
    "ff_dim": ["ff_dim"],

    # Métricas
    "mse": ["mse", "MSE"],
    "rmse": ["rmse", "RMSE"],
    "mae": ["mae", "MAE"],
    "mape": ["mape", "MAPE"],
    "r2": ["r2", "R2", "r^2", "R^2", "r2_score", "R2_Score"],
    "cvrmse": ["cvrmse", "CVRMSE", "cvrmse_percent", "CV(RMSE)"],

    # Tiempo y overfitting
    "training_time_s": [
        "Training Time (s)", "Training Time(s)", "training_time_s", "fit_time",
        "Training_Time_s", "Training_Time_(s)", "training_time_sec", "time_train_s"
    ],
    "overfit_gap": ["Overfit Gap", "Overfit_Gap", "overfit_gap"],
    "loss_ratio": ["Loss Ratio", "Loss_Ratio", "loss_ratio"],

    # Campos que se deben excluir si aparecen
    "train_loss_final": ["Train Loss Final", "Train_Loss_Final", "train_loss_final"],
    "val_loss_final": ["Validation Loss Final", "Val_Loss_Final", "Validation_Loss_Final", "val_loss_final"],
}

# ---------- Utilidades ----------

LATEX_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # ¡No escapamos la barra invertida para permitir \newline en celdas!
    # "\\": r"\textbackslash{}",
}

def latex_escape(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    for ch, rep in LATEX_SPECIALS.items():
        s = s.replace(ch, rep)
    return s

def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map: Dict[str, str] = {}
    for canonical, cand in ALIASES.items():
        found = first_present(df, cand)
        if found is not None:
            col_map[found] = canonical
    return df.rename(columns=col_map)

def detect_architecture(df_norm: pd.DataFrame, file_stem: str) -> str:
    cols = set(df_norm.columns)
    # Detectamos por columnas primero
    if {"architecture", "d_model", "num_heads"} & cols:
        return "Transformers"
    if {"units_lstm1", "units_lstm2"} & cols:
        return "BiLSTM"
    if "model" in cols and "params" in cols:
        return "Machine learning"

    # Fallback por nombre de archivo
    s = file_stem.lower()
    if "lstmcnn" in s:
        return "LSTMCNN"
    if "ccnlstm" in s:
        return "CCNLSTM"
    if "bilstm" in s:
        return "BiLSTM"
    if "transformer" in s:
        return "Transformers"
    if "stacked" in s or "stackedlstm" in s:
        return "Stacked LSTM"
    return "Stacked LSTM"

HP_KEYS_BY_ARCH = {
    "Stacked LSTM": [
        ("num_layers", "num_layers"), ("dropout", "dropout"),
        ("epochs", "epochs"), ("batch_size", "batch_size")
    ],
    "LSTMCNN": [
        ("num_layers", "num_layers"), ("dropout", "dropout"),
        ("epochs", "epochs"), ("batch_size", "batch_size")
    ],
    "CCNLSTM": [
        ("num_layers", "num_layers"), ("dropout", "dropout"),
        ("epochs", "epochs"), ("batch_size", "batch_size")
    ],
    "BiLSTM": [
        ("units_lstm1", "units_lstm1"), ("units_lstm2", "units_lstm2"),
        ("dropout_rate", "dropout_rate"), ("feature_att", "feature_att"),
        ("temporal_att", "temporal_att"), ("epochs", "epochs"),
        ("batch_size", "batch_size")
    ],
    "Transformers": [
        ("architecture", "architecture"), ("d_model", "d_model"),
        ("num_heads", "num_heads"), ("ff_dim", "ff_dim"),
        ("num_layers", "num_layers"), ("dropout_rate", "dropout_rate"),
        ("epochs", "epochs"), ("batch_size", "batch_size")
    ],
    "Machine learning": []  # Usa Model + Params
}

def guess_model_group(df_norm: pd.DataFrame, file_stem: str) -> str:
    if "model" in df_norm.columns:
        vals = [str(v) for v in df_norm["model"].dropna().astype(str)]
        if vals:
            moda, _ = Counter(vals).most_common(1)[0]
            return moda
    return file_stem


def params_to_items(ps: str) -> List[str]:
    s = str(ps).strip()
    # normaliza comillas “ ” ’ -> " '
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    # intenta parsear como dict de Python
    try:
        import ast
        d = ast.literal_eval(s)
    except Exception:
        d = None

    items: List[str] = []
    if isinstance(d, dict):
        for k, v in d.items():
            v_str = str(v).strip().strip('"').strip("'")
            items.append(f"{k}={v_str}")
        return items

    # fallback: quitar llaves y trocear manualmente
    s = s.strip("{} \n")
    tokens = re.split(r",\s*|\n+", s)
    for t in tokens:
        if not t:
            continue
        if ":" in t:
            k, v = t.split(":", 1)
            k = k.strip().strip("'").strip('"')
            v = v.strip().strip("'").strip('"')
            items.append(f"{k}={v}")
        else:
            items.append(t.strip().strip("'").strip('"'))
    return items

def pretty_params_from_row(row: pd.Series, arch_name: str) -> str:
    """
    Construye 'Model and Config.':
    - Si hay 'params', los usa (troceados por comas), anteponiendo 'model' si existe.
    - Si no, lista hiperparámetros según la arquitectura; prefija con el 'model' si existe,
      en su defecto con el nombre de la arquitectura.
    """
    parts: List[str] = []

    model_name = str(row.get("model", "")).strip()
    has_model = bool(model_name) and model_name.lower() != "nan"

    # Caso con Params (típico Machine learning)
    params_str = row.get("params", None)
    params_str = row.get("params", None)
    if pd.notna(params_str):
        ps = str(params_str).strip()
        if ps and ps.lower() != "nan":
            items = [latex_escape(it) for it in params_to_items(ps)]
            if has_model:
                return f"{latex_escape(model_name)}:\\newline " + ",\\newline ".join(items)
            else:
                return ",\\newline ".join(items)

    # Sin Params: armamos desde hp por arquitectura
    hp_keys = HP_KEYS_BY_ARCH.get(arch_name, [
        ("num_layers", "num_layers"), ("dropout", "dropout"),
        ("epochs", "epochs"), ("batch_size", "batch_size")
    ])

    hp_items = []
    for key, label in hp_keys:
        if key in row.index and pd.notna(row.get(key, None)):
            hp_items.append(f"{label}={row[key]}")

    if hp_items:
        hp_items = [latex_escape(x) for x in hp_items]
        prefix = latex_escape(model_name) if has_model else latex_escape(arch_name)
        # return (prefix + ":\\newline " if prefix else "") + ",\\newline ".join(hp_items)
        return ",\\newline ".join(hp_items)


    # Fallback
    if has_model:
        return latex_escape(model_name)
    return latex_escape(arch_name)

def format_numeric(x):
    """Por defecto, muestra hasta 2 decimales (sin ceros ni punto finales)."""
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    try:
        xf = float(x)
    except Exception:
        return latex_escape(str(x))
    s = f"{xf:.2f}".rstrip("0").rstrip(".")
    return s

def build_table_block(df_norm: pd.DataFrame, arch_name: str, label_slug: str) -> str:
    """Tabla multipágina con cabecera repetida y anchos ajustados, caption por arquitectura."""

    need_map = {
        "MSE": "mse",
        "RMSE": "rmse",
        "MAE": "mae",
        "MAPE": "mape",
        r"R\textsuperscript{2}": "r2",
        "CVRMSE": "cvrmse",
        "Training Time (s)": "training_time_s",
        "Overfit Gap": "overfit_gap",
        "Loss Ratio": "loss_ratio",
    }

    rows = []
    for idx, row in df_norm.iterrows():
        rid = idx + 1
        mac = pretty_params_from_row(row, arch_name) or "—"

        # 2 decimales en general; Overfit Gap exactamente 6
        vals = {
            out: (
                format_numeric(row.get(key, ""))
                if out != "Overfit Gap"
                else ("" if (pd.isna(row.get(key, "")) or str(row.get(key, "")).strip() == "")
                      else f"{float(row.get(key, '')):.6f}")
            )
            for out, key in need_map.items()
        }

        rows.append([str(rid), mac] + [vals[k] for k in [
            "MSE", "RMSE", "MAE", "MAPE", r"R\textsuperscript{2}",
            "CVRMSE", "Training Time (s)", "Overfit Gap", "Loss Ratio"
        ]])

    body_lines = []
    for r in rows:
        body_lines.append(" & ".join([r[0], r[1]] + r[2:]) + r" \\")

    caption = f"Results of the ablation study of the {latex_escape(arch_name)} architecture."
    label = f"tab:{label_slug}"

    # 🔧 Claves para que quepa: letra pequeña + tabcolsep bajo + columnas estrechas.
    header_block = r"""
\begingroup
\scriptsize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{0.55}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}m{#1}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash\tiny}m{#1}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash\tiny\setlength{\baselineskip}{0.4\baselineskip}}m{#1}}
\captionsetup{justification=centering,singlelinecheck=false,margin=0pt}

\begin{longtable}{C{0.8cm} L{2cm} C{1.15cm} C{1.15cm} C{1.15cm} C{1.15cm} C{0.85cm} C{1.25cm} C{1.2cm} C{1.5cm} C{1.0cm}}
\caption{""" + caption + r"""}\label{""" + label + r"""}\\
\toprule
\textbf{ID} & \textbf{Config.} & \textbf{MSE} & \textbf{RMSE} & \textbf{MAE} & \textbf{MAPE} & \textbf{R\textsuperscript{2}} & \textbf{CVRMSE} & \textbf{Training\newline Time (s)} & \textbf{Overfit\newline Gap} & \textbf{Loss\newline Ratio} \\
\midrule
\endfirsthead

\caption[]{""" + caption + r"""}\\
\toprule
\textbf{ID} & \textbf{Config.} & \textbf{MSE} & \textbf{RMSE} & \textbf{MAE} & \textbf{MAPE} & \textbf{R\textsuperscript{2}} & \textbf{CVRMSE} & \textbf{Training\newline Time (s)} & \textbf{Overfit\newline Gap} & \textbf{Loss\newline Ratio} \\
\midrule
\endhead

\midrule
\endfoot

\bottomrule
\endlastfoot
""".lstrip("\n")

    footer_block = r"""
\end{longtable}
\endgroup
""".lstrip("\n")

    return header_block + "\n".join(body_lines) + "\n" + footer_block

def process_csv(path: str) -> str:
    df = pd.read_csv(path)
    df = normalize_columns(df)

    # Excluir columnas no deseadas si están (ya normalizadas a nombres canónicos)
    for ex in ("train_loss_final", "val_loss_final"):
        if ex in df.columns:
            df = df.drop(columns=[ex])

    df = df.reset_index(drop=True)

    stem = os.path.splitext(os.path.basename(path))[0]
    arch_name = detect_architecture(df, stem)
    label_slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem.lower()).strip("_")

    return build_table_block(df, arch_name=arch_name, label_slug=label_slug)

def main():
    ap = argparse.ArgumentParser(description="Genera tablas LaTeX (longtable) por cada CSV con métricas de modelos.")
    ap.add_argument("--input-dir", type=str, default="./output/ResultsML", help="Directorio con .csv")
    ap.add_argument("--output", type=str, default="AnexoML.tex", help="Fichero .tex de salida")
    args = ap.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_paths:
        raise SystemExit(f"No se encontraron .csv en: {args.input_dir}")

    blocks = []
    for p in csv_paths:
        try:
            blocks.append(process_csv(p))
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo procesar {p}: {e}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("% --- Archivo generado automáticamente con script de python ---\n\n \chapter*{Anexo}\n\n")
        for b in blocks:
            f.write(b)
            f.write("\n\n")

    print(f"Generado: {args.output} (tablas: {len(blocks)})")

if __name__ == "__main__":
    main()
