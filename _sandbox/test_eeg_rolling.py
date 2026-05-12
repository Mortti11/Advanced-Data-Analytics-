"""
Sandbox test for Issue #2 (EEG rolling features after row removal).
Compares macro-F1 for three pipelines:
  (A) OLD : drop anomalies, then rolling(64) on the squeezed frame.
  (B) FIX1: rolling(64) computed BEFORE anomaly removal, then drop rows.
  (C) FIX2: keep anomaly rows but set their channel values to NaN, so rolling
            sees gaps; min_periods=1 keeps stats valid.
The four hard-coded anomaly rows are the same as the notebook (>8000 or <1000).
We skip the MNE epoch-based dropping for speed; the principle is identical.

NO original files are modified.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

DATA = r"d:\ML_DE\Advanced-Data-Analytics-\Third_assignment\Datasets\EEG_Eye+State.csv"


def load():
    data, _ = arff.loadarff(DATA)
    df = pd.DataFrame(data)
    df["eyeDetection"] = df["eyeDetection"].astype(int)
    return df


def signal_cols(df):
    return [c for c in df.columns if c != "eyeDetection"]


def rolling_feats(df, cols, window=64):
    rm = df[cols].rolling(window, min_periods=1).mean()
    rs = df[cols].rolling(window, min_periods=1).std(ddof=0).fillna(0)
    rm.columns = [f"{c}_mean" for c in cols]
    rs.columns = [f"{c}_std" for c in cols]
    return pd.concat([rm, rs], axis=1)


def macro_f1(X, y, split=0.8, label=""):
    cut = int(split * len(X))
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y.iloc[:cut], y.iloc[cut:]
    clf = RandomForestClassifier(n_estimators=200, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    f1 = f1_score(yte, pred, average="macro")
    print(f"[{label:50s}] rows={len(X):5d}  macro-F1={f1:.3f}")
    return f1


def main():
    df = load()
    cols = signal_cols(df)
    anomaly = (df[cols] > 8000).any(axis=1) | (df[cols] < 1000).any(axis=1)
    print(f"Total rows: {len(df)}  Anomaly rows: {int(anomaly.sum())}")

    # (A) OLD: drop first, then roll
    df_a = df.loc[~anomaly].reset_index(drop=True)
    feats_a = rolling_feats(df_a, cols)
    f1_a = macro_f1(feats_a, df_a["eyeDetection"], label="OLD: drop -> rolling(64)")

    # (B) FIX1: roll first, then drop
    feats_b_full = rolling_feats(df, cols)
    feats_b = feats_b_full.loc[~anomaly].reset_index(drop=True)
    y_b = df.loc[~anomaly, "eyeDetection"].reset_index(drop=True)
    f1_b = macro_f1(feats_b, y_b, label="FIX1: rolling(64) on full, then drop")

    # (C) FIX2: NaN out anomaly rows, then roll (min_periods=1 keeps stats)
    df_c = df.copy()
    df_c.loc[anomaly, cols] = np.nan
    feats_c = rolling_feats(df_c, cols).fillna(method="ffill").fillna(method="bfill")
    # also drop anomaly rows for the model (they have no valid label/feature pair)
    feats_c = feats_c.loc[~anomaly].reset_index(drop=True)
    y_c = df.loc[~anomaly, "eyeDetection"].reset_index(drop=True)
    f1_c = macro_f1(feats_c, y_c, label="FIX2: NaN-out anomalies, rolling(64)")

    print()
    print(f"Delta F1 (FIX1 - OLD): {f1_b - f1_a:+.4f}")
    print(f"Delta F1 (FIX2 - OLD): {f1_c - f1_a:+.4f}")


if __name__ == "__main__":
    main()
