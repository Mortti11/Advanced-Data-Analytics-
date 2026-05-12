"""
Sandbox test for Issue #1 (PGCB leakage).
Compares the original 'global preprocessing then TimeSeriesSplit' pipeline
against a leakage-safe variant where interpolation, grouped-median fill,
and IQR clipping are fit on each training fold only.

NO original files are modified.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit

DATA = r"d:\ML_DE\Advanced-Data-Analytics-\Third_assignment\Datasets\PGCB_date_power_demand.xlsx"


def load_raw():
    df = pd.read_excel(DATA)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    # Same drops as the notebook
    suspicious = df.index[df["demand_mw"] > 20000]
    df = df.drop(index=suspicious)
    df = df.drop(columns=["remarks", "nepal", "india_adani", "wind",
                          "generation_mw", "load_shedding"])
    # Same dedup + hourly resample + year filter
    df = df.groupby("datetime").mean().reset_index()
    df = df.set_index("datetime").resample("h").mean()
    df = df[df.index.year >= 2018]
    return df


def add_calendar(df):
    df = df.copy()
    h = df.index.hour
    m = df.index.month
    d = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["dow_sin"] = np.sin(2 * np.pi * d / 7)
    df["dow_cos"] = np.cos(2 * np.pi * d / 7)
    return df


def add_lags(df, target="demand_mw"):
    df = df.copy()
    df["lag_1"] = df[target].shift(1)
    df["lag_24"] = df[target].shift(24)
    df["lag_168"] = df[target].shift(168)
    df["roll_24"] = df[target].shift(1).rolling(24).mean()
    df["diff_1"] = df["lag_1"] - df[target].shift(2)
    return df


# ---------- ORIGINAL (leaky) preprocessing ----------
def preprocess_global(df_raw):
    df = df_raw.copy()
    # global time-interpolation + global hour/month median fill (LEAKY)
    df = df.interpolate(method="time", limit=6)
    fill = df.groupby([df.index.hour, df.index.month]).transform("median")
    df = df.fillna(fill)
    # global IQR clipping on numeric cols except solar (LEAKY)
    cols = [c for c in df.columns if c != "solar"]
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    upper = q3 + 3 * (q3 - q1)
    for c in upper.index:
        if q3[c] > q1[c]:
            df.loc[df[c] > upper[c], c] = np.nan
    df = df.interpolate(method="time", limit=3)
    fill2 = df.groupby([df.index.hour, df.index.month]).transform("median")
    df = df.fillna(fill2)
    return df


# ---------- LEAKAGE-SAFE preprocessing (causal / fold-local) ----------
def preprocess_causal(df_raw):
    """Forward-only fill; fold-local stats applied later inside CV.

    For interpolation we use ffill which is strictly causal. The grouped-median
    fallback and IQR thresholds are computed per training fold by the CV loop.
    """
    df = df_raw.copy()
    # causal forward-fill with a 6h limit (no future leakage)
    df = df.ffill(limit=6)
    return df


def fit_train_only_imputer(df_train):
    """Returns (group_medians, iqr_upper) computed only on training rows."""
    group_medians = (df_train
                     .groupby([df_train.index.hour, df_train.index.month])
                     .median())
    cols = [c for c in df_train.columns if c != "solar"]
    q1 = df_train[cols].quantile(0.25)
    q3 = df_train[cols].quantile(0.75)
    upper = q3 + 3 * (q3 - q1)
    return group_medians, upper, cols


def apply_train_only_imputer(df_part, group_medians, upper, cols):
    df = df_part.copy()
    # fill remaining NaNs using train-only grouped medians
    keys = list(zip(df.index.hour, df.index.month))
    fill_frame = group_medians.reindex(keys).set_index(df.index)
    df = df.fillna(fill_frame)
    # clip with train-only IQR
    for c in upper.index:
        if c in df.columns:
            df.loc[df[c] > upper[c], c] = np.nan
    df = df.fillna(fill_frame)
    return df


def evaluate(df_full, label, n_splits=5):
    df_full = add_calendar(df_full)
    df_full = add_lags(df_full)
    df_full = df_full.iloc[168:]            # drop initial NaNs from lags
    df_full = df_full.dropna(subset=["demand_mw"])
    feats = [c for c in df_full.columns if c != "demand_mw"]
    X = df_full[feats].values
    y = df_full["demand_mw"].values
    tss = TimeSeriesSplit(n_splits=n_splits)
    r2s, maes = [], []
    for tr, te in tss.split(X):
        m = make_pipeline(StandardScaler(), Ridge())
        # final NaN guard (rare): replace with column train medians
        Xtr = pd.DataFrame(X[tr]).fillna(pd.DataFrame(X[tr]).median()).values
        Xte = pd.DataFrame(X[te]).fillna(pd.DataFrame(X[tr]).median()).values
        m.fit(Xtr, y[tr])
        p = m.predict(Xte)
        r2s.append(1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
        maes.append(np.mean(np.abs(y[te] - p)))
    print(f"[{label:35s}] R2={np.mean(r2s):+.3f} ± {np.std(r2s):.3f}  "
          f"MAE={np.mean(maes):6.0f} ± {np.std(maes):.0f}")
    return np.mean(r2s), np.mean(maes)


def evaluate_fold_local(df_raw, label, n_splits=5):
    """Re-fit imputer per fold (truly leakage-safe)."""
    df_raw = df_raw.copy()
    # causal pre-step common to both train and test segments
    df_raw = df_raw.ffill(limit=6)
    # add features that don't depend on future
    df_full = add_calendar(df_raw)
    df_full = add_lags(df_full)
    df_full = df_full.iloc[168:].copy()
    df_full = df_full.dropna(subset=["demand_mw"])  # keep DatetimeIndex
    feats = [c for c in df_full.columns if c != "demand_mw"]
    y = df_full["demand_mw"].values
    tss = TimeSeriesSplit(n_splits=n_splits)
    r2s, maes = [], []
    for tr, te in tss.split(df_full):
        train_part = df_full.iloc[tr][feats].copy()
        test_part = df_full.iloc[te][feats].copy()
        gm, upper, cols = fit_train_only_imputer(train_part)
        train_imp = apply_train_only_imputer(train_part, gm, upper, cols)
        test_imp = apply_train_only_imputer(test_part, gm, upper, cols)
        # any residual NaN -> column median of training fold
        train_med = train_imp.median()
        train_imp = train_imp.fillna(train_med)
        test_imp = test_imp.fillna(train_med)
        m = make_pipeline(StandardScaler(), Ridge())
        m.fit(train_imp.values, y[tr])
        p = m.predict(test_imp.values)
        ytrue = y[te]
        r2s.append(1 - ((ytrue - p) ** 2).sum() / ((ytrue - ytrue.mean()) ** 2).sum())
        maes.append(np.mean(np.abs(ytrue - p)))
    print(f"[{label:35s}] R2={np.mean(r2s):+.3f} ± {np.std(r2s):.3f}  "
          f"MAE={np.mean(maes):6.0f} ± {np.std(maes):.0f}")
    return np.mean(r2s), np.mean(maes)


def main():
    raw = load_raw()
    print(f"Rows after resample/year-filter: {len(raw)}")
    print(f"NaNs after resample (per col):\n{raw.isna().sum().to_string()}\n")

    # 1. Old behavior: global imputation then TimeSeriesSplit
    df_global = preprocess_global(raw)
    print(f"Rows after global preprocessing : {len(df_global)}, "
          f"NaNs left: {int(df_global.isna().sum().sum())}")
    r2_old, mae_old = evaluate(df_global, "OLD (global preprocessing)")

    # 2. Fixed: causal ffill + fold-local imputation
    r2_new, mae_new = evaluate_fold_local(raw, "NEW (fold-local, train-only)")

    print()
    print(f"Delta R2 : {r2_new - r2_old:+.3f}")
    print(f"Delta MAE: {mae_new - mae_old:+.0f} MW")


if __name__ == "__main__":
    main()
