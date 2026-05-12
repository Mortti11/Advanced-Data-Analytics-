"""
Sandbox test for Issue #3 (Steel regression: missing final test evaluation).
Reproduces the notebook's preprocessing minimally and reports MAE/RMSE/R2 on
the same untouched temporal test set for:
  (a) baseline-train (raw, only encoded + CO2 dropped)
  (b) optimized-train (after IQR ∩ IsolationForest outlier removal)
CTGAN augmentation is intentionally skipped to keep the test cheap; we add a
note about how to plug it in.

NO original files are modified.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA = r"d:\ML_DE\Advanced-Data-Analytics-\Second_Assignment\datasets\Steel_industry_data.csv"


def load():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("date").reset_index(drop=True)
    df["hour"] = df["date"].dt.hour
    df["month"] = df["date"].dt.month
    df = pd.get_dummies(df, columns=["WeekStatus", "Day_of_week", "Load_Type"],
                        drop_first=True, dtype="int")
    df = df.drop(columns=["date", "CO2(tCO2)"])
    return df


def temporal_split(df, frac=0.7):
    cut = int(frac * len(df))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def remove_outliers_train_only(train_df, target="Usage_kWh", contamination=0.05):
    q1, q3 = train_df[target].quantile([0.25, 0.75])
    iqr_mask = (train_df[target] < q1 - 1.5 * (q3 - q1)) | (train_df[target] > q3 + 1.5 * (q3 - q1))
    feats = [c for c in train_df.columns if c != target]
    iso_mask = IsolationForest(contamination=contamination, random_state=42).fit_predict(train_df[feats]) == -1
    drop = iqr_mask & iso_mask
    return train_df.loc[~drop].reset_index(drop=True), int(drop.sum())


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def evaluate(model_factory, X_tr, y_tr, X_te, y_te, label):
    model = model_factory()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae, rmse, r2 = metrics(y_te, pred)
    print(f"[{label:55s}] MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:+.3f}")
    return mae, rmse, r2


def main():
    df = load()
    print(f"Total rows: {len(df)}")
    train_df, test_df = temporal_split(df, 0.7)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    target = "Usage_kWh"
    feats = [c for c in df.columns if c != target]

    X_test, y_test = test_df[feats], test_df[target]

    # baseline (raw temporal split, no outlier removal)
    X_base, y_base = train_df[feats], train_df[target]

    # optimized = outlier removal on TRAIN ONLY
    train_opt, n_drop = remove_outliers_train_only(train_df, target=target)
    print(f"Outliers removed from train: {n_drop} ({100*n_drop/len(train_df):.1f}%)")
    X_opt, y_opt = train_opt[feats], train_opt[target]

    print()
    print("=== Ridge ===")
    ridge = lambda: make_pipeline(StandardScaler(), Ridge())
    evaluate(ridge, X_base, y_base, X_test, y_test, "baseline-train (raw)              -> test")
    evaluate(ridge, X_opt,  y_opt,  X_test, y_test, "optimized-train (IQR ∩ IsoForest) -> test")

    print()
    print("=== RandomForest (n=300) ===")
    rf = lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    evaluate(rf, X_base, y_base, X_test, y_test, "baseline-train (raw)              -> test")
    evaluate(rf, X_opt,  y_opt,  X_test, y_test, "optimized-train (IQR ∩ IsoForest) -> test")

    print()
    print("Note: CTGAN augmentation is NOT included here. To extend, train CTGAN on")
    print("train_opt[train_opt[target] >= train_opt[target].quantile(0.95)], sample,")
    print("then concatenate with train_opt and retrain. Augmentation should be judged")
    print("by whether it improves THESE same test metrics.")


if __name__ == "__main__":
    main()
