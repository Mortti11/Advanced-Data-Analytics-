# Advanced Data Analytics

This repo contains my work for an Advanced Data Analytics course. It is organized into three assignments, each with a classification and a regression notebook. The datasets come from UCI Machine Learning Repository and OpenML.

---

## First Assignment — EDA and Modeling

The focus here was on exploratory data analysis using profiling tools like `ydata_profiling`, `sweetviz`, and `autoviz`, then building some baseline models.

**Classification** — Predictive Maintenance dataset (AI4I 2020, UCI)
- 10,000 rows, synthetic industrial data simulating machine failures
- Target: whether a machine failed or not (binary)
- Explored failure types: heat dissipation failure, power failure, overstrain failure, tool wear failure
- Used Logistic Regression, Random Forest, and Gradient Boosting
- Handled class imbalance with sample weights and stratified cross-validation

**Regression** — Iranian Telecom Customer Churn dataset (UCI)
- 3,150 customers, 13 features
- The dataset is originally for churn classification, but I used it for regression with `Customer Value` as the target
- Used Linear Regression, Ridge, Huber, and Random Forest regressors
- Preprocessing with `RobustScaler` inside a pipeline

---

## Second Assignment — Data Quality and Sampling

This assignment dealt with messier problems: class imbalance, noise, overlapping classes, and outliers. The goal was to try different data preparation strategies and compare results.

**Classification** — Codon Usage dataset (UCI)
- Genomic data mapping codon usage frequencies to biological kingdoms (bacteria, virus, plant, mammal, etc.)
- Large dataset with strong class imbalance
- Tried sampling methods: `RandomOverSampler`, `SMOTENC`, `ADASYN`, `TomekLinks`, `RandomUnderSampler`
- Feature selection with mutual information and VIF to reduce multicollinearity
- Models: Random Forest, KNN inside an imbalanced pipeline

**Regression** — Steel Industry Energy Consumption dataset (UCI)
- Electricity usage data from a steel plant in South Korea
- Target: `Usage_kWh`
- Worked on outlier detection and removal using `IsolationForest`, `RANSAC`, and `EllipticEnvelope`
- Used `CTGAN` for synthetic data generation
- Models: Random Forest Regressor

---

## Third Assignment — Feature Engineering

The focus shifted to feature selection, multicollinearity, dimensionality reduction, and building new features.

**Classification** — EEG Eye State dataset (OpenML)
- EEG recordings from a single 117-second session using an Emotiv headset
- Target: whether eyes are open or closed (binary)
- Time-series structure, loaded from ARFF format using `scipy`
- Feature selection with mutual information and `phik` correlation
- Applied PCA and created time-based features
- Used `TimeSeriesSplit` for cross-validation

**Regression** — PGCB Hourly Generation dataset (UCI)
- Hourly electricity generation, demand, and loadshedding data from Bangladesh's national grid
- Target: `demand_mw`
- Feature selection using mutual information and F-test
- Created interaction features and cluster-based profiles
- Checked redundancy with PCA and VIF
- Models: Random Forest Regressor

---

## Structure

```
First-Assignment/
    Classification.ipynb     # Predictive Maintenance (machine failure)
    Regression.ipynb         # Customer Value prediction
    Datasets/

Second_Assignment/
    classification.ipynb     # Codon Usage (biological kingdoms)
    Regression.ipynb         # Steel Industry energy consumption
    datasets/

Third_assignment/
    Eeg_Eye_state_classification.ipynb   # EEG eye open/closed
    PGCB_Regession.ipynb                 # Bangladesh electricity demand
    Datasets/
```

---

## Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `ydata_profiling`, `sweetviz`, `autoviz`
- `scikit-learn`, `imbalanced-learn`
- `phik`, `statsmodels`
- `ctgan` (synthetic data generation)
- `mne`, `scipy` (EEG data handling)
