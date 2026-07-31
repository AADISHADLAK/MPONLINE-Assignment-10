"""
generate_dataset.py
--------------------
Generates a synthetic but clinically realistic Heart Disease dataset that
follows the exact same column schema as the popular Kaggle dataset:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Columns:
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
    oldpeak, slope, ca, thal, target

NOTE: This script is provided so the project is fully reproducible even
without internet access to Kaggle. If you have internet access, simply
download the original heart.csv from the Kaggle link above and replace
the generated file with it -- the rest of the pipeline (train_model.py,
app.py) works identically either way since the column schema matches.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1025  # same size as the original Kaggle dataset

# Start by deciding the target (1 = heart disease present, 0 = absent)
target = np.random.binomial(1, 0.51, N)

age = np.where(
    target == 1,
    np.random.normal(56, 8, N),
    np.random.normal(50, 9, N),
).clip(29, 77).astype(int)

sex = np.random.binomial(1, 0.68, N)  # 1 = male, 0 = female

cp = np.where(
    target == 1,
    np.random.choice([0, 1, 2, 3], N, p=[0.55, 0.18, 0.17, 0.10]),
    np.random.choice([0, 1, 2, 3], N, p=[0.20, 0.30, 0.30, 0.20]),
)

trestbps = np.where(
    target == 1,
    np.random.normal(134, 18, N),
    np.random.normal(128, 16, N),
).clip(94, 200).astype(int)

chol = np.where(
    target == 1,
    np.random.normal(250, 52, N),
    np.random.normal(238, 45, N),
).clip(126, 564).astype(int)

fbs = np.random.binomial(1, 0.15, N)

restecg = np.random.choice([0, 1, 2], N, p=[0.48, 0.50, 0.02])

thalach = np.where(
    target == 1,
    np.random.normal(139, 22, N),
    np.random.normal(158, 19, N),
).clip(71, 202).astype(int)

exang = np.where(
    target == 1,
    np.random.binomial(1, 0.55, N),
    np.random.binomial(1, 0.14, N),
)

oldpeak = np.where(
    target == 1,
    np.random.exponential(1.4, N),
    np.random.exponential(0.6, N),
).round(1).clip(0, 6.2)

slope = np.where(
    target == 1,
    np.random.choice([0, 1, 2], N, p=[0.10, 0.55, 0.35]),
    np.random.choice([0, 1, 2], N, p=[0.05, 0.30, 0.65]),
)

ca = np.where(
    target == 1,
    np.random.choice([0, 1, 2, 3, 4], N, p=[0.35, 0.30, 0.20, 0.10, 0.05]),
    np.random.choice([0, 1, 2, 3, 4], N, p=[0.65, 0.20, 0.10, 0.04, 0.01]),
)

thal = np.where(
    target == 1,
    np.random.choice([0, 1, 2, 3], N, p=[0.02, 0.05, 0.25, 0.68]),
    np.random.choice([0, 1, 2, 3], N, p=[0.02, 0.10, 0.68, 0.20]),
)

df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
    "target": target,
})

# shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("heart.csv", index=False)
print("heart.csv generated with shape:", df.shape)
print(df.head())
print("\nTarget distribution:\n", df["target"].value_counts())
