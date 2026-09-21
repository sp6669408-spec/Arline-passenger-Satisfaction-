"""
MINI PROJECT: Predicting Airline Passenger Satisfaction
Supervised Learning | Python | Individual Mini Project

Dataset: "Airline Passenger Satisfaction" (Kaggle, public secondary data)
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

Place the real CSV as 'airline_passenger_satisfaction.csv' in this folder
before running for a genuine submission. (A synthetic stand-in with the
identical schema is used here only to demonstrate the pipeline.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

sns.set_style("whitegrid")
PLOT_DIR = "plots"
import os
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("airline_passenger_satisfaction.csv")
print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# 2. DATASET DESCRIPTION
# ---------------------------------------------------------------
print("\n--- Column info ---")
print(df.dtypes)
print("\n--- Target distribution ---")
print(df["satisfaction"].value_counts())

# ---------------------------------------------------------------
# 3. DATA PREPROCESSING
# ---------------------------------------------------------------
# drop identifier column - not predictive
if "id" in df.columns:
    df = df.drop(columns=["id"])
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("\nMissing values per column:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Impute missing numeric values with the median (robust to outliers)
num_cols_with_na = df.columns[df.isnull().any()]
for c in num_cols_with_na:
    df[c] = df[c].fillna(df[c].median())

# Outlier handling on delay columns: cap at the 99th percentile (winsorizing)
for c in ["Departure Delay in Minutes", "Arrival Delay in Minutes"]:
    cap = df[c].quantile(0.99)
    n_capped = (df[c] > cap).sum()
    df[c] = np.where(df[c] > cap, cap, df[c])
    print(f"Capped {n_capped} outlier values in '{c}' at {cap:.0f} minutes")

# Encode categorical variables
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != "satisfaction"]
print("\nCategorical columns encoded:", cat_cols)

df_enc = df.copy()
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df_enc[c] = le.fit_transform(df_enc[c])
    encoders[c] = le

target_le = LabelEncoder()
df_enc["satisfaction"] = target_le.fit_transform(df_enc["satisfaction"])
print("Target classes:", list(target_le.classes_), "-> encoded as", list(range(len(target_le.classes_))))

# ---------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------
print("\n--- Descriptive statistics ---")
print(df.describe(include="all").T)

# Target balance
plt.figure(figsize=(5, 4))
df["satisfaction"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.title("Target class balance: satisfaction")
plt.ylabel("Passenger count")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_target_balance.png", dpi=130)
plt.close()

# Age distribution by satisfaction
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="Age", hue="satisfaction", kde=True, element="step")
plt.title("Age distribution by satisfaction")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_age_distribution.png", dpi=130)
plt.close()

# Satisfaction rate by Class
plt.figure(figsize=(6, 4))
rate_by_class = df.groupby("Class")["satisfaction"].apply(lambda s: (s == "satisfied").mean())
rate_by_class.sort_values().plot(kind="barh", color="#55A868")
plt.xlabel("Proportion satisfied")
plt.title("Satisfaction rate by travel Class")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_satisfaction_by_class.png", dpi=130)
plt.close()

# Satisfaction rate by Type of Travel
plt.figure(figsize=(6, 4))
rate_by_travel = df.groupby("Type of Travel")["satisfaction"].apply(lambda s: (s == "satisfied").mean())
rate_by_travel.sort_values().plot(kind="barh", color="#C44E52")
plt.xlabel("Proportion satisfied")
plt.title("Satisfaction rate by Type of Travel")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_satisfaction_by_travel_type.png", dpi=130)
plt.close()

# Correlation heatmap of service ratings + numeric variables
plt.figure(figsize=(10, 8))
corr = df_enc.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3, cbar_kws={"shrink": 0.7})
plt.title("Correlation heatmap (all encoded variables)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_correlation_heatmap.png", dpi=130)
plt.close()

# Boxplot: Flight Distance vs satisfaction (check for outliers too)
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="satisfaction", y="Flight Distance")
plt.title("Flight Distance by satisfaction")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_flightdistance_boxplot.png", dpi=130)
plt.close()

print("\nEDA plots saved to ./plots/")

# ---------------------------------------------------------------
# 5. MODEL BUILDING
# ---------------------------------------------------------------
X = df_enc.drop(columns=["satisfaction"])
y = df_enc["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model A: Logistic Regression (interpretable baseline)
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
pred_lr = logreg.predict(X_test_scaled)
proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]

# Model B: Random Forest (non-linear, handles interactions)
rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------
# 6. MODEL EVALUATION
# ---------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print(classification_report(y_true, y_pred, target_names=target_le.classes_))
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

results = []
results.append(evaluate("Logistic Regression", y_test, pred_lr, proba_lr))
results.append(evaluate("Random Forest", y_test, pred_rf, proba_rf))
results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)
print("\n", results_df)

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, pred) in zip(axes, [("Logistic Regression", pred_lr), ("Random Forest", pred_rf)]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=target_le.classes_, yticklabels=target_le.classes_)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_confusion_matrices.png", dpi=130)
plt.close()

# ROC curves
plt.figure(figsize=(6, 5))
for name, proba in [("Logistic Regression", proba_lr), ("Random Forest", proba_rf)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/08_roc_curves.png", dpi=130)
plt.close()

# Feature importance (Random Forest) - interpretation aid
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(7, 6))
importances.head(12).sort_values().plot(kind="barh", color="#8172B2")
plt.title("Top 12 features driving satisfaction (Random Forest)")
plt.xlabel("Relative importance")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/09_feature_importance.png", dpi=130)
plt.close()

print("\nTop 10 most important features:")
print(importances.head(10))

print("\nAll plots saved in ./plots/  |  metrics saved to model_results.csv")
