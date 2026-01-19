# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# 04_inference_demo.ipynb
# -----------------------
# Notebook for testing a trained model on new data

# + tags=["parameters"]
import os
import sys
import glob
import pandas as pd
import joblib

# -------------------------------
# REPO-ROOT HANDLING (stable)
# -------------------------------
NOTEBOOK_DIR = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, ".."))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

print("Repo root:", REPO_ROOT)
print("Current working directory:", os.getcwd())

# + imports
from src.feature_eng import create_features

# +
# Step 1: Load the latest trained model
model_files = sorted(glob.glob(os.path.join(REPO_ROOT, "models", "*.joblib")))
if not model_files:
    raise FileNotFoundError("No model found in models/ folder!")

latest_model_file = model_files[-1]
print("Loading model:", latest_model_file)
model = joblib.load(latest_model_file)

# +
# Step 2: Load new raw data
# Make sure this CSV has the same columns as training data (without target)
new_data_path = os.path.join(REPO_ROOT, "data", "raw", "iris_new.csv")

if not os.path.exists(new_data_path):
    # If file doesn't exist, create dummy new data (same as training format)
    from sklearn.datasets import load_iris
    iris = load_iris()
    df_new = pd.DataFrame(iris.data, columns=iris.feature_names)
    os.makedirs(os.path.dirname(new_data_path), exist_ok=True)
    df_new.to_csv(new_data_path, index=False)
    print("Created dummy new data:", new_data_path)
else:
    df_new = pd.read_csv(new_data_path)

df_new.head()
# -

# Step 3: Create features
X_new = create_features(df_new).drop("target", axis=1, errors="ignore")
X_new.head()

# +
# Step 4: Make predictions
y_pred = model.predict(X_new)
print("Predictions:", y_pred)

# Optional: probabilities (if classifier supports it)
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_new)
    print("Prediction probabilities:\n", y_proba)
# -

# Step 5: Save predictions
df_new["prediction"] = y_pred
predictions_path = os.path.join(REPO_ROOT, "data", "predictions.csv")
df_new.to_csv(predictions_path, index=False)
print("Predictions saved to:", predictions_path)

