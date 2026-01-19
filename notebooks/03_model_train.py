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
# # + tags=["parameters"]
import os
import sys

# -------------------------------
# REPO-ROOT HANDLING (stable)
# -------------------------------

# Get absolute path to repo root based on the notebooks folder
# notebooks folder is fixed, so ".." from notebooks is repo root
NOTEBOOK_DIR = os.path.dirname(os.path.abspath("__file__"))  # fallback, ignored
try:
    # __file__ does not exist in notebook, so fallback to current working dir
    NOTEBOOK_DIR = os.getcwd()
except NameError:
    pass

REPO_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, ".."))  # stable, one level up
os.chdir(REPO_ROOT)                 # set working dir to repo root
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)      # add src/ to Python path

print("Repo root:", REPO_ROOT)
print("Current working directory:", os.getcwd())
# -

import pandas as pd
from src.utils import load_processed
from src.train import train_model, save_model
from src.eval import evaluate_model, update_registry
from sklearn.model_selection import train_test_split

df = load_processed(REPO_ROOT, "data/processed/features.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_model(X_train, y_train)
model_file = save_model(model, "random_forest", REPO_ROOT)
metrics = evaluate_model(model, X_test, y_test)
update_registry(model_file, metrics, REPO_ROOT, "registry/models.json")

# + editable=true slideshow={"slide_type": ""}
print(f"Model saved to {model_file}")
print("Metrics:", metrics)
# -


