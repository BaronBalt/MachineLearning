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

from src.utils import load_processed
from src.feature_eng import create_features, save_processed

# + editable=true slideshow={"slide_type": ""}
df = load_processed(REPO_ROOT, "data/raw/iris.csv")
df_feat = create_features(df)
save_processed(df_feat, root=REPO_ROOT)
df_feat.head()
# -


