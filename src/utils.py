import pandas as pd

def load_processed(root=".", path="data/processed/features.csv"):
    import os, pandas as pd
    full_path = os.path.join(root, path)
    return pd.read_csv(full_path)
