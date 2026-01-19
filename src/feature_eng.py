import pandas as pd
import os

def create_features(df):
    # Lägger till ett exempelfeature: sum of all numeric columns
    df_feat = df.copy()
    numeric_cols = df_feat.select_dtypes(include='number').columns.tolist()
    df_feat['sum_feature'] = df_feat[numeric_cols].sum(axis=1)
    return df_feat

def save_processed(df, root=".", path="data/processed/features.csv"):
    full_path = os.path.join(root, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df.to_csv(full_path, index=False)
    return full_path