import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE

def load_and_split_data(data_path):
    df = pd.read_csv(data_path)

    if "target" not in df.columns:
        raise ValueError("CSV must contain a 'target' column")

    X = df.drop(columns=["target"])
    y = df["target"]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
