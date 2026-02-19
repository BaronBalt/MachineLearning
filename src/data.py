import io
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from db.database import Parameter
from src.config import RANDOM_STATE, TEST_SIZE


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

def load_and_split_bin_csv(data, logger):
    stream = io.BytesIO(data)
    
    df = pd.read_csv(stream)
    logger.info(f"Columns in CSV: {df.columns}")
    logger.info(f"First few rows of CSV:\n{df.head()}")
    

    if "target" not in df.columns:
        raise ValueError("CSV must contain a 'target' column")

    X = df.drop(columns=["target"])
    y = df["target"]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

def get_columns_from_csv(data) -> List[Parameter]:
    stream = io.BytesIO(data)
    
    df = pd.read_csv(stream)
    df.drop(columns=["target"], inplace=True, errors="ignore")

    params = [Parameter(name=col, value=str(df[col].values[0])) for col in df.columns]
    return params
