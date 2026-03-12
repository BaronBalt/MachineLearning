import io
from typing import List, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from db.models import Parameter
from src.config import RANDOM_STATE, TEST_SIZE


def load_and_split(source: Union[bytes, str], logger=None):
    """
    Load a CSV from either raw bytes or a file path, drop ID-like columns,
    and split into train/val sets.
    """
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    df = pd.read_csv(stream)

    if logger:
        logger.info(f"Columns in CSV: {df.columns.tolist()}")

    if "target" not in df.columns:
        raise ValueError("CSV must contain a 'target' column")

    X = df.drop(columns=["target"])
    X = X.loc[:, ~X.columns.str.match(r'^id', case=False)]
    y = df["target"]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def get_columns_from_csv(data: bytes) -> List[Parameter]:
    df = pd.read_csv(io.BytesIO(data))
    df.drop(columns=["target"], inplace=True, errors="ignore")
    df = df.loc[:, ~df.columns.str.match(r'^id', case=False)]
    return [Parameter(name=col, value=str(df[col].values[0])) for col in df.columns]

