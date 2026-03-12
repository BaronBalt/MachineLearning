from src.model import create_model, train_model, Algorithms
import pandas as pd
import numpy as np

def test_train_model():
    X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    model = create_model(Algorithms.RANDOM_FOREST)
    trained = train_model(model, X, y, is_continuation=False)
    assert trained is not None