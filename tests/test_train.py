from src.train import train_model
import pandas as pd

def test_train_model():
    X = pd.DataFrame({"f1":[1,2,3], "f2":[4,5,6]})
    y = pd.Series([0,1,0])
    model = train_model(X, y)
    assert model is not None