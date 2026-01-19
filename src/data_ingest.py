import pandas as pd
from sklearn.datasets import load_iris
import os

def save_iris_csv(root=".", raw_path="data/raw/iris.csv"):
    import os
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    full_path = os.path.join(root, raw_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df.to_csv(full_path, index=False)
    return df
