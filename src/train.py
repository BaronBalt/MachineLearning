import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime
import os

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def save_model(model, model_name="random_forest", root="."):
    import os
    import joblib
    from datetime import datetime

    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    version = datetime.now().strftime("%Y%m%d%H%M")
    filename = os.path.join(root, "models", f"{model_name}_v{version}.joblib")
    joblib.dump(model, filename)
    return filename
