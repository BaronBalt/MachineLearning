import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import json
import os

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc}

def update_registry(model_file, metrics, root=".", registry_file="registry/models.json"):
    import os, json
    import pandas as pd

    full_path = os.path.join(root, registry_file)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    try:
        with open(full_path, "r") as f:
            registry = json.load(f)
    except FileNotFoundError:
        registry = []

    entry = {
        "model_file": model_file,
        "trained_on": str(pd.Timestamp.now()),
        "metrics": metrics
    }
    registry.append(entry)

    with open(full_path, "w") as f:
        json.dump(registry, f, indent=2)

    return registry
