import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime
import psycopg
import os

DB_URL = "postgresql://mluser:mlpass@localhost:5432/mlregistry"

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def save_model(model, model_name="random_forest", root=".", accuracy=None):
    import os
    import joblib
    from datetime import datetime

    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    version = datetime.now().strftime("%Y%m%d%H%M")
    filename = os.path.join(root, "models", f"{model_name}_v{version}.joblib")
    joblib.dump(model, filename)
    if accuracy is not None:
        save_model_to_db(
            name=model_name,
            version=version,
            algorithm=type(model).__name__,
            artifact_path=filename,
            accuracy=accuracy
        )
    return filename

def save_model_to_db(name, version, algorithm, artifact_path, accuracy):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model (name, version, algorithm, artifact_path, accuracy)
                VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (name, version) DO NOTHING
                """,
                (name, version, algorithm, artifact_path, accuracy)
            )