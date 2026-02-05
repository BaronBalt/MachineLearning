import json
import joblib
import os
from pathlib import Path
from datetime import datetime, UTC

try:
    import psycopg
except ImportError:
    psycopg = None

# The ML_DB_URL variable is used, the  local path is a fallback
DB_URL = os.getenv("ML_DB_URL", "postgresql://mluser:mlpass@localhost:5432/mlregistry")

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


def save_model(model, output_dir, *, model_name="random_forest", accuracy=None, version=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(output_dir) / "model.pkl"
    joblib.dump(model, model_path)
    # only register to DB if we actually have accuracy
    if accuracy is not None:
        artifact_path = model_path.as_posix()
        save_model_to_db(
            name=model_name,
            version=version,
            algorithm=type(model).__name__,
            artifact_path=artifact_path,
            accuracy=accuracy,
        )

    return model_path

def save_metrics(metrics, output_dir: str):
    metrics["timestamp"] = datetime.now(UTC).isoformat()
    metrics_path = Path(output_dir) / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics_path

def load_model(model_path: str):
    model = joblib.load(model_path)
    return model
