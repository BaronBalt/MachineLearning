import json
from datetime import datetime, UTC

import joblib

from src.utils import get_next_model_path


def save_model(model, model_name: str, output_dir: str = "models"):
    model_path = get_next_model_path(model_name, output_dir)
    joblib.dump(model, model_path)
    return model_path

def save_metrics(metrics, model_name: str, output_dir: str = "artifacts"):
    metrics["timestamp"] = datetime.now(UTC).isoformat()
    metrics_path = get_next_model_path(model_name, output_dir, ".json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics_path

def load_model(model_path):
    model = joblib.load(model_path)
    return model