import json
import joblib
from pathlib import Path
from datetime import datetime, UTC

def save_model(model, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(output_dir) / "model.pkl"
    joblib.dump(model, model_path)
    return model_path

def save_metrics(metrics, output_dir):
    metrics["timestamp"] = datetime.now(UTC).isoformat()
    metrics_path = Path(output_dir) / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics_path
