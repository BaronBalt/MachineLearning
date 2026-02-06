import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from src.utils import get_next_model_path
from db.database import save_model_db


# The ML_DB_URL variable is used, the  local path is a fallback



def save_model(
    model, output_dir, *, model_name="random_forest", accuracy=None, version=None
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(output_dir) / "model.pkl"
    joblib.dump(model, model_path)
    # only register to DB if we actually have accuracy
    # if accuracy is not None:
    #     artifact_path = model_path.as_posix()
    #     save_model_db(
    #         name=model_name,
    #         version=version,
    #         algorithm=type(model).__name__,
    #         artifact_path=artifact_path,
    #         accuracy=accuracy,
    #     )

    return model_path


def save_metrics(metrics, model_name: str, output_dir: str = "artifacts"):
    metrics["timestamp"] = datetime.now(UTC).isoformat()
    metrics_path = get_next_model_path(model_name, output_dir, ".json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics_path



