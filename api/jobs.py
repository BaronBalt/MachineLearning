import io
import threading
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from db.database import load_training, save_model_db, save_parameters
from src.data import get_columns_from_csv, load_and_split
from src.evaluate import evaluate_model
from src.model import impute_data, train_model

_training_jobs: dict = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        _training_jobs[job_id].update(kwargs)


def start_training_job(model, trees, classes, is_continuation,
                       training_data_name, model_name, version, algorithm, logger) -> str:
    """Register a new job and kick off the background thread. Returns the job_id."""
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _training_jobs[job_id] = {"status": "pending", "result": None, "error": None}

    thread = threading.Thread(
        target=_run_training,
        args=(job_id, model, trees, classes, is_continuation,
              training_data_name, model_name, version, algorithm, logger),
        daemon=True,
    )
    thread.start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return _training_jobs.get(job_id)


def _run_training(job_id, model, trees, classes, is_continuation,
                  training_data_name, model_name, version, algorithm, logger):
    _update_job(job_id, status="running")
    try:
        training_data, training_id = load_training(training_data_name)
        if training_data is None or training_id is None:
            _update_job(job_id, status="failed", error="Training data not found")
            return

        X_train, X_val, y_train, y_val = load_and_split(training_data, logger)
        X_train_processed, X_val_processed, preprocessor = impute_data(X_train, X_val)

        trained_model = train_model(
            model, X_train_processed, y_train, is_continuation,
            int(trees) if trees else 0, classes,
        )
        evaluation_metrics = evaluate_model(trained_model, X_val_processed, y_val)

        pipeline = Pipeline([("preprocess", preprocessor), ("model", trained_model)])
        params = get_columns_from_csv(training_data)
        features = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else params

        model_bytes = io.BytesIO()
        joblib.dump({"pipeline": pipeline, "features": features}, model_bytes)

        new_id = save_model_db(
            model_name, version, algorithm,
            evaluation_metrics.accuracy,
            evaluation_metrics.precision,
            evaluation_metrics.recall,
            model_bytes.getvalue(),
            training_id,
        )
        if new_id is None:
            _update_job(job_id, status="failed", error="Model with this name and version already exists")
            return

        save_parameters(new_id, params)
        _update_job(job_id, status="complete", result={
            "evaluation_metrics": evaluation_metrics.to_dict(),
            "model": model_name,
            "version": version,
        })
    except Exception as e:
        logger.exception("Background training failed")
        _update_job(job_id, status="failed", error=str(e))
