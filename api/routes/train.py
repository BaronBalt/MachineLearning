import io
import json

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, request
from werkzeug.datastructures import FileStorage

from api.jobs import get_job, start_training_job
from db.database import (
    does_model_exist,
    get_all_models_info,
    last_model_version,
    load_model,
    load_training_by_id,
    save_training,
)
from src.config import UPLOAD_FOLDER
from src.model import Algorithms, const_algorithms, create_model

bp = Blueprint("train", __name__)


def _save_uploaded_file(file: FileStorage, target_column: str) -> str:
    filename = file.filename or ""
    if not filename:
        raise ValueError("No file provided")
    data = file.read()
    df = pd.read_csv(io.BytesIO(data))
    df = df.rename(columns={target_column: "target"})
    out = io.BytesIO()
    df.to_csv(out, index=False)
    save_training(filename, out.getvalue())
    return filename


@bp.route("/api/train", methods=["POST", "PUT"])
def post_train():
    """
    Validates inputs and starts model training in a background thread.
    Returns {"job_id": "..."} with HTTP 202 immediately.
    Poll GET /api/train/status/<job_id> for completion.
    """
    model_name = request.form.get("model")
    algorithm = request.form.get("algorithm")
    trees = request.form.get("trees")
    classes = request.form.get("classes")
    classes = json.loads(classes) if classes else None

    current_app.logger.info(f"Train request — model: {model_name}, algorithm: {algorithm}")

    if classes is not None and not isinstance(classes, list):
        return jsonify({"error": "classes must be a list"}), 400
    classes = np.asarray(classes) if classes is not None else None

    last_version = last_model_version(model_name)

    if last_version == 0:
        # New model
        if "file" in request.files:
            target_column = request.form.get("target_column", "target")
            training_data_name = _save_uploaded_file(request.files["file"], target_column)
        elif "file_name" in request.form:
            training_data_name = request.form.get("file_name", "")
        else:
            return jsonify({"error": "No training data provided. Supply a file or file_name."}), 400

        if algorithm not in [a.name for a in Algorithms]:
            valid = ", ".join(a.name for a in Algorithms)
            return jsonify({"error": f"Invalid algorithm '{algorithm}'. Valid options: {valid}"}), 400

        if trees:
            model = create_model(
                Algorithms.RANDOM_FOREST if algorithm == Algorithms.RANDOM_FOREST.name else Algorithms.EXTRA_TREES
            )
        elif classes is not None:
            model = create_model(
                Algorithms.LOGISTIC_REGRESSION if algorithm == Algorithms.LOGISTIC_REGRESSION.name else Algorithms.SGD
            )
        else:
            return jsonify({"error": "For tree-based algorithms provide 'trees'; for incremental algorithms provide 'classes'."}), 400

        version = 1
        is_continuation = False
    else:
        # Continue training
        full_model = load_model(model_name, last_version)
        if full_model is None:
            return jsonify({"error": "Model not found for training continuation"}), 404

        algorithm = full_model.algorithm
        if algorithm == Algorithms.LOGISTIC_REGRESSION.name:
            return jsonify({"error": "Logistic regression models cannot be further trained"}), 400

        training_data_name = load_training_by_id(full_model.training_data_id)
        loaded_pipeline, _ = full_model.to_prediction_model()
        model = loaded_pipeline.named_steps["model"]
        version = int(last_version) + 1
        is_continuation = True

    job_id = start_training_job(
        model, trees, classes, is_continuation,
        training_data_name, model_name, version, algorithm,
        current_app.logger,
    )
    current_app.logger.info(f"Job {job_id} started — model: {model_name} v{version}")
    return jsonify({"job_id": job_id}), 202


@bp.route("/api/train/status/<job_id>", methods=["GET"])
def get_train_status(job_id: str):
    """Poll the status of a background training job."""
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@bp.route("/api/train", methods=["GET"])
def get_train_info():
    """
    Returns information needed to train or further-train a model.
    With no query params, returns the list of available algorithms.
    """
    model_name = request.args.get("model")

    if model_name and does_model_exist(model_name):
        models = get_all_models_info()
        for model in models:
            if model.name != model_name:
                continue
            alg = model.algorithm
            if alg in (Algorithms.RANDOM_FOREST.name, Algorithms.EXTRA_TREES.name):
                return jsonify({
                    "version": "version to further train",
                    "trees": "amount of trees to add",
                    "parameters": [p.to_dict() for p in model.parameters],
                })
            if alg == Algorithms.SGD.name:
                return jsonify({
                    "version": "version to further train",
                    "parameters": [p.to_dict() for p in model.parameters],
                })
            if alg == Algorithms.LOGISTIC_REGRESSION.name:
                return jsonify({"info": "This algorithm cannot be further trained"})

    return jsonify([a.to_dict() for a in const_algorithms])
