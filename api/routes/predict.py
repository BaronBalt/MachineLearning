import pandas as pd
from flask import Blueprint, jsonify, request

from db.database import load_model

bp = Blueprint("predict", __name__)

_predict_input_example = {"input": [0, 1, 2, 3, 4, 5]}


@bp.route("/api/predict", methods=["POST"])
def post_predict():
    """Run a prediction on a given model and version."""
    model_name = request.args.get("model")
    version = request.args.get("version")
    body = request.get_json(force=True) or {}

    model = load_model(model_name, int(version) if version else 0)
    if model is None:
        return jsonify({"error": "Model not found"}), 404

    values = body.get("input", [])
    if not values:
        return jsonify({"error": "No input data provided"}), 400

    prediction_pipeline, features = model.to_prediction_model()

    if len(features) != len(values):
        return jsonify({"error": f"Expected {len(features)} values, got {len(values)}"}), 400

    X_raw = pd.DataFrame([values], columns=features)
    prediction = prediction_pipeline.predict(X_raw).tolist()
    result = str(prediction[0]) if len(prediction) == 1 else str(prediction)

    return jsonify({"model": model.name, "version": version, "result": result})


@bp.route("/api/predict", methods=["GET"])
def get_predict_info():
    """Return the expected input format for a model."""
    model_name = request.args.get("model")
    version = request.args.get("version")
    return jsonify({
        "model": model_name,
        "version": version,
        "input_example": _predict_input_example,
        "description": "",
    })
