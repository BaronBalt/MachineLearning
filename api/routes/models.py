from flask import Blueprint, jsonify

from db.database import get_all_models_info, get_training_files_db

bp = Blueprint("models", __name__)


@bp.route("/healthz", methods=["GET"])
def get_health():
    """Health check for docker/kubernetes."""
    return jsonify({"status": "healthy"})


@bp.route("/api/models", methods=["GET"])
def get_models():
    models_info = get_all_models_info()
    return jsonify([m.to_dict() for m in models_info])


@bp.route("/api/training-files", methods=["GET"])
def get_training_files():
    training_files = get_training_files_db()
    return jsonify([f.to_dict() for f in training_files])
