import io
import json

import joblib
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.datastructures import FileStorage

from db.database import load_model, load_training, save_model_db, save_parameters, save_training, get_all_models_info, \
    does_model_exist
from src.data import get_columns_from_csv, load_and_split_bin_csv
from src.evaluate import evaluate_model
from src.model import train_model, Algorithms, create_model

UPLOAD_FOLDER = "data"
ALLOWED_EXTENSIONS = {"csv", "txt"}
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

predict_result = {
    "model": "test",
    "version": "1",
    "result": {"result_key": "result_value"},
}

predict_body_example = {"input": [0, 1, 2, 3, 4, 5]}


@app.route("/healthz", methods=["GET"])
def get_health():
    """
    Health check for docker/kubernetes
    """
    return jsonify({"status": "healthy"})


@app.route("/api/predict", methods=["POST"])
def post_predict():
    """
    This route is to run a prediction on a given model & version.
    Info about the input format is available with the GET method.
    """
    model = request.args.get("model")
    version = request.args.get("version")
    body = request.get_json() or predict_body_example
    print("")
    model = load_model(model, int(version) if version else 0)

    if model is None:
        return jsonify({"error": "Model not found"}), 404

    values = body.get("input", [])
    if len(values) == 0:
        return jsonify({"error": "No input data provided"}), 400

    X_input = [values]
    predictionModel = joblib.load(io.BytesIO(model.data))
    predict_result["result"] = predictionModel.predict(X_input).tolist()

    # fetch from model form db or something and predict based on body
    print(f"Received model: {model}, version: {version}")
    print(f"received body: {body}")
    predict_result["model"] = model.name
    predict_result["version"] = version

    return jsonify(predict_result)


@app.route("/api/predict", methods=["GET"])
def get_predict_info():
    """
    This route should return the input format that this model expects.
    """
    model = request.args.get("model")
    version = request.args.get("version")

    # fetch from model form db or something and return info
    print(f"Received model: {model}, version: {version}")

    return jsonify(
        {
            "model": model,
            "version": version,
            "input_example": predict_body_example,
            "description": "",
        }
    )


def save_uploaded_file(file: FileStorage) -> str:
    """
    Function to save a training file to the database.
    """

    filename = file.filename or ""
    if filename == "":
        raise ValueError("No file provided")
    data = file.read()
    save_training(filename, data)
    return filename


@app.route("/api/train", methods=["PUT"])
def put_train():
    """
    This route trains a new model or version of model
    The parameters for the training method are in the url parameters.
    To further train:
        model: name of the model
        version: version to further train
        trees: only if tree based
        file: training data in .csv file
    To create new model:
        model: name of the model
        algorithm: algorithm to use for the model
        trees: if tree based, otherwise
        classes: classes for incremental models like, e.g. ["cat", "dog"] or [1, 2, 3]
        file: training data in .csv file
    """

    model_name = request.args.get("model")
    algorithm = request.args.get("algorithm")
    trees = request.args.get("trees")
    version = request.args.get("version")

    if version is None:
        version = 1
    else:
        version = int(version)

    payload = None
    training_data_name = ""

    if "file" in request.files:
        file = request.files["file"]
        save_uploaded_file(file)
        training_data_name = file.filename

        payload_raw = request.form.get("payload")
        if payload_raw:
            payload = json.loads(payload_raw)

    elif body := request.get_json(silent=True):
        payload = body
        training_data_name = body.get("data_name", "")

    else:
        return jsonify({"error": "No data provided for training"}), 400

    if not training_data_name:
        return jsonify({"error": "No data name provided"}), 400

    # Extract classes safely, classes must be the same as y values in training data
    classes = None
    if payload and "classes" in payload:
        classes = payload["classes"]

        if not isinstance(classes, list):
            return jsonify({"error": "classes must be a list"}), 400

    if classes is not None:
        classes = np.asarray(classes)

    training_data = load_training(training_data_name)

    X_train, X_val, y_train, y_val = load_and_split_bin_csv(training_data)

    is_continuation = False
    found_model = load_model(model_name, int(version) if version else 0)
    if found_model is not None:
        if found_model.algorithm == Algorithms.LOGISTIC_REGRESSION.name:
            return jsonify({"error": "Logistic regression model cannot be further trained"}), 400
        print("Found model")
        is_continuation = True
        model_bytes = io.BytesIO(found_model.data)
        model = joblib.load(model_bytes)
        trained_model = train_model(model, X_train, y_train, is_continuation, int(trees) if trees is not None else 0)
        algorithm = found_model.algorithm
        version += 1
    elif algorithm and trees:
        print("Creating new tree-based model")
        model = create_model(Algorithms.RANDOM_FOREST if algorithm == Algorithms.RANDOM_FOREST.name else Algorithms.EXTRA_TREES)
        trained_model = train_model(model, X_train, y_train, is_continuation, int(trees))
    elif algorithm and classes is not None:
        print("Creating new incremental model")
        model = create_model(Algorithms.LOGISTIC_REGRESSION if algorithm == Algorithms.LOGISTIC_REGRESSION.name else Algorithms.SGD)
        trained_model = train_model(model, X_train, y_train, is_continuation, 0, classes)
    else:
        return jsonify({"error": "Missing info to create new model or further train"}), 400

    evaluation_metrics = evaluate_model(trained_model, X_val, y_val)
    model_bytes = io.BytesIO()
    joblib.dump(trained_model, model_bytes)

    new_id = save_model_db(
        model_name,
        version,
        algorithm,
        evaluation_metrics.accuracy,
        evaluation_metrics.precision,
        evaluation_metrics.recall,
        model_bytes.getvalue(),
    )
    if new_id is None:
        return jsonify({"error": "Model with this name and version already exists"}), 400
    params = get_columns_from_csv(training_data)
    save_parameters(new_id, params)

    # trigger training for model version
    print(f"Training model: {model_name}, version: {version}")

    return jsonify(
        {
            "evaluation_metrics": evaluation_metrics.to_dict(),
            "model": model_name,
            "version": version,
        }
    )


@app.route("/api/train", methods=["GET"])
def get_train_info():
    """
    Returns either existing model parameters or information needed
    to train a new model from scratch.
    Query param:
        model: name of the model
        algorithm: algorithm used for the model
    """
    model_name = request.args.get("model")
    model_algorithm = request.args.get("algorithm")

    # If existing model was provided return what is needed to further train
    if does_model_exist(model_name):
        models = get_all_models_info()
        for model in models:
            if model.name == model_name and model.algorithm == model_algorithm:
                if model.algorithm == Algorithms.RANDOM_FOREST.name or model.algorithm == Algorithms.EXTRA_TREES.name:
                    return jsonify({
                        "version": "version to further train",
                        "trees": "amount of trees to add",
                        "parameters": [param.to_dict() for param in model.parameters]
                    })
                elif model.algorithm == Algorithms.SGD.name:
                    return jsonify({
                        "version": "version to further train",
                        "parameters": [param.to_dict() for param in model.parameters]
                    })
                elif model.algorithm == Algorithms.LOGISTIC_REGRESSION.name:
                    return jsonify({"info:" "This algorithm cannot be further trained"})

    # Train tree based model from scratch
    elif model_algorithm == Algorithms.RANDOM_FOREST.name or model_algorithm == Algorithms.EXTRA_TREES.name:
        return jsonify({
            "name": "e.g. irismodel",
            "trees": "number of trees",
            "data": "data in .csv file"
        })

    # Train incremental model from scratch
    elif model_algorithm == Algorithms.LOGISTIC_REGRESSION.name or model_algorithm == Algorithms.SGD.name:
        return jsonify({
            "name": "e.g. irismodel",
            "classes": "e.g. \"dog\", \"cat\"",
            "data": "data in .csv file"
        })

    # Available algorithms
    else:
        available_algorithms = [algo.name for algo in Algorithms]
        return jsonify({
            "algorithm_options": available_algorithms
        })


@app.route("/api/models", methods=["GET"])
def get_models():
    models_info = get_all_models_info()

    return jsonify([model_info.to_dict() for model_info in models_info])


if __name__ == "__main__":
    app.run(debug=True)
