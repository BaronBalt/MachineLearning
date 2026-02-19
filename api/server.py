import io
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.datastructures import FileStorage

from db.database import (
    does_model_exist,
    get_all_models_info,
    get_training_files_db,
    last_model_version,
    load_model,
    load_training,
    load_training_by_id,
    save_model_db,
    save_parameters,
    save_training,
)
from src.config import MODEL_CONFIG, RANDOM_STATE, TEST_SIZE
from src.data import get_columns_from_csv, load_and_split_bin_csv
from src.evaluate import evaluate_model
from src.model import Algorithms, const_algorithms, create_model, train_model, impute_data

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


def save_uploaded_file(file: FileStorage, target_column: str) -> str:
    """
    Function to save a training file to the database.
    """

    filename = file.filename or ""
    if filename == "":
        raise ValueError("No file provided")

    data = file.read()  # bytes

    df = pd.read_csv(io.BytesIO(data))

    df = df.rename(columns={target_column: "target"})
    output = io.BytesIO()
    df.to_csv(output, index=False)

    new_bytes = output.getvalue()
    save_training(filename, new_bytes)
    return filename


@app.route("/api/train", methods=["PUT", "POST"])
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

    model_name = request.form.get("model")
    algorithm = request.form.get("algorithm")
    trees = request.form.get("trees")
    version = request.form.get("version")
    classes = request.form.get("classes")
    classes = json.loads(classes) if classes else None

    last_version = last_model_version(model_name)
    is_continuation = False

    training_data_name = ""

    # Extract classes safely, classes must be the same as y values in training data
    if classes is not None and not isinstance(classes, list):
        return jsonify({"error": "classes must be a list"}), 400

    classes = np.asarray(classes)

    model = None

    if last_version == 0:
        # new model

        version = 1
        # Here we need training data to create the first version of the model, so we check if it's provided
        if "file" in request.files:
            target_column = request.form.get("target_column", "target")
            save_uploaded_file(request.files["file"], target_column)
            training_data_name = request.files["file"].filename
        elif "file_name" in request.form:
            training_data_name = request.form.get("file_name", "")
        else:
            return jsonify(
                {
                    "error": "Model does not exist, therefore training data must be provided either with a file_name or a raw csv file"
                }
            ), 400

        app.logger.info("new model")
        # model creation
        if algorithm and trees:
            app.logger.info("tree")
            print("Creating new tree-based model")
            model = create_model(
                Algorithms.RANDOM_FOREST
                if algorithm == Algorithms.RANDOM_FOREST.name
                else Algorithms.EXTRA_TREES
            )
            # trained_model = train_model(model, X_train, y_train, is_continuation, int(trees))
        elif algorithm and classes is not None:
            app.logger.info("incremental")
            app.logger.info(f"classes: {classes}")
            print("Creating new incremental model")
            model = create_model(
                Algorithms.LOGISTIC_REGRESSION
                if algorithm == Algorithms.LOGISTIC_REGRESSION.name
                else Algorithms.SGD
            )
            app.logger.info(f"model: {model}")
            # trained_model = train_model(model, X_train, y_train, is_continuation, 0, classes)
        is_continuation = False
    else:
        # Further training
        full_model = load_model(model_name, last_version)
        if full_model is None:
            return jsonify({"error": "Model not found for training continuation"}), 404

        if full_model.algorithm == Algorithms.LOGISTIC_REGRESSION.name:
            return jsonify(
                {"error": "Logistic regression model cannot be further trained"}
            ), 400
        is_continuation = True
        model = full_model.to_prediction_model()

    training_data, training_id = load_training(training_data_name)
    if training_data is None or training_id is None:
        return jsonify({"error": "Training data not found"}), 404

    X_train, X_val, y_train, y_val = load_and_split_bin_csv(training_data, app.logger)

    X_train_processed, X_val_processed = impute_data(X_train, X_val)
    # app.logger.info(f"y_train: {y_train}, y_val: {y_val}")
    
    trained_model = train_model(
        model, X_train_processed, y_train, is_continuation, int(trees) if trees else 0, classes
    )

    evaluation_metrics = evaluate_model(trained_model, X_val_processed, y_val)
    # save model to db in form of bytes
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
        training_id,
    )

    if new_id is None:
        return jsonify(
            {"error": "Model with this name and version already exists"}
        ), 400

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
    app.logger.info(f"Model {model_name}")
    if does_model_exist(model_name):
        app.logger.info(
            f"Model {model_name} exists, fetching info for further training"
        )
        models = get_all_models_info()
        for model in models:
            # if model.name == model_name and model.algorithm == model_algorithm:
            if model.name == model_name:
                if (
                    model.algorithm == Algorithms.RANDOM_FOREST.name
                    or model.algorithm == Algorithms.EXTRA_TREES.name
                ):
                    return jsonify(
                        {
                            "version": "version to further train",
                            "trees": "amount of trees to add",
                            "parameters": [
                                param.to_dict() for param in model.parameters
                            ],
                        }
                    ), 200
                elif model.algorithm == Algorithms.SGD.name:
                    return jsonify(
                        {
                            "version": "version to further train",
                            "parameters": [
                                param.to_dict() for param in model.parameters
                            ],
                        }
                    ), 200
                elif model.algorithm == Algorithms.LOGISTIC_REGRESSION.name:
                    return jsonify(
                        {"info":"This algorithm cannot be further trained"}
                    ), 200

    # Train tree based model from scratch
    elif (
        model_algorithm == Algorithms.RANDOM_FOREST.name
        or model_algorithm == Algorithms.EXTRA_TREES.name
    ):
        return jsonify(
            {
                "name": "e.g. irismodel",
                "trees": "number of trees",
                "data": "data in .csv file",
            }
        )

    # Train incremental model from scratch
    elif (
        model_algorithm == Algorithms.LOGISTIC_REGRESSION.name
        or model_algorithm == Algorithms.SGD.name
    ):
        return jsonify(
            {
                "name": "e.g. irismodel",
                "classes": 'e.g. "dog", "cat"',
                "data": "data in .csv file",
            }
        )

    # Available algorithms
    else:
        available_algorithms = [a.to_dict() for a in const_algorithms]
        return available_algorithms,200
    return jsonify({"error": "Invalid query parameters"}), 400


@app.route("/api/models", methods=["GET"])
def get_models():
    models_info = get_all_models_info()

    return jsonify([model_info.to_dict() for model_info in models_info])


@app.route("/api/training-files", methods=["GET"])
def get_training_files():

    training_files = get_training_files_db()

    return jsonify([file.to_dict() for file in training_files])


if __name__ == "__main__":
    app.run(debug=True)
