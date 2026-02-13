import io
from os import path
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier
from werkzeug.datastructures import FileStorage

from db.database import load_model, load_training, save_model_db, save_parameters, save_training, get_all_models_info
from src.config import MODEL_CONFIG, RANDOM_STATE, TEST_SIZE
from src.data import get_columns_from_csv, load_and_split_bin_csv
from src.evaluate import evaluate_model
from src.model import train_model
from src.utils import get_latest_model_path

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


@app.route("/api/train", methods=["PUT", "POST"])
def put_train():
    """
    This route trains a new model or version of model
    The parameters for the training method are in the url parameters.
    """
    training_data_name = ""
    print(list(request.form.keys()))

    model_name = request.form.get("model")
    version = request.form.get("version")
    print(f"Received training request for model: {model_name}, version: {version}")
    print(f"Received files: {request.files}")
    if version is None:
        # fetch latest version and increment
        version = 1
    if "file" in request.files:
        save_uploaded_file(request.files["file"])
        training_data_name = request.files["file"].filename
    elif "file_name" in request.form:
        training_data_name = request.form.get("file_name", "")
    else:
        return jsonify({"error": "No data provided for training"}), 400
    if training_data_name == "":
        return jsonify({"error": "No data name provided in form"}), 400

    training_data = load_training(training_data_name)



    

    X_train, X_val, y_train, y_val = load_and_split_bin_csv(training_data)


    is_continuation = False
    found_model = load_model(model_name, int(version) if version else 0)
    model = RandomForestClassifier(
        **MODEL_CONFIG,
        random_state=RANDOM_STATE,
        warm_start=True,
    )
    if found_model and found_model.version == int(version):
        is_continuation = True
        model = joblib.load(found_model.data)


    trained_model = train_model(model, X_train, y_train, is_continuation)
    evaluation_metrics = evaluate_model(trained_model, X_val, y_val)
    model_bytes = io.BytesIO()
    joblib.dump(trained_model, model_bytes)

    new_id = save_model_db(
        model_name,
        version,
        "model",
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
    This route returns the various training parameters that can be used
    TODO: add things here
    """
    return jsonify({})


@app.route("/api/models", methods=["GET"])
def get_models():

    models_info = get_all_models_info()

    return jsonify([model_info.to_dict() for model_info in models_info])


if __name__ == "__main__":
    app.run(debug=True)
