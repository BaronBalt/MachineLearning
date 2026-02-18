import io
from os import path
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier
from werkzeug.datastructures import FileStorage

from db.database import load_model, load_training, load_training_by_id, save_model_db, save_parameters, save_training, get_all_models_info, last_model_version
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
    last_version = last_model_version(model_name)
    is_continuation = False
    
    if last_version == 0:
    # Model doesn't exist, so version must be 1 or not provided
        version = 1
        # Here we need training data to create the first version of the model, so we check if it's provided
        if "file" in request.files:
            save_uploaded_file(request.files["file"])
            training_data_name = request.files["file"].filename
        elif "file_name" in request.form:
            training_data_name = request.form.get("file_name", "")
        else: 
            return jsonify({"error": "Model does not exist, therefore training data must be provided either with a file_name or a raw csv file"}), 400
        
        model = RandomForestClassifier(
            **MODEL_CONFIG,
            random_state=RANDOM_STATE,
            warm_start=True,
        )
        is_continuation = False

    else:
        version = last_version + 1
        # use last trainig data if version is not provided, otherwise use provided version and check if it exists
        model = load_model(model_name, last_version) 
        if model is None:
            return jsonify({"error": "Model not found for training continuation"}), 404
        training_data_name = load_training_by_id(model.training_data_id)
        is_continuation = True




    training_data, training_id = load_training(training_data_name)
    if training_data is None or training_id is None:
        return jsonify({"error": "Training data not found"}), 404

    # split training data
    X_train, X_val, y_train, y_val = load_and_split_bin_csv(training_data)

    # train and valuate
    trained_model = train_model(model, X_train, y_train, is_continuation)
    evaluation_metrics = evaluate_model(trained_model, X_val, y_val)
    # save model to db in form of bytes
    model_bytes = io.BytesIO()
    joblib.dump(trained_model, model_bytes)
    new_id = save_model_db(
        model_name,
        version,
        "RainforestClassifier",
        evaluation_metrics.accuracy,
        evaluation_metrics.precision,
        evaluation_metrics.recall,
        model_bytes.getvalue(),
        training_id
    )

    if new_id is None:
        return jsonify({"error": "Model with this name and version already exists"}), 400

    params = get_columns_from_csv(training_data)
    save_parameters(new_id, params)

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
