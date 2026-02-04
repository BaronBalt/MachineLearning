from os import path

from flask import Flask, jsonify, request

from src.artifacts import load_model, save_metrics, save_model
from src.data import load_and_split_data
from src.evaluate import evaluate_model
from src.model import train_model

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv', 'txt'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    loaded_model = load_model(f"artifacts/{model}/v{version}/model.pkl")

    values = body.get("input", [])
    if len(values) == 0:
        return jsonify({"error": "No input data provided"}), 400

    X_input = [values]
    predict_result["result"] = loaded_model.predict(X_input).tolist()

    # fetch from model form db or something and predict based on body
    print(f"Received model: {model}, version: {version}")
    print(f"received body: {body}")
    predict_result["model"] = model
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


@app.route("/api/train", methods=["PUT"])
def put_train():
    """
    This route trains a new model or version of model
    The parameters for the training method are in the url parameters.
    """
    data_path = ""

    model = request.args.get("model")
    version = request.args.get("version")
    if version is None:
        # fetch latest version and increment
        version = 1
    if "file" in request.files:
        print("file")
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        # if path.exists(f"data/{file.filename}"):
        #     return jsonify({"error": "File already exists"}), 400
        file.save(f"data/{file.filename}")
        data_path = f"data/{file.filename}"
    else:
        # Fy fan va skit
        body = request.get_json() or {}
        print(f"received body: {body}")
        if body.get("data_path"):
            data_path = body.get("data_path")

    
    if not data_path or not isinstance(data_path, str) or not path.exists(data_path):
        return jsonify({"error": "No valid file to train on was found"}), 400
    X_train, X_val, y_train, y_val = load_and_split_data(data_path)

    trained_model = train_model(X_train, y_train)
    evaluation_metrics = evaluate_model(trained_model, X_val, y_val)
    save_model(trained_model, f"model/{model}/v{version}")
    save_metrics(evaluation_metrics, f"artifacts/{model}/v{version}")

    # trigger training for model version
    print(f"Training model: {model}, version: {version}")

    return jsonify({"evaluation_metrics": evaluation_metrics, "model": model, "version": version})


@app.route("/api/train", methods=["GET"])
def get_train_info():
    """
    This route returns the various training parameters that can be used
    TODO: add things here
    """
    return jsonify({})


@app.route("/api/models", methods=["GET"])
def get_models():
    new_models = []
    if models := request.args.getlist("model"):
        for model in models:
            # fetch model info from db or something
            model_info = {"model": model, "version": 1, "status": "available"}
            new_models.append(model_info)
        return jsonify({"models": new_models})
    # get all models
    return jsonify({"models": new_models})


if __name__ == "__main__":
    app.run(debug=True)
