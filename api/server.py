from flask import Flask, jsonify, request

app = Flask(__name__)

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
    model = request.args.get("model")
    version = request.args.get("version")
    if version is None:
        # fetch latest version and increment
        version = 1
    # training data to train on
    # maybe could also point to file on server?
    body = request.get_json() or {}

    # trigger training for model version
    print(f"Training model: {model}, version: {version}")
    print(f"received body: {body}")

    return jsonify({"status": "training started", "model": model, "version": version})


@app.route("/api/train", methods=["GET"])
def get_train_info():
    """
    This route returns the various training parameters that can be used
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
