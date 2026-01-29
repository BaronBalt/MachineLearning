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
    return jsonify({"status": "healthy"})


@app.route("/api/predict", methods=["POST"])
def post_predict():
    model = request.args.get("model")
    version = request.args.get("version")
    body = request.get_json() or predict_body_example

    # fetch from model form db or something and predict based on body
    print(f"Received model: {model}, version: {version}")
    print(f"received body: {body}")
    predict_result["model"] = model
    predict_result["version"] = version

    return jsonify(predict_result)


@app.route("/api/train", methods=["PUT"])
def put_train():
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
