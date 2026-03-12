from flask import Flask

from src.config import UPLOAD_FOLDER


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    from api.routes.models import bp as models_bp
    from api.routes.predict import bp as predict_bp
    from api.routes.train import bp as train_bp

    app.register_blueprint(models_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(train_bp)

    return app
