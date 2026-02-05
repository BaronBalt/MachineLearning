import joblib
from sklearn.ensemble import RandomForestClassifier
from src.config import MODEL_CONFIG, RANDOM_STATE
from src.utils import get_latest_model_path

def get_or_init_model(model_name: str, models_dir: str):
    latest_path = get_latest_model_path(model_name, models_dir)

    if latest_path is not None:
        model = joblib.load(latest_path)

        # Ensure warm_start is enabled
        model.set_params(warm_start=True)

        print(f"Loaded model {latest_path.name}")
        return model, True
    else:
        model = RandomForestClassifier(
            **MODEL_CONFIG,
            random_state=RANDOM_STATE,
            warm_start=True,
        )
        print("No existing model found, starting fresh")
        return model, False


def train_model(model, x_train, y_train, is_continuation: bool, n_new_trees: int = 50):
    if is_continuation:
        model.set_params(
            n_estimators=model.n_estimators + n_new_trees
        )

    model.fit(x_train, y_train)
    return model
