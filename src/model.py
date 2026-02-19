from enum import Enum

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

from db.database import load_model, get_all_models_info
from src.config import MODEL_CONFIG, RANDOM_STATE


class Algorithms(Enum):
    RANDOM_FOREST = "RANDOM_FOREST",
    EXTRA_TREES = "EXTRA_TREES",
    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION",
    SGD = "SGD"


def get_or_init_model(algorithm: Algorithms, name: str, version: int = 0):
    models = get_all_models_info()
    for m in models:
        if m.name == name:
            model = load_model(name, version)
            print(f"Loaded model {model.name}")
            return model, True

    model = create_model(algorithm)
    print("No existing model found, starting fresh")
    return model, False

def create_model(algorithm: Algorithms):
    """Factory function to create a model based on the selected algorithm."""
    if algorithm == Algorithms.RANDOM_FOREST:
        return RandomForestClassifier(
            **MODEL_CONFIG,
            random_state=RANDOM_STATE
        )

    if algorithm == Algorithms.EXTRA_TREES:
        return ExtraTreesClassifier(
            **MODEL_CONFIG,
            random_state=RANDOM_STATE
        )

    if algorithm == Algorithms.LOGISTIC_REGRESSION:
        return LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000  # increase if needed
        )

    if algorithm == Algorithms.SGD:
        return SGDClassifier(
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def train_model(model, x_train, y_train, is_continuation: bool, n_new_trees: int = 50, classes: np.ndarray | None = None):
    """
    Train a model, supporting continuation for different types of algorithms.

    Parameters
    ----------
    model:
        sklearn estimator
    x_train:
        training data
    y_train:
        training data
    is_continuation: bool
        whether to continue training an existing model
    n_new_trees: int
        number of new trees to add for tree-based ensembles
    classes:
        required for partial_fit models (SGD, LogisticRegression)
    """

    # 1. Tree-based ensembles: RandomForest, ExtraTrees
    if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
        if is_continuation:
            # Increase number of estimators and reuse existing trees
            model.set_params(n_estimators=model.n_estimators + n_new_trees, warm_start=True)
        else:
            model.set_params(warm_start=True)

        model.fit(x_train, y_train)

    # 2. Incremental models: SGD, LogisticRegression
    elif isinstance(model, (SGDClassifier, LogisticRegression)):
        # LogisticRegression and SGD can use partial_fit for continuation
        if is_continuation:
            if classes is None:
                # derive classes from data if not provided
                classes = np.unique(y_train)
            model.partial_fit(x_train, y_train, classes=classes)
        else:
            if hasattr(model, "partial_fit"):
                # initial training with partial_fit
                if classes is None:
                    classes = np.unique(y_train)
                model.partial_fit(x_train, y_train, classes=classes)
            else:
                model.fit(x_train, y_train)

    else:
        # fallback: just fit
        model.fit(x_train, y_train)

    return model
