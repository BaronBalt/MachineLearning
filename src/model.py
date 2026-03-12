from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import MODEL_CONFIG, RANDOM_STATE


class Algorithms(Enum):
    RANDOM_FOREST = "RANDOM_FOREST"
    EXTRA_TREES = "EXTRA_TREES"
    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"
    SGD = "SGD"


class AlgorithmParameters:
    def __init__(self, name: str, label: str, default_value: str, param_type: str):
        self.name = name
        self.label = label
        self.default_value = default_value
        self.param_type = param_type

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "default_value": self.default_value,
            "param_type": self.param_type,
        }


class Algorithm:
    def __init__(self, id: Algorithms, name: str, parameters: List[AlgorithmParameters]):
        self.id = id
        self.name = name
        self.parameters = parameters

    def to_dict(self):
        return {
            "id": self.id.value,
            "name": self.name,
            "parameters": [p.to_dict() for p in self.parameters],
        }


const_algorithms = [
    Algorithm(Algorithms.RANDOM_FOREST, "Random Forest", [
        AlgorithmParameters("trees", "Amount of trees to add on continuation", "50", "number"),
    ]),
    Algorithm(Algorithms.EXTRA_TREES, "Extra Trees", [
        AlgorithmParameters("trees", "Amount of trees to add on continuation", "50", "number"),
    ]),
    Algorithm(Algorithms.LOGISTIC_REGRESSION, "Logistic Regression", [
        AlgorithmParameters("classes", "Classes (List)", "[1,2,3]", "text"),
    ]),
]

_MODEL_FACTORY = {
    Algorithms.RANDOM_FOREST:        lambda: RandomForestClassifier(**MODEL_CONFIG, random_state=RANDOM_STATE),
    Algorithms.EXTRA_TREES:          lambda: ExtraTreesClassifier(**MODEL_CONFIG, random_state=RANDOM_STATE),
    Algorithms.LOGISTIC_REGRESSION:  lambda: LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    Algorithms.SGD:                  lambda: SGDClassifier(random_state=RANDOM_STATE),
}


def create_model(algorithm: Algorithms):
    """Factory: return a fresh sklearn estimator for the given algorithm."""
    factory = _MODEL_FACTORY.get(algorithm)
    if factory is None:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return factory()


def _train_tree(model, X_train, y_train, is_continuation: bool, n_new_trees: int):
    if is_continuation:
        model.set_params(n_estimators=model.n_estimators + n_new_trees, warm_start=True)
    else:
        model.set_params(warm_start=True)
    model.fit(X_train, y_train)
    return model

def _train_incremental(model, X_train, y_train, classes: Optional[np.ndarray]):
    if hasattr(model, "partial_fit"):
        known = np.unique(y_train)
        all_classes = np.union1d(classes, known) if classes is not None else known
        model.partial_fit(X_train, y_train, classes=all_classes)
    else:
        model.fit(X_train, y_train)
    return model


def train_model(model, X_train, y_train, is_continuation: bool,
                n_new_trees: int = 50, classes: Optional[np.ndarray] = None):
    if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
        return _train_tree(model, X_train, y_train, is_continuation, n_new_trees)
    if isinstance(model, (SGDClassifier, LogisticRegression)):
        return _train_incremental(model, X_train, y_train, classes)
    # fallback
    model.fit(X_train, y_train)
    return model


def impute_data(X_train, X_test):
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="mean"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_test_processed, preprocessor

