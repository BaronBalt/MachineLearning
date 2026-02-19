from enum import Enum

import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from db.database import load_model, get_all_models_info
from src.config import MODEL_CONFIG, RANDOM_STATE


class Algorithms(Enum):
    RANDOM_FOREST = "RANDOM_FOREST",
    EXTRA_TREES = "EXTRA_TREES",
    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION",
    SGD = "SGD"

class AlgorithmParameters:
    name: str
    label: str
    default_value: str
    param_type: str
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
            "param_type": self.param_type
        }

class Algorithm:
    id: Algorithms
    name: str
    parameters: List[AlgorithmParameters]
    def __init__(self, id: Algorithms, name: str, parameters: List[AlgorithmParameters]):
        self.id = id
        self.name = name
        self.parameters = parameters

    def to_dict(self):
        return {
            "id": self.id.value[0],
            "name": self.name,
            "parameters": [p.__dict__ for p in self.parameters]
        }

const_algorithms = [
    Algorithm(Algorithms.RANDOM_FOREST, "Random Forest", []),
    Algorithm(Algorithms.EXTRA_TREES, "Extra Trees", []),
    Algorithm(Algorithms.LOGISTIC_REGRESSION, "Logistic Regression", [AlgorithmParameters("classes", "Classes (List)", "[1,2,3]", "text")]),
    # Algorithm(Algorithms.SGD, "SGD", [AlgorithmParameters("classes", "Classes (List)", "[1,2,3]", "text")]),
]
    

def get_or_init_model(algorithm: Algorithms, name: str, version: int = 0):
    models = get_all_models_info()
    for m in models:
        if m.name == name:
            model = load_model(name, version)
            print(f"Loaded model {model.name if model else 'unknown'}")
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


def impute_data(x_train, x_test):

    # Convert x_train to DataFrame if not already
    if not isinstance(x_train, pd.DataFrame):
        x_train = pd.DataFrame(x_train)

    # Detect column types
    numeric_features = x_train.select_dtypes(include=['number']).columns.tolist()
    categorical_features = x_train.select_dtypes(exclude=['number']).columns.tolist()

    # Define transformers
    numeric_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit and transform x_train
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    return x_train_processed, x_test_processed

