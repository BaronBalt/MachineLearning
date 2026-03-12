import io
from dataclasses import dataclass, field
from typing import List

import joblib


@dataclass
class TrainingFile:
    filename: str
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.filename.split(".")[0].capitalize()

    def to_dict(self):
        return {"name": self.name, "filename": self.filename}


@dataclass
class Parameter:
    name: str
    value: str

    def to_dict(self):
        return {"name": self.name, "value": self.value}


@dataclass
class Model:
    id: str
    name: str
    data: bytes
    version: int
    algorithm: str
    training_data_id: str

    def to_prediction_model(self):
        bundle = joblib.load(io.BytesIO(self.data))
        return bundle["pipeline"], bundle["features"]


@dataclass
class ModelInfo:
    id: int
    name: str
    description: str
    version: List[str]
    algorithm: str
    parameters: List[Parameter]
    url: str

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "algorithm": self.algorithm,
            "parameters": [p.to_dict() for p in self.parameters],
            "url": self.url,
        }
