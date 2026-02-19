import io
import os
from typing import List

import joblib
import psycopg

DB_URL = os.getenv("ML_DB_URL", "postgresql://mluser:mlpass@localhost:5432/mlregistry")


class Model:
    id: str  # uuid
    name: str
    data: bytes
    version: int
    training_data_id: str # uuid

    def __init__(self, id, name, data, version, training_data_id):
        self.id = id
        self.name = name
        self.data = data
        self.version = version
        self.training_data_id = training_data_id

    def to_prediction_model(self):
        return joblib.load(io.BytesIO(self.data))


class Parameter:
    value: str
    name: str

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
        }


class ModelInfo:
    id: int
    name: str
    description: str
    version: List[str]
    parameters: List[Parameter]
    url: str

    def __init__(self,id, name, description, version, parameters, url):
        self.id = id
        self.name = name
        self.description = description
        self.version = version
        self.parameters = parameters
        self.url = url

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": [param.to_dict() for param in self.parameters],
            "url": self.url,
        }


def save_parameters(model_id, parameters: List[Parameter]):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            args = [(model_id, param.name, param.value) for param in parameters]
            cur.executemany(
                """
                INSERT INTO model_parameters (model_id, name, value)
                VALUES (%s,%s,%s)
                    ON CONFLICT (model_id, name, value) DO NOTHING
                """,
                args,
            )


def get_parameters_for_model(model_id) -> List[Parameter]:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, value FROM model_parameters
                WHERE model_id = %s
                """,
                (model_id,),
            )
            results = cur.fetchall()

    params = []
    for row in results:
        rows_param = Parameter(name=row[0], value=row[1])
        params.append(rows_param)

    return params


def save_model_db(name, version, algorithm, accuracy, precision, recall, model_data, training_data_id):
    """
    SAVE TRANING DATA BEFORE THIS
    """
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model (name, version, algorithm, accuracy, precision, recall, model_data, training_data_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name, version) DO NOTHING
                    RETURNING id
                """,
                (name, version, algorithm, accuracy, precision, recall, model_data, training_data_id),
            )
            result = cur.fetchone()
            if result:
                return result[0]  # newly inserted model id

    return None


def load_model(name, version: int = 0) -> Model | None:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            if version != 0:
                cur.execute(
                    """
                    SELECT id, model_data, version, training_data_id FROM model
                    WHERE name = %s AND version = %s
                    """,
                    (name, str(version)),
                )
            else:
                cur.execute(
                    """
                    SELECT id, model_data, version, training_data_id FROM model
                    WHERE name = %s
                    ORDER BY version DESC
                    """,
                    (name,),
                )
            result = cur.fetchone()
            predictionModel = joblib.load(io.BytesIO(result[1]))
            model = Model(result[0], name, result[1], result[2], result[3]) if result else None
            print(
                f"Loaded model: {model.name}, version: {model.version}"
                if model
                else "No model found"
            )
            return model


def get_all_models_info() -> List[ModelInfo]:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, version FROM model
                """
            )
            models = cur.fetchall()

    modelDict = {}
    for model in models:
        model_id, name, version = model
        if not modelDict.get(name):
            modelDict[name] = {"versions": [version], "id": model_id}
        else:
            modelDict[name]["versions"].append(version)

    output = []
    id = 0
    for model in modelDict:
        print(modelDict[model]["id"])
        params = get_parameters_for_model(str(modelDict[model]["id"]))
        modelDict[model]["parameters"] = params
        modelInfo = ModelInfo(
            id,
            model,
            "description",
            modelDict[model]["versions"],
            params,
            "/api/predict",
        )
        id += 1
        output.append(modelInfo)
    return output


def does_model_exist(name, version):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM model
                WHERE name = %s AND version = %s
                """,
                (name, version),
            )
            result = cur.fetchone()
            return result is not None


def last_model_version(name):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT version FROM model
                WHERE name = %s
                """,
                (name,),
            )

            result = cur.fetchone()
            return result[0] if result else 0


def save_training(name, data):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_data (name, data)
                VALUES (%s, %s)
                    ON CONFLICT (name) DO NOTHING
                RETURNING id
                """,
                (name, data),
            )
            result = cur.fetchone()
            if result:
                return result[0]  # newly inserted training_data id

            


def load_training(name):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT data, id FROM training_data
                WHERE name = %s
                """,
                (name,),
            )
            result = cur.fetchone()
            return result[0] if result else None, result[1] if result else None

def load_training_by_id(id):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name FROM training_data
                WHERE id = %s
                """,
                (id,),
            )
            result = cur.fetchone()
            return result[0] if result else None
