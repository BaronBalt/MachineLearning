import os
from typing import List, Tuple

import psycopg

DB_URL = os.getenv("ML_DB_URL", "postgresql://mluser:mlpass@localhost:5432/mlregistry")


class Model:
    id: str  # uuid
    name: str
    data: bytes
    version: int

    def __init__(self, id, name, data, version):
        self.id = id
        self.name = name
        self.data = data
        self.version = version


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
    algorithm: str
    parameters: List[Parameter]
    url: str

    def __init__(self,id, name, description, version, algorithm, parameters, url):
        self.id = id
        self.name = name
        self.description = description
        self.version = version
        self.algorithm = algorithm
        self.parameters = parameters
        self.url = url

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "algorithm": self.algorithm,
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


def save_model_db(name, version, algorithm, accuracy, precision, recall, model_data):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model (name, version, algorithm, accuracy, precision, recall, model_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name, version) DO NOTHING
                    RETURNING id
                """,
                (name, version, algorithm, accuracy, precision, recall, model_data),
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
                    SELECT id, model_data, version FROM model
                    WHERE name = %s AND version = %s
                    """,
                    (name, str(version)),
                )
            else:
                cur.execute(
                    """
                    SELECT id, model_data, version FROM model
                    WHERE name = %s
                    ORDER BY version DESC
                    """,
                    (name,),
                )
            result = cur.fetchone()
            model = Model(result[0], name, result[1], result[2]) if result else None
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
                SELECT id, name, version, algorithm FROM model
                """
            )
            models = cur.fetchall()

    model_dict = {}

    for model_id, name, version, algorithm in models:
        key: Tuple[str, str] = (name, algorithm)

        if key not in model_dict:
            model_dict[key] = {
                "ids": [model_id],
                "name": name,
                "algorithm": algorithm,
                "versions": [version],
            }
        else:
            model_dict[key]["ids"].append(model_id)
            model_dict[key]["versions"].append(version)



    output: List[ModelInfo] = []
    model_info_id = 0

    for (_, _), data in model_dict.items():
        params = get_parameters_for_model(str(data["ids"][0]))

        model_info = ModelInfo(
            model_info_id,
            data["name"],
            "description",
            data["versions"],
            data["algorithm"],
            params,
            "/api/predict",
        )

        output.append(model_info)
        model_info_id += 1

    return output


def does_model_exist(name, version=1) -> bool:
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM model WHERE name = %s AND version = %s",
                    (name, str(version))
                )
                row = cur.fetchone()
                return row is not None
    except psycopg.Error as e:
        print("Database error in does_model_exist:", e)
        return False




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
                """,
                (name, data),
            )


def load_training(name):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT data FROM training_data
                WHERE name = %s
                """,
                (name,),
            )
            result = cur.fetchone()
            return result[0] if result else None
