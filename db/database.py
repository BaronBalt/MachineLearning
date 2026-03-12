import os
from typing import List, Optional

import psycopg

from db.models import Model, ModelInfo, Parameter, TrainingFile

DB_URL = os.getenv("ML_DB_URL", "postgresql://mluser:mlpass@localhost:5432/mlregistry")


def _query(sql: str, params: tuple = ()):
    """Execute a SELECT and return all rows."""
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def _execute(sql: str, params: tuple = ()) -> Optional[tuple]:
    """Execute a write statement and return the first row (e.g. RETURNING id)."""
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def save_parameters(model_id, parameters: List[Parameter]):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO model_parameters (model_id, name, value)
                VALUES (%s, %s, %s)
                ON CONFLICT (model_id, name, value) DO NOTHING
                """,
                [(model_id, p.name, p.value) for p in parameters],
            )


def get_parameters_for_model(model_id) -> List[Parameter]:
    rows = _query(
        "SELECT name, value FROM model_parameters WHERE model_id = %s",
        (model_id,),
    )
    return [Parameter(name=row[0], value=row[1]) for row in rows]


def save_model_db(name, version, algorithm, accuracy, precision, recall, model_data, training_data_id) -> Optional[str]:
    """Insert a model row and return the new id, or None if it already exists."""
    result = _execute(
        """
        INSERT INTO model (name, version, algorithm, accuracy, precision, recall, model_data, training_data_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (name, version) DO NOTHING
        RETURNING id
        """,
        (name, version, algorithm, accuracy, precision, recall, model_data, training_data_id),
    )
    return result[0] if result else None


def load_model(name, version: int = 0) -> Optional[Model]:
    if version != 0:
        sql = "SELECT id, model_data, version, algorithm, training_data_id FROM model WHERE name = %s AND version = %s"
        params = (name, str(version))
    else:
        sql = "SELECT id, model_data, version, algorithm, training_data_id FROM model WHERE name = %s ORDER BY version DESC"
        params = (name,)

    rows = _query(sql, params)
    if not rows:
        return None
    row = rows[0]
    return Model(id=row[0], name=name, data=row[1], version=row[2], algorithm=row[3], training_data_id=row[4])


def get_all_models_info() -> List[ModelInfo]:
    rows = _query("SELECT id, name, version, algorithm FROM model")

    grouped: dict = {}
    for model_id, name, version, algorithm in rows:
        key = (name, algorithm)
        if key not in grouped:
            grouped[key] = {"first_id": model_id, "name": name, "algorithm": algorithm, "versions": []}
        grouped[key]["versions"].append(version)

    output = []
    for idx, data in enumerate(grouped.values()):
        params = get_parameters_for_model(str(data["first_id"]))
        output.append(ModelInfo(
            id=idx,
            name=data["name"],
            description="description",
            version=data["versions"],
            algorithm=data["algorithm"],
            parameters=params,
            url="/api/predict",
        ))
    return output


def does_model_exist(name, version=None) -> bool:
    try:
        if version is None:
            rows = _query("SELECT 1 FROM model WHERE name = %s", (name,))
        else:
            rows = _query("SELECT 1 FROM model WHERE name = %s AND version = %s", (name, str(version)))
        return len(rows) > 0
    except psycopg.Error as e:
        print("Database error in does_model_exist:", e)
        return False


def last_model_version(name) -> int:
    rows = _query(
        "SELECT version FROM model WHERE name = %s ORDER BY version DESC LIMIT 1",
        (name,),
    )
    return rows[0][0] if rows else 0


def save_training(name, data) -> Optional[str]:
    result = _execute(
        """
        INSERT INTO training_data (name, data)
        VALUES (%s, %s)
        ON CONFLICT (name) DO NOTHING
        RETURNING id
        """,
        (name, data),
    )
    return result[0] if result else None


def load_training(name):
    rows = _query("SELECT data, id FROM training_data WHERE name = %s", (name,))
    if not rows:
        return None, None
    return rows[0][0], rows[0][1]


def load_training_by_id(id) -> Optional[str]:
    rows = _query("SELECT name FROM training_data WHERE id = %s", (id,))
    return rows[0][0] if rows else None


def get_training_files_db() -> List[TrainingFile]:
    rows = _query("SELECT name FROM training_data")
    return [TrainingFile(filename=row[0]) for row in rows]

