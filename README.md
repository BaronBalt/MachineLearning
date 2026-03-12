# Backend

A Flask REST API for training, versioning, and serving machine learning models, backed by PostgreSQL.

## Running

**Start the API server**
```bash
python app.py
```

The server runs on `http://localhost:5000` by default.

**With Docker Compose** (starts API + PostgreSQL together)
```bash
docker-compose up
```

## Setup

**Install dependencies**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate
pip install -r requirements.txt
```

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `ML_DB_URL` | `postgresql://mluser:mlpass@localhost:5432/mlregistry` | PostgreSQL connection string |
| `APP_ENV` | `development` | Set to `production` to disable debug mode |

## API Endpoints

### Models
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/models` | List all trained models with versions and parameters |
| `GET` | `/api/training-files` | List available training datasets |
| `GET` | `/healthz` | Health check |

### Training
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/train` | Start a background training job — returns `{"job_id": "..."}` immediately |
| `GET` | `/api/train` | Get available algorithms (or info for further-training an existing model) |
| `GET` | `/api/train/status/<job_id>` | Poll job status: `pending`, `running`, `complete`, or `failed` |

Training runs in a background thread so requests never time out. Poll the status endpoint until `status` is `complete` or `failed`.

### Prediction
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/predict?model=<name>&version=<n>` | Run inference on a model version |
| `GET` | `/api/predict?model=<name>&version=<n>` | Get expected input format for a model |

## Features

- **Async training** — training jobs run in background threads; the API returns a `job_id` immediately and the client polls for completion.
- **Model versioning** — each training run creates a new version; tree-based models support continuation training (adding more trees).
- **Preprocessing pipeline** — numeric imputation and one-hot encoding are fitted at train time and bundled with the model for consistent inference.
- **Multiple algorithms** — Random Forest, Extra Trees, Logistic Regression, and SGD Classifier.
- **CSV upload or existing data** — training data can be uploaded as a CSV file or selected from previously uploaded datasets stored in the database.

## Project Structure

```
├── app.py                  Entry point
├── requirements.txt
├── api/
│   ├── __init__.py         Flask app factory (create_app)
│   ├── jobs.py             Background job store and training runner
│   └── routes/
│       ├── train.py        /api/train routes
│       ├── predict.py      /api/predict routes
│       └── models.py       /api/models, /training-files, /healthz
├── db/
│   ├── models.py           Data classes (Model, ModelInfo, Parameter, TrainingFile)
│   └── database.py         PostgreSQL query functions
├── src/
│   ├── config.py           Constants and configuration
│   ├── data.py             CSV loading and train/val splitting
│   ├── model.py            Algorithm definitions and training logic
│   └── evaluate.py         Accuracy, precision, recall evaluation
└── tests/
    └── test_train.py
```

## Tech Stack

| | |
|---|---|
| Framework | [Flask](https://flask.palletsprojects.com/) 3.1 |
| ML | [scikit-learn](https://scikit-learn.org/) 1.8 |
| Database | PostgreSQL via [psycopg](https://www.psycopg.org/) 3 |
| Serialisation | [joblib](https://joblib.readthedocs.io/) |
| Data | [pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) |

