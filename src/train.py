import argparse

from src.artifacts import save_model, save_metrics
from src.data import load_and_split_data
from src.evaluate import evaluate_model
from src.model import train_model, get_or_init_model


def main(model_name, algorithm, data_path):
    x_train, x_val, y_train, y_val = load_and_split_data(data_path)

    model, is_continuation = get_or_init_model(algorithm, model_name)

    model = train_model(model, x_train, y_train, is_continuation)

    metrics = evaluate_model(model, x_val, y_val)

    model_path = save_model(model, model_name)
    metrics_path = save_metrics(metrics, model_name)

    print("Training complete")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelName", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--dataPath", required=True)

    args = parser.parse_args()
    main(args.modelName, args.algorithm, args.dataPath)
