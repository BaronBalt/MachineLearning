import argparse
from src.data import load_and_split_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.artifacts import save_model, save_metrics

def main(data_path, output_dir):
    X_train, X_val, y_train, y_val = load_and_split_data(data_path)

    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_val, y_val)

    model_path = save_model(model, output_dir)
    metrics_path = save_metrics(metrics, output_dir)

    print("Training complete")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="artifacts")

    args = parser.parse_args()
    main(args.data_path, args.output_dir)
