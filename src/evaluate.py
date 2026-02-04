from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="weighted"),
        "recall": recall_score(y_val, y_pred, average="weighted"),
    }
