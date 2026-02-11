from sklearn.metrics import accuracy_score, precision_score, recall_score


class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float

    def __init__(self, accuracy: float, precision: float, recall: float):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
    def to_dict(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
        }
    

def evaluate_model(model, X_val, y_val) -> EvaluationMetrics:
    y_pred = model.predict(X_val)

    return  EvaluationMetrics(
        accuracy=accuracy_score(y_val, y_pred),
        precision=precision_score(y_val, y_pred, average="weighted"),
        recall=recall_score(y_val, y_pred, average="weighted"),
    )
