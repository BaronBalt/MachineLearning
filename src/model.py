from sklearn.ensemble import RandomForestClassifier
from src.config import MODEL_CONFIG, RANDOM_STATE

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        **MODEL_CONFIG,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model
