import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_DIR = "bank_marketing_preprocessing"


def load_data():
    x_train_path = os.path.join(DATA_DIR, "X_train.csv")
    x_test_path = os.path.join(DATA_DIR, "X_test.csv")
    y_train_path = os.path.join(DATA_DIR, "y_train.csv")
    y_test_path = os.path.join(DATA_DIR, "y_test.csv")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    return X_train, X_test, y_train, y_test


def main():
    # IMPORTANT:
    # Do not use mlflow.set_experiment() inside MLflow Project.
    # mlflow run already creates/manages the active run.

    X_train, X_test, y_train, y_test = load_data()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("Training completed successfully.")
        print(f"Accuracy : {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall   : {recall}")
        print(f"F1-score : {f1}")


if __name__ == "__main__":
    main()