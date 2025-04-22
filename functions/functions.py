import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelResults:
    def __init__(self):
        self.results = pd.DataFrame()

    def add(self, model, model_name, X_train, y_train, X_test, y_test, average='binary'):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "Model": model_name,
            "Train Accuracy": accuracy_score(y_train, y_train_pred),
            "Test Accuracy": accuracy_score(y_test, y_test_pred),
            "Train Precision": precision_score(y_train, y_train_pred, average=average),
            "Test Precision": precision_score(y_test, y_test_pred, average=average),
            "Train Recall": recall_score(y_train, y_train_pred, average=average),
            "Test Recall": recall_score(y_test, y_test_pred, average=average),
            "Train F1": f1_score(y_train, y_train_pred, average=average),
            "Test F1": f1_score(y_test, y_test_pred, average=average)
        }

        self.results = pd.concat([self.results, pd.DataFrame([metrics])], ignore_index=True)

        # Optional: print inline
        self._print_summary(model_name, metrics)

    def _print_summary(self, model_name, metrics):
        print(f"\n= CLASSIFICATION METRICS COMPARISON: {model_name} =")
        print(f"{'Metric':<15} | {'Train':>8} | {'Test':>8}")
        print("-" * 40)
        for key in ["Accuracy", "Precision", "Recall", "F1"]:
            print(f"{key:<15} | {metrics[f'Train {key}']:.4f} | {metrics[f'Test {key}']:.4f}")

    def get_results(self):
        return self.results

    def reset(self):
        self.results = pd.DataFrame()        