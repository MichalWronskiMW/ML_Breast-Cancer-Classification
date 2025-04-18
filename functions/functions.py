import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_classifier(model, model_name, X_train, y_train, X_test, y_test, average='binary', save_to_table=True):
    """
    Evaluate classification model, print metrics, and optionally save to results table.

    Args:
        model: Trained model
        model_name: String name of the model
        X_train, y_train: Training data
        X_test, y_test: Test data
        average: Averaging method for precision, recall, F1
        save_to_table: Whether to save results to the global model_results DataFrame

    Returns:
        None (prints metrics and optionally updates the results table)
    """
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

    # Print comparison table
    print(f"\n= CLASSIFICATION METRICS COMPARISON: {model_name} =")
    print(f"{'Metric':<15} | {'Train':>8} | {'Test':>8}")
    print("-" * 40)
    for key in ["Accuracy", "Precision", "Recall", "F1"]:
        print(f"{key:<15} | {metrics[f'Train {key}']:.4f} | {metrics[f'Test {key}']:.4f}")

    # Save to global table
    if save_to_table:
        global model_results
        model_results = pd.concat([model_results, pd.DataFrame([metrics])], ignore_index=True)
        
        