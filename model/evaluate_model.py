import pandas as pd
import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load saved model
model_path = os.path.join(
    BASE_DIR, "saved_model", "random_forest_model.pkl"
)

info_path = os.path.join(
    BASE_DIR, "saved_model", "model_info.pkl"
)

model = joblib.load(model_path)
model_info = joblib.load(info_path)

threshold = model_info["threshold"]

# Load untouched test data
test_data = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "processed", "test.csv")
)

X_test = test_data.drop(columns=["default payment next month"])
y_test = test_data["default payment next month"]

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Apply optimized threshold
predictions = (probabilities >= threshold).astype(int)

# Calculate metrics
print("\n========== FINAL TEST RESULTS ==========")

print("Model:", model_info["model_name"])
print("Decision Threshold:", threshold)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 Score:", round(f1_score(y_test, predictions), 4))
print("ROC AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("PR AUC:", round(average_precision_score(y_test, probabilities), 4))

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Calculate test cost
fn_cost = model_info["false_negative_cost"]
fp_cost = model_info["false_positive_cost"]

total_cost = (fn * fn_cost) + (fp * fp_cost)

print("\nTotal Cost:", total_cost)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nFINAL MODEL EVALUATION COMPLETED!")