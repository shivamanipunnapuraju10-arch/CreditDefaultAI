import pandas as pd
import os
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load training and validation data
train_data = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "processed", "train.csv")
)

val_data = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "processed", "validation.csv")
)

X_train = train_data.drop(columns=["default payment next month"])
y_train = train_data["default payment next month"]

X_val = val_data.drop(columns=["default payment next month"])
y_val = val_data["default payment next month"]

# Define cost assumptions
FN_COST = 5
FP_COST = 1

# Define models
models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

all_results = []

# Train and find best threshold
for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    best_cost = float("inf")
    best_threshold = None
    best_metrics = None

    for threshold in np.arange(0.05, 0.96, 0.01):

        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1]
        ).ravel()

        total_cost = (FN_COST * fn) + (FP_COST * fp)

        if total_cost < best_cost:

            best_cost = total_cost
            best_threshold = threshold

            best_metrics = {
                "Precision": precision_score(
                    y_val, predictions, zero_division=0
                ),

                "Recall": recall_score(
                    y_val, predictions, zero_division=0
                ),

                "F1 Score": f1_score(
                    y_val, predictions, zero_division=0
                ),

                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp
            }

    all_results.append({
        "Model": name,
        "Best Threshold": best_threshold,
        "Total Cost": best_cost,
        **best_metrics
    })

# Display results
results_df = pd.DataFrame(all_results)

print("\n========== COST-SENSITIVE MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

print("\nCOST-SENSITIVE THRESHOLD OPTIMIZATION COMPLETED!")