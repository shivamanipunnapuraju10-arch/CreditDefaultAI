import pandas as pd
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load datasets
train_data = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "processed", "train.csv")
)

val_data = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "processed", "validation.csv")
)

# Separate features and target
X_train = train_data.drop(columns=["default payment next month"])
y_train = train_data["default payment next month"]

X_val = val_data.drop(columns=["default payment next month"])
y_val = val_data["default payment next month"]

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

# Train and evaluate models
results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    probabilities = model.predict_proba(X_val)[:, 1]

    results.append({

        "Model": name,

        "Accuracy": accuracy_score(y_val, predictions),

        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),

        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),

        "F1 Score": f1_score(
            y_val, predictions, zero_division=0
        ),

        "ROC AUC": roc_auc_score(y_val, probabilities),

        "PR AUC": average_precision_score(y_val, probabilities)
    })

# Display comparison
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

print("\nMODEL TRAINING AND COMPARISON COMPLETED!")