import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_data = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "processed", "train.csv")
)

X_train = train_data.drop(columns=["default payment next month"])
y_train = train_data["default payment next month"]

# Train selected model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save model
model_folder = os.path.join(BASE_DIR, "saved_model")
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, "random_forest_model.pkl")

joblib.dump(model, model_path)

# Save threshold and cost assumptions
model_info = {
    "threshold": 0.34,
    "false_negative_cost": 5,
    "false_positive_cost": 1,
    "model_name": "Random Forest",
    "features": list(X_train.columns)
}

joblib.dump(
    model_info,
    os.path.join(model_folder, "model_info.pkl")
)

print("BEST MODEL SAVED SUCCESSFULLY!")
print("Model: Random Forest")
print("Threshold: 0.34")
print("Model location:", model_path)