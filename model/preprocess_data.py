
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split

# Find project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Find dataset
dataset_folder = os.path.join(BASE_DIR, "dataset")
excel_files = glob.glob(os.path.join(dataset_folder, "*.xls"))

if len(excel_files) == 0:
    print("Dataset not found!")
    exit()

# Load dataset
data = pd.read_excel(excel_files[0])

# Correct column names
data.columns = data.iloc[0]
data = data.iloc[1:].reset_index(drop=True)

# Remove ID column
data = data.drop(columns=["ID"])

# Convert columns into numeric values
data = data.apply(pd.to_numeric, errors="coerce")

# Separate input and output
X = data.drop(columns=["default payment next month"])
y = data["default payment next month"]

# Split into training (60%) and temporary data (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Split temporary data into validation (20%) and testing (20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Create processed data folder
output_folder = os.path.join(BASE_DIR, "dataset", "processed")
os.makedirs(output_folder, exist_ok=True)

# Save datasets
train_data = X_train.copy()
train_data["default payment next month"] = y_train

val_data = X_val.copy()
val_data["default payment next month"] = y_val

test_data = X_test.copy()
test_data["default payment next month"] = y_test

train_data.to_csv(os.path.join(output_folder, "train.csv"), index=False)
val_data.to_csv(os.path.join(output_folder, "validation.csv"), index=False)
test_data.to_csv(os.path.join(output_folder, "test.csv"), index=False)

# Display results
print("DATA SPLITTING COMPLETED!")

print("\nTraining Shape:", train_data.shape)
print("Validation Shape:", val_data.shape)
print("Testing Shape:", test_data.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nValidation Target Distribution:")
print(y_val.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nAll three datasets saved successfully!")
