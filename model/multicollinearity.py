import pandas as pd
import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load training dataset only
file_path = os.path.join(
    BASE_DIR, "dataset", "processed", "train.csv"
)

data = pd.read_csv(file_path)

# Separate input features
X = data.drop(columns=["default payment next month"])

print("MULTICOLLINEARITY ANALYSIS")
print("Number of Features:", X.shape[1])

# STEP 1: Correlation Analysis
correlation = X.corr()

print("\nHighly Correlated Feature Pairs:")

found = False

for i in range(len(correlation.columns)):
    for j in range(i + 1, len(correlation.columns)):

        value = correlation.iloc[i, j]

        if abs(value) > 0.80:
            print(
                correlation.columns[i],
                "and",
                correlation.columns[j],
                "Correlation:",
                round(value, 3)
            )
            found = True

if not found:
    print("No highly correlated pairs found.")

# STEP 2: VIF Analysis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

vif_data = pd.DataFrame()

vif_data["Feature"] = X.columns

vif_data["VIF"] = [
    variance_inflation_factor(X_scaled, i)
    for i in range(X_scaled.shape[1])
]

vif_data = vif_data.sort_values(
    by="VIF",
    ascending=False
)

print("\nVARIANCE INFLATION FACTOR (VIF)")
print(vif_data.to_string(index=False))

print("\nMULTICOLLINEARITY ANALYSIS COMPLETED!")