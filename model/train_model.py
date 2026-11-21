import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load Dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_folder = os.path.join(BASE_DIR, "dataset")

excel_files = glob.glob(os.path.join(dataset_folder, "*.xls"))

if len(excel_files) == 0:
    print("Dataset not found!")
    exit()

file_path = excel_files[0]

data = pd.read_excel(file_path)

print("Dataset Loaded Successfully!")

# STEP 2: Data Cleaning

data.columns = data.iloc[0]
data = data.iloc[1:].reset_index(drop=True)

data = data.drop(columns=["ID"])

data = data.apply(pd.to_numeric, errors="coerce")

print("\nDATA CLEANING COMPLETED")
print("Dataset Shape:", data.shape)

# STEP 3: Data Analysis

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Records:")
print(data.duplicated().sum())

print("\nDefault Distribution:")
print(data["default payment next month"].value_counts())

# STEP 4: Exploratory Data Analysis

# Graph 1: Default Distribution

plt.figure(figsize=(6, 4))
sns.countplot(x="default payment next month", data=data)
plt.title("Default vs Non-Default Customers")
plt.show()

# Graph 2: Age Distribution

plt.figure(figsize=(7, 4))
sns.histplot(data["AGE"], bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.show()

# Graph 3: Credit Limit Distribution

plt.figure(figsize=(7, 4))
sns.histplot(data["LIMIT_BAL"], bins=30, kde=True)
plt.title("Credit Limit Distribution")
plt.show()

# Graph 4: Repayment Status vs Default

plt.figure(figsize=(7, 4))
sns.countplot(
    x="PAY_0",
    hue="default payment next month",
    data=data
)
plt.title("Repayment Status vs Default")
plt.show()

# Graph 5: Correlation Heatmap

plt.figure(figsize=(14, 10))
sns.heatmap(data.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

print("\nEDA ANALYSIS COMPLETED SUCCESSFULLY!")