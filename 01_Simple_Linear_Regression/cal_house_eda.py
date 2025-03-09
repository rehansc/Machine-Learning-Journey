import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.datasets import fetch_california_housing

cal = fetch_california_housing()
df = pd.DataFrame(cal.data, columns=cal.feature_names)
df["Target"] = cal.target

print(df.head())

missing_values = df.isnull().sum()
print(missing_values)

describe = df.describe()

print(describe)

# Check data types
info = df.info()
print(info)

cor_matrix = df.corr()
plt.figure(figsize=(10,6))
sb.heatmap(cor_matrix, annot=True, cmap='coolwarm',fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()


# Histogram for all features
df.hist(figsize=(12, 8), bins=30, edgecolor="black")
plt.suptitle("Feature Distributions")
plt.show()

plt.figure(figsize=(8,5))
sb.scatterplot(x=df["MedInc"], y=df["Target"])
plt.xlabel("Median Income")
plt.ylabel("Target")
plt.title("Relationship between Median Income and Target")
plt.show()


# Boxplot to detect outliers
plt.figure(figsize=(12, 6))
sb.boxplot(data=df, orient="h")
plt.title("Boxplot of Features (Outlier Detection)")
plt.show()


Q1 = df["Population"].quantile(0.25)  # 25th percentile
Q3 = df["Population"].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile Range

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
df_no_outliers = df[(df["Population"] >= lower_bound) & (df["Population"] <= upper_bound)]

print(f"Original dataset size: {df.shape[0]} rows")
print(f"Dataset size after removing outliers: {df_no_outliers.shape[0]} rows")


# Boxplot to detect outliers
plt.figure(figsize=(12, 6))
sb.boxplot(data=df_no_outliers, orient="h")
plt.title("Boxplot of Features (Outlier Detection)")
plt.show()

