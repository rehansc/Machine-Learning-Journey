# Overview

This project applies simple linear regression to predict California housing prices using the MedInc (median income) feature, which has the highest correlation with the target variable (price). The goal is to perform Exploratory Data Analysis (EDA), train a regression model, evaluate it, and visualize the results.

## Dataset Used

Source: Scikit-learn's fetch_california_housing() dataset

Data Set Characteristics:
  - Number of Instances: 20640
  - Number of Attributes: 8 numeric, predictive attributes and the target
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
  - Missing Attribute Values:None
- Target: Median house price

## Exploratory Data Analysis (EDA)Before running regression, I conducted EDA:

The python code to conduct the eda:
```python
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

```
  - Correlation Analysis: MedInc had the highest correlation (0.69) with housing prices. 

![Correlation Matrix](https://github.com/rehansc/Machine-Learning-Journey/blob/main/01_Simple_Linear_Regression/Corr.png?raw=True)

  - Box Plot Analysis: Detected outliers in Population, which were handled. There are outliers in other features as well as the target value. Need to address that in the later time.

![Box Plot](https://github.com/rehansc/Machine-Learning-Journey/blob/main/01_Simple_Linear_Regression/box_plot.png?raw=True)

  - Histogram Distribution: Verified skewness in certain features. The MedInc does show a normal distribution with some skewness. The other features do not show any normal distribution. 

![Histrogram Plot](https://github.com/rehansc/Machine-Learning-Journey/blob/main/01_Simple_Linear_Regression/feature_dist.png?raw=True)

## Simple Linear Regression

Since there is a relative high correlation between MedInc and Target, I am going to conduct simple linear regression. The following is the python code.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
cal = fetch_california_housing()

df = pd.DataFrame(cal.data, columns=cal.feature_names)

# Add the target variable (house prices)
df["Target"] = cal.target

# Display the first 5 rows
print(df.head())
X = cal.data
y = cal.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ---- VISUALIZATION ----

# 1. Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue", label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed", label="Perfect Fit")  # y = x line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.show()

# 2. Residual Plot
residuals = y_test - y_pred  # Errors
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.5, color="green")
plt.axhline(y=0, color="red", linestyle="dashed")  # Zero error line
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()
```
The result is:

Mean Absolute Error (MAE): 0.5332
- This means, on average, the predictions are off by about 0.53 units (in 100,000s USD).

R² Score: 0.5758

- This indicates that 57.58% of the variance in house prices is explained by median income alone.

Analysis & Next Steps
- The model captures a moderate relationship but isn't highly predictive, meaning other factors significantly impact house prices.
- Consider using multiple linear regression to incorporate other features (HouseAge, AveRooms, etc.).
- Perform feature engineering (e.g., creating interaction terms, handling outliers more rigorously).

![Simple Linear Regression](https://github.com/rehansc/Machine-Learning-Journey/blob/main/01_Simple_Linear_Regression/simple_linear.png?raw=True)

The residual plot is shown below:

![Residual Plot](https://github.com/rehansc/Machine-Learning-Journey/blob/main/01_Simple_Linear_Regression/residual_plot.png?raw=True)




