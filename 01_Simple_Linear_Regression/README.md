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

## Exploratory Data Analysis (EDA)Before running regression, we conducted EDA:
  - Correlation Analysis: MedInc had the highest correlation (0.69) with housing prices.
  - Box Plot Analysis: Detected outliers in Population, which were handled.
  - Histogram Distribution: Verified skewness in certain features.


