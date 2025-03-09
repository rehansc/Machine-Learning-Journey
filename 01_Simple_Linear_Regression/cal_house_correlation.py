import pandas as pd
from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
cal = fetch_california_housing()
df = pd.DataFrame(cal.data, columns=cal.feature_names)
df["Target"] = cal.target  # Add target column

# Compute correlation with target
correlation_with_target = df.corr()["Target"].sort_values(ascending=False)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()
