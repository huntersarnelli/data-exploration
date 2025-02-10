import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression  # as an example model

predictor_cols = df_train.columns.tolist()
target_col = 'Price'

# 1. Prepare the data: select predictors and drop rows with missing values
X = df_train[predictor_cols].dropna()
y = df_train.loc[X.index, target_col]  # align target with the cleaned predictor DataFrame

# 2. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Detect outliers using Isolation Forest
#   - contamination: estimated proportion of outliers in your data (adjust if you have domain knowledge)
iso_forest = IsolationForest(n_estimators=100, contamination=0.00005, random_state=42)
iso_forest.fit(X_scaled)
# Predict returns 1 for inliers and -1 for outliers.
outlier_labels = iso_forest.predict(X_scaled)

# Get indices of detected outliers
outlier_indices = X.index[outlier_labels == -1]
print(f"Number of outliers detected: {len(outlier_indices)}")

# 4. (Optional) Remove outliers
# You might want to compare model performance with and without outlier removal.
train_df_clean = df_train.drop(index=outlier_indices)

# 5. Validate the impact on your model performance
# Example: Use a simple linear regression model with cross-validation.

# Prepare data with outliers removed
X_clean = train_df_clean[predictor_cols]
y_clean = train_df_clean[target_col]

# Scale the cleaned data
X_clean_scaled = scaler.fit_transform(X_clean)

X_clean_original = scaler.inverse_transform(X_clean_scaled)

#Visualize the outliers using PCA

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming X_scaled is your scaled data and outlier_labels from IsolationForest
# Create a boolean mask for outliers: True for outliers (-1), False for inliers (1)
mask = (outlier_labels == -1)

# Reduce the high-dimensional data to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the data: inliers in blue, outliers in red
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[~mask, 0], X_pca[~mask, 1],
            c='blue', alpha=0.5, label='Inliers')
plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
            c='red', alpha=0.8, label='Outliers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Data with Outliers Highlighted')
plt.legend()
plt.show()


#proceed to train the model

# Split data into features (X) and target (y)
X = train_df_clean.drop(['Price','id'], axis=1)
y = train_df_clean['Price']

# Suppose X_clean_scaled was obtained from scaling the cleaned data:
X_clean_scaled = scaler.fit_transform(X)

# To go back to the original scale:
X_clean_original = scaler.inverse_transform(X_clean_scaled)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_clean_original, y, test_size=0.2, random_state=42)
