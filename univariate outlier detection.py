import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
#from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.cluster import DBSCAN


def detect_outliers_2d(df, column_x, column_y, eps=1.5, min_samples=5):
    # Extract the two columns and drop any rows with NaN values
    data = df[[column_x, column_y]].dropna()

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Initialize and fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    # Plotting the results
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6, edgecolor='k')
    plt.xlabel(f'{column_x} (scaled)')
    plt.ylabel(f'{column_y} (scaled)')
    plt.title('DBSCAN Clustering Results')
    plt.colorbar(label='Cluster ID')
    plt.show()

    # Identify and return outlier indices
    outlier_indices = data.index[clusters == -1]
    #print(f"Outliers detected in {column_x}: {outlier_indices}")
    return outlier_indices

outlier_indices = set()

y = df_train['Price']

X_train, X_val, y_train, y_val = train_test_split(df_train, y, test_size=0.9, random_state=42)

# Loop through all numeric columns except 'SalePrice'
for column in X_train.select_dtypes(include=['number']).columns:
    if column != 'Price' and column !='id':  # Avoid using 'SalePrice' as the X variable
        new_outliers = detect_outliers_2d(X_train, column, 'Price', eps=2, min_samples=5)
        outlier_indices.update(new_outliers)

print(f"Total unique outliers detected: {len(outlier_indices)}")
