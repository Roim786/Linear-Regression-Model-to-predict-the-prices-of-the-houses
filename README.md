# Linear-Regression-Model-to-predict-the-prices-of-the-houses
In this house price prediction project, a Linear Regression model was used with squared and interaction features. The data was standardized, and outliers were removed based on Z-scores. The model's performance was evaluated using MSE, RMSE, MAE, and R-squared, and it was also applied to predict prices for new data.
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
from itertools import combinations

# Load the dataset

train_file_path = '/content/drive/My Drive/house-prices-advanced-regression-techniques/train.csv'

test_file_path = '/content/drive/My Drive/house-prices-advanced-regression-techniques/test.csv'

df_train = pd.read_csv(train_file_path)

df_test = pd.read_csv(test_file_path)

# Display the first few rows of the dataset 
print(df_train.head())

# Select features and target variable
features = [
    'TotalSF', 'OverallQual', 'TotalBsmtSF',
    'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF'
]

# Add squared features

for feature in features:

    df_train[f'{feature}_squared'] = df_train[feature] ** 2
    
    df_test[f'{feature}_squared'] = df_test[feature] ** 2

# Add interaction features

for feature1, feature2 in combinations(features, 2):

    df_train[f'{feature1}_{feature2}'] = df_train[feature1] * df_train[feature2]
    
    df_test[f'{feature1}_{feature2}'] = df_test[feature1] * df_test[feature2]

# Update the features list to include squared and interaction features

features_with_squared_and_interactions = features + [f'{feature}_squared' for feature in features] + 

[f'{feature1}_{feature2}' for feature1, feature2 in combinations(features, 2)]

# Identify and remove outliers based on Z-score

from scipy.stats import zscore

df_train = df_train[(np.abs(zscore(df_train[features_with_squared_and_interactions])) < 3).all(axis=1)]

X_train = df_train[features_with_squared_and_interactions]

y_train = df_train['SalePrice']

X_test = df_test[features_with_squared_and_interactions]

y_test = df_test['SalePrice']

# Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Create the linear regression model

model = LinearRegression()

# Start timing the training process

start_time = time.time()

# Train the model

model.fit(X_train_scaled, y_train)

# End timing the training process

end_time = time.time()

# Calculate training time

training_time = end_time - start_time

print(f"Training completed in {training_time:.4f} seconds")

# Make predictions on the test set

y_pred = model.predict(X_test_scaled)

# Evaluate the model

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')

print(f'Root Mean Squared Error: {rmse}')

print(f'Mean Absolute Error: {mae}')

print(f'R-squared: {r2}')

# Plot actual vs. predicted prices

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs. Predicted Prices")

plt.show()

new_data = pd.DataFrame({

    'TotalSF': [1671],
    'OverallQual': [6],
    'TotalBsmtSF': [959],
    'GrLivArea': [1671],
    'FullBath': [2],
    'HalfBath': [1],
    'BedroomAbvGr': [3],
    'GarageArea': [472],
    'WoodDeckSF': [0],
    'OpenPorchSF': [38]
})

# Transform the new data using the same polynomial features

new_data_poly = poly.transform(new_data)

# Make predictions on the new data

predicted_prices = model.predict(new_data_poly)

# Print the predicted prices

print(f'Predicted Prices: {predicted_prices}')

Training completed in 0.0048 seconds
Mean Squared Error: 1540615411.0968738
Root Mean Squared Error: 39250.67402092445
Mean Absolute Error: 24543.969571903188
R-squared: 0.7694647193545581
