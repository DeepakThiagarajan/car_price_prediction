import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time

# Load and preprocess dataset
data_path = r'D:\Capstone projects\cardekho\car_dekho_cleaned_dataset.csv'
data = pd.read_csv(data_path)

# Attempt to load preprocessing steps (Label encoders and scalers)
label_encoders_path = r'D:\Capstone projects\cardekho\label_encoders.pkl'
scalers_path = r'D:\Capstone projects\cardekho\scalers.pkl'

try:
    label_encoders = joblib.load(label_encoders_path)
    scalers = joblib.load(scalers_path)
    print("Preprocessing files loaded successfully.")
except FileNotFoundError as e:
    print(f"Warning: {e}")
    print("Proceeding without preprocessing steps.")

# Feature Engineering
data['car_age'] = 2024 - data['modelYear']
brand_popularity = data.groupby('oem')['price'].mean().to_dict()
data['brand_popularity'] = data['oem'].map(brand_popularity)
data['mileage_normalized'] = data['mileage'] / data['car_age']

# Define features and target
features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 
            'mileage', 'Seats', 'car_age', 'brand_popularity', 'mileage_normalized']
X = data[features]
y = data['price']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Define hyperparameter grid for Random Search
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform Randomized Search for hyperparameter tuning
start_time = time.time()
rf_random = RandomizedSearchCV(
    rf_model, 
    param_distributions=param_dist, 
    n_iter=20, 
    cv=5, 
    scoring='neg_mean_squared_error', 
    random_state=42
)
rf_random.fit(X_train, y_train)
end_time = time.time()

# Retrieve the best model
best_rf_model = rf_random.best_estimator_

# Cross-validation to evaluate the model's performance
rf_cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f'Random Forest CV Mean MSE: {-rf_cv_scores.mean():.4f}')

# Make predictions on the test set
start_predict_time = time.time()
y_pred_rf = best_rf_model.predict(X_test)
end_predict_time = time.time()

# Model evaluation metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest - MSE: {mse_rf:.4f}, MAE: {mae_rf:.4f}, R²: {r2_rf:.4f}')
print(f'Training Time: {end_time - start_time:.2f} seconds')
print(f'Prediction Time: {end_predict_time - start_predict_time:.2f} seconds')

# Evaluate model performance on older cars
older_cars = data[data['car_age'] > 10]
X_older = older_cars[features]
y_older = older_cars['price']

y_pred_older = best_rf_model.predict(X_older)
mse_older = mean_squared_error(y_older, y_pred_older)
mae_older = mean_absolute_error(y_older, y_pred_older)
r2_older = r2_score(y_older, y_pred_older)

print(f'Older Cars - MSE: {mse_older:.4f}, MAE: {mae_older:.4f}, R²: {r2_older:.4f}')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest: Actual vs Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.show()

# Save the trained model
model_save_path = r'D:\Capstone projects\cardekho\car_price_prediction_model.pkl'
joblib.dump(best_rf_model, model_save_path)

print(f"Model training complete. Model saved as '{model_save_path}'.")