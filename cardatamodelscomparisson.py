import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r'D:\Capstone projects\cardekho\car_dekho_cleaned_dataset.csv', 
    low_memory=False)
print(df.head())

# Define the feature set (X) and the target variable (y)
features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage', 'Seats']
X = df[features]
y = df['price']

# One-hot encode categorical variables, dropping the first category to avoid multicollinearity
X_encoded = pd.get_dummies(X, drop_first=True)

# Perform the train-test split with 80% for training and 20% for testing
X_train, x_test, Y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Display the sizes of the train and test sets
print(f"Training set size (X_train): {len(X_train)}")
print(f"Test set size (x_test): {len(x_test)}")
print(f"Training labels size (Y_train): {len(Y_train)}")
print(f"Test labels size (y_test): {len(y_test)}")

# Handle missing values in features using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train-Test Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

#Linear Regression with Cross-Validation and Regularization (Hyperparameter Tuning for Ridge and Lasso)# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Cross-Validation to evaluate model performance
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_mse = -cv_scores.mean()
print(f'\nLinear Regression CV Mean MSE: {mean_cv_mse:.4f}')

# Make predictions on the test set
y_pred_lr = lr_model.predict(X_test)

# Evaluate model performance on the test set
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f'Linear Regression - MSE: {mse_lr:.4f}, MAE: {mae_lr:.4f}, R²: {r2_lr:.4f}')

# Plotting Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_lr)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Linear Regression: Actual vs Predicted Prices')
plt.show()

# Hyperparameter Tuning for Ridge and Lasso Regression using Grid Search

# Ridge Regression with Grid Search for optimal alpha
ridge = Ridge()
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
best_ridge_alpha = ridge_grid.best_params_['alpha']
print(f'*****Best Ridge Alpha:***** {best_ridge_alpha}')

# Lasso Regression with Grid Search for optimal alpha
lasso = Lasso()
lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
best_lasso_alpha = lasso_grid.best_params_['alpha']
print(f'*****Best Lasso Alpha:***** {best_lasso_alpha}\n')

#Gradient Boosting with Cross-Validation and Random Hyperparameter Search
# Initialize and train the Gradient Boosting Regressor model
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train, y_train)

# Cross-Validation to evaluate model performance
cv_scores = cross_val_score(gbr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_mse = -cv_scores.mean()
print(f'Gradient Boosting CV Mean MSE: {mean_cv_mse:.4f}')

# Make predictions on the test set
y_pred_gbr = gbr_model.predict(X_test)

# Evaluate model performance on the test set
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

print(f'Gradient Boosting - MSE: {mse_gbr:.4f}, MAE: {mae_gbr:.4f}, R²: {r2_gbr:.4f}')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_gbr)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Gradient Boosting: Actual vs Predicted Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.show()

# Hyperparameter Tuning using Randomized Search
param_distributions = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Reduce the number of iterations for faster results (e.g., n_iter=10)
random_search = RandomizedSearchCV(
    estimator=gbr_model,
    param_distributions=param_distributions,
    n_iter=10,  # Adjust as needed for a balance between speed and thoroughness
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1  # Utilize all available CPUs for faster computation
)
random_search.fit(X_train, y_train)

# Output the best hyperparameters found
best_params = random_search.best_params_
print(f'*****Best Gradient Boosting Hyperparameters:***** {best_params}\n')

#Decision Tree with Cross-Validation and Grid Search for Hyperparameter Tuning
# Initialize and train the Decision Tree Regressor model with pruning
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# Cross-Validation to evaluate model performance
cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_mse = -cv_scores.mean()
print(f'Decision Tree CV Mean MSE: {mean_cv_mse:.4f}')

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluate model performance on the test set
mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f'Decision Tree - MSE: {mse_dt:.4f}, MAE: {mae_dt:.4f}, R²: {r2_dt:.4f}')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_dt)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Decision Tree: Actual vs Predicted Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Add reference line
plt.show()

# Hyperparameter Tuning using Grid Search
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)

# Output the best hyperparameters found
best_params = grid_search.best_params_
print(f'*****Best Decision Tree Hyperparameters:***** {best_params}\n')


#Random Forest with Cross-Validation and Hyperparameter Tuning (Random Search)
# Initialize and train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model performance using Cross-Validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_mse = -cv_scores.mean()
print(f'Random Forest CV Mean MSE: {mean_cv_mse:.4f}')

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate model performance on the test set
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest - MSE: {mse_rf:.4f}, MAE: {mae_rf:.4f}, R²: {r2_rf:.4f}')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest: Actual vs Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.show()

# Hyperparameter Tuning using Random Search
rf_param_distributions = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_param_distributions,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42
)
rf_random_search.fit(X_train, y_train)

# Output the best hyperparameters found
best_rf_params = rf_random_search.best_params_
print(f'*****Best Random Forest Hyperparameters:***** {best_rf_params}\n')


#Comparison and summary
# Store model evaluation metrics
model_results = {
    'Model': ['Linear Regression', 'Gradient Boosting', 'Decision Tree', 'Random Forest'],
    'MSE': [mse_lr, mse_gbr, mse_dt, mse_rf],
    'MAE': [mae_lr, mae_gbr, mae_dt, mae_rf],
    'R²': [r2_lr, r2_gbr, r2_dt, r2_rf]
}

# Create a DataFrame to compare model performance
comparison_df = pd.DataFrame(model_results)

# Display the Model Comparison Table
print("Model Comparison Table:")
print(comparison_df.to_string(index=False))

# Identify the best model based on the highest R² and the lowest MSE/MAE
best_model_idx = comparison_df['R²'].idxmax()
best_model = comparison_df.iloc[best_model_idx]

# Print the summary of the best model
print("\nBest Model Summary:")
print(f"Best Model: {best_model['Model']}")
print(f"MSE: {best_model['MSE']:.4f}")
print(f"MAE: {best_model['MAE']:.4f}")
print(f"R²: {best_model['R²']:.4f}")