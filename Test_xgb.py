import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV

# Load the stock data
file_path = "AAPL.csv"
data = pd.read_csv(file_path, parse_dates=['Date'])

# Add weekday feature (0 = Monday, ..., 6 = Sunday)
data['Weekday'] = data['Date'].dt.dayofweek

# Re-split data into training and test sets (training on the first 13 years, testing on the last year)
train_data = data[data['Date'] < '2023-09-07']
test_data = data[data['Date'] >= '2023-09-07']

X_train = train_data.drop(columns=['Date', 'Close', 'Adj Close'])
y_train = train_data['Close']
X_test = test_data.drop(columns=['Date', 'Close', 'Adj Close'])
y_test = test_data['Close']

# Standardization + PCA
scaler = StandardScaler()
pca = PCA(n_components=0.95)  # Keep 95% of the variance
preprocessor = Pipeline(steps=[('scaler', scaler), ('pca', pca)])

# Apply PCA transformation on the training and test sets
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Get the number of features after PCA (3 features left)
pca_feature_count = X_train_transformed.shape[1]

# Use new column names for the PCA features
pca_columns = [f'PCA_{i + 1}' for i in range(pca_feature_count)]

# Convert the transformed data back to DataFrame
X_train = pd.DataFrame(X_train_transformed, columns=pca_columns)
X_test = pd.DataFrame(X_test_transformed, columns=pca_columns)

# Define XGBoost hyperparameters for tuning
xgboost_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 1.5]
}

# Perform Grid SearchCV (without early stopping)
xgboost_grid = GridSearchCV(XGBClassifier(objective='reg:squarederror', eval_metric='rmse'),
                            xgboost_params,
                            cv=5,
                            scoring='neg_mean_squared_error')

xgboost_grid.fit(X_train, y_train)

# Retrieve the best estimator directly
best_xgboost = xgboost_grid.best_estimator_

# Fit the best model with early stopping on the test data
eval_set = [(X_test, y_test)]
best_xgboost.fit(X_train, y_train,
                 early_stopping_rounds=10,
                 eval_metric="rmse",
                 eval_set=eval_set,
                 verbose=True)


# Function for cross-validation evaluation
def cross_val_evaluate(model, X_train, y_train, model_name):
    # TimeSeriesSplit is used instead of KFold for time series data
    tscv = TimeSeriesSplit(n_splits=5)
    cv_mse = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    cv_r2 = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
    print(f"{model_name} Cross-Validated MSE: {abs(np.mean(cv_mse)):.4f}, R²: {np.mean(cv_r2):.4f}")


# Cross-validation evaluation
cross_val_evaluate(best_xgboost, X_train, y_train, "XGBoost")


# Evaluate on the training set
def evaluate_train(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Train MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")


y_train_pred = best_xgboost.predict(X_train)
evaluate_train(y_train, y_train_pred, "XGBoost")


# Evaluate on the test set
def evaluate_test(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Test MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")


y_test_pred = best_xgboost.predict(X_test)
evaluate_test(y_test, y_test_pred, "XGBoost")


# Plot actual vs predicted values (Train and Test)
def plot_pred_vs_actual_full(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name):
    plt.figure(figsize=(12, 6))

    # Training set actual vs predicted
    plt.plot(y_train_true.values, label="Actual (Train)", color='blue', linestyle='--')
    plt.plot(y_train_pred, label=f"{model_name} Prediction (Train)", color='green')

    # Test set actual vs predicted
    plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
             y_test_true.values, label="Actual (Test)", color='orange', linestyle='--')
    plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
             y_test_pred, label=f"{model_name} Prediction (Test)", color='red')

    # Mark the boundary between training and test sets
    plt.axvline(x=len(y_train_true), color='black', linestyle='--', label="Train/Test Split")

    plt.title(f"Actual vs Predicted - {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


# Plot predictions vs actual values
plot_pred_vs_actual_full(y_train, y_train_pred, y_test, y_test_pred, "XGBoost")


# Residual analysis for both training and test sets
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))

    # Residuals vs Predictions Plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f"Residuals vs Predictions - {model_name}")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")

    # Residuals Distribution Plot
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title(f"Residuals Distribution - {model_name}")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# Residual analysis for training and test sets
print("\n-------------------Residuals analysis on train set--------------------------")
plot_residuals(y_train, y_train_pred, "XGBoost on Train set")

print("\n-------------------Residuals analysis on test set--------------------------")
plot_residuals(y_test, y_test_pred, "XGBoost on Test set")
