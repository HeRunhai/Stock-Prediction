"""
Model Improvement -  Ensemble learning models: Stacking (instead of simple weighted average)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import StackingRegressor

# Load the stock data
file_path = "AAPL.csv"
data = pd.read_csv(file_path, parse_dates=['Date'])

# Add weekday feature (0 = Monday, ..., 6 = Sunday)
data['Weekday'] = data['Date'].dt.dayofweek

# Train/test split (Instead, we use the first 13 years for training and predict the next year.)
train_data = data[data['Date'] < '2023-09-07']
test_data = data[data['Date'] >= '2023-09-07']

X_train = train_data.drop(columns=['Date', 'Close', 'Adj Close'])
y_train = train_data['Close']
X_test = test_data.drop(columns=['Date', 'Close', 'Adj Close'])
y_test = test_data['Close']

# Standardization + PCA (keeping 95% variance)
scaler = StandardScaler()
pca = PCA(n_components=0.95)
preprocessor = Pipeline(steps=[('scaler', scaler), ('pca', pca)])

# Apply PCA transformation to training and test sets
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Extract PCA feature count
pca_feature_count = X_train_transformed.shape[1]

# Create new DataFrames with PCA features
pca_columns = [f'PCA_{i + 1}' for i in range(pca_feature_count)]
X_train = pd.DataFrame(X_train_transformed, columns=pca_columns)
X_test = pd.DataFrame(X_test_transformed, columns=pca_columns)

from sklearn.model_selection import GridSearchCV

# Ridge Regression (L2 Regularization) with GridSearchCV
ridge = Ridge()
param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')
grid_search_ridge.fit(X_train, y_train)

# Lasso Regression (L1 Regularization) with GridSearchCV
lasso = Lasso()
param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
grid_search_lasso = GridSearchCV(lasso, param_grid_lasso, cv=5, scoring='neg_mean_squared_error')
grid_search_lasso.fit(X_train, y_train)

from sklearn.svm import SVR

# SVM 超参数调优
svm_params = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1]
}
svm_grid = GridSearchCV(SVR(), svm_params, cv=5, scoring='neg_mean_squared_error')
svm_grid.fit(X_train, y_train)
best_svr = svm_grid.best_estimator_

# Building a stacking model using Ridge as a metamodel
estimators = [
    ('Ridge', grid_search_ridge.best_estimator_),
    ('Lasso', grid_search_lasso.best_estimator_),
    ('SVM', best_svr)
]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())

# Training the stacking model
stacking_model.fit(X_train, y_train)


def evaluate(y_true, y_pred, model_name, data_type):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} {data_type} MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")


# Evaluating the performance of the stacking model
y_stack_train_pred = stacking_model.predict(X_train)
y_stack_test_pred = stacking_model.predict(X_test)
print("\n-------------------Stacking Ensemble Model Evaluation--------------------------")
evaluate(y_train, y_stack_train_pred, "Stacking Model", "Train")
evaluate(y_test, y_stack_test_pred, "Stacking Model", "Test")


# Plot function to compare actual vs predicted
def plot_pred_vs_actual_full(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name):
    plt.figure(figsize=(12, 6))
    # Set the global font size
    plt.rcParams.update({'font.size': 14})

    # Plot actual vs predicted for train set
    # plt.plot(y_train_true.values, label="Actual (Train)", color='blue', linestyle='--')
    # plt.plot(y_train_pred, label=f"{model_name} Prediction (Train)", color='green')

    # Plot actual vs predicted for test set
    plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
             y_test_true.values, label="Actual (Test)", color='green', linestyle='--')
    plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
             y_test_pred, label=f"{model_name} Prediction (Test)", color='red')

    # Mark train/test split
    # plt.axvline(x=len(y_train_true), color='black', linestyle='--', label="Train/Test Split")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


# Forecast Plot
y_train_pred = stacking_model.predict(X_train)
y_test_pred = stacking_model.predict(X_test)
plot_pred_vs_actual_full(y_train, y_train_pred, y_test, y_test_pred, "Stacking Model")


# Residual analysis to check whether it is overfitting; if the residual graph is close to the standard normal
# distribution (mean is 0, no obvious trend), it means that the model fits well, otherwise it is underfitting or overfitting

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))

    # 残差随预测值的分布
    """
    Residuals vs Predictions Plot:
    这张图显示了预测值和残差之间的关系。如果模型拟合良好，残差应随机分布，没有明显的模式或趋势。
    如果残差图呈现出某种模式（如漏斗形），则表明模型可能有一些非线性趋势未被捕捉，或者存在过拟合或欠拟合的情况
    """

    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f"Residuals vs Predictions - {model_name}")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")

    # 残差的直方图
    """
    Residuals Distribution Plot:
    这张图展示了残差的分布情况。理想情况下，残差应呈正态分布，且均值应接近于零。
    如果残差的分布偏离正态分布，可能表明模型对某些区域的拟合不佳
    """
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title(f"Residuals Distribution - {model_name}")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# 训练集残差分析
print("\n-------------------Residuals analysis on train set--------------------------")
plot_residuals(y_train, y_train_pred, "Stacking on Train set")

# 测试集残差分析
print("\n-------------------Residuals analysis on test set--------------------------")
plot_residuals(y_test, y_test_pred, "Stacking on Test set")
