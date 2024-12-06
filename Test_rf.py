import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV

# Load the stock data
file_path = "AAPL.csv"
data = pd.read_csv(file_path, parse_dates=['Date'])

# Add weekday feature (0 = Monday, ..., 6 = Sunday)
data['Weekday'] = data['Date'].dt.dayofweek

# # Expanding lagged features (增加滞后期为 1 到 30 天)
# for lag in range(1, 31):  # Add 1 to 30 days lag
#     data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
#
# # Adding rolling average and volatility with different windows (扩展滚动窗口)
# rolling_windows = [5, 20, 60]
# for window in rolling_windows:
#     data[f'Rolling_{window}'] = data['Close'].rolling(window=window).mean()
#     data[f'Volatility_{window}'] = data['Close'].rolling(window=window).std()
#
# # Drop missing values caused by lagging
# data = data.dropna()

# Split the data into training and test sets (前13年训练，后1年测试)
train_data = data[data['Date'] < '2023-09-07']
test_data = data[data['Date'] >= '2023-09-07']

X_train = train_data.drop(columns=['Date', 'Close', 'Adj Close'])
y_train = train_data['Close']
X_test = test_data.drop(columns=['Date', 'Close', 'Adj Close'])
y_test = test_data['Close']

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Random Forest model with tuned parameters (调整超参数，防止过拟合)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],  # 更加保守的深度
    'min_samples_split': [10, 20],  # 增加分裂的最小样本数
    'min_samples_leaf': [5, 10],  # 增加叶子节点的最小样本数
    'max_features': ['sqrt']  # 采用 sqrt 特征选择来减少特征空间
}

rf_grid = GridSearchCV(RandomForestRegressor(random_state=123), rf_params, cv=5, scoring='neg_mean_squared_error')
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_


# Cross-validated evaluation
def cross_val_evaluate(model, X_train, y_train, model_name):
    tscv = TimeSeriesSplit(n_splits=5)
    cv_mse = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    cv_r2 = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
    print(f"{model_name} Cross-Validated MSE: {abs(np.mean(cv_mse)):.4f}, R²: {np.mean(cv_r2):.4f}")


# 评估交叉验证表现
print("\n-------------------Performance on cross-validation--------------------------")
cross_val_evaluate(best_rf, X_train_scaled, y_train, "Random Forest")


# Training set evaluation
def evaluate_train(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Train MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")


# 训练集上的预测并评估表现
print("\n-------------------Performance on train set--------------------------")
y_train_pred = best_rf.predict(X_train_scaled)
evaluate_train(y_train, y_train_pred, "Random Forest")


# Test set evaluation
def evaluate_test(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Test MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")


# 使用测试集进行预测并计算各个指标
print("\n-------------------Performance on test set--------------------------")
y_test_pred = best_rf.predict(X_test_scaled)
evaluate_test(y_test, y_test_pred, "Random Forest")


# 实际值 vs 预测值对比图 (合并训练集和测试集的预测值, 并区分颜色)
def plot_pred_vs_actual_full(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name):
    plt.figure(figsize=(12, 6))
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})

    # 训练集实际值和预测值
    plt.plot(y_train_true.values, label="Actual (Train)", color='blue', linestyle='--')
    plt.plot(y_train_pred, label=f"{model_name} Prediction (Train)", color='orange')

    # 测试集实际值和预测值
    plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
             y_test_true.values, label="Actual (Test)", color='green', linestyle='--')
    plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
             y_test_pred, label=f"{model_name} Prediction (Test)", color='red')

    # 标记训练集与测试集的分界线
    plt.axvline(x=len(y_train_true), color='black', linestyle='--', label="Train/Test Split")

    plt.title(f"Actual vs Predicted - {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


# Plot actual vs predicted
plot_pred_vs_actual_full(y_train, y_train_pred, y_test, y_test_pred, "Random Forest")


# 残差分析，检验是否过拟合; 如果残差图表现为接近标准正态分布（均值为0，无明显趋势），说明模型拟合良好，否则就是欠拟合或过拟合

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
plot_residuals(y_train, y_train_pred, "Random Forest on Train")

# 测试集残差分析
print("\n-------------------Residuals analysis on test set--------------------------")
plot_residuals(y_test, y_test_pred, "Random Forest on Test")
