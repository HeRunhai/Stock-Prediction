import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

"""
1. Data Preparation
"""

# Load the stock data
file_path = "AAPL.csv"
data = pd.read_csv(file_path, parse_dates=['Date'])

# Add weekday feature (0 = Monday, ..., 6 = Sunday)
data['Weekday'] = data['Date'].dt.dayofweek

# Splitting data into train (first 10 years) and test (last 4 years)
train_data = data[data['Date'] < '2020-09-07']
test_data = data[data['Date'] >= '2020-09-07']

# Features and target for prediction (using Close price as the target)
X_train = train_data.drop(columns=['Date', 'Close', 'Adj Close'])
y_train = train_data['Close']
X_test = test_data.drop(columns=['Date', 'Close', 'Adj Close'])
y_test = test_data['Close']

# 标准化 + PCA
scaler = StandardScaler()
pca = PCA(n_components=0.95)  # 保留95%的方差
preprocessor = Pipeline(steps=[('scaler', scaler), ('pca', pca)])

# 对训练集和测试集应用PCA变换
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# 获取PCA之后的特征数量 (只剩三个feature)
pca_feature_count = X_train_transformed.shape[1]

# 使用新的列名
pca_columns = [f'PCA_{i+1}' for i in range(pca_feature_count)]

# 将转换后的数据保持为 DataFrame 格式
X_train = pd.DataFrame(X_train_transformed, columns=pca_columns)
X_test = pd.DataFrame(X_test_transformed, columns=pca_columns)

"""
2. Model Training and Prediction
由于股票数据已经是按交易日排列的（即跳过周末），所以训练和预测都是在交易日上进行的
"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error

MSE_list = []


# 超参数调优
from sklearn.model_selection import GridSearchCV

# Random Forest 超参数调优
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=123), rf_params, cv=5, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# XGBoost 超参数调优
xgboost_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 1.5]
}
xgboost_grid = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), xgboost_params, cv=5,
                            scoring='neg_mean_squared_error')
xgboost_grid.fit(X_train, y_train)
best_xgboost = xgboost_grid.best_estimator_

# SVM 超参数调优
svm_params = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1]
}
svm_grid = GridSearchCV(SVR(), svm_params, cv=5, scoring='neg_mean_squared_error')
svm_grid.fit(X_train, y_train)
best_svr = svm_grid.best_estimator_

# KNN 超参数调优
knn_params = {
    'n_neighbors': [5, 15, 25, 35],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40],
    'p': [1, 2]
}
knn_grid = GridSearchCV(KNeighborsRegressor(), knn_params, cv=5, scoring='neg_mean_squared_error')
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_

# 调整后的模型训练和交叉验证
lr = LinearRegression()
models = [(lr, "Linear Regression"),
          (best_rf, "Random Forest"),
          (best_xgboost, "XGBoost"),
          (best_svr, "SVM"),
          (best_knn, "KNN")]

# 评估交叉验证表现
print("\n-------------------Performance on cross-validation--------------------------")

# 使用交叉验证来评估模型性能
def cross_val_evaluate(model, X_train, y_train, model_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    cv_mse = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    print(f"{model_name} Cross-Validated MSE: {abs(np.mean(cv_mse)):.4f}, R²: {np.mean(cv_r2):.4f}")
    MSE_list.append(abs(np.mean(cv_mse)))

for model, name in models:
    model.fit(X_train, y_train)
    cross_val_evaluate(model, X_train, y_train, name)


# 训练集上的预测并评估表现
print("\n-------------------Performance on train set--------------------------")

def evaluate_train(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Train MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

for model, name in models:
    y_train_pred = model.predict(X_train)
    evaluate_train(y_train, y_train_pred, name)

# 使用测试集进行预测并计算各个指标
print("\n-------------------Performance on test set--------------------------")

def evaluate_test(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Test MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")


for model, name in models:
    y_test_pred = model.predict(X_test)
    evaluate_test(y_test, y_test_pred, name)


# 实际值 vs 预测值对比图 (合并训练集和测试集的预测值, 并区分颜色)
def plot_pred_vs_actual_full(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name):
    plt.figure(figsize=(12, 6))

    # 训练集实际值和预测值
    plt.plot(y_train_true.values, label="Actual (Train)", color='blue', linestyle='--')
    plt.plot(y_train_pred, label=f"{model_name} Prediction (Train)", color='green')

    # 测试集实际值和预测值
    plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
             y_test_true.values, label="Actual (Test)", color='orange', linestyle='--')
    plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
             y_test_pred, label=f"{model_name} Prediction (Test)", color='red')

    # 标记训练集与测试集的分界线
    plt.axvline(x=len(y_train_true), color='black', linestyle='--', label="Train/Test Split")

    plt.title(f"Actual vs Predicted - {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


# 在训练集和测试集上绘制实际值 vs 预测值的图
for model, name in models:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    plot_pred_vs_actual_full(y_train, y_train_pred, y_test, y_test_pred, name)

"""
3.1 Model Improvement - Ensemble learning models: weighted averaging
"""

# 根据交叉验证的 MSE 来调整权重
mse_values = MSE_list  # 来自交叉验证的 MSE
weights = [1 / mse for mse in mse_values]  # MSE 越低，权重越大
normalized_weights = [w / sum(weights) for w in weights]  # 归一化权重
print("Normalized Weights:", normalized_weights)

# 使用加权平均进行预测
ensemble_train_pred = (normalized_weights[0] * lr.predict(X_train) +
                       normalized_weights[1] * best_rf.predict(X_train) +
                       normalized_weights[2] * best_xgboost.predict(X_train) +
                       normalized_weights[3] * best_svr.predict(X_train) +
                       normalized_weights[4] * best_knn.predict(X_train))

ensemble_test_pred = (normalized_weights[0] * lr.predict(X_test) +
                      normalized_weights[1] * best_rf.predict(X_test) +
                      normalized_weights[2] * best_xgboost.predict(X_test) +
                      normalized_weights[3] * best_svr.predict(X_test) +
                      normalized_weights[4] * best_knn.predict(X_test))

# 评估集成模型的训练和测试集表现
print("\n-------------------Weighted Averaging Ensemble Model--------------------------")
evaluate_train(y_train, ensemble_train_pred, "Weighted Ensemble")
evaluate_test(y_test, ensemble_test_pred, "Weighted Ensemble")

# 绘制集成模型的预测图
plot_pred_vs_actual_full(y_train, ensemble_train_pred, y_test, ensemble_test_pred, "Weighted Ensemble")

"""
3.2 Model Improvement -  Ensemble learning models: Stacking (instead of simple weighted average)
"""

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# 构建 stacking 模型，使用 Ridge 作为元模型
estimators = [
    ('lr', lr),
    ('rf', best_rf),
    ('xgb', best_xgboost),
    ('svr', best_svr),
    ('knn', best_knn)
]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())

# 训练 stacking 模型
stacking_model.fit(X_train, y_train)

# 评估 stacking 模型表现
y_stack_pred = stacking_model.predict(X_test)
print("\n-------------------Stacking Ensemble Model--------------------------")
evaluate_test(y_test, y_stack_pred, "Stacking Model")

# 预测图
y_train_pred = stacking_model.predict(X_train)
y_test_pred = stacking_model.predict(X_test)
plot_pred_vs_actual_full(y_train, y_train_pred, y_test, y_test_pred, "Stacking Model")

# Plot
# 各个模型的MSE值
mse_values = [
    mean_squared_error(y_test, lr.predict(X_test)),
    mean_squared_error(y_test, best_rf.predict(X_test)),
    mean_squared_error(y_test, best_xgboost.predict(X_test)),
    mean_squared_error(y_test, best_svr.predict(X_test)),
    mean_squared_error(y_test, best_knn.predict(X_test)),
    mean_squared_error(y_test, stacking_model.predict(X_test))
]
model_names = ["Linear Regression", "Random Forest", "XGBoost", "SVM", "KNN", "Stacking"]

# 绘制MSE对比柱状图
plt.figure(figsize=(10, 6))
plt.bar(model_names, mse_values, color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
plt.title("Mean Squared Error (MSE) Comparison")
plt.ylabel("MSE")
plt.xticks(rotation=45)
plt.show()

# """
# 6. Model Improvement - LSTM-based time series forecasting model
# """
#
# # 6.1 Data preprocessing
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
#
# # 缩放数据到 [0, 1] 区间
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data[['Close']].values)
#
#
# # 创建时间序列数据
# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(seq_length, len(data)):
#         X.append(data[i - seq_length:i, 0])
#         y.append(data[i, 0])
#     return np.array(X), np.array(y)
#
#
# # 选择时间窗口长度
# seq_length = 60
#
# # 使用前10年数据进行训练
# train_data_scaled = scaled_data[:len(train_data)]
# X_train_lstm, y_train_lstm = create_sequences(train_data_scaled, seq_length)
#
# # 使用最后4年数据进行测试
# test_data_scaled = scaled_data[len(train_data) - seq_length:]
# X_test_lstm, y_test_lstm = create_sequences(test_data_scaled, seq_length)
#
# # 将数据调整为适合 LSTM 模型输入的三维格式 [样本数, 时间步数, 特征数]
# X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
# X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
#
# # 6.2 构建 LSTM 模型
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
# model.add(LSTM(units=50))
# model.add(Dense(1))  # 输出一个预测值（下一天的 Close 价格）
#
# # 编译模型
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# # 训练模型
# model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32)
#
# # 用测试数据进行预测
# lstm_pred_scaled = model.predict(X_test_lstm)
#
# # 将预测结果缩放回原始范围
# lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
#
# # 计算LSTM模型的均方误差（MSE）
# from sklearn.metrics import mean_squared_error
#
# lstm_mse = mean_squared_error(y_test_lstm, lstm_pred)
# print(f"LSTM Model Mean Squared Error: {lstm_mse:.4f}")
#
# # 6.3 Combine LSTM and other models
# # 将 LSTM 预测结果与其他模型结合（同样使用加权平均法）
# weights = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]  # LSTM, Linear Regression, Random Forest, XGBoost
#
# # 集成预测
# final_ensemble_pred = (weights[0] * lstm_pred.squeeze() +
#                        weights[1] * lr_pred +
#                        weights[2] * rf_pred +
#                        weights[3] * xgb_pred +
#                        weights[3] * svr_pred +
#                        weights[4] * knn_pred)
#
# # 计算最终集成模型的均方误差（MSE）
# final_ensemble_mse = mean_squared_error(y_test[-len(final_ensemble_pred):], final_ensemble_pred)
# print(f"Final Ensemble Model MSE: {final_ensemble_mse:.4f}")
