import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Load the stock data
file_path = "AAPL.csv"
data = pd.read_csv(file_path, parse_dates=['Date'])

# Add weekday feature (0 = Monday, ..., 6 = Sunday)
data['Weekday'] = data['Date'].dt.dayofweek

# Re-split data into training and test sets (改为用前13年训练，预测后1年)
train_data = data[data['Date'] < '2023-09-07']
test_data = data[data['Date'] >= '2023-09-07']

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
pca_columns = [f'PCA_{i + 1}' for i in range(pca_feature_count)]

# 将转换后的数据保持为 DataFrame 格式
X_train = pd.DataFrame(X_train_transformed, columns=pca_columns)
X_test = pd.DataFrame(X_test_transformed, columns=pca_columns)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

lr = LinearRegression()

# 评估交叉验证表现
print("\n-------------------Performance on cross-validation--------------------------")


def cross_val_evaluate(model, X_train, y_train, model_name):
    # 定义TimeSeriesSplit代替KFold, 更适用于时间序列
    tscv = TimeSeriesSplit(n_splits=5)
    cv_mse = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    cv_r2 = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
    print(f"{model_name} Cross-Validated MSE: {abs(np.mean(cv_mse)):.4f}, R²: {np.mean(cv_r2):.4f}")


lr.fit(X_train, y_train)
cross_val_evaluate(lr, X_train, y_train, "Linear Regression")

# 训练集上的预测并评估表现
print("\n-------------------Performance on train set--------------------------")


def evaluate(y_true, y_pred, model_name, data_type):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} {data_type} MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")


y_train_pred = lr.predict(X_train)
evaluate(y_train, y_train_pred, "Linear Regression","Train")

# 使用测试集进行预测并计算各个指标
print("\n-------------------Performance on test set--------------------------")

y_test_pred = lr.predict(X_test)
evaluate(y_test, y_test_pred, "Linear Regression", "Train")


# 实际值 vs 预测值对比图 (合并训练集和测试集的预测值, 并区分颜色)
def plot_pred_vs_actual_full(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name):
    plt.figure(figsize=(12, 6))
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 16})

    # 训练集实际值和预测值
    # plt.plot(y_train_true.values, label="Actual (Train)", color='blue', linestyle='--')
    # plt.plot(y_train_pred, label=f"{model_name} Prediction (Train)", color='green')

    # 测试集实际值和预测值
    plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
             y_test_true.values, label="Actual Value", color='green', linestyle='--')
    plt.plot(range(len(y_train_true), len(y_train_true) + len(y_test_true)),
             y_test_pred, label=f"{model_name} Prediction", color='red')

    # 标记训练集与测试集的分界线
    # plt.axvline(x=len(y_train_true), color='black', linestyle='--', label="Train/Test Split")

    plt.title(f"Actual vs Predicted on Test set- {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
plot_pred_vs_actual_full(y_train, y_train_pred, y_test, y_test_pred, "Linear Regression")


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
plot_residuals(y_train, y_train_pred, "LR on Train set")

# 测试集残差分析
print("\n-------------------Residuals analysis on test set--------------------------")
plot_residuals(y_test, y_test_pred, "LR on Test set")
