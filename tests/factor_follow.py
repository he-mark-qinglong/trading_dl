import numpy as np  
import pandas as pd  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import r2_score, mean_squared_error  

# 模拟市场数据  
np.random.seed(42)  
data = {  
    "price": np.cumsum(np.random.normal(0, 1, 10000)) + 100,  # 模拟价格  
    "volume": np.random.randint(1, 100, 10000)  # 模拟成交量  
}  
df = pd.DataFrame(data)  

# 计算 VWAP  
df["vwap"] = (df["price"] * df["volume"]).expanding().sum() / df["volume"].expanding().sum()  

# 计算因子  
df["price_deviation"] = (df["price"] - df["vwap"]) / df["vwap"]  # 价格乖离率  
df["momentum"] = df["price"].diff(5) / df["price"].shift(5)  # 价格动量  
df["volume_change"] = df["volume"].diff(5) / df["volume"].shift(5)  # 成交量变化率  
df["volatility"] = df["price"].rolling(10).std()  # 价格波动率  

# 因子归一化（Z-Score 标准化）  
for col in ["price_deviation", "momentum", "volume_change", "volatility"]:  
    df[col + "_norm"] = (df[col] - df[col].mean()) / df[col].std()  

# 因子的跟随性分析（相关性）  
correlations = {}  
for col in ["price_deviation_norm", "momentum_norm", "volume_change_norm", "volatility_norm"]:  
    correlations[col] = df[col].corr(df["vwap"])  
print("因子与 VWAP 的相关性：")  
print(correlations)  

# Prepare X and y  
X = df[["price_deviation_norm", "momentum_norm", "volume_change_norm", "volatility_norm"]]  # 当前时刻的因子值  
y = df["vwap"].shift(-1)  # 下一时刻的 VWAP  

# Drop rows with NaN values to ensure alignment  
X = X.iloc[:-1]  # 去掉最后一行，因为 y 的最后一行是 NaN  
y = y.iloc[:-1]  # 去掉最后一行，保持与 X 对齐
# Linear regression model  
model = LinearRegression()  
model.fit(X, y)  
y_pred = model.predict(X)  

# Evaluate prediction performance  
r2 = r2_score(y, y_pred)  
mse = mean_squared_error(y, y_pred)  
print(f"Prediction Performance: R² = {r2:.4f}, MSE = {mse:.4f}")  
