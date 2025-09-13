
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

class MarketTrendPredictor:
    def __init__(self, timeframes=[1, 5, 15], model_params=None):
        """
        初始化 MarketTrendPredictor 类
        :param timeframes: 分时线的时间周期（单位：分钟），如 [1, 5, 15]
        :param model_params: XGBoost 模型的参数
        """
        self.timeframes = timeframes
        self.models = {tf: XGBClassifier(**(model_params or {})) for tf in timeframes}
        self.trend_labels = {0: "Downtrend", 1: "Uptrend", 2: "Sideways"}
    
    def preprocess_data(self, data, timeframe):
        """
        数据预处理：生成特征和标签
        :param data: DataFrame，包含时间序列数据（如开盘价、收盘价、最高价、最低价、成交量等）
        :param timeframe: 当前分时线的时间周期
        :return: 特征矩阵 X 和标签 y
        """
        # 计算技术指标作为特征
        data['return'] = data['close'].pct_change()  # 收益率
        data['volatility'] = data['close'].rolling(window=timeframe).std()  # 波动率
        data['momentum'] = data['close'] - data['close'].shift(timeframe)  # 动量
        data['sma'] = data['close'].rolling(window=timeframe).mean()  # 简单移动平均线
        data['ema'] = data['close'].ewm(span=timeframe).mean()  # 指数移动平均线
        
        # 标签：根据未来价格变化定义趋势
        data['future_return'] = data['close'].shift(-timeframe) / data['close'] - 1
        data['trend'] = np.where(data['future_return'] > 0.002, 1,  # 上涨
                                 np.where(data['future_return'] < -0.002, 0, 2))  # 下跌或横盘
        
        # 删除缺失值
        data = data.dropna()
        
        # 特征和标签
        features = ['return', 'volatility', 'momentum', 'sma', 'ema']
        X = data[features]
        y = data['trend']
        return X, y

    def train(self, data_dict):
        """
        训练模型
        :param data_dict: 包含多个分时线数据的字典，格式为 {timeframe: DataFrame}
        """
        for timeframe, data in data_dict.items():
            print(f"Training model for {timeframe}-minute timeframe...")
            X, y = self.preprocess_data(data, timeframe)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.models[timeframe].fit(X_train, y_train)
            y_pred = self.models[timeframe].predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy for {timeframe}-minute model: {accuracy:.2f}")

    def predict_trend(self, data, timeframe):
        """
        预测市场趋势
        :param data: DataFrame，包含当前分时线的市场数据
        :param timeframe: 当前分时线的时间周期
        :return: 趋势预测结果和反转概率
        """
        X, _ = self.preprocess_data(data, timeframe)
        model = self.models[timeframe]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # 获取最后一个时间点的预测结果
        last_prediction = predictions[-1]
        last_probabilities = probabilities[-1]
        
        trend = self.trend_labels[last_prediction]
        reversal_probability = max(last_probabilities)  # 反转概率
        return trend, reversal_probability

    def predict_multiple_timeframes(self, data_dict):
        """
        对多个分时线进行趋势预测
        :param data_dict: 包含多个分时线数据的字典，格式为 {timeframe: DataFrame}
        :return: 每个分时线的趋势预测结果和反转概率
        """
        results = {}
        for timeframe, data in data_dict.items():
            trend, reversal_probability = self.predict_trend(data, timeframe)
            results[timeframe] = {
                "trend": trend,
                "reversal_probability": reversal_probability
            }
        return results

