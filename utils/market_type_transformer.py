import torch  
import torch.nn as nn  
import torch.optim as optim  
import pandas as pd  
import numpy as np  
from typing import Dict, Tuple, List  
from typing import Dict  
import pandas as pd  

#A1.外部真实使用
def classify_trend_from_factors(factors: Dict[str, pd.Series], 
                                # future_df:pd.DataFrame, current_price: float, 
                                index: int = -1) -> int:  
    """  
    根据因子值分类趋势，返回单个时间点的趋势类型  
    :param factors: 包含因子名称和对应 pd.Series 的字典  
    :param index: 当前需要判断趋势的时间点索引  
    :return: 趋势类型（0: 上升趋势, 1: 震荡趋势, 2: 下降趋势, 3: 高波动趋势）  
    """  
    try:
        # min_percent = future_df['low'].min()/current_price
        # high_percent = future_df['high'].max()/current_price

        # 提取需要的因子值  
        obv_slope = factors.get('obv_slope', pd.Series([0])).iloc[index]  
        ma_10_slope = factors.get('ma_10_slope', pd.Series([0])).iloc[index]  
        adx = factors.get('adx', pd.Series([0])).iloc[index]  
        plus_di = factors.get('+di', pd.Series([0])).iloc[index]  
        minus_di = factors.get('-di', pd.Series([0])).iloc[index]  
        supertrend_direction = factors.get('supertrend_direction', pd.Series([0])).iloc[index]  
        atr = factors.get('atr', pd.Series([0])).iloc[index]  
        bb_width = factors.get('bb_width', pd.Series([0])).iloc[index]  
        # 上升趋势逻辑  
        if (  
            obv_slope > 0 and  
            ma_10_slope > 0 and  
            adx > 25 and  
            plus_di > minus_di and  
            supertrend_direction > 0  
        ):  
            return 0 #[0, min_percent, high_percent]  # 上升趋势 (uptrend)  

        # 下降趋势逻辑  
        if (  
            obv_slope < 0 and  
            ma_10_slope < 0 and  
            adx > 25 and  
            plus_di < minus_di and  
            supertrend_direction < 0  
        ):  
            return 2 #[2, min_percent, high_percent]  # 下降趋势 (downtrend)  

        # 高波动趋势逻辑  
        if (  
            adx < 25 and  
            atr > pd.Series(atr).mean() and  
            bb_width > pd.Series(bb_width).mean()  
        ):  
            return 3 #[3, min_percent, high_percent]  # 高波动趋势 (volatile)  

        # 默认震荡趋势  
        return 1 #[1, min_percent, high_percent]  # 震荡趋势 (range)\
    except Exception as e:
        print(f'classify_trend_from_factors-iloc keys:{e}')
        return 1 #[1, min_percent, high_percent]    

class CNNTrendTransformer(nn.Module):  
    def __init__(self, num_features, num_classes, d_model=128, nhead=8, num_layers=2, dropout=0.1):  
        """  
        基于 Transformer 的多因子趋势检测模型，添加 CNN 模块  
        :param num_features: 输入因子数量  
        :param num_classes: 输出趋势类别数量  
        :param d_model: Transformer 的嵌入维度  
        :param nhead: 多头注意力的头数  
        :param num_layers: Transformer Encoder 的层数  
        :param dropout: Dropout 概率  
        """  
        super(CNNTrendTransformer, self).__init__()  
        
        # CNN 模块：提取局部特征  
        self.cnn = nn.Sequential(  
            nn.Conv1d(in_channels=num_features, out_channels=d_model, kernel_size=3, stride=1, padding=1),  # 保持时间维度不变  
            nn.ReLU(),  # 激活函数  
            nn.MaxPool1d(kernel_size=2, stride=2),  # 时间维度减半  
        )  
        
        # 线性嵌入层  
        self.embedding = nn.Linear(d_model, d_model)  
        self.embedding_norm = nn.LayerNorm(d_model)  # 对嵌入后的特征进行 LayerNorm  
        self.embedding_dropout = nn.Dropout(dropout)  # 嵌入后增加 Dropout  

        # Transformer Encoder  
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)  
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  

        # 输出分类层  
        self.fc = nn.Linear(d_model, num_classes)  
        self.output_norm = nn.LayerNorm(num_classes)  # 对输出进行 LayerNorm  
        self.output_dropout = nn.Dropout(dropout)  # 输出层增加 Dropout  

    def forward(self, x):  
        """  
        前向传播  
        :param x: 输入数据，形状为 (batch_size, sequence_length, num_features)  
        :return: 分类概率，形状为 (batch_size, num_classes)  
        """  
        # CNN 模块：提取局部特征  
        # 输入形状 (batch_size, sequence_length, num_features)  
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, num_features, sequence_length) 以适配 Conv1d  
        x = self.cnn(x)  # 通过 CNN 提取特征  
        x = x.permute(0, 2, 1)  # 转换回 (batch_size, sequence_length, d_model)  

        # 将输入嵌入到高维空间  
        embedded = self.embedding(x)  # (batch_size, sequence_length, d_model)  
        embedded = self.embedding_norm(embedded)  # 对嵌入结果进行 LayerNorm  
        embedded = self.embedding_dropout(embedded)  # Dropout  
        
        # Transformer Encoder  
        transformer_output = self.transformer_encoder(embedded)  # (batch_size, sequence_length, d_model)  

        # 残差连接：直接加回嵌入层的输出（保留原始特征）  
        transformer_output = transformer_output + embedded  

        # 取最后一个时间步的输出  
        x = transformer_output[:, -1, :]  # (batch_size, d_model)  

        # 全连接分类层  
        x = self.fc(x)  # (batch_size, num_classes)  
        x = self.output_norm(x)  # 对输出进行 LayerNorm  
        x = self.output_dropout(x)  # Dropout  

        return x  


class _TrendDetector:  
    def __init__(self, config: Dict, num_features: int, num_classes: int = 4):  
        """  
        初始化趋势检测器  
        :param config: 配置字典，包括基础阈值和窗口参数  
        :param num_features: 输入因子数量  
        :param num_classes: 输出趋势类别数量（默认为 4，包括 volatile）  
        """  
        self.config = config  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.model = MH_CNNTrendTransformer(num_features=num_features, num_classes=num_classes).to(self.device)  
        self.criterion = nn.CrossEntropyLoss()  
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005)  

    def train(self, symbol, timeframe, train_data: torch.Tensor, train_labels: torch.Tensor, epochs: int = 10):  
        """  
        训练模型  
        :param factors: 输入因子字典  
        :param df: 包含 'close' 列的 DataFrame，用于生成标签  
        :param epochs: 训练轮数  
        """   

        # 模型训练  
        self.model.train()  
        epoch = 0

        while True:
            train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)  
            # 检查 train_data 是否包含 NaN 或 Inf  
            if torch.isnan(train_data).any() or torch.isinf(train_data).any():  
                print("train_data contains NaN or Inf values!")  
                print(train_data)  

            # 检查 train_labels 是否包含 NaN 或 Inf  
            if torch.isnan(train_labels).any() or torch.isinf(train_labels).any():  
                print("train_labels contains NaN or Inf values!")  
                print(train_labels)  
            outputs = self.model(train_data)  
            loss = self.criterion(outputs, train_labels)  
            self.optimizer.zero_grad()  
            loss.backward()  
            self.optimizer.step()  

            epoch += 1
            if epoch % 10 == 0:
                print(f"symbol:{symbol} timeframe:{timeframe} Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")  
            if  loss.item() < 0.01 or epoch > epochs:
                break

    def predict(self, factors: Dict[str, pd.Series]) -> Tuple[str, float]:  
        """  
        预测趋势  
        :param factors: 输入因子字典  
        :return: (预测趋势类别, 概率)  
        """   

        # 模型预测  
        self.model.eval()  
        realtime_data = factors.to(self.device)  
        with torch.no_grad():  
            logits = self.model(realtime_data)  
            probabilities = torch.softmax(logits, dim=1)  
            predicted_class = torch.argmax(probabilities, dim=1).item()  
            confidence = probabilities[0, predicted_class].item()  

        trend_labels = ['uptrend', 'range', 'downtrend', 'volatile']  
        return trend_labels[predicted_class], confidence


    """
    1. **上升趋势（uptrend）**：
    - 在上升趋势中，成交量与价格的同步性非常重要，因此 `volume_price` 和 `macd_volume` 的权重较高。
    - OBV 作为资金流动的中期趋势指标，也有一定权重。
    - RSI 和资金流入流出背离的权重较低，因为市场情绪和短期资金流动对趋势的影响较小。
    2. **下降趋势（downtrend）**：
    - 在下降趋势中，资金流入流出（`price_fund_flow`）和成交量（`volume_price`）的背离更为重要，因为资金的撤离速度和成交量的变化可以反映趋势的强弱。
    - MACD 作为趋势指标也有较高权重。
    - RSI 的权重略高于上升趋势，因为超卖信号可能预示反弹。
    3. **震荡行情（range）**：
    - 在震荡行情中，市场情绪（`rsi_volume`）和资金流动（`price_fund_flow`）的背离更为重要，因为震荡行情中趋势指标的作用较弱。
    - 成交量（`volume_price`）和 OBV 的权重适中，用于判断震荡区间的突破或假突破。
    4. **高波动行情（volatile）**：
    - 在高波动行情中，成交量（`volume_price`）和资金流动（`price_fund_flow`）的背离是核心，因为波动性往往伴随着资金的大量流入或流出。
    - RSI 的权重较高，因为市场情绪在高波动行情中起到重要作用。
    - MACD 和 OBV 的权重较低，因为趋势指标在高波动中可能滞后。
    """
    def detect_weighted_divergence(  self,
        factors: Dict[str, pd.Series],  
        df: pd.DataFrame,  
        market_trend:str,  
        threshold: float = 0.6,  
        methods: List[str] = ["volume_price", "obv_price", "macd_volume", "rsi_volume", "price_fund_flow"]  
    ) -> bool:  
        """  
        综合多指标加权判断是否存在真实背离。  
        
        :param factors: 技术因子字典，键为因子名称，值为 pandas.Series。  
        :param df: 历史数据 DataFrame，包含基础数据（如 open、close、volume 等）。  
        :param weights: 每个背离判断方式的权重，键为背离方式名称，值为权重（0~1）。  
        :param threshold: 综合评分的阈值，超过该值则认为存在真实背离。  
        :param methods: 要检测的量价背离方式列表。  
        :return: 布尔值，表示是否存在真实背离。  
        """  
        def detect_volume_price_divergence(  
            factors: Dict[str, pd.Series],  
            df: pd.DataFrame,  
            methods: List[str] = ["volume_price", "obv_price", "macd_volume", "rsi_volume", "price_fund_flow", "pvt_price", "vwap_price"],  
            threshold: float = 0.1  # 背离的阈值，例如加密货币 10%  
        ) -> Dict[str, bool]:  
            """  
            检测量价背离的函数，支持多种量价背离方式。  
            
            :param factors: 技术因子字典，键为因子名称，值为 pandas.Series。  
                            必须包含以下因子：'ema_10', 'obv', 'macd', 'rsi', 'fund_flow'。  
            :param df: 历史数据 DataFrame，包含基础数据（如 open、close、volume 等）。  
            :param methods: 要检测的量价背离方式列表，默认为五种方式。  
            :param threshold: 背离的阈值，表示价格和指标之间的变化差异超过多少比例时认为存在背离。  
            :return: 一个字典，键为背离方式，值为布尔值，表示是否存在背离。  
            """  
            results = {}  

            # 1. 成交量与价格趋势背离  
            if "volume_price" in methods:  
                price_change = factors["ema_10"].pct_change().iloc[-1]  # 最近 ema_10 的变化率  
                volume_change = df["volume"].pct_change().iloc[-1]  # 最近成交量的变化率  
                # 判断 ema_10 和成交量的变化方向是否一致  
                results["volume_price"] = abs(price_change - volume_change) > threshold  

            # 2. OBV（能量潮）与价格背离  
            if "obv_price" in methods:  
                price_change = factors["ema_10"].pct_change().iloc[-1]  # 最近 ema_10 的变化率  
                obv_change = factors["obv"].pct_change().iloc[-1]  # 最近 OBV 的变化率  
                # 判断 ema_10 和 OBV 的变化方向是否一致  
                results["obv_price"] = abs(price_change - obv_change) > threshold  

            # 3. MACD 与成交量背离  
            if "macd_volume" in methods:  
                macd_change = factors["macd"].pct_change().iloc[-1]  # 最近 MACD 的变化率  
                volume_change = df["volume"].pct_change().iloc[-1]  # 最近成交量的变化率  
                # 判断 MACD 和成交量的变化方向是否一致  
                results["macd_volume"] = abs(macd_change - volume_change) > threshold  

            # 4. RSI 与成交量背离  
            if "rsi_volume" in methods:  
                rsi_value = factors["rsi"].iloc[-1]  # 最近 RSI 值  
                volume_change = df["volume"].pct_change().iloc[-1]  # 最近成交量的变化率  
                # 判断 RSI 是否处于超买/超卖区域，且成交量未同步放大  
                results["rsi_volume"] = ((rsi_value > 70 or rsi_value < 30) and abs(volume_change) < threshold)  

            # 替代 5. 价格与资金流入流出背离  
            if "price_fund_flow" in methods:  
                price_change = factors["ema_10"].pct_change().iloc[-1]  # 最近 ema_10 的变化率  

                # 使用 VWAP 偏离替代 fund_flow  
                vwap_deviation = factors["vwap_deviation"].iloc[-1]  # 最近 VWAP 偏离值  
                results["price_fund_flow"] = abs(vwap_deviation) > threshold  

            # 6. PVT 与价格背离  
            if "pvt_price" in methods:  
                price_change = factors["ema_10"].pct_change().iloc[-1]  # 最近 ema_10 的变化率  
                pvt_change = factors["pvt"].pct_change().iloc[-1]  # 最近 PVT 的变化率  
                results["pvt_price"] = abs(price_change - pvt_change) > threshold  

            # 7. VWAP 偏离与价格背离  
            if "vwap_price" in methods:  
                vwap_deviation = factors["vwap_deviation"].iloc[-1]  # 最近 VWAP 偏离值  
                results["vwap_price"] = abs(vwap_deviation) > threshold

            return results 
        # 检测每种背离方式的结果  
        divergence_results = detect_volume_price_divergence(factors, df, methods)  

        #分类市场类型的weights
        def get_market_weights(market_type: str) -> Dict[str, float]:  
            weights = {  
                "uptrend": {  
                    "volume_price": 0.3,  
                    "obv_price": 0.2,  
                    "macd_volume": 0.25,  
                    "rsi_volume": 0.1,  
                    "price_fund_flow": 0.15,  # 替代为 CMF 或 VWAP  
                    "pvt_price": 0.1,        # 新增 PVT  
                    "vwap_price": 0.1,       # 新增 VWAP 偏离  
                },  
                "downtrend": {  
                    "volume_price": 0.25,  
                    "obv_price": 0.2,  
                    "macd_volume": 0.2,  
                    "rsi_volume": 0.15,  
                    "price_fund_flow": 0.2,  # 替代为 CMF 或 VWAP  
                    "pvt_price": 0.1,        # 新增 PVT  
                    "vwap_price": 0.1,       # 新增 VWAP 偏离  
                },  
                "range": {  
                    "volume_price": 0.2,  
                    "obv_price": 0.2,  
                    "macd_volume": 0.2,  
                    "rsi_volume": 0.2,  
                    "price_fund_flow": 0.15,  # 替代为 CMF 或 VWAP  
                    "pvt_price": 0.15,        # 新增 PVT  
                    "vwap_price": 0.1,        # 新增 VWAP 偏离  
                },  
                "volatile": {  
                    "volume_price": 0.35,  
                    "obv_price": 0.15,  
                    "macd_volume": 0.15,  
                    "rsi_volume": 0.15,  
                    "price_fund_flow": 0.1,  # 替代为 CMF 或 VWAP  
                    "pvt_price": 0.1,        # 新增 PVT  
                    "vwap_price": 0.1,       # 新增 VWAP 偏离  
                },  
            }  
            return weights.get(market_type, {})
        weights = get_market_weights(market_trend)
        # 计算综合评分  
        score = 0.0  
        for method, result in divergence_results.items():  
            if method in weights:  
                score += weights[method] * (1 if result else 0)  

        # 判断综合评分是否超过阈值  
        return score >= threshold