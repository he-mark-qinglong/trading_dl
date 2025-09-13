import pandas as pd  
import numpy as np  
from typing import Dict  
from scipy.stats import linregress  
import pandas as pd  
from typing import Dict, Tuple, List  

 

class TrendDetector:  
    def __init__(self, 
                 config: Dict):  
        """  
        初始化趋势检测器  
        :param config: 配置字典，包括基础阈值和窗口参数  
        """  
        self.config = config  
        self.base_di = config.get('base_di', 3) 
        self.di_multiplier = config.get('di_multiplier', 50) 
        self.base_adx = config.get('base_adx', 20)  
        self.base_atr_pct = config.get('base_atr_pct', 2)  
        self.base_bollinger_width = config.get('base_bollinger_width', 0.05)  
        self.base_trend_score_threshold = config.get('base_trend_score_threshold', 0.65)  
        self.adx_multiplier = config.get('adx_multiplier', 5)  
        self.atr_multiplier = config.get('atr_multiplier', 10)  
        self.bollinger_multiplier = config.get('bollinger_multiplier', 0.01)  
        self.trend_score_multiplier = config.get('trend_score_multiplier', 0.1)  

    def calculate_dynamic_thresholds(self, 
                                     factors: Dict[str, pd.Series]) -> Dict[str, float]:  
        """  
        根据因子数据计算动态阈值  
        :param factors: 因子字典，包含 ATR 百分比、布林带宽度等  
        :return: 动态阈值字典  
        """  
        atr_pct = factors.get('atr_pct', pd.Series([0])).iloc[-1]  
        bollinger_width = factors.get('bollinger_width', pd.Series([0])).iloc[-1]  

        thresholds = {  
            'adx_threshold': self.base_adx + atr_pct * self.adx_multiplier,  
            'di_threshold': self.base_di + atr_pct * self.di_multiplier, 
            'atr_pct_threshold': self.base_atr_pct + bollinger_width * self.atr_multiplier,  
            'bollinger_width_threshold': self.base_bollinger_width - atr_pct * self.bollinger_multiplier,  
            'trend_score_threshold': self.base_trend_score_threshold + atr_pct * self.trend_score_multiplier  
        }  
        return thresholds  

    def detect_market_state(self, 
                            factors: Dict[str, pd.Series], thresholds: Dict[str, float], current_price) -> str:  
        """  
        检测当前市场状态（上升趋势、下降趋势、震荡、高波动）  
        使用 ADX、DI+、DI- 作为主要判断依据，均线作为备选判断方式  
        
        :param factors: 因子字典，包含：  
            - adx: ADX指标  
            - +di: DI+指标 (Positive Directional Indicator)  
            - -di: DI-指标 (Negative Directional Indicator)  
            - atr_pct: ATR百分比  
            - bollinger_width: 布林带宽度  
            - ma_20: 20周期均线  
            - ma_60: 60周期均线  
            - price: 当前价格  
        :param thresholds: 动态阈值字典，包含：  
            - adx_threshold: ADX趋势强度阈值  
            - di_threshold: DI交叉判断阈值  
            - atr_pct_threshold: ATR波动阈值  
            - bollinger_width_threshold: 布林带宽度阈值  
        :return: 市场状态（'uptrend', 'downtrend', 'range', 'volatile'）  
        """  
        # 获取最新的指标值  
        adx = factors.get('adx', pd.Series([0])).iloc[-1]  
        di_plus = factors.get('+di', pd.Series([0])).iloc[-1]  
        di_minus = factors.get('-di', pd.Series([0])).iloc[-1]  
        atr_pct = factors.get('atr_pct', pd.Series([0])).iloc[-1]  
        bollinger_width = factors.get('bollinger_width', pd.Series([0])).iloc[-1]  
        
        # 1. 首先判断是否为趋势市场（使用ADX）  
        if adx > thresholds['adx_threshold']:  
            # 使用DI+和DI-判断趋势方向  
            di_diff = di_plus - di_minus  # DI+和DI-的差值  
            
            if abs(di_diff) > thresholds['di_threshold']:  # 差值显著  
                if di_plus > di_minus:  
                    return 'uptrend_type'  # DI+ > DI-，上升趋势  
                else:  
                    return 'downtrend_type'  # DI- > DI+，下降趋势  
            else:  
                # DI+和DI-差值不显著，使用均线作为备选判断方式    
                ma_20 = factors.get('ma_20', pd.Series([0])).iloc[-1]  
                ma_60 = factors.get('ma_60', pd.Series([0])).iloc[-1]  
                
                if current_price > ma_20 > ma_60:  
                    return 'uptrend_type'  
                elif current_price < ma_20 < ma_60:  
                    return 'downtrend_type'  
                else:  
                    return 'range_type'  # 均线关系不明确，视为震荡  
        
        # 2. 判断是否为高波动市场  
        elif atr_pct > thresholds['atr_pct_threshold']:  
            return 'volatile_type'  
        
        # 3. 判断是否为震荡市场  
        elif bollinger_width > thresholds['bollinger_width_threshold']:  
            return 'range_type'  
        
        # 4. 默认返回震荡市场  
        return 'range_type'  

    def get_dynamic_weights(self, 
                            market_state: str) -> Dict[str, float]:  
        """  
        根据市场状态动态调整因子权重  
        :param market_state: 当前市场状态  
        :return: 因子权重字典  
        """  

        if market_state == 'uptrend_type':  
            return {  
                '+di': 0.3,        # DI+在上升趋势中更重要  
                'ma_20_slope': 0.25,      # 均线斜率确认趋势  
                'macd_diff': 0.2,      # MACD差值确认动量  
                'obv_slope': 0.15,     # 成交量趋势确认  
                'rsi': 0.1             # RSI确认强度  
            }  
        
        elif market_state == 'downtrend_type':  
            return {  
                '-di': 0.3,       # DI-在下降趋势中更重要  
                'ma_20_slope': 0.25,      # 均线斜率确认趋势  
                'macd_diff': 0.2,      # MACD差值确认动量  
                'obv_slope': 0.15,     # 成交量趋势确认  
                'rsi': 0.1             # RSI确认强度  
            }  
        
        elif market_state == 'range_type':  
            return {  
                'bb_width': 0.3,       # 布林带宽度更重要  
                'rsi': 0.25,           # RSI超买超卖  
                'stoch_k': 0.2,        # 随机指标  
                'cci': 0.15,           # CCI确认区间  
                'cmf': 0.1             # 资金流确认  
            }  
        
        elif market_state == 'volatile_type':  
            return {  
                'atr_pct': 0.35,       # ATR百分比反映价格波动  
                'bb_width': 0.25,      # 布林带宽度反映波动范围  
                'leverage_ratio': 0.25,    # 杠杆率指标  
                'force_index': 0.2,    # 强力指数替代volume_std，反映成交量和价格的综合波动  
                'vwap_deviation': 0.15, # VWAP偏离度反映价格异常波动  
                'obv_slope': 0.05      # OBV斜率反映成交量趋势变化  
            }  
        
    def get_dynamic_divergence_weights(self, 
                                       market_state: str, divergence: str) -> Dict[str, float]:  
        """  
        根据市场状态和背离信号动态调整因子权重，并确保权重总和为 1  
        :param market_state: 当前市场状态  
        :param divergence: 背离信号（'bullish_divergence', 'bearish_divergence', 'no_divergence'）  
        :return: 修改后的权重副本（归一化后）  
        """  
        # 获取基础权重（默认权重）  
        base_weights = self.get_dynamic_weights(market_state)  

        # 创建权重的副本，避免修改原始权重  
        adjusted_weights = base_weights.copy()  

        # 根据背离信号调整权重  
        if divergence == 'bullish_divergence':  
            adjusted_weights['macd_diff'] += 0.1  # 增强 MACD 差值的权重  
            adjusted_weights['rsi'] += 0.1        # 增强 RSI 的权重  
        elif divergence == 'bearish_divergence':  
            adjusted_weights['macd_diff'] += 0.1  
            adjusted_weights['rsi'] += 0.1  

        # 重新归一化权重  
        total_weight = sum(adjusted_weights.values())  
        normalized_weights = {factor: weight / total_weight for factor, weight in adjusted_weights.items()}  

        return normalized_weights  

    def calculate_trend_score(self, 
                              factors: Dict[str, pd.Series], weights: Dict[str, float]) -> float:  
        """  
        根据因子值和权重计算趋势评分  
        :param factors: 因子字典  
        :param weights: 因子权重字典  
        :return: 趋势评分  
        """  
        trend_score = 0.0  
        for factor, weight in weights.items():  
            value = factors.get(factor, pd.Series([0])).iloc[-1]  
            trend_score += value * weight  
        return trend_score  

    def detect_trend(self,   
                    factors: Dict[str, pd.Series],   
                    current_price,   
                    df: pd.DataFrame = None) -> Tuple[str, Dict[str, bool]]:  
        """  
        检测市场趋势（上升、下降、震荡）  
        :param factors: 因子字典  
        :param current_price: 当前价格  
        :param df: 历史数据  
        :return: (趋势方向, {市场状态, 背离信号})  
        """  
        # Step 1: 计算动态阈值  
        thresholds = self.calculate_dynamic_thresholds(factors)  

        # Step 2: 检测市场状态  
        market_state = self.detect_market_state(factors, thresholds, current_price)  
        
        # Step 3: 判断背离信号  
        divergence = self.detect_weighted_divergence(factors, df, market_state)  
        
        # Step 4: 获取动态权重  
        if divergence == False:  
            weights = self.get_dynamic_weights(market_state)   
        else:  
            weights = self.get_dynamic_divergence_weights(market_state,   
                                                        divergence=divergence)   

        # Step 5: 计算趋势评分  
        trend_score = self.calculate_trend_score(factors, weights)  

        # Step 6: 判断趋势方向  
        trend_score_threshold = thresholds['trend_score_threshold']   
        
        if trend_score > trend_score_threshold:  
            return 'uptrend', {"market_state": market_state, "divergence": divergence}  
        elif trend_score < -trend_score_threshold:  
            return 'downtrend', {"market_state": market_state, "divergence": divergence}  
        else:  
            return 'range', {"market_state": market_state, "divergence": divergence}  
    
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
            methods: List[str] = ["volume_price", "obv_price", "macd_volume", "rsi_volume", "price_fund_flow"],  
            threshold: float = 0.05  # 背离的阈值，例如 5%  
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

            # 5. 价格与资金流入流出背离  
            if "price_fund_flow" in methods:  
                price_change = factors["ema_10"].pct_change().iloc[-1]  # 最近 ema_10 的变化率  
                fund_flow_change = factors["fund_flow"].pct_change().iloc[-1]  # 最近资金流入流出的变化率  
                # 判断 ema_10 和资金流入流出的变化方向是否一致  
                results["price_fund_flow"] = abs(price_change - fund_flow_change) > threshold  

            return results 
        # 检测每种背离方式的结果  
        divergence_results = detect_volume_price_divergence(factors, df, methods)  

        #分类市场类型的weights
        def get_market_weights(market_type: str) -> Dict[str, float]:  
            """  
            根据市场类型返回对应的权重分配。  
            
            :param market_type: 市场类型，可选值为 "uptrend", "downtrend", "range", "volatile"。  
            :return: 对应市场类型的权重字典。  
            """  
            weights = {  
                "uptrend": {  
                    "volume_price": 0.35,  
                    "obv_price": 0.25,  
                    "macd_volume": 0.3,  
                    "rsi_volume": 0.1,  
                    "price_fund_flow": 0.2,  
                },  
                "downtrend": {  
                    "volume_price": 0.3,  
                    "obv_price": 0.2,  
                    "macd_volume": 0.25,  
                    "rsi_volume": 0.15,  
                    "price_fund_flow": 0.3,  
                },  
                "range": {  
                    "volume_price": 0.25,  
                    "obv_price": 0.2,  
                    "macd_volume": 0.2,  
                    "rsi_volume": 0.25,  
                    "price_fund_flow": 0.3,  
                },  
                "volatile": {  
                    "volume_price": 0.4,  
                    "obv_price": 0.15,  
                    "macd_volume": 0.15,  
                    "rsi_volume": 0.2,  
                    "price_fund_flow": 0.35,  
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

    
def test_trend():
    # 示例因子数据  
    factors = {  
        'adx': pd.Series([30]),  
        'atr_pct': pd.Series([2.5]),  
        'bollinger_width': pd.Series([0.04]),  
        'ma_20_slope': pd.Series([0.2]),  
        'macd_diff': pd.Series([0.1]),  
        'supertrend_direction': pd.Series([1]),  
        'obv_slope': pd.Series([0.05]),  
        'rsi': pd.Series([50]),  
        'stoch_k': pd.Series([20]),  
        'cci': pd.Series([100]),  
        'atr': pd.Series([1.5]),  
        'volatility': pd.Series([0.3]),  
        'price_range': pd.Series([0.2]),  
        'cmf': pd.Series([0.1])  
    }  

    # 初始化趋势检测器  
    config = {  
        'base_adx': 20,  
        'base_atr_pct': 2,  
        'base_di': 3,
        'di_multiplier': 50,
        'base_bollinger_width': 0.05,  
        'base_trend_score_threshold': 0.65,  
        'adx_multiplier': 5,  
        'atr_multiplier': 10,  
        'bollinger_multiplier': 0.01,  
        'trend_score_multiplier': 0.1  
    }  
    trend_detector = TrendDetector(config)  

    # 检测趋势  
    trend, market_state = trend_detector.detect_trend(factors)  
    print(f"当前市场趋势: {trend}, 市场状态: {market_state}")  