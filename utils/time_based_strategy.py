import pandas as pd
from typing import Dict, List
from .market_trend_detect import TrendDetector

class TimeBasedStrategy:  
    def __init__(self):  
        self.current_position = None  # 当前持仓方向 ("long", "short", None)  
        self.position_size = 0        # 当前持仓量  
        self.last_signal = None       # 上一个交易信号  
        self.trend_detector = None    # 趋势检测器实例  

    def generate_signal(self, factors: Dict[str, pd.Series], current_price: float, df: pd.DataFrame) -> Dict[str, Any]:  
        """  
        根据时间策略生成交易信号  
        :param factors: 因子字典  
        :param current_price: 当前价格  
        :param df: 历史数据  
        :return: 信号字典  
        """  
        # Step 1: 检测市场趋势和状态  
        trend_direction, state_info = self.trend_detector.detect_trend(factors, current_price, df)  
        market_state = state_info["market_state"]  
        divergence = state_info["divergence"]  

        # Step 2: 信号生成逻辑  
        signal = None  
        reason = None  

        # 当前无持仓  
        if self.current_position is None:  
            if trend_direction == "uptrend" and not divergence:  
                signal = "open_long"  
                reason = "trend_up_no_divergence"  
                self.current_position = "long"  
                self.position_size = 1  # 假设固定开仓量  
            elif trend_direction == "downtrend" and not divergence:  
                signal = "open_short"  
                reason = "trend_down_no_divergence"  
                self.current_position = "short"  
                self.position_size = 1  

        # 当前持有多仓  
        elif self.current_position == "long":  
            if trend_direction == "downtrend" or divergence:  
                signal = "close_long"  
                reason = "trend_down_or_divergence"  
                self.current_position = None  
                self.position_size = 0  
            elif trend_direction == "range":  
                signal = "close_long"  
                reason = "trend_range"  
                self.current_position = None  
                self.position_size = 0  

        # 当前持有空仓  
        elif self.current_position == "short":  
            if trend_direction == "uptrend" or divergence:  
                signal = "close_short"  
                reason = "trend_up_or_divergence"  
                self.current_position = None  
                self.position_size = 0  
            elif trend_direction == "range":  
                signal = "close_short"  
                reason = "trend_range"  
                self.current_position = None  
                self.position_size = 0  

        # Step 3: 返回信号  
        return {  
            "timestamp": pd.Timestamp.now(),  
            "signal": signal,  
            "reason": reason,  
            "market_state": market_state,  
            "trend_direction": trend_direction,  
            "divergence": divergence  
        }