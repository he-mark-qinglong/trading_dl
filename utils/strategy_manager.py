from typing import Dict, Optional, List  
import pandas as pd  
from .multi_strategy import BaseStrategy, RangeBreakoutStrategy, AntiMartingaleStrategy, UptrendPullbackStrategy, DowntrendReboundStrategy, MomentumBreakoutStrategy

class CombinedRangeStrategy(BaseStrategy):  
    def __init__(self, leverage: int = 10, risk_per_trade: float = 0.02):  
        super().__init__(leverage, risk_per_trade)  
        self.breakout = RangeBreakoutStrategy(leverage, risk_per_trade)  
        self.martingale = AntiMartingaleStrategy(leverage, risk_per_trade)  

    def check_conditions(self) -> bool:  
        """检查策略条件是否满足"""  
        return True  

    def setup_data(self, df: pd.DataFrame, factors: Dict[str, pd.Series], trend: str):  
        """重写setup_data以同时设置两个子策略的数据"""  
        super().setup_data(df, factors, trend)  
        self.breakout.setup_data(df, factors, trend)  
        self.martingale.setup_data(df, factors, trend)  
    
    def generate_signal(self, symbol: str, balance: float) -> Optional[Dict]:  
        if self.trend != 'range':  
            return None  
        
        # 在区间中部用反马丁格尔  
        if self.martingale.check_conditions():  
            return self.martingale.generate_signal(symbol, balance)  
            
        # 在区间边缘用突破策略  
        if self.breakout.check_conditions():  
            return self.breakout.generate_signal(symbol, balance)  
        
        return None 
    

class StrategyManager:  
    def __init__(self):  
        self.strategies = {  
            'uptrend': UptrendPullbackStrategy(leverage=50),  
            'downtrend': DowntrendReboundStrategy(leverage=50),  
            'range': CombinedRangeStrategy(leverage=50),
            'volatile':MomentumBreakoutStrategy(leverage=50)
        }  
        
    def process_data(self, trend: str, factors: Dict[str, pd.Series],   
                    df: pd.DataFrame, symbol: str, balance: float) -> List[Dict]:  
        """  
        处理数据并返回所有有效的交易信号  
        """  
        
        strategy = self.strategies.get(trend)  
        if strategy:  
            try:
                strategy.setup_data(df, factors, trend)
                signal = strategy.generate_signal(symbol, balance)  
                return signal
            except Exception as e: 
                print(f"generate_signal 数据时出错: {e}")  
        return None