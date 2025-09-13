import os  
from dataclasses import dataclass  
"""
apikey = "d015efca-a5a2-4c60-8fd5-d8981d0dd4e8"
secretkey = "0584BD1C059E26C0F415D79825DC4711"
IP = ""
备注名 = "强化学习+动态策略"
权限 = "读取/交易"
"""
@dataclass  
class ExchangeConfig:  
    api_key: str = os.getenv('OKX_API_KEY', '62b5698a-1359-4717-9521-a92376038a26')  
    secret_key: str = os.getenv('OKX_SECRET_KEY', '87C28F978AB815D90B932C65F578E149')  
    passphrase: str = os.getenv('OKX_PASSPHRASE', 'Q1/w2/e3/')  

@dataclass  
class TradingEnvConfig:  
    symbol: str = "PEPE/USDT:USDT"  
    timeframe: str = "1m"  
    initial_balance: float = 10000.0  
    max_position: float = 0.5 
    commission_rate: float = 0.01 * 2
    leverage: int = 100  
    window_size: int = 24

trend_detect_config = {  
    'base_adx': 20,  #对于加密货币波动较大，从25降低为20可以提高敏感度.
    'base_atr_pct': 2,  
    'base_di': 3,
    'di_multiplier': 50,
    'base_bollinger_width': 0.05,  
    'base_trend_score_threshold': 0.7,  
    'adx_multiplier': 5,  
    'atr_multiplier': 10,  
    'bollinger_multiplier': 0.01,  
    'trend_score_multiplier': 0.1  
}