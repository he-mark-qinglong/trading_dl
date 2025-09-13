import ccxt  
from typing import Dict, Optional  
from config import ExchangeConfig, TradingEnvConfig  

class ExchangeManager:  
    def __init__(self, trading_config: TradingEnvConfig, exchange_config: Optional[ExchangeConfig] = None):  
        self.config = trading_config  
        self.exchange = self._init_exchange(exchange_config)  
        self.market = self._init_market()  
        
    def _init_exchange(self, exchange_config: Optional[ExchangeConfig]) -> ccxt.Exchange:  
        """初始化交易所接口"""  
        config = {  
            'enableRateLimit': True,  
            'options': {  
                'defaultType': 'swap',  
                'adjustForTimeDifference': True,
            },
            'proxies': {  
                'http': 'http://127.0.0.1:7890',  # clash 默认 HTTP 代理端口  
                'https': 'http://127.0.0.1:7890'  # clash 默认 HTTPS 代理端口  
            }    
        }  
        
        if exchange_config:  
            config.update({  
                'apiKey': exchange_config.api_key,  
                'secret': exchange_config.secret_key,  
                'password': exchange_config.passphrase  
            })  
            
        exchange = ccxt.okx(config)  
        self._setup_trading_mode(exchange)  
        return exchange  
    
    def _init_market(self) -> Dict:  
        """初始化市场信息"""  
        try:  
            market = self.exchange.market(self.config.symbol)  
            return market  
        except Exception as e:  
            print(f"获取市场信息失败: {e}")  
            return {'limits': {'amount': {'min': 0.06}}} 
         
    def get_account_balance(self) -> float:  
        """获取账户余额"""  
        try:  
            balance = self.exchange.fetch_balance()  
            return balance['total']['USDT']  # 假设账户以 USDT 计价  
        except Exception as e:  
            print(f"获取账户余额失败: {e}")  
            return 0.0  

    def get_positions(self) -> Dict:  
        """获取当前持仓信息"""  
        try:  
            positions = self.exchange.fetch_positions([self.config.symbol])  
            return positions  
        except Exception as e:  
            print(f"获取持仓信息失败: {e}")  
            return {}  

    def _setup_trading_mode(self, exchange: ccxt.Exchange):  
        """设置交易模式和杠杆"""  
        try:  
            exchange.set_position_mode(True, symbol=self.config.symbol)  
            
            for pos_side in ['long', 'short']:  
                exchange.set_leverage(  
                    self.config.leverage,  
                    self.config.symbol,  
                    params={  
                        'mgnMode': 'cross',  
                        'posSide': pos_side  
                    }  
                )  
            print(f"交易模式设置成功: 双向持仓, {self.config.leverage}x杠杆")  
        except Exception as e:  
            print(f"设置交易模式失败: {e}")  

    @property  
    def min_amount(self) -> float:  
        """获取最小交易量"""  
        return self.market['limits']['amount']['min']