import os
import ccxt
import time
from typing import Optional, Dict
import random

class ExchangeConfig:
    api_key: str = os.getenv('OKX_API_KEY', '62b5698a-1359-4717-9521-a92376038a26')
    secret_key: str = os.getenv('OKX_SECRET_KEY', '87C28F978AB815D90B932C65F578E149')
    passphrase: str = os.getenv('OKX_PASSPHRASE', 'Q1/w2/e3/')

class BinanceExchangeConfig(ExchangeConfig):
    api_key:str = "Cy4lhqGCE4rg9OvhxCDE7PRuxPQIfE6mWyqpk3S8tPw2yauX3bip9Z5gpZy63YYv"
    secret_key:str = "nKhvh8Vfw08YPyOSQ4iu9xWUPuJI97c3TnfJyuaqc812IbIYOf87gxxxT7xC7ONe"


class TradingEnvConfig:
    def __init__(self, symbol: str, leverage: int):
        self.symbol = symbol
        self.leverage = leverage


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
                'defaultType': 'future', #'swap',  # 默认使用合约交易, okx是swap，binance是future
                'adjustForTimeDifference': True,
            },
            'proxies': {
                'http': 'http://127.0.0.1:7890',  # Clash 默认 HTTP 代理端口
                'https': 'http://127.0.0.1:7890'  # Clash 默认 HTTPS 代理端口
            }
        }

        if exchange_config:
            config.update({
                'apiKey': exchange_config.api_key,
                'secret': exchange_config.secret_key,
                # 'password': exchange_config.passphrase  #only okx required this
            })

        # exchange = ccxt.okx(config)
        exchange = ccxt.binance(config)
        self._setup_trading_mode(exchange)
        return exchange

    def _init_market(self) -> Dict:
        """初始化市场信息"""
        try:
            market = self.exchange.market(self.config.symbol)
            return market
        except Exception as e:
            print(f"获取市场信息失败: {e}")
            return {'limits': {'amount': {'min': 1}}}  # 默认最小交易量

    def _setup_trading_mode(self, exchange: ccxt.Exchange):
        """设置交易模式和杠杆"""
        # 加载市场信息（必须步骤）  
        exchange.load_markets()  
        try:
            # exchange.set_position_mode(True)

            for pos_side in ['long', 'short']:
                exchange.set_leverage(
                    self.config.leverage,
                    self.config.symbol,
                    params={
                        'mgnMode': 'cross',  # 全仓模式
                        'posSide': pos_side
                    }
                )
            print(f"交易模式设置成功:{self.config.leverage}x杠杆")
        except Exception as e:
            print(f"设置交易模式失败: {e}")