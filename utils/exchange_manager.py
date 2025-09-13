from typing import Optional, Dict
import ccxt

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
                'defaultType': 'swap',  # 默认使用合约交易
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
            return {'limits': {'amount': {'min': 1}}}  # 默认最小交易量

    def _setup_trading_mode(self, exchange: ccxt.Exchange):
        """设置交易模式和杠杆"""
        try:
            exchange.set_position_mode(True, symbol=self.config.symbol)

            for pos_side in ['long', 'short']:
                exchange.set_leverage(
                    self.config.leverage,
                    self.config.symbol,
                    params={
                        'mgnMode': 'cross',  # 全仓模式
                        'posSide': pos_side
                    }
                )
            print(f"交易模式设置成功: 双向持仓, {self.config.leverage}x杠杆")
        except Exception as e:
            print(f"设置交易模式失败: {e}")
