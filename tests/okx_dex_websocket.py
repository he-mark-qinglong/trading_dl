import os  
import ccxt  
import time  
from typing import Optional, Dict  
import random  
import keyboard  # 用于捕获键盘输入事件  
import threading  # 用于实现持续开仓的线程  

class ExchangeConfig:  
    api_key: str = os.getenv('OKX_API_KEY', '62b5698a-1359-4717-9521-a92376038a26')  
    secret_key: str = os.getenv('OKX_SECRET_KEY', '87C28F978AB815D90B932C65F578E149')  
    passphrase: str = os.getenv('OKX_PASSPHRASE', 'Q1/w2/e3/')  


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


class TimerTrader:  
    def __init__(self, exchange_manager: ExchangeManager,  
                 take_profit: float, fixed_usdt_amount: float,  
                 taker_fee: float, stop_loss: float):  
        self.exchange_manager = exchange_manager  
        self.take_profit = take_profit  # 止盈百分比（如 0.3%）  
        self.stop_loss = stop_loss  # 止损百分比（如 1.5%）  
        self.fixed_usdt_amount = fixed_usdt_amount  # 固定开仓金额（如 5 USDT）  
        self.taker_fee = taker_fee  # 吃单手续费（如 0.05%）  
        self.leverage = exchange_manager.config.leverage  
        self.running = False  # 控制持续开仓的标志  

    def open_position(self, side: str):  
        """  
        开仓逻辑：开多或开空，并设置自动止盈止损  
        """  
        try:  
            # 获取市场价格  
            market_price = self.get_market_price()  
            if market_price <= 0:  
                raise ValueError("无法获取市场价格！")  

            # 根据固定金额计算开仓量  
            amount = self.fixed_usdt_amount * self.leverage / market_price  
            print(f"准备开仓: {side}, 开仓金额: {self.fixed_usdt_amount} USDT, 开仓量: {amount:.4f} {self.exchange_manager.config.symbol}")  

            # 计算止盈和止损价格  
            if side == "buy":  # 开多  
                take_profit_price = market_price * (1 + self.take_profit / 100)  
                stop_loss_price = market_price * (1 - self.stop_loss / 100)  
                pos_side = "long"  
            else:  # 开空  
                take_profit_price = market_price * (1 - self.take_profit / 100)  
                stop_loss_price = market_price * (1 + self.stop_loss / 100)  
                pos_side = "short"  

            # 开仓  
            order = self.exchange_manager.exchange.create_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="market",  
                side=side,  
                amount=amount,  
                params={  
                    "reduceOnly": False,  # 确保是开仓  
                    "posSide": pos_side,  # 指定仓位方向  
                    "takeProfitPrice": take_profit_price,  # 自动止盈价格  
                    "stopLossPrice": stop_loss_price  # 自动止损价格  
                }  
            )  
            print(f"开仓成功: {order}")  

        except Exception as e:  
            print(f"开仓失败: {e}")  

    def close_position(self, pos_side: str):  
        """  
        平仓逻辑：平多或平空  
        """  
        try:  
            # 获取当前持仓量  
            positions = self.exchange_manager.exchange.fetch_positions([self.exchange_manager.config.symbol])  
            for position in positions:  
                if position['side'] == pos_side and float(position['contracts']) > 0:  
                    amount = float(position['contracts'])  
                    print(f"准备平仓: {pos_side}, 平仓量: {amount:.4f} {self.exchange_manager.config.symbol}")  

                    # 平仓  
                    exit_side = "sell" if pos_side == "long" else "buy"  
                    order = self.exchange_manager.exchange.create_order(  
                        symbol=self.exchange_manager.config.symbol,  
                        type="market",  
                        side=exit_side,  
                        amount=amount,  
                        params={  
                            "reduceOnly": True,  # 确保是平仓  
                            "posSide": pos_side  # 指定仓位方向  
                        }  
                    )  
                    print(f"平仓成功: {order}")  
                    return  

            print(f"没有找到需要平仓的 {pos_side} 持仓")  
        except Exception as e:  
            print(f"平仓失败: {e}")  

    def get_market_price(self) -> float:  
        """获取市场价格"""  
        try:  
            ticker = self.exchange_manager.exchange.fetch_ticker(self.exchange_manager.config.symbol)  
            return float(ticker['last'])  
        except Exception as e:  
            print(f"获取市场价格失败: {e}")  
            return 0.0  

    def start_trading(self, side: str):  
        """持续开仓逻辑"""  
        self.running = True  
        while self.running:  
            self.open_position(side)  
            time.sleep(random.uniform(1, 3))  # 随机间隔 1 到 3 秒  

    def stop_trading(self):  
        """停止持续开仓"""  
        self.running = False  


# ==========================  
# Main Execution with Keyboard Control  
# ==========================  
if __name__ == "__main__":  
    # 配置交易对和杠杆  
    trading_config = TradingEnvConfig(  
        symbol='AIXBT-USDT-SWAP',  
        leverage=50  
    )  

    # 配置 API 密钥  
    exchange_config = ExchangeConfig()  

    # 初始化交易所管理器  
    exchange_manager = ExchangeManager(trading_config, exchange_config)  

    # 初始化交易器  
    timer_trader = TimerTrader(  
        exchange_manager=exchange_manager,  
        take_profit=0.3,  # 止盈 0.3%  
        stop_loss=1.5,  # 止损 1.5%  
        fixed_usdt_amount=1,  
        taker_fee=0.05 / 100  
    )  

    print("按键控制交易程序已启动：")  
    print("上键：持续开多")  
    print("下键：持续开空")  
    print("左键：平空")  
    print("右键：平多")  
    print("空格键：停止持续开仓")  

    while True:  
        if keyboard.is_pressed("up"):  # 上键持续开多  
            print("检测到上键：持续开多")  
            threading.Thread(target=timer_trader.start_trading, args=("buy",)).start()  
            time.sleep(0.5)  

        if keyboard.is_pressed("down"):  # 下键持续开空  
            print("检测到下键：持续开空")  
            threading.Thread(target=timer_trader.start_trading, args=("sell",)).start()  
            time.sleep(0.5)  

        if keyboard.is_pressed("left"):  # 左键平空  
            print("检测到左键：平空")  
            timer_trader.close_position("short")  
            time.sleep(0.5)  

        if keyboard.is_pressed("right"):  # 右键平多  
            print("检测到右键：平多")  
            timer_trader.close_position("long")  
            time.sleep(0.5)  

        if keyboard.is_pressed("space"):  # 空格键停止持续开仓  
            print("检测到空格键：停止持续开仓")  
            timer_trader.stop_trading()  
            time.sleep(0.5)