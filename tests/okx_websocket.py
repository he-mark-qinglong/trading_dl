import os
import ccxt
import time
from typing import Optional, Dict
import random

from exchange_manager import ExchangeManager, ExchangeConfig, TradingEnvConfig

class TimerTrader:
    def __init__(self, exchange_manager: ExchangeManager,
                take_profit: float, fixed_usdt_amount: float, 
                taker_fee: float, stop_loss:float):
        self.exchange_manager = exchange_manager
        self.take_profit = take_profit  # 止盈百分比（0.3%）
        self.stop_loss = stop_loss #止损百分比，20倍杠杆条件下40%
        self.fixed_usdt_amount = fixed_usdt_amount  # 固定开仓金额（0.5 USDT）
        self.taker_fee = taker_fee  # 吃单手续费（0.05%）
        self.leverage = exchange_manager.config.leverage

    def execute_trade(self, side: str):
        """
        定时执行开仓交易，并设置止盈
        """
        try:
            # 获取市场价格
            market_price = self.get_market_price()
            if market_price <= 0:
                raise ValueError("无法获取市场价格！")

            # 根据固定金额计算开仓量
            amount = self.fixed_usdt_amount * random.uniform(1, 1.4) # (self.fixed_usdt_amount * self.leverage) / market_price
            print(f"准备开仓: {side}, 开仓金额: {self.fixed_usdt_amount} USDT, 开仓量: {amount:.4f} {self.exchange_manager.config.symbol}")

            # 开仓
            pos_side = "long" if side == "buy" else "short"
            order = self.exchange_manager.exchange.create_order(
                symbol=self.exchange_manager.config.symbol,
                type="market",
                side=side,
                amount=amount,
                params={
                    "reduceOnly": True,  # 确保只减少持仓  
                    "posSide": pos_side  # 指定仓位方向
                }
            )
            print(f"开仓成功: {order}")

            # 获取开仓价格
            entry_price = self.get_entry_price(pos_side)
            # if entry_price is None:
            #     raise ValueError("无法获取开仓价格！")

            print(f"开仓价格: {entry_price:.2f} USDT")

            # 设置止盈
            self.set_take_profit(pos_side, amount, entry_price)
            
            # 设置止损  
            #self.set_stop_loss(pos_side, amount, entry_price)  

        except Exception as e:
            print(f"交易执行失败: {e}")

    def get_market_price(self) -> float:
        """获取市场价格"""
        try:
            ticker = self.exchange_manager.exchange.fetch_ticker(self.exchange_manager.config.symbol)
            return float(ticker['last'])
        except Exception as e:
            print(f"获取市场价格失败: {e}")
            return 0.0

    def get_entry_price(self, pos_side: str) -> Optional[float]:
        """获取开仓价格"""
        try:
            positions = self.exchange_manager.exchange.fetch_positions([self.exchange_manager.config.symbol])
            for position in positions:
                if position['side'] == pos_side:
                    return float(position['entryPrice'])
            return None
        except Exception as e:
            print(f"获取开仓价格失败: {e}")
            return None

    def set_take_profit(self, pos_side: str, amount: float, entry_price: float):
        """设置止盈订单"""
        try:
            # 考虑手续费的止盈价格
            effective_take_profit = self.take_profit*random.uniform(1,1.2) + (self.taker_fee * 2 * 100 / self.leverage)
            if pos_side == "long":
                take_profit_price = entry_price * (1 + effective_take_profit / 100)
            else:
                take_profit_price = entry_price * (1 - effective_take_profit / 100)

            # 挂止盈单
            exit_side = "sell" if pos_side == "long" else "buy"
            order = self.exchange_manager.exchange.create_order(
                symbol=self.exchange_manager.config.symbol,
                type="limit",
                side=exit_side,
                amount=amount,
                price=take_profit_price,
                params={
                    "posSide": pos_side
                }
            )
            print(f"止盈订单已挂单: {order}")
        except Exception as e:
            print(f"设置止盈失败: {e}")
    def set_stop_loss(self, pos_side: str, amount: float, entry_price: float):  
        """设置止损订单"""  
        try:  
            # 考虑手续费的止损价格  
            effective_stop_loss = self.stop_loss - (self.taker_fee * 2 * 100 / self.leverage)  
            if pos_side == "long":  
                # 多头止损价格：低于开仓价格  
                stop_loss_price = entry_price * (1 - effective_stop_loss / 100)  
            else:  
                # 空头止损价格：高于开仓价格  
                stop_loss_price = entry_price * (1 + effective_stop_loss / 100)  

            # 挂止损单  
            exit_side = "sell" if pos_side == "long" else "buy"  
            order = self.exchange_manager.exchange.create_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="stopMarket",  # 使用市价止损单  
                side=exit_side,  
                amount=amount,  
                params={  
                    "posSide": pos_side,  
                    "stopPx": stop_loss_price  # 设置触发价格  
                }  
            )  
            print(f"止损订单已挂单: {order}")  
        except Exception as e:  
            print(f"设置止损失败: {e}")

# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    # 配置交易对和杠杆
    trading_config = TradingEnvConfig(
        # symbol="TRUMP-USDT-SWAP",  # 交易对
        # symbol = 'AIXBT-USDT-SWAP',
        symbol="ETH-USDT-SWAP",
        # symbol="NC-USDT-SWAP",
        leverage=100  # 杠杆倍数
    )

    # 配置 API 密钥
    exchange_config = ExchangeConfig()

    # 初始化交易所管理器
    exchange_manager = ExchangeManager(trading_config, exchange_config)

    # 初始化定时交易器
    timer_trader = TimerTrader(
        exchange_manager=exchange_manager,
        take_profit=0.1,  # 止盈 0.2%（基础值）
        stop_loss=0.1, 
        fixed_usdt_amount=0.08, #NC
        # fixed_usdt_amount=0.02, #SOL
        # fixed_usdt_amount=4, #AIXBT
        taker_fee=0.05 / 100  # 吃单手续费率（0.05%）
    )

    # 定时执行交易
    while True:
        # timer_trader.execute_trade("sell")
        timer_trader.execute_trade("buy")  
        
        #只考虑单边行情，因为RRR高。并且在更大的分时图上容易找到Stoch RSI的波动周期。
        #核心思想是柱子会随着RSI突破前高、低点。
        #观察K线在哪个分时线上连续x个形态后就会回到趋势上，然后以这个次数比如10，
        #对应这个分时级别-比如3分钟，算出时间为10次操作在平均3分钟内。
        #所以就是3*60 / 10=18s，做一次开仓
        
        #如果要做30分钟的图形--考虑top ACC/LS主动买入的反转，既然总共可以开仓152手左右，那么分配到30分钟内就是30/150=1/5分钟一次
        _5m = 0.1 * random.uniform(30, 90)
        time.sleep(5/150 *  _5m)  