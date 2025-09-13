from envs.trading_core import TradingCore
from envs.exchange_manager import ExchangeManager
from config import TradingEnvConfig
from typing import Dict
import pandas as pd

class AntiMartingaleTradingCore(TradingCore):  
    def __init__(self, trading_config: TradingEnvConfig, exchange_manager: ExchangeManager):  
        super().__init__(trading_config, exchange_manager)  
        self.max_add_times = 7  # 最大加仓次数  
        self.add_times = 0  # 当前加仓次数  
        self.trend = None  # 当前趋势  

    def execute_strategy(self, trend: str, current_price: float, factors: Dict[str, pd.Series]) -> None:  
        """  
        执行反马丁格尔策略逻辑  
        :param trend: 当前趋势（由外部传入，'uptrend' 或 'downtrend'）  
        :param current_price: 当前市场价格  
        :param factors: 因子数据（由外部传入）  
        """  
        # 动态计算止盈比例和加仓比例  
        take_profit_pct = self._calculate_take_profit_pct(factors)  
        add_position_pct = self._calculate_add_position_pct(factors)  

        if self.position == 0:  
            # 如果没有持仓，开仓  
            self._open_position(trend, current_price)  
        elif trend != self.trend:  
            # 如果趋势发生变化，平仓并重新开仓  
            self.close_position()  
            self._open_position(trend, current_price)  
        elif (self.trend == "uptrend" and current_price >= self.entry_price * (1 + take_profit_pct)) or \
             (self.trend == "downtrend" and current_price <= self.entry_price * (1 - take_profit_pct)):  
            # 如果达到止盈目标，加仓  
            if self.add_times < self.max_add_times:  
                self._add_position(current_price, add_position_pct)  
            else:  
                print("已达到最大加仓次数，等待趋势变化或平仓")  

    def _calculate_take_profit_pct(self, factors: Dict[str, pd.Series]) -> float:  
        """  
        根据因子动态计算止盈比例  
        :param factors: 因子数据  
        :return: 动态止盈比例  
        """  
        # 使用最新的 ATR 值动态计算止盈比例  
        atr = factors.get('atr', pd.Series([0])).iloc[-1]  # 获取最新 ATR 值  
        return min(max(atr / 100, 0.01), 0.1)  # 将 ATR 转换为止盈比例，限制在 1%-10% 之间  

    def _calculate_add_position_pct(self, factors: Dict[str, pd.Series]) -> float:  
        """  
        根据因子动态计算加仓比例  
        :param factors: 因子数据  
        :return: 动态加仓比例  
        """  
        # 使用最新的波动率因子动态计算加仓比例  
        volatility = factors.get('volatility', pd.Series([0])).iloc[-1]  # 获取最新波动率值  
        return min(max(volatility / 50, 0.1), 0.5)  # 将波动率转换为加仓比例，限制在 10%-50% 之间  

    def _open_position(self, trend: str, current_price: float) -> None:  
        """  
        开仓逻辑  
        :param trend: 当前趋势  
        :param current_price: 当前市场价格  
        """  
        trade_size = self.balance / current_price  # 全仓开仓  
        if not self.execute_trade(trade_size if trend == "uptrend" else -trade_size, current_price):  
            print("开仓失败")  
            return  

        # 更新策略状态  
        self.trend = trend  
        self.add_times = 0  
        self.entry_price = current_price  
        print(f"开仓成功: {'多头' if trend == 'uptrend' else '空头'}，价格: {current_price}, 仓位: {self.position}")  

    def _add_position(self, current_price: float, add_position_pct: float) -> None:  
        """  
        加仓逻辑  
        :param current_price: 当前市场价格  
        :param add_position_pct: 动态加仓比例  
        """  
        add_size = self.position * add_position_pct  # 计算加仓数量  
        if not self.execute_trade(add_size if self.trend == "uptrend" else -add_size, current_price):  
            print("加仓失败")  
            return  

        # 更新策略状态  
        self.add_times += 1  
        self.entry_price = (self.entry_price * self.position + current_price * add_size) / (self.position + add_size)  
        print(f"加仓成功: {'多头' if self.trend == 'uptrend' else '空头'}，价格: {current_price}, 新仓位: {self.position}")  

    def close_position(self) -> None:  
        """  
        平仓逻辑  
        """  
        if super().close_position():  
            print(f"平仓成功，当前余额: {self.balance}")  
        else:  
            print("平仓失败")