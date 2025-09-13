from envs.trading_core import TradingCore  
from envs.exchange_manager import ExchangeManager  
from config import TradingEnvConfig  
from typing import Dict  
import pandas as pd  


class MartingaleTradingCore(TradingCore):  
    def __init__(self, trading_config: TradingEnvConfig, exchange_manager: ExchangeManager):  
        super().__init__(trading_config, exchange_manager)  
        self.max_add_times = 7  # 最大加仓次数  
        self.add_times = 0  # 当前加仓次数  
        self.opened_trend = None  # 当前趋势  
        self.price_drop_threshold = 0.005  # 每次加仓的价格跌幅阈值（0.5%）  
        self.last_add_price = None  # 上一次加仓的价格  
        self.entry_atr = None  # 记录开仓时的ATR值  

    def execute_strategy(self, trend: str, current_price: float, factors: Dict[str, pd.Series]) -> None:  
        """  
        执行马丁格尔策略逻辑  
        :param trend: 当前趋势（由外部传入，'uptrend' 或 'downtrend'）  
        :param current_price: 当前市场价格  
        :param factors: 因子数据（由外部传入）  
        """  
        # 获取当前ATR值（假设从factors中传入）  
        atr = factors.get("atr", pd.Series()).iloc[-1] if "atr" in factors else 0  
        fee_rate = 0.001  # 手续费率（千分之一）  

        if self.position == 0:  
            # 如果没有持仓，开仓  
            self._open_position(trend, current_price, atr)  
        elif trend != self.opened_trend:  
            # 如果趋势发生变化，平仓并重新开仓  
            print(f'close reason:如果趋势发生变化，平仓并重新开仓 ')
            self.close_position()  
            self._open_position(trend, current_price, atr)  
        elif self._should_close_for_profit(current_price, fee_rate):  
            # 如果达到盈利目标，平仓  
            print(f'close reason:达到盈利目标，平仓')
            self.close_position()  
        elif self._should_add_position(current_price):  
            # 如果达到加仓条件，加仓  
            if self.add_times < self.max_add_times:  
                self._add_position(current_price)  
            else:  
                print("已达到最大加仓次数，等待趋势变化或平仓")  

    def _should_close_for_profit(self, current_price: float, fee_rate: float) -> bool:  
        """  
        判断是否达到盈利目标  
        :param current_price: 当前市场价格  
        :param fee_rate: 手续费率  
        :return: 是否满足平仓条件  
        """  
        if self.entry_atr is None or self.position == 0:  
            return False  

        # 计算ATR目标盈利和手续费成本  
        atr_target = self.entry_atr * 2 / self.config.leverage  
        fee_cost = self.entry_price * fee_rate  
        total_target = atr_target + fee_cost  

        # 计算当前浮动盈利  
        floating_profit = (current_price - self.entry_price) * (1 if self.opened_trend == "uptrend" else -1)  
        return floating_profit >= total_target  

    def _should_add_position(self, current_price: float) -> bool:  
        """  
        判断是否满足加仓条件（基于价格跌幅）  
        :param current_price: 当前市场价格  
        :return: 是否满足加仓条件  
        """  
        if self.last_add_price is None:  
            self.last_add_price = self.entry_price  # 初始化为开仓价格  

        # 计算价格跌幅（多头）或涨幅（空头）  
        if self.opened_trend == "uptrend":  
            price_drop = (self.last_add_price - current_price) / self.last_add_price  
            return price_drop >= self.price_drop_threshold  
        elif self.opened_trend == "downtrend":  
            price_rise = (current_price - self.last_add_price) / self.last_add_price  
            return price_rise >= self.price_drop_threshold  
        return False  

    def _open_position(self, trend: str, current_price: float, atr: float) -> None:  
        """  
        开仓逻辑  
        :param trend: 当前趋势  
        :param current_price: 当前市场价格  
        :param atr: 当前的ATR值  
        """  
        trade_size = self.balance / current_price  # 全仓开仓  
        if not self.execute_trade(trade_size if trend == "uptrend" else -trade_size, current_price):  
            print("开仓失败")  
            return  

        # 更新策略状态  
        self.opened_trend = trend  
        self.add_times = 0  
        self.entry_price = current_price  
        self.last_add_price = current_price  # 初始化加仓价格  
        self.entry_atr = atr  # 记录开仓时的ATR值  
        print(f"开仓成功: {'多头' if trend == 'uptrend' else '空头'}，价格: {current_price}, 仓位: {self.position}")  

    def _add_position(self, current_price: float) -> None:  
        """  
        加仓逻辑（基于价格跌幅）  
        :param current_price: 当前市场价格  
        """  
        # 动态计算加仓比例（可根据价格跌幅调整）  
        add_position_pct = self._calculate_add_position_pct(current_price)  
        add_size = self.position * add_position_pct  # 计算加仓数量  

        if not self.execute_trade(add_size if self.opened_trend == "uptrend" else -add_size, current_price):  
            print("加仓失败")  
            return  

        # 更新策略状态  
        self.add_times += 1  
        self.last_add_price = current_price  # 更新上一次加仓价格  
        self.entry_price = (self.entry_price * self.position + current_price * add_size) / (self.position + add_size)  
        print(f"加仓成功: {'多头' if self.opened_trend == 'uptrend' else '空头'}，价格: {current_price}, 新仓位: {self.position}")  

    def _calculate_add_position_pct(self, current_price: float) -> float:  
        """  
        动态计算加仓比例（可根据价格跌幅调整）  
        :param current_price: 当前市场价格  
        :return: 动态加仓比例  
        """  
        # 示例：价格跌幅越大，加仓比例越高  
        price_drop = abs((self.last_add_price - current_price) / self.last_add_price)  
        return min(max(price_drop * 5, 0.1), 0.7)  # 加仓比例限制在 10%-70% 之间  

    def close_position(self) -> None:  
        """  
        平仓逻辑  
        """  
        if super().close_position():  
            print(f"平仓成功，当前余额: {self.balance}")  
        else:  
            print("平仓失败")