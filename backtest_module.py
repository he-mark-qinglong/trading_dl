import ccxt  
import pandas as pd  
from datetime import datetime, timedelta  
import time  
from typing import Dict, Optional  
from factors import FactorManager  
import threading  
from queue import Queue  
from utils.history_data import DataManager, HistoricalDataLoader  
from utils.strategy_module import DynamicMultiTimeframeStrategy
from utils.market_trend_detect import TrendDetector

class BacktestModule:  
    def __init__(self, symbol: str, timeframes: list, start_date: str, end_date: str, initial_balance: float = 10000.0):  
        """  
        回测模块  
        :param symbol: 交易对（如 'BTC/USDT'）  
        :param timeframes: 时间框架列表（如 ['1m', '5m', '15m']）  
        :param start_date: 回测开始日期（如 '2023-01-01'）  
        :param end_date: 回测结束日期（如 '2023-12-31'）  
        :param initial_balance: 初始账户余额（默认 10000 USDT）  
        """  
        self.symbol = symbol  
        self.timeframes = timeframes  
        self.start_date = pd.to_datetime(start_date)  
        self.end_date = pd.to_datetime(end_date)  
        self.balance = initial_balance  
        self.position = 0  # 当前持仓数量  
        self.history = []  # 记录交易历史  
        self.data = {}  # 存储多时间框架的历史数据  
        self.leverage = 10
        self.strategy_module = DynamicMultiTimeframeStrategy(leverage=self.leverage)  
        self.trend_detector = TrendDetector({  
            'base_adx': 20,  
            'base_atr_pct': 2,  
            'base_bollinger_width': 0.05,  
            'base_trend_score_threshold': 0.6,  
            'adx_multiplier': 5,  
            'atr_multiplier': 10,  
            'bollinger_multiplier': 0.01,  
            'trend_score_multiplier': 0.1  
        })  
        self.factor_manager = FactorManager()  
        self._load_historical_data()  

    def _load_historical_data(self):  
        """加载历史数据"""  
        loader = HistoricalDataLoader('okx')  
        for timeframe in self.timeframes:  
            self.data[timeframe] = loader.fetch_historical_data(self.symbol, timeframe, DataManager())  
            print(f'timeframe:{timeframe} self.data[timeframe].index:{self.data[timeframe].index[0]} start_date:{self.start_date}')
            self.data[timeframe] = self.data[timeframe][(self.data[timeframe].index >= self.start_date) &  
                                                        (self.data[timeframe].index <= self.end_date)]  
            print(f'self.data[timeframe].index[-1] >= self.start_date = {self.data[timeframe].index[-1] >= self.start_date}')
    def _simulate_step(self, current_time: pd.Timestamp):  
        """模拟单个时间步"""  
        for timeframe in self.timeframes:  
            # 获取当前时间窗口内的数据  
            df = self.data[timeframe]  
            # 打印调试信息  
            print(f"Loaded data for {timeframe}: {df.index[0]} current_time{current_time}") 
            df = df[df.index <= current_time]  

            if not df.empty:  
                # 计算因子  
                factors, df = self.factor_manager.calculate_factors(df)  

                # 检测趋势  
                trend, market_state = self.trend_detector.detect_trend(factors)  

                # 生成交易信号  
                current_price = df['close'].iloc[-1]  
                self.strategy_module.process_data(trend, factors, current_price, self.symbol, self.balance)  
                signals = self.strategy_module.get_signals()  

                # 执行交易信号  
                for signal in signals:  
                    self._execute_trade(signal, current_price, current_time)  

    def _execute_trade(self, signal: Dict, current_price: float, current_time: pd.Timestamp):  
        """执行交易信号"""  
        action = signal['signal_type']  
        size = signal['size']  

        if action == 'buy':  
            # 买入逻辑  
            cost = size * current_price  
            if self.balance * self.leverage >= cost:  
                self.position += size  
                self.balance -= cost  
                self.history.append({  
                    'timestamp': current_time,  
                    'action': 'buy',  
                    'price': current_price,  
                    'size': size,  
                    'balance': self.balance,  
                    'position': self.position  
                })  
        elif action == 'sell':  
            # 卖出逻辑  
            if self.position >= size:  
                revenue = size * current_price  
                self.position -= size  
                self.balance += revenue  
                self.history.append({  
                    'timestamp': current_time,  
                    'action': 'sell',  
                    'price': current_price,  
                    'size': size,  
                    'balance': self.balance,  
                    'position': self.position  
                })  

    def run_backtest(self):  
        """运行回测"""  
        
        max_timeframe_minutes = 15  # 最大时间框架为 15 分钟  
        window_size = 60  # 滑动窗口需要 20 根 K 线  
        # current_time = self.start_date  #for the reason of needing history data to caculate factors, window should move afterward from the start point.
        current_time = self.start_date + timedelta(minutes=max_timeframe_minutes * window_size)  # 向前推移时间  

        while current_time <= self.end_date:  
            self._simulate_step(current_time)  
            current_time += timedelta(minutes=1)  # 时间步进 1 分钟  

        # 输出回测结果  
        self._print_results()  

    def _print_results(self):  
        """打印回测结果"""  
        print("回测完成！")  
        print(f"初始余额: 100 USDT")  
        print(f"最终余额: {self.balance:.2f} USDT")  
        print(f"最终持仓: {self.position:.4f} {self.symbol.split('/')[0]}")  
        print("交易历史:")  
        for trade in self.history:  
            print(trade)


if __name__ == "__main__":
    # 初始化回测模块  
    backtest = BacktestModule(  
        symbol='PEPE-USDT-SWAP',  
        timeframes=['1m', '5m'],  
        start_date='2025-01-03',  
        end_date='2025-01-05',  
        initial_balance=100.0  
    )  

    # 运行回测  
    backtest.run_backtest()