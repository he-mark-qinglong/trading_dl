import ccxt  
import pandas as pd  
from datetime import datetime, timedelta  
import time  
from typing import Dict, Optional  
from factors import FactorManager  
import threading  
from queue import Queue  
from concurrent.futures import ThreadPoolExecutor 

from .history_data import DataManager, HistoricalDataLoader  

from .strategy_module import DynamicMultiTimeframeStrategy
from .strategy_manager import StrategyManager 

from .market_trend_detect import TrendDetector

from .anti_MartingaleTradingCore import AntiMartingaleTradingCore
from .normal_MartingaleTradingCore import MartingaleTradingCore

from config import TradingEnvConfig, ExchangeConfig, trend_detect_config
from envs.exchange_manager import ExchangeManager

class RealtimeDataManager:  
    def __init__(self, symbol: str, timeframes: list = None, history_days: int = 3000):  
        """  
        实时数据管理器，结合历史数据初始化和实时更新  
        :param symbol: 交易对  
        :param timeframes: 时间框架列表（如 ['1m',  '5m', '15m']）  
        :param history_days: 初始化时加载的历史天数  
        """  
        self.symbol = symbol  
        self.history_days = history_days
        self.timeframes = timeframes or ["1m", "3m", "5m", "15m", "30m", "1h", "4h"]  # 支持的时间周期 
        self.timeframe_counters = {timeframe: 0 for timeframe in self.timeframes}  # 每个时间周期的计数器  
        self.timeframe_intervals = {  # 每个时间周期对应的触发间隔  
            "1m": 1,       # 每次进入都计算  
            "3m": 3,       # 每3次进入计算一次  
            "5m": 5,       # 每5次进入计算一次  
            "15m": 15,     # 每15次进入计算一次  
            "30m": 30,     # 每30次进入计算一次  
            "1h": 60,      # 每60次进入计算一次  
            "4h": 60, #240,     # 每240次进入计算一次  
            "1d": 60, #1440     # 每1440次进入计算一次  
        }  

        self.exchange = ccxt.binance({  
            'enableRateLimit': True,  
            'options': {'defaultType': 'swap'},  
            'proxies': {  
                'http': 'http://127.0.0.1:7890',  
                'https': 'http://127.0.0.1:7890'  
            }  
        })
        # self.exchange = ccxt.okx({  
        #     'enableRateLimit': True,  
        #     'options': {'defaultType': 'swap'},  
        #     'proxies': {  
        #         'http': 'http://127.0.0.1:7890',  
        #         'https': 'http://127.0.0.1:7890'  
        #     }  
        # })  

        # 初始化数据管理器和数据存储  
        self.data_manager = DataManager()  
        self.data: Dict[str, pd.DataFrame] = {}  

        # 初始化因子管理器  
        self.factors: Dict[str, Dict] = {tf: {} for tf in self.timeframes}  
        self.factor_manager = FactorManager()  
        self.data_cache = {tf: None for tf in timeframes}  # 缓存数据和因子  
        self.data_cache_lock = threading.Lock() 

        self.strategy_module = DynamicMultiTimeframeStrategy()
        self.multi_strategy_manager = StrategyManager()

        self.trend_detector = TrendDetector(trend_detect_config)

        # 用于实时更新的队列和事件  
        self.update_queue = Queue(maxsize=100)  # 限制队列大小，防止内存泄漏  
        self.stop_event = threading.Event()  
        self.update_thread = None  

        self._initialize_data()  
    def _initialize_data(self):  
        """  
        初始化数据，优先使用本地数据，不足部分从交易所获取  
        :param days: 初始化加载的历史天数  
        """  
        loader = HistoricalDataLoader('binance')
        for timeframe in self.timeframes:  
            df = loader.fetch_historical_data(self.symbol, timeframe, self.data_manager, limit=self.history_days)
            self.data[timeframe] = df
            print(f'{timeframe} data len={len(df)}, required={self.history_days}')
            self._caculate_signals_and_trend(timeframe)

        print(f'time of last {self.timeframes[0]}:')
        print('\t', self.data[self.timeframes[0]].index[-1])
        # MARK: 需要异常处理,如果网路异常拿不到数据怎么办？
    
    def start_realtime_updates(self):  
        """  
        启动实时更新线程  
        """  
        if self.update_thread is None:  
            self.stop_event.clear()  
            self.update_thread = threading.Thread(target=self._update_loop)  
            self.update_thread.daemon = True  
            self.update_thread.start()  

    def stop_realtime_updates(self):  
        """  
        停止实时更新  
        """  
        if self.update_thread is not None:  
            self.stop_event.set()  
            self.update_thread.join()  
            self.update_thread = None  

    def _update_loop(self):  
        """  
        实时更新循环  
        """  
        print('------------------_update_loop excuted')
        while not self.stop_event.is_set():  
            try:  
                # self._update_timeframe('1m')  
                self._initialize_data()
                time.sleep(50 * 60)  
            except Exception as e:  
                print(f"更新数据时出错: {e}")  
    def transfer_symbol(self, exchange_id='okx'):
        if exchange_id == 'okx':
            return self.symbol
        else:
            formatted_symbol = self.symbol.replace('/', '').replace('-', '').replace('SWAP', '').replace('USDT', '') + 'USDT'
            return formatted_symbol.upper()  # 转换为大写 (可选)
    def get_processed_ohlcv(self, timeframe='1m', limit=200):  

        # 获取原始成交数据  
        raw_trades = self.exchange.fetch_trades(self.transfer_symbol('binance'),
                                                limit=limit)  
        
        # 解析taker标记（支持Binance/OKX通用解析）  
        parsed_trades = []  
        for trade in raw_trades:  
            is_taker = False  
            if self.exchange.id == 'binance':  
                pass
                # is_taker = not trade['info']['maker']  
            elif self.exchange.id == 'okx':  
                pass
                # import json
                # print(json.dumps(trade, indent=4))  
                # is_taker = (trade['info']['execType'] == 'T')  
            
            parsed_trades.append({  
                'timestamp': trade['timestamp'],  
                'price': float(trade['price']),  
                'amount': float(trade['amount']),  
                # 'is_taker': is_taker  #由于trades里面有takerOrMaker但是又是none，fee也是空缺的，导致看不出是taker还是maker。
            }) 
        
        # 创建DataFrame并处理时间  
        df = pd.DataFrame(parsed_trades)  
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')  
        df.set_index('datetime', inplace=True)  
        
        # 定义时间频率映射，推荐使用 'min' 替代 'T'  
        timeframe_mapping = {  
            '1m': 'min',  
            '5m': '5min',  
            '15m': '15min',  
            '30m': '30min',  
            '1h': '1H',  
            '4h': '4H',  
            '1d': '1D',  
            '1w': '1W'  
        } 
        
        # 重采样计算基础OHLC  
        resampled = df['price'].resample(timeframe_mapping[timeframe]).agg(  
            open=('first'),  
            high=('max'),  
            low=('min'),  
            close=('last')  
        )  

        total_vol = df['amount'].resample(timeframe_mapping[timeframe]).sum()

        # 合并最终数据  
        new_data = pd.DataFrame({  
            'timestamp': resampled.index.astype('int64') // 10**6,  # 转回毫秒时间戳  
            'open': resampled['open'],  
            'high': resampled['high'],  
            'low': resampled['low'],  
            'close': resampled['close'],  
            'volume': total_vol, 
        }).dropna().reset_index(drop=True)  
        
        # 处理可能的空值  
        if not new_data.empty:  
            new_data['volume'] = new_data['volume'].fillna(0)  
            new_data = new_data.ffill()  # 前向填充OHLC  
            
        return new_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  
    
    def _update_timeframe(self, timeframe: str):  
        """  
        更新 1m 数据并动态生成高时间框架数据  
        """  
        try:  
            limit = 100 
            new_data = self.get_processed_ohlcv(timeframe=timeframe, limit=limit)
            
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')  
            new_data.set_index('timestamp', inplace=True)  
            
            if not self.data['1m'].empty:  #是更新最后一个，没有这个时间才追加。
                # 对历史数据和新数据的时间戳取整到分钟  
                self.data['1m'].index = self.data['1m'].index.floor('min')  # 'T' 表示分钟  
                new_data.index = new_data.index.floor('min')  
                
                last_timestamp = self.data['1m'].index[-1]  
                new_data = new_data[new_data.index > last_timestamp]  
                combined_data = pd.concat([self.data['1m'], new_data])  
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]  
                combined_data.sort_index(inplace=True)  
                self.data['1m'] = combined_data  
            else:  
                self.data['1m'] = new_data  

            for higher_timeframe in self.timeframes:  
                if higher_timeframe == '1m':  
                    continue  
                # 生成新的高时间框架数据  
                new_higher_timeframe_data = self._generate_higher_timeframe_data(  
                    self.data['1m'], higher_timeframe  
                )  

                # 合并已有数据和新生成的数据  
                if not self.data[higher_timeframe].empty:  
                    combined_data = pd.concat([self.data[higher_timeframe], new_higher_timeframe_data])  
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]  
                    combined_data.sort_index(inplace=True)  
                    self.data[higher_timeframe] = combined_data  
                else:  
                    # 如果没有已有数据，直接赋值  
                    self.data[higher_timeframe] = new_higher_timeframe_data    
            
            #self.calculate_all_signals_and_trends()
            
        except Exception as e:  
            print(f"_update_timeframe 更新{timeframe}数据时出错: {e}")  
    
    def calculate_all_signals_and_trends(self):  
        """  
        使用多线程计算所有时间周期的信号和趋势，基于计数器控制计算频率  
        """ 
        with ThreadPoolExecutor() as executor:  
            try:
                futures = []
                if all(self.timeframe_counters[timeframe] == 0 for timeframe in self.timeframes):
                    for timeframe in self.timeframes:  
                        futures.append(executor.submit(self._caculate_signals_and_trend, timeframe)) 
                    for future in futures:    
                        future.result()  # 获取线程执行结果，捕获异常  
                else:
                    for timeframe in self.timeframes:  
                        # 增加计数器  
                        self.timeframe_counters[timeframe] += 1  

                        # 检查计数器是否达到触发间隔  
                        if self.timeframe_counters[timeframe] >= self.timeframe_intervals[timeframe]:  
                            # 重置计数器并触发计算  
                            self.timeframe_counters[timeframe] = 0  
                            futures.append(executor.submit(self._caculate_signals_and_trend, timeframe))  

                    # 等待所有线程完成  
                    for future in futures:  
                        future.result()  # 获取线程执行结果，捕获异常  
            except Exception as e:  
                print(f"Error in thread execution: {e}")  
    
    def _caculate_signals_and_trend(self, timeframe):
        """
        多线程安全没有考虑在内的，因为多个df的更新可能会穿插到一起，哪怕都是1m的factors更新，也可能会打断data_cache内的数据.
        导致数据混乱，这是有可能的。
        """
        try:
            df = self.get_latest_data(timeframe)  
            # df = df.iloc[0:2000].copy()
            if df is not None and not df.empty:  
                from utils.signal_backtester import  Backtester, simple_twvwap_signal
                from utils.signal import analyze_price_distribution, calculate_dynamic_period
                # df = df.iloc[-5000:-1].copy()
                df = df.iloc[300+655000 : 655000 + 1000000].copy() #持续上涨到暴跌的过程.
                # df = df.iloc[300+655000 +32000 : 655000 + 32000 + 200000].copy() #测试1月8号的横盘整理后的暴跌。
                if True or timeframe != '1m':
                    result_dict = Backtester().run_backtest(df, timeframe, signal_func=simple_twvwap_signal, 
                                            factor_manager=FactorManager(),
                                            calculate_dynamic_period_func=calculate_dynamic_period, 
                                            analyze_price_distribution_func=analyze_price_distribution)
                    print(f'backtest timeframe {timeframe}:\n result_dict: {result_dict}')
                signal, trend, market_type = None, None, None

                # 计算因子  
                manager = FactorManager()
                factors, df = manager.calculate_factors(df) 
                factors = manager.update_twvwap(df, factors)

                # 缓存数据和因子  
                self.data_cache_lock.acquire()
                self.data_cache[timeframe] = {"df":df, "factors":factors, 
                                              "signal":signal, "trend":trend, 
                                              "market_type":market_type}  
                self.data_cache_lock.release()
        except Exception as e: 
            print(f"_caculate_signals_and_trend {timeframe}数据时出错: {e}")  
    def _generate_higher_timeframe_data(self, base_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:  
        """  
        动态生成高时间框架数据  
        """  
        #举例：对 "15m"字符串 进行切片（排除末尾的 "m"），剩下 "15"，再转成整数得到 15。
        timeframe_minutes = int(timeframe[:-1])  

        resampled_data = base_data.resample(f"{timeframe_minutes}min").agg({  
            'open': 'first',  
            'high': 'max',  
            'low': 'min',  
            'close': 'last',  
            'volume': 'sum'  
        }).dropna()  
        return resampled_data  

    def get_latest_data(self, timeframe: str) -> Optional[pd.DataFrame]:  
        """  
        获取指定时间框架的最新数据  
        """  
        return self.data.get(timeframe, pd.DataFrame())
    
    def get_latest_data_with_factors(self, timeframe):  
        """  
        获取最新数据，并计算因子。  
        :param timeframe: 时间周期  
        :return: 数据框和因子字典  
        """  
        self.data_cache_lock.acquire()
        d = self.data_cache[timeframe].copy()
        self.data_cache_lock.release()

        return d
    
    