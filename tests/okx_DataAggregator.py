import datetime  
import time  
import numpy as np  
import pandas as pd  
import json
from tests.mock_trade import LeveragedMockTrader
from typing import List, Dict, Tuple 
from collections import deque 
from tests.order_filter import OrderFilter
from tests.dual_positionTrader import DualPositionTrader
from sklearn.linear_model import LinearRegression 

class OkxDataAggregator:  
    max_window_keep_len = 50 #episode_data中的数组最多保留这个长度，每次折半存文件，剩下的折半保留。
    _seconds_per_episode: float = 5
    data_window = 30
    """  
    数据聚合器：  
    xxx 1. 维护秒级数据(second_data)，包括每秒买卖成交量、价格  
    2. 维护分钟级数据(episode_data)，包括OHLCV、VWAP、深度变化率  
    3. 维护 Maker 的订单簿累计分布  
    4. 维护 Taker 的累计成交量分布  
    5. 提供更新和清理方法，供 WebSocket 客户端在接收数据后调用  
    6. 通过线程定时调度来生成每分钟 K 线  
    """  

    def __init__(self, trader:DualPositionTrader):  
        self.trader = trader

        self.current_second = {  
            "timestamp": None,  
            "price": None,  
            "buy": 0.0,  
            "sell": 0.0  
        }  

        # 分钟级数据  
        self.all_episode_data = {  
            "timestamps": [],  
            "open": [],  
            "high": [],  
            "low": [],  
            "close": [],  
            "volume": [],  
            "vwap": [],  
            "vwap_smooth":[],

            "market_price": [],

            "gamma_factor": [],
            "block_impact": [],
            "vwap_slope": [],
            "depth_rate": [],
            "ask_volatility":[],
            "bid_volatility":[],
        }  
        self.current_episode_data = {  
            "pv_t":[], #{"price":p, "volume",v, "timestamp":t}
            
            "bid_depth_start": 0.0,  
            "ask_depth_start": 0.0,  
            "bid_depth_end": 0.0,  
            "ask_depth_end": 0.0,  
        }  

        self.maker_depth_stats = {  
            'bid_avg': [],  
            'ask_avg': [],  
            'bid_std': [],  
            'ask_std': []  
        }  

        self.keep_history_seconds = self._seconds_per_episode * 15  

        from tests.obpi import  OrderBookPressureIndex
        self.obpi_calculator = OrderBookPressureIndex()
        # 盘口(books)最新状态  
        self.order_book_data = {"bids": {}, "asks": {}}  

        # Maker 数据：存储过去一段时间的订单簿数据  
        self.order_book_history = []  # 每个元素是 {"timestamp": ..., "bids": ..., "asks": ...}  
        self.order_book_window = self.keep_history_seconds  # 只保留最近 x 秒的数据  

        # Taker 数据：存储过去一段时间的成交量数据  
        self.taker_trade_history = []  # 每个元素是 {"timestamp": ..., "buy_volume": ..., "sell_volume": ...}  
        self.taker_trade_window = self.keep_history_seconds  # 只保留最近 x 秒的数据  

        # ======= 新增：时空加权相关设置 =======  
        self.half_life = self._seconds_per_episode * 20  # 半衰期（秒），可根据实际需求调整  
        self.decay_dict_bids = {}  # 用于存储时空加权后的买单  
        self.decay_dict_asks = {}  # 用于存储时空加权后的卖单  

        # 冲击系数参数。
        self._gamma_threshold = 0.3  # 动态调整参数  
        # self.impact_count_window = deque(maxlen=60)  # 1小时冲击计数 
        self.impact_count_window = deque(maxlen=self.keep_history_seconds)  # 历史窗口冲击计数 

        self.mock_trader = LeveragedMockTrader(  
            initial_equity=100,  
            fee_rate=0.0004,  #纯maker
            take_profit_pct=0.01,  # 5%止盈  
            stop_loss_pct=0.02,    # 50%止损  
            total_stop_loss_pct=0.30, 
            leverage=20,  
            maintenance_margin=0.01  # 1%维持保证金  
        )  

    def update_maker_order_book(self, bids, asks):  
        """  
        更新订单簿数据，并保留首次出现的时间戳。  
        :param bids: list，包含 (price, volume) 元组  
        :param asks: list，包含 (price, volume) 元组  
        """  
        now = datetime.datetime.now()  

        def update_with_timestamps(current_orders, new_orders):  
            """  
            更新挂单的时间戳和成交量。  
            :param current_orders: dict，以价格 (float) 为键，值是包含 volume 和 time 的字典  
            :param new_orders: list，包含 (price, volume) 的元组  
            :return: dict，更新后的挂单数据  
            """  
            updated_orders = {}  

            if not isinstance(new_orders, list) or not all(  
                isinstance(order, (list, tuple)) and len(order) == 2 for order in new_orders  
            ):  
                print(f"Invalid new_orders format: {new_orders}")  
                return updated_orders  

            for price, volume in new_orders:  
                if price in current_orders:  
                    # 如果挂单已存在，更新其 volume  
                    updated_orders[price] = current_orders[price]  
                    updated_orders[price]["volume"] = volume  
                else:  
                    # 如果挂单是新出现的，记录首次出现时间  
                    updated_orders[price] = {"volume": volume, "time": now}  

            return updated_orders  

        # 更新 bids 和 asks 的时间戳  
        self.order_book_data["bids"] = update_with_timestamps(  
            self.order_book_data.get("bids", {}), bids  
        )  
        self.order_book_data["asks"] = update_with_timestamps(  
            self.order_book_data.get("asks", {}), asks  
        )  

        # 存储到历史记录  
        self.order_book_history.append({  
            "timestamp": now,  
            "bids": [{"price": p, **d} for p, d in self.order_book_data["bids"].items()],  
            "asks": [{"price": p, **d} for p, d in self.order_book_data["asks"].items()]  
        })  
        # 更新 OBPI 数据  
        self.obpi_calculator.update_obpi(bids, asks)  

        # 移除超过时间窗口的数据  
        self.order_book_history = [  
            entry for entry in self.order_book_history  
            if (now - entry["timestamp"]).total_seconds() <= self.order_book_window  
        ]  

    def _calculate_cumulative_order_book(self) -> Tuple[List[Dict], List[Dict]]:  
        """  
        计算过去 self.order_book_window 秒内的累计挂单分布  
        """  
        cumulative_bids = {}  
        cumulative_asks = {}  

        for entry in self.order_book_history:  
            for bid in entry["bids"]:  
                price = bid["price"]  
                volume = bid["volume"]  
                t = bid.get("time")  
                if price not in cumulative_bids:  
                    cumulative_bids[price] = {"volume": 0, "time": t}  
                cumulative_bids[price]["volume"] += volume  
                # 记录最早时间戳  
                if cumulative_bids[price]["time"]:  
                    earliest_time = min(cumulative_bids[price]["time"], t)  
                else:  
                    earliest_time = t  
                cumulative_bids[price]["time"] = earliest_time  

            for ask in entry["asks"]:  
                price = ask["price"]  
                volume = ask["volume"]  
                t = ask.get("time")  
                if price not in cumulative_asks:  
                    cumulative_asks[price] = {"volume": 0, "time": t}  
                cumulative_asks[price]["volume"] += volume  
                # 记录最早时间戳  
                if cumulative_asks[price]["time"]:  
                    earliest_time = min(cumulative_asks[price]["time"], t)  
                else:  
                    earliest_time = t  
                cumulative_asks[price]["time"] = earliest_time  

        # 转换为列表并排序（买单从高到低、卖单从低到高）  
        cumulative_bids_list = sorted(  
            [  
                {"price": p, "volume": d["volume"], "time": d["time"]}  
                for p, d in cumulative_bids.items()  
            ],  
            key=lambda x: x["price"],  
            reverse=True  
        )  
        cumulative_asks_list = sorted(  
            [  
                {"price": p, "volume": d["volume"], "time": d["time"]}  
                for p, d in cumulative_asks.items()  
            ],  
            key=lambda x: x["price"]  
        )  

        return cumulative_bids_list, cumulative_asks_list  

    def get_market_median_price(self) -> float:  
        """  
        获取市场中位数价格  
        :return: 中位数价格  
        """  
        all_prices = self.get_all_order_prices()  
        sorted_prices = sorted(all_prices)  
        n = len(sorted_prices)  

        if n == 0:  
            return 0  
        elif n % 2 == 1:  
            return sorted_prices[n // 2]  
        else:  
            return (sorted_prices[n // 2 - 1] + sorted_prices[n // 2]) / 2  

    def get_all_order_prices(self):  
        """  
        获取当前订单簿中所有挂单的价格（买单、卖单）  
        """  
        bids = self.order_book_data.get("bids", {})  
        asks = self.order_book_data.get("asks", {})  

        if isinstance(bids, dict):  
            bid_prices = list(bids.keys())  
        else:  
            bid_prices = [order["price"] for order in bids]  

        if isinstance(asks, dict):  
            ask_prices = list(asks.keys())  
        else:  
            ask_prices = [order["price"] for order in asks]  

        return bid_prices + ask_prices  

    def get_cumulative_order_book(self, apply_time_weighting: bool = False) -> Tuple[List[Dict], List[Dict]]:  
        """  
        计算过去 self.order_book_window 秒内的累计挂单分布，并执行过滤+（可选）时空加权。  
        :param apply_time_weighting: 是否在过滤之后进行时空加权  
        :return: 最终处理后的买单、卖单列表  
        """  
        # (1) 先计算累计订单簿  
        cumulative_bids, cumulative_asks = self._calculate_cumulative_order_book()  

        # (2) 先过滤  
        # 2.1 过滤短时间内撤销的挂单  
        cumulative_bids, cumulative_asks = OrderFilter.filter_short_lived_orders(  
            cumulative_bids,  
            cumulative_asks  
        )  
        # 2.2 过滤异常巨量挂单  
        cumulative_bids, cumulative_asks = OrderFilter.filter_abnormal_orders(  
            cumulative_bids, cumulative_asks  
        )  
        # 2.3 过滤价格异常的挂单  
        median_price = self.get_market_median_price()  
        cumulative_bids, cumulative_asks = OrderFilter.filter_outlier_prices(  
            cumulative_bids, cumulative_asks, median_price  
        )  

        # (3) 如果需要，再进行时空加权折叠  
        if apply_time_weighting:  
            weighted_bids = self._apply_time_spatial_folding(cumulative_bids, is_bid=True, ref_price=median_price)  
            weighted_asks = self._apply_time_spatial_folding(cumulative_asks, is_bid=False, ref_price=median_price)  
            return weighted_bids, weighted_asks  
        else:  
            return cumulative_bids, cumulative_asks  

    # =========== 新增：核心时空加权方法 ===========  
    def _apply_time_spatial_folding(self, orders: List[Dict], is_bid: bool, ref_price: float) -> List[Dict]:  
        """  
        对过滤后的累计订单列表进行时空加权折叠。  
        :param orders: [{"price":..., "volume":..., "time":...}, ...]  
        :param is_bid: True表示买单，False表示卖单  
        :param ref_price: 用于价格距离权重的参考价格（可用中位价或最新成交价等）  
        :return: 加权折叠后的订单列表  
        """  
        now_ts = time.time()  

        # 先把order转换为更便于处理的dict结构: price -> {"acc_volume":..., "last_time":...}  
        weighting_dict = {}  

        for od in orders:  
            price = od["price"]  
            volume = od["volume"]  
            time_diff_sec = now_ts - od["time"].timestamp()  # 距离当前的秒数  

            # 计算时间衰减因子  
            time_weight = self._time_decay_weight(time_diff_sec, self.half_life)  
            # 计算价格距离权重  
            spread = abs(ref_price - price) if ref_price else 1  # 避免除0  
            price_weight = self._price_distance_weight(price, ref_price, spread)  

            # 叠加最终权重  
            weighted_volume = volume * time_weight * price_weight  

            if price not in weighting_dict:  
                weighting_dict[price] = {"acc_volume": 0.0, "time": od["time"]}  
            weighting_dict[price]["acc_volume"] += weighted_volume  

        # 将结果转换回列表并排序  
        result_list = []  
        for p, data in weighting_dict.items():  
            result_list.append({  
                "price": p,  
                "volume": data["acc_volume"],  
                "time": data["time"]  
            })  

        # 买单降序、卖单升序  
        sorted_list = sorted(  
            result_list,  
            key=lambda x: x["price"],  
            reverse=is_bid  
        )  
        return sorted_list  

    def generate_weighted_heatmap_data(  
        self, orders: List[Dict], is_bid: bool, ref_price: float, bins: int = 50  
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  
        """  
        生成加权热力图数据。  
        :param orders: [{"price": ..., "volume": ..., "time": ...}, ...]  
        :param is_bid: True 表示买单，False 表示卖单  
        :param ref_price: 用于价格距离权重的参考价格（如中位价）  
        :param bins: 热力图的价格区间分箱数量  
        :return: (heatmap_matrix, price_bins, time_bins)  
            - heatmap_matrix: 二维数组，表示每个价格区间和时间区间的加权成交量  
            - price_bins: 价格区间的分箱  
            - time_bins: 时间区间的分箱  
        """  
        now_ts = time.time()  

        # 准备价格和时间数据  
        prices = [order["price"] for order in orders]  
        volumes = [order["volume"] for order in orders]  
        times = [order["time"].timestamp() for order in orders]  

        # 计算时间衰减权重  
        elapsed_seconds = [now_ts - t for t in times]  
        time_weights = [self._time_decay_weight(t, self.half_life) for t in elapsed_seconds]  

        # 计算价格距离权重  
        spreads = [abs(ref_price - p) for p in prices]  
        price_weights = [self._price_distance_weight(p, ref_price, max(spreads)) for p in prices]  

        # 计算最终加权成交量  
        weighted_volumes = [v * tw * pw for v, tw, pw in zip(volumes, time_weights, price_weights)]  

        if len(prices) < 2:
            return None, None, None
        # 构建二维热力图矩阵  
        price_bins = np.linspace(min(prices), max(prices), bins)  
        time_bins = np.linspace(min(times), max(times), bins)  
        heatmap_matrix, _, _ = np.histogram2d(prices, times, bins=[price_bins, time_bins], weights=weighted_volumes)  

        return heatmap_matrix, price_bins, time_bins 

    @staticmethod  
    def _time_decay_weight(elapsed_seconds: float, half_life: float) -> float:  
        """  
        根据时间衰减函数计算权重: w = e^(-lambda * t)，其中 lambda = ln(2)/half_life  
        :param elapsed_seconds: 距离当前的秒数  
        :param half_life: 半衰期  
        :return: 时间衰减权重  
        """  
        if half_life <= 0:  
            return 1.0  # 不衰减  
        decay_rate = np.log(2) / half_life  
        return np.exp(-decay_rate * elapsed_seconds)  

    @staticmethod  
    def _price_distance_weight(price: float, ref_price: float, spread: float) -> float:  
        """  
        价格距离权重示例：距离越接近ref_price，权重越高  
        :param price: 挂单价格  
        :param ref_price: 参考价格（如中间价或中位价）  
        :param spread: abs(ref_price - price)的基准，可按需优化  
        :return: 价格步进权重(示例：1 / (1 + distance_ratio))  
        """  
        if spread == 0:  
            return 1.0  # 避免除0  
        distance = abs(price - ref_price)  
        distance_ratio = distance / spread  
        return 1.0 / (1.0 + distance_ratio) 

    def update_tickers(self, ts: str, price: float, size: float):  
        """  
        处理ticker级别的逐笔成交(Trade)数据
        """  
        
        # 转换为秒级时间戳（浮点数）  
        timestamp_seconds = int(ts) / 1000  

        # 转换为 datetime 对象  
        dt_object = datetime.datetime.fromtimestamp(timestamp_seconds)  
        # 更新 current_episode_data 中的 pv_t 数据（包含撮合价格、成交量和时间戳）  
        self.current_episode_data["pv_t"].append({  
            "price": price,  
            "volume": size,  
            "timestamp": dt_object  
        })
        
        # 根据更新的价格和成交量触发后续回调逻辑  
        self.mock_trader.on_price_tick(price, size) 

    def get_episode_data(self):  
        """  
        供 Dash 回调读取分钟级数据  
        """  
        return self.all_episode_data  

    def get_episode_data_as_dataframe(self):  
        """  
        将分钟级数据转换为 DataFrame  
        """  
        # 提取 OBPI 数据  
        obpi_data = self.obpi_calculator.get_obpi_data()  

        df = pd.DataFrame({  
            "timestamps": self.all_episode_data["timestamps"],  
            "open": self.all_episode_data["open"],  
            "high": self.all_episode_data["high"],  
            "low": self.all_episode_data["low"],  
            "close": self.all_episode_data["close"],  
            "volume": self.all_episode_data["volume"],  
            "vwap": self.all_episode_data["vwap"],
            "vwap_smooth":self.all_episode_data["vwap_smooth"],
            "vwap_slope":self.all_episode_data["vwap_slope"],
            "market_price":self.all_episode_data['market_price'], 
            "ask_volatility":self.all_episode_data["ask_volatility"],
            "bid_volatility":self.all_episode_data["bid_volatility"]
        })  
        df.set_index("timestamps", inplace=True)  

        # 创建 OBPI DataFrame，并设置时间戳为索引  
        obpi_df = pd.DataFrame({  
            "timestamps": [entry["timestamp"] for entry in obpi_data],  
            "obpi_ema_fast": [entry["obpi_ema_fast"] for entry in obpi_data],  
            "obpi_ema_slow": [entry["obpi_ema_slow"] for entry in obpi_data],
            "obpi_diff": [entry["obpi_diff"] for entry in obpi_data],

            "obpi": [entry["obpi"] for entry in obpi_data],
            "signal_line": [entry["signal_line"] for entry in obpi_data],

        })  

        obpi_df.set_index("timestamps", inplace=True)  
        
        # # 合并两个 DataFrame，采用外连接  
        # merged_df = obpi_df.merge(df, how='outer', left_index=True, right_index=True)
        # # 使用前向填充填充缺失值  
        # merged_df.ffill(inplace=True)  

        # 检查 DataFrame 长度并填补  
        if len(df) > len(obpi_df):  
            # 在 OBPI 中添加缺失的时间戳  
            obpi_df = obpi_df.reindex(df.index, method='ffill')  
        else:  
            # 在主 DataFrame 中添加缺失的时间戳  
            df = df.reindex(obpi_df.index, method='ffill')  

        # 合并两个 DataFrame，采用外连接  
        merged_df = df.join(obpi_df, how='outer')  

        return merged_df  

    def normalize_gamma_to_strength(self, gamma_factor: float) -> float:  
        """将gamma因子标准化为0-100的信号强度"""  
        # 方法1：基于历史数据的动态标准化  
        if not self.all_episode_data["gamma_factor"]:  
            max_gamma = max(abs(self._gamma_threshold) * 2, abs(gamma_factor))  
        else:  
            max_gamma = max(  
                max(abs(g) for g in self.all_episode_data["gamma_factor"]),  
                abs(gamma_factor)  
            )  
        
        # 将gamma标准化到0-100范围  
        normalized_strength = (abs(gamma_factor) / max_gamma) * 100  
        return min(normalized_strength, 100)  # 确保不超过100

    def calculate_vwap_slope(self, window_size_small=14):  
        """计算 VWAP 的斜率"""  
        using_size = window_size_small
        v_len = len(self.all_episode_data["vwap"])
        if v_len < window_size_small:  
            return 0  # 数据不足，返回0  

        # 获取最近 window_size 个 VWAP 值  
        recent_vwap = self.all_episode_data["vwap"][-using_size:]  
        timestamps = np.arange(len(recent_vwap)).reshape(-1, 1)  # 创建时间序列  

        # 使用线性回归计算斜率  
        model = LinearRegression()  
        model.fit(timestamps, recent_vwap)  
        return model.coef_[0]  # 返回斜率  
    def calc_tvwap(self, total_volume, prices, volumes):
        """
        use tvwap to replace normal vwap:
        vwap = 0.0  
        if total_volume > 0:  
            vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume   
        """
        import math  
        # 计算 TVWAP  
        tvwap = 0.0  
        if total_volume > 0:  
            weighted_price_sum = 0.0  
            weighted_volume_sum = 0.0  
            # 遍历每个价格和成交量  
            for i, (p, v) in enumerate(zip(prices, volumes)):  
                # 计算权重：这里假设最新的数据权重更高，权重根据距离窗口末尾的距离衰减:   
                # 衰减因子 = exp(-tau * (self.data_window - i))  
                weight = math.exp(-0.03 * (self.data_window - i))  
                weighted_price_sum += p * v * weight  
                weighted_volume_sum += v * weight  
            if weighted_volume_sum > 0:  
                tvwap = weighted_price_sum / weighted_volume_sum 
        return tvwap
    
    def episode_task(self):  
        """  
        每分钟执行一次，计算并保存 K 线和深度指标，然后重置 current_episode_data  
        """  
        while True:  
            time.sleep(self._seconds_per_episode)  # 每分钟执行一次  
            # 等待数据积累  
            if len(self.maker_depth_stats['bid_std']) < self._seconds_per_episode or len(self.maker_depth_stats['ask_std']) < self._seconds_per_episode:  
                print("等待数据积累中...")  
                time.sleep(1)
                continue  
            if self.current_episode_data["pv_t"]:
                # 先将值从字典数组中提取到独立的列表中
                prices = []
                volumes = []
                timestamps = [] # 用来后续维护索引

                for item in self.current_episode_data["pv_t"]:  
                    prices.append(item["price"])  
                    volumes.append(item["volume"])  
                    timestamps.append(item["timestamp"])  

                # 根据提取出来的数据计算 OHLCV  
                open_price = prices[0]  
                high_price = max(prices)  
                low_price = min(prices)  
                close_price = prices[-1]  
                total_volume = sum(volumes)  

                # 计算 VWAP  
                vwap = self.calc_tvwap(total_volume, prices, volumes)
                # 深度变化率  
                bid_depth_change = (  
                    self.current_episode_data["bid_depth_end"]  
                    - self.current_episode_data["bid_depth_start"]  
                ) / self._seconds_per_episode  
                ask_depth_change = (  
                    self.current_episode_data["ask_depth_end"]  
                    - self.current_episode_data["ask_depth_start"]  
                ) / self._seconds_per_episode  

                # 保存到 episode_data  
                self.all_episode_data["timestamps"].append(datetime.datetime.now())  
                self.all_episode_data["open"].append(open_price)  
                self.all_episode_data["high"].append(high_price)  
                self.all_episode_data["low"].append(low_price)  
                self.all_episode_data["close"].append(close_price)  
                self.all_episode_data["volume"].append(total_volume)  
                self.all_episode_data["vwap"].append(vwap) 
                
                smoth_vwap = pd.Series(self.all_episode_data["vwap"]).ewm(span=self.data_window*5, adjust=False).mean()
                self.all_episode_data["vwap_smooth"] = smoth_vwap.values

                depth_bid_minus_ask_delta = bid_depth_change - ask_depth_change 
                self.all_episode_data["depth_rate"].append((bid_depth_change, ask_depth_change, 
                                                       depth_bid_minus_ask_delta)) 
                
                market_price = self.trader.get_market_price()
                self.all_episode_data["market_price"].append(market_price)

                data_window = self.data_window
                vwap_slope = self.calculate_vwap_slope(window_size_small=data_window)
                self.all_episode_data['vwap_slope'].append(vwap_slope)
                # 计算深度波动率    
                if len(self.maker_depth_stats['bid_std']) >= self._seconds_per_episode:  
                    bid_volatility = np.nanmean(self.maker_depth_stats['bid_std'][-self._seconds_per_episode:])  # 使用 np.nanmean 忽略 NaN  
                else:  
                    bid_volatility = 0  # 或者设置为默认值  

                if len(self.maker_depth_stats['ask_std']) >= self._seconds_per_episode:  
                    ask_volatility = np.nanmean(self.maker_depth_stats['ask_std'][-self._seconds_per_episode:])  # 使用 np.nanmean 忽略 NaN  
                else:  
                    ask_volatility = 0  # 或者设置为默认值  

                # 检查计算结果是否为 NaN  
                bid_volatility = bid_volatility if not np.isnan(bid_volatility) else 0  
                ask_volatility = ask_volatility if not np.isnan(ask_volatility) else 0  

                self.all_episode_data["bid_volatility"].append(bid_volatility)
                self.all_episode_data["ask_volatility"].append(ask_volatility)
                  
                
                
                # 新增大单冲击检测模块  
                def detect_block_trade(volumes):  
                    """基于成交量突变的冲击检测"""  
                    volume_series = np.array(volumes)  
                    if len(volume_series) == 0:  
                        return np.array([])  
                    
                    median = np.median(volume_series)  
                    mad = np.median(np.abs(volume_series - median))  
                    if mad == 0:  
                        return np.array([])  
                    
                    z_scores = np.abs((volume_series - median) / mad)  
                    return np.where(z_scores > 5)[0]  

                # 检测本分钟内的冲击时点  
                volumes = [item["volume"] for item in self.current_episode_data["pv_t"]]
                block_indices = detect_block_trade(volumes)  
                
                # 安全获取最后冲击时间  
                last_impact_time = None  
                if len(block_indices) > 0 and len(self.current_episode_data["pv_t"]) > 0:  
                    try:  
                        last_index = min(block_indices[-1], len(self.current_episode_data["pv_t"])-1)  
                        last_impact_time = self.current_episode_data["pv_t"][last_index]["timestamp"]  
                    except IndexError:  
                        last_impact_time = None

                        
                # 计算复合Gamma因子  
                def compute_gamma(bid_vol, ask_vol, last_impact):  
                    """改进的复合方向指标"""  
                    # 计算深度波动率导数  
                    delta_bid = bid_vol[-1] - bid_vol[-2] if len(bid_vol)>=2 else 0  
                    delta_ask = ask_vol[-1] - ask_vol[-2] if len(ask_vol)>=2 else 0  
                    
                    # 计算成交量分布曲率  
                    curvature = (self.current_episode_data["ask_depth_end"] - 2*self.current_episode_data["bid_depth_end"]   
                                + self.current_episode_data["ask_depth_start"]) / (self._seconds_per_episode**2)  
                    
                    # 时间衰减因子  
                    lambda_decay = 0.1 if last_impact else 0  
                    time_factor = np.exp(-lambda_decay * (datetime.datetime.now() - last_impact).seconds) if last_impact else 1  
                    
                    return (delta_bid - delta_ask) * np.sign(curvature) * time_factor  

                # 获取原始波动率数据  
                bid_vol_series = self.maker_depth_stats['bid_std'][-self._seconds_per_episode:]  
                ask_vol_series = self.maker_depth_stats['ask_std'][-self._seconds_per_episode:]  
                
                # 计算Gamma因子  
                gamma_factor = compute_gamma(bid_vol_series, ask_vol_series, last_impact_time)  

                # 更新存储结构（新增字段）  
                self.all_episode_data["gamma_factor"].append(gamma_factor)  
                self.all_episode_data["block_impact"].append(len(block_indices))  # 冲击次数  




                # 计算 ATR  
                def calculate_atr(prices, period=14):  
                    tr_values = []  
                    for i in range(1, len(prices)):  
                        high = prices[i]  
                        low = prices[i]  
                        previous_close = prices[i - 1]  
                        
                        # 计算 TR  
                        tr = max(high - low, abs(high - previous_close), abs(low - previous_close))  
                        tr_values.append(tr)  

                    # 计算 ATR  
                    if len(tr_values) < period:  
                        return None  # 不足以计算 ATR  
                    atr = sum(tr_values[-period:]) / period  # 简单移动平均  
                    return atr  

                # 计算当前的 ATR  
                tvwap_for_trade = self.all_episode_data['vwap_smooth'][-1]
                atr_value = calculate_atr(prices) 
                self.trader.update_tvwap(tvwap_for_trade, atr_value)
                
                # 标准化处理并发送信号  
                if abs(gamma_factor) > self._gamma_threshold:  
                    pass
                    strength = self.normalize_gamma_to_strength(gamma_factor)  
                    def are_signs_consistent(num1: float, num2: float) -> bool:  
                        return (num1 < 0 and num2 < 0)  or (num1 > 0 and num2 > 0)
                    
                    #策略解释：利用vwap和gamma同向，但是订单深度变化有反方向的缺口，主动去填补订单缺口（优势：这样就不是taker了，一个是手续费低了，一个是价格可降低一个盘口深度的价格）
                    # 计算统计信息  
                    min_depth_rate_delta_requirment = 100
                    if len(self.all_episode_data["depth_rate"]) > data_window*2 and abs(depth_bid_minus_ask_delta) > min_depth_rate_delta_requirment:
                        depth_rate = self.all_episode_data["depth_rate"][-data_window:]
                        depth_deltas = depth_rate[2]
                        depth_deltas_mean = np.mean(depth_deltas)  
                        depth_deltas_std = np.std(depth_deltas)  
                        # 计算乖离率  
                        depth_rate_bias_ratio = (depth_bid_minus_ask_delta - depth_deltas_mean) / depth_deltas_std 

                        vwap_slopes = self.all_episode_data['vwap_slope'][:]
                        mean = np.mean(vwap_slopes)  
                        std = np.std(vwap_slopes)  
                        # 计算乖离率  
                        slops_bias_ratio = (vwap_slope - mean) / std 

                        wantted_slop_bias_ratio = 1.3
                        wantted_depth_rate_ratio = 1.8  #参考bias_ratio_example.py文件额运行视觉效果。

                        avg_price = close_price #vwap #np.average(self.current_episode_data["pv_t"][-data_window:]["price"])

                        # 使用乖离率判断阈值 乖离率超过 wantted_ratio 倍标准差 ------
                        # MARK: 这个非常重要，后续会作为强化学习调参的action参数的。 

                        #为了不混淆概念，一切以vwap_slope也就是量价的同向趋势为核心。
                        #这是盘口和量价相反的条件，所以补充一下（比如上涨的时候它会视图回踩确认有没有支撑，我们就是给他补充支撑的，这样可以顺势）

                        #same的逻辑：既然要单边发展，那发展到新的位置结果maker的单不够用了，就主动补充过来帮助趋势延续。
                        # 内部有反转平仓逻辑，如果出现了另一个方向的趋势，当前的仓位就以盘内加平仓。  
                        # 比如原来多仓、结果现在出现盘口是空头了--也就是顺势方向的挂单量不足，就顺势追挂仓。
                        if abs(slops_bias_ratio) > 1.8:
                            def calculate_value_from_deviation(deviation, mean, std_dev):  
                                """  
                                根据给定的乖离率、均值和标准差计算对应的数值  
                                :param deviation: 乖离率  
                                :param mean: 均值  
                                :param std_dev: 标准差  
                                :return: 对应的数值  
                                """  
                                return deviation * std_dev + mean  
                            
                            deviation_depth_bid_minus_ask_delta = calculate_value_from_deviation(slops_bias_ratio, depth_deltas_mean, depth_deltas_std)
                            if not are_signs_consistent(depth_bid_minus_ask_delta, deviation_depth_bid_minus_ask_delta):
                                deviation_depth_bid_minus_ask_delta = -deviation_depth_bid_minus_ask_delta
                            
                            #self.trader.execute_trade(long_minus_short_delta=deviation_depth_bid_minus_ask_delta, strength=strength, depth_price=avg_price)
                        elif (\
                            abs(depth_rate_bias_ratio) > wantted_depth_rate_ratio \
                                and abs(slops_bias_ratio) > wantted_slop_bias_ratio  \
                                    and are_signs_consistent(depth_rate_bias_ratio, slops_bias_ratio) \
                                        and not are_signs_consistent(depth_bid_minus_ask_delta, vwap_slope) \
                                            ):  
                            """
                            组合条件，要么斜率常见但是盘面显示倾斜不足以支撑趋势，要么斜率比较乖离短期内大概率趋势会延续。
                            """
                            if depth_bid_minus_ask_delta > 0:
                                #买票更多，补充卖单(利用正态分布的均值回归)
                                avg_depth_price = self.maker_depth_stats['bid_avg'][-1]
                            else:
                                #卖盘更多，补充买单(利用正态分布的均值回归)
                                avg_depth_price = self.maker_depth_stats['ask_avg'][-1]
                            
                            #self.trader.execute_trade(long_minus_short_delta=depth_bid_minus_ask_delta, strength=strength, depth_price=avg_price)
                                    
                #动态条件修正gamma冲击灵敏度。
                if len(block_indices) > 0:  # 仅在 block_indices 非空时追加
                    self.impact_count_window.append(len(block_indices))  
                if len(self.impact_count_window) > 0 and np.mean(self.impact_count_window) > data_window:  # 高频冲击期  
                    self._gamma_threshold *= 0.9  # 降低灵敏度  
                else:  
                    self._gamma_threshold = 0.3  # 重置阈值 
                    
                # 重置 current_episode_data  
                atr_window = self.data_window
                self.current_episode_data = {  
                    #保留最低要求的长度以计算atr用.
                    "pv_t":self.current_episode_data["pv_t"][-min(atr_window, len(self.current_episode_data["pv_t"])):],
                    "bid_depth_start": 0.0,  
                    "ask_depth_start": 0.0,  
                    "bid_depth_end": 0.0,  
                    "ask_depth_end": 0.0  
                }
                self.save_and_trim_data(self.max_window_keep_len)
    
    def save_and_trim_data(self, max_window_keep_len):  
        """  
        将 self.current_episode_data 中长度达到 max_window_keep_len 的数组前半部分存入 JSONL 文件，并保留后半部分  
        """  
        for key, data in self.current_episode_data.items():  
            if isinstance(data, list) and len(data) >= max_window_keep_len:  
                # 计算前半部分和后半部分的分界点  
                split_index = len(data) // 2  

                # 保存前半部分到 JSONL 文件  
                with open(f"{key}_data.jsonl", "a") as f:  # 追加模式写入文件  
                    for item in data[:split_index]:  
                        f.write(f"{json.dumps(item)}\n")  # 将每个元素写入文件，逐行存储  

                # 保留后半部分  
                self.current_episode_data[key] = data[split_index:]  
                
    def update_current_episode_maker_price(self, bids, asks, five_minutes_supertrend):
        """
        根据 websocket 快照数据，
        利用买盘或卖盘前5档数据中价格区间最短的那一边累计撮合出价格和成交量，
        如果两边都有数据，则选择价格波动较小的一边进行计算；
        若只有一侧数据，则直接采用该侧数据。
        """
        try:
            self.trader.real_time_hedgeing(five_minutes_supertrend)
            
            current_time = datetime.datetime.now() # 获取当前时间
            candidate_orders = []  
            
            accumulate_height = 2
            if len(bids) > 0 and len(asks) > 0:  
                # 取各自最多前5档数据  
                bid_count = min(accumulate_height, len(bids))  
                bid_candidates = bids[:bid_count]  # 假设 bids 数据已按价格降序排列（买一在最前）  
                ask_count = min(accumulate_height, len(asks))  
                ask_candidates = asks[:ask_count]  # 假设 asks 数据已按价格升序排列（卖一在最前）  

                # 计算各自的价格区间  
                # 对于 bids，最高买价 - 第五档买价；对于 asks，第五档卖价 - 最低卖价  
                bid_range = bid_candidates[0][0] - bid_candidates[-1][0]  
                ask_range = ask_candidates[-1][0] - ask_candidates[0][0]  
                
                # 选择价格区间较小的一边进行撮合  
                if bid_range <= ask_range:  
                    candidate_orders = bid_candidates  
                else:  
                    candidate_orders = ask_candidates  
                for pv in candidate_orders:
                    if pv[1] == 0:
                        print(f"A candidate_orders{candidate_orders}！！！！！！！！！！！！！！") 
            elif len(bids) > 0:  
                candidate_orders = bids[:min(accumulate_height, len(bids))]
                for pv in candidate_orders:
                    if pv[1] == 0:
                        print(f"B candidate_orders{candidate_orders}！！！！！！！！！！！！！！") 
            elif len(asks) > 0:  
                candidate_orders = asks[:min(accumulate_height, len(asks))]  
                for pv in candidate_orders:
                    if pv[1] == 0:
                        print(f"C candidate_orders{candidate_orders}！！！！！！！！！！！！！！") 
            else:  
                print("无有效盘口数据。")  
                return  

            # 累计计算 VWAP（量价加权平均价格）和累计成交量  
            total_volume = 0.0  
            weighted_price_sum = 0.0  
            for order in candidate_orders:  
                price, volume = order  # 每个订单结构为 [price, volume]  
                weighted_price_sum += price * volume  
                total_volume += volume  
                
            matched_price = weighted_price_sum / total_volume if total_volume > 0 else 0.0  

            
        except Exception as e:  
            print(f"未知错误: {e}. 请检查update_current_episode_maker_price代码逻辑。")  

    def update_maker_depth(self, bids, asks):  
        """  
        更新当前分钟的深度数据  
        :param bids: 买盘深度数据  
        :param asks: 卖盘深度数据  
        """  
        if len(bids)>0:
            bid_depths = [b[1] for b in bids]  
            bid_depth = sum(bid_depths) 
            if self.current_episode_data["bid_depth_start"] == 0.0:  
                # 初始化起始深度  
                self.current_episode_data["bid_depth_start"] = bid_depth 
            else:  
                # 更新结束深度  
                self.current_episode_data["bid_depth_end"] = bid_depth  

            self.maker_depth_stats['bid_avg'].append(np.mean(bid_depths))
            self.maker_depth_stats['bid_std'].append(np.std(bid_depths))

        if len(asks) > 0:
            ask_depths = [a[1] for a in asks]  
            ask_depth = sum(ask_depths)  

            if self.current_episode_data["ask_depth_start"] == 0.0:  
                # 初始化起始深度  
                self.current_episode_data["ask_depth_start"] = ask_depth  
            else:  
                # 更新结束深度  
                self.current_episode_data["ask_depth_end"] = ask_depth  
            
            self.maker_depth_stats['ask_avg'].append(np.mean(ask_depths))   
            self.maker_depth_stats['ask_std'].append(np.std(ask_depths))  
 