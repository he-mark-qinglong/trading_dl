
from factors import FactorManager
import pandas as pd
from tests.okx_DataAggregator import OkxDataAggregator
import json
import websocket
from typing import Dict
import requests


class DataProcessor:  
    def __init__(self, aggregator: OkxDataAggregator):  
        self.aggregator = aggregator  
        self.maker_price_limit = 6  # 默认保留的价格数量   
        self.newest_timeframe_supertrend = {}  

    def process_taker_trade(self, trades):  
        """  
        处理Binance交易数据，并更新到aggregator  
        :param trade_data: Binance交易数据  
        """  
        for d in trades:
            # Binance的trade数据格式与OKX不同  
            ts = d["time"]  # 事件时间  
            price = float(d["price"])  # 价格  
            size = float(d["quoteQty"])  # 数量  
            self.aggregator.update_tickers(ts, price, size)  

    def process_maker_book(self, bids, asks):  
        """  
        处理订单簿数据，并更新到aggregator  
        :param bids: 买盘数据  
        :param asks: 卖盘数据  
        """  
        # 筛选逻辑  
        bids = self.filter_maker_prices(bids, "bids")  
        asks = self.filter_maker_prices(asks, "asks")  

        # 更新到aggregator  
        try:  
            self.aggregator.update_maker_order_book(bids, asks)  
        except Exception as e:  
            print(f'!!!!!!!!!!!!!!update_maker_order_book {e}')  

        try:  
            # # 从Flask服务获取数据  
            # response = requests.get("http://127.0.0.1:5001/fetch")  
            # if response.status_code == 200:  
            #     data = response.json()  
            #     if len(data.keys()) == 0:  
            #         five_minutes_trend = None  
            #     else:  
            #         self.newest_timeframe_supertrend = data  
            #         five_minutes_trend = self.newest_timeframe_supertrend['1m']  
            # else:  
            #     five_minutes_trend = None  
            #     print('request to local supertrend failed:', response.status_code)   
            five_minutes_trend = None
            self.aggregator.update_current_episode_maker_price(bids, asks, five_minutes_trend)  
        except Exception as e:  
            print(f'!!!!!!!!!!!!!!update_current_episode_maker_price {e}')  

        try:  
            self.aggregator.update_maker_depth(bids, asks)  
        except Exception as e:  
            print(f'!!!!!!!!!!!!!!update_maker_depth {e}')  

    def filter_maker_prices(self, orders, order_type):  
        """  
        筛选Maker价格，保留指定数量的价格  
        :param orders: 当前订单列表，例如bids或asks  
        :param order_type: "bids"或"asks"  
        :return: 筛选后的订单列表  
        """  
        # 如果订单数量不在5到10之间，直接返回  
        if not (5 < len(orders) < 10):  
            return orders  

        # 根据订单类型进行排序和筛选  
        if order_type == "bids":  
            # 买入订单：保留价格较高的  
            orders = sorted(orders, key=lambda x: float(x[0]), reverse=True)[:self.maker_price_limit]  
        elif order_type == "asks":  
            # 卖出订单：保留价格较低的  
            orders = sorted(orders, key=lambda x: float(x[0]))[:self.maker_price_limit]  

        return orders  

    

import time  
from binance import Client  
import json  

class BinanceExchangeConfig:  
    api_key: str = "xxx"  
    secret_key: str = "yyy"  

class BinanceDataGateway:  
    def __init__(self, aggregator: OkxDataAggregator, symbol="BTCUSDT", limit=20, interval=5):  
        """  
        :param data_processor: 提供 process_maker_book 接口的处理器实例  
        :param symbol: 交易对，例如 "BTCUSDT"  
        :param limit: 深度订单数据深度  
        :param interval: 轮询间隔（秒）  
        """  
        self.aggregator = aggregator   
        
        self.data_processor = DataProcessor(aggregator) 
        self.symbol = symbol  
        self.trade_limit = 10
        self.limit = limit  
        self.interval = interval  
        self.client = Client(BinanceExchangeConfig.api_key, BinanceExchangeConfig.secret_key)  

    def fetch_and_process_depth(self):  
        try:  
            depth = self.client.get_order_book(symbol=self.symbol, limit=self.limit)  
            # 深度数据中包含有 "lastUpdateId"，按此判断数据类型  
            if "lastUpdateId" in depth:  
                # 将订单簿数据整理为列表形式，转换为 float 类型，并处理数量为 0 的情况  
                bids = [[float(item[0]), float(item[1])] for item in depth.get("bids", [])]  
                asks = [[float(item[0]), float(item[1])] for item in depth.get("asks", [])]  

                # 处理数量为 0 的情况，替换为 0.0005  
                for pv in bids:  
                    if pv[1] == 0:  
                        pv[1] = 0.0005  
                for pv in asks:  
                    if pv[1] == 0:  
                        pv[1] = 0.0005  

                # 模拟 OKX websocket 数据格式（这里只关注处理深度数据）  
                ws_like_message = {  
                    "lastUpdateId": depth.get("lastUpdateId"),  
                    "bids": bids,  
                    "asks": asks  
                }  
                # 将数据转换为 JSON 格式（如果后续接口需要）  
                message = json.dumps(ws_like_message)  
                # 调用接口，传入处理好的数据  
                self.data_processor.process_maker_book(bids, asks)  
        except Exception as e:  
            print(f"获取或处理深度数据出错: {e}")  

    def fetch_and_process_trades(self):  
        try:  
            # 获取最近交易数据（成交数据）  
            trades = self.client.get_recent_trades(symbol=self.symbol, limit=self.trade_limit)
            self.data_processor.process_taker_trade(trades)  
        except Exception as e:  
            print(f"获取或处理成交数据出错: {e}") 

    def run_forever(self):  
        while True:  
            self.fetch_and_process_depth() 
            self.fetch_and_process_trades() 
            time.sleep(self.interval)  