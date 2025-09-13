
from factors import FactorManager
import pandas as pd
from tests.okx_DataAggregator import OkxDataAggregator
import json
import websocket
from typing import Dict

class DataProcessor:  
    def __init__(self, aggregator: OkxDataAggregator):  
        self.aggregator = aggregator  
        self.maker_price_limit = 6  # 默认保留的价格数量   
        self.newest_timeframe_supertrend = {}

    def process_taker_trade(self, trades):  
        """  
        处理交易数据，并更新到 aggregator  
        :param trades: 交易数据列表  
        """  

        for trade in trades:  
            ts = trade["ts"]  # "buy" or "sell"  
            price = float(trade["last"])  
            size = float(trade["lastSz"])  
            self.aggregator.update_tickers(ts, price, size)  

    def process_maker_book(self, bids, asks):  
        """  
        处理订单簿数据，并更新到 aggregator  
        :param bids: 买盘数据  
        :param asks: 卖盘数据  
        """  
        # 筛选逻辑  
        bids = self.filter_maker_prices(bids, "bids")  
        asks = self.filter_maker_prices(asks, "asks")  

        # 更新到 aggregator  
        try:
            self.aggregator.update_maker_order_book(bids, asks)  
        except Exception as e:
            print(f'!!!!!!!!!!!!!!update_maker_order_book {e}')

        try:
            #from 1m, 5m, 15m show_history.py
            # 从 Flask 服务获取数据  
            import requests
            response = requests.get("http://127.0.0.1:5001/fetch")  
            if response.status_code == 200:  
                data = response.json()  
                if len(data.keys()) == 0:
                    five_minutes_trend = None
                else:
                    self.newest_timeframe_supertrend = data
                    five_minutes_trend = self.newest_timeframe_supertrend['1m']
            else:
                five_minutes_trend = None
                print('request to local supertrend failed:', response.status_code) 
            
            self.aggregator.update_current_episode_maker_price(bids, asks, five_minutes_trend)
        except Exception as e:
            print(f'!!!!!!!!!!!!!!update_current_episode_maker_price {e}')

        try:
            self.aggregator.update_maker_depth(bids, asks)
        except Exception as e:
            print(f'!!!!!!!!!!!!!!update_maker_depth {e}')

    def filter_maker_prices(self, orders, order_type):  
        """  
        筛选 Maker 价格，保留指定数量的价格  
        :param orders: 当前订单列表，例如 bids 或 asks  
        :param order_type: "bids" 或 "asks"  
        :return: 筛选后的订单列表  
        """  
        # 如果订单数量不在 5 到 10 之间，直接返回  
        if not (5 < len(orders) < 10):  
            return orders  

        # 根据订单类型进行排序和筛选  
        if order_type == "bids":  
            # 买入订单：保留价格较高的  
            orders = sorted(orders, key=lambda x: x[0], reverse=True)[:self.maker_price_limit]  
        elif order_type == "asks":  
            # 卖出订单：保留价格较低的  
            orders = sorted(orders, key=lambda x: x[0])[:self.maker_price_limit]  

        return orders  
    
        
class OkxWebSocketClient:
    """
    WebSocket客户端，基于websocket.WebSocketApp
    1. 负责连接OKX公共频道 (books / trades)
    2. 解析数据后调用 DataProcessor 去处理aggregator 的 update_taker_trade / update_maker_order_book / update_maker_depth
    """
    def __init__(self, aggregator: OkxDataAggregator, inst_id=None):
        self.aggregator = aggregator
        self.inst_id = inst_id
        # self.url = "wss://ws.okx.com:8443/ws/v5/business"
        self.url = 'wss://wsaws.okx.com:8443/ws/v5/public'
        self.ws_app = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
            )
        self.data_processor = DataProcessor(aggregator)

        self.ws_app.on_open = self.on_open

        self.books_channel = "books"
        self.trades_channel = "trades"
        self.tickers_channel = "tickers"

    def on_open(self, ws):   
        print("WebSocket opened: subscribing to trades & books...")  
        params = {  
            "op": "subscribe",  
            "args": [  
                {"channel": self.books_channel, "instId": self.inst_id},  
                {"channel": self.trades_channel, "instId": self.inst_id},
                {"channel": self.tickers_channel, "instId": self.inst_id}  
            ]  
        }  
        ws.send(json.dumps(params))  

    def on_message(self, ws, message):  
        try:
            data = json.loads(message)  

            # 如果包含 "event"，一般是订阅或心跳事件，无需处理  
            if "event" in data:  
                return  

            # 分频道解析  
            if "arg" in data and "channel" in data["arg"]:  
                channel = data["arg"]["channel"]  
                #if channel == self.trades_channel and "data" in data:  
                if channel == self.tickers_channel and "data" in data: 
                    try:
                        self.data_processor.process_taker_trade(data["data"])
                    except Exception as e:
                        print(f'on_message process_taker_trade, Exception {e}')
                elif channel == self.books_channel and "data" in data:  
                    try:
                        # 只处理前25档，或按需更改  
                        bids = [[float(x[0]), float(x[1])] for x in data["data"][0]["bids"][:25]]  
                        asks = [[float(x[0]), float(x[1])] for x in data["data"][0]["asks"][:25]]  
                    except Exception as e:
                        print(f'on_message process bids and asks, Exception {e}')
                    for pv in bids:
                        if pv[1] == 0:
                            pv[1] = 0.0005
                            # print(f'bids pv = {data["data"][0]["bids"]}')
                    for pv in asks:
                        if pv[1] == 0:
                            pv[1] = 0.0005
                            # print(f'asks pv = {data["data"][0]["asks"]}')
                    self.data_processor.process_maker_book(bids, asks)
        except Exception as e:
            print(f'on_message Exception {e}')

    def on_error(self, ws, error):  
        print(f"WebSocket error: {error}")  

    def on_close(self, ws, code, reason):  
        print(f"WebSocket closed: {code}, {reason}")  

    def run_forever(self):  
        self.ws_app.run_forever()  