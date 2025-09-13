import json
import threading
import websocket
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import time
import datetime

# 全局变量，用于存储实时盘口和成交单数据  
order_book_data = {"bids": [], "asks": []}  
trade_data = {"buy": [], "sell": []}  

# 分钟级数据  
minute_data = {  
    "timestamps": [],  
    "open": [],  
    "high": [],  
    "low": [],  
    "close": [],  
    "volume": [],  
    "vwap": [],  
    "depth_rate": []  
}  

current_minute_data = {  
    "prices": [],  
    "volumes": [],  
    "bid_depth_start": 0,  
    "ask_depth_start": 0,  
    "bid_depth_end": 0,  
    "ask_depth_end": 0  
}  

# 秒级数据（修正后）  
second_data = {  
    "timestamps": [],    
    "prices": [],        # 新增价格存储  
    "buy_volumes": [],     
    "sell_volumes": []     
}  

current_second = {  
    "timestamp": None,     
    "price": None,        # 存储当前秒最新价格  
    "buy": 0,              
    "sell": 0              
}  

def on_message(ws, message):  
    global second_data, current_second, order_book_data, current_minute_data  

    data = json.loads(message)  
    if 'event' not in data:  
        if data["arg"]["channel"] == "trades":  
            for trade in data["data"]:  
                now = datetime.datetime.now().replace(microsecond=0)  
                side = trade["side"]  
                price = float(trade["px"])  
                size = float(trade["sz"])  

                # 更新当前秒价格  
                current_second["price"] = price  

                # 秒切换处理  
                if current_second["timestamp"] != now:  
                    if current_second["timestamp"] is not None:  
                        # 保存完整秒数据  
                        second_data["timestamps"].append(current_second["timestamp"])  
                        second_data["prices"].append(current_second["price"])  
                        second_data["buy_volumes"].append(current_second["buy"])  
                        second_data["sell_volumes"].append(current_second["sell"])  

                        # 保持60秒数据  
                        if len(second_data["timestamps"]) > 60:  
                            for key in ["timestamps", "prices", "buy_volumes", "sell_volumes"]:  
                                second_data[key].pop(0)  

                    # 重置当前秒  
                    current_second = {  
                        "timestamp": now,  
                        "price": price,  
                        "buy": 0,  
                        "sell": 0  
                    }  

                # 累计成交量  
                if side == "buy":  
                    current_second["buy"] += size  
                elif side == "sell":  
                    current_second["sell"] += size  

                # 更新分钟数据  
                current_minute_data["prices"].append(price)  
                current_minute_data["volumes"].append(size)  

        elif data["arg"]["channel"] == "books":  
            bids = [[float(x[0]), float(x[1])] for x in data["data"][0]["bids"][:25]]  
            asks = [[float(x[0]), float(x[1])] for x in data["data"][0]["asks"][:25]]  
            order_book_data["bids"] = sorted(bids, reverse=True)  
            order_book_data["asks"] = sorted(asks) 


def on_error(ws, error):
    """
    处理 WebSocket 错误
    """
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    """
    处理 WebSocket 关闭
    """
    print(f"WebSocket Closed {close_msg}")

def on_open(ws):
    """
    WebSocket 连接成功后订阅数据
    """
    # 订阅 OKX 的盘口数据和成交单数据
    params = {
        "op": "subscribe",
        "args": [
            {"channel": "books", "instId": "SOL-USDT-SWAP"},  # 订阅盘口数据
            {"channel": "trades", "instId": "SOL-USDT-SWAP"}  # 订阅成交单数据
        ]
    }
    ws.send(json.dumps(params))
    print("Subscribed to Order Book and Trade Data")

def start_websocket():
    """
    启动 WebSocket 客户端
    """
    # OKX WebSocket 地址
    url = "wss://ws.okx.com:8443/ws/v5/public"

    # 创建 WebSocket 连接
    ws = websocket.WebSocketApp(
        url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open

    # 运行 WebSocket 客户端
    ws.run_forever()

# 启动 WebSocket 客户端线程
threading.Thread(target=start_websocket, daemon=True).start()

# 等待 WebSocket 初始化
time.sleep(2)

# 定时任务：每分钟计算 K 线数据
def calculate_minute_data():
    global minute_data, current_minute_data

    while True:
        time.sleep(60)  # 每分钟执行一次

        # 如果当前分钟有数据，计算 K 线和 VWAP
        if current_minute_data["prices"]:
            open_price = current_minute_data["prices"][0]
            high_price = max(current_minute_data["prices"])
            low_price = min(current_minute_data["prices"])
            close_price = current_minute_data["prices"][-1]
            total_volume = sum(current_minute_data["volumes"])
            vwap = sum(p * v for p, v in zip(current_minute_data["prices"], current_minute_data["volumes"])) / total_volume

            # 计算盘口深度变化率
            bid_depth_change = (current_minute_data["bid_depth_end"] - current_minute_data["bid_depth_start"]) / 60
            ask_depth_change = (current_minute_data["ask_depth_end"] - current_minute_data["ask_depth_start"]) / 60

            # 保存到 minute_data
            minute_data["timestamps"].append(datetime.datetime.now())
            minute_data["open"].append(open_price)
            minute_data["high"].append(high_price)
            minute_data["low"].append(low_price)
            minute_data["close"].append(close_price)
            minute_data["volume"].append(total_volume)
            minute_data["vwap"].append(vwap)
            minute_data["depth_rate"].append((bid_depth_change, ask_depth_change))

            # 清空当前分钟数据
            current_minute_data = {
                "prices": [],
                "volumes": [],
                "bid_depth_start": 0,
                "ask_depth_start": 0,
                "bid_depth_end": 0,
                "ask_depth_end": 0
            }

# 启动定时任务线程
threading.Thread(target=calculate_minute_data, daemon=True).start()

# Dash 应用程序
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("OKX 实时盘口与成交单数据", style={"textAlign": "center"}),

    # 分离的图表
    dcc.Graph(id="kline-graph", style={"display": "inline-block", "width": "48%"}),
    dcc.Graph(id="vwap-graph", style={"display": "inline-block", "width": "48%"}),
    dcc.Graph(id="depth-rate-graph", style={"display": "inline-block", "width": "48%"}),

    # Taker 秒级图
    dcc.Graph(id="taker-price-graph", style={"width": "100%"}),
    dcc.Graph(id="taker-volume-graph", style={"width": "100%"}),

    # 定时刷新组件
    dcc.Interval(
        id="interval-component",
        interval=1000,  # 每秒刷新一次
        n_intervals=0
    )
])

@app.callback(  
    [Output("kline-graph", "figure"),  
     Output("vwap-graph", "figure"),  
     Output("depth-rate-graph", "figure"),  
     Output("taker-price-graph", "figure"),  
     Output("taker-volume-graph", "figure")],  
    [Input("interval-component", "n_intervals")]  
)  
def update_graphs(n):  
    global minute_data, second_data  

    # K线图  
    kline_figure = go.Figure(data=[  
        go.Candlestick(  
            x=minute_data["timestamps"],  
            open=minute_data["open"],  
            high=minute_data["high"],  
            low=minute_data["low"],  
            close=minute_data["close"],  
            name="K线"  
        )  
    ]).update_layout(title="1 分钟 K 线图", template="plotly_dark")  

    # VWAP图  
    vwap_figure = go.Figure(data=[  
        go.Scatter(  
            x=minute_data["timestamps"],  
            y=minute_data["vwap"],  
            mode="lines",  
            name="VWAP",  
            line=dict(color="orange")  
        )  
    ]).update_layout(title="VWAP 指标", template="plotly_dark")  

    # 深度变化率图  
    depth_rate_figure = go.Figure()  
    if minute_data["depth_rate"]:  
        depth_rate_figure.add_traces([  
            go.Scatter(  
                x=minute_data["timestamps"],  
                y=[r[0] for r in minute_data["depth_rate"]],  
                name="买单深度变化率",  
                line_color="green"  
            ),  
            go.Scatter(  
                x=minute_data["timestamps"],  
                y=[r[1] for r in minute_data["depth_rate"]],  
                name="卖单深度变化率",  
                line_color="red"  
            )  
        ]).update_layout(title="盘口深度变化率", template="plotly_dark")  

    # Taker价格图  
    price_figure = go.Figure(  
        data=[go.Scatter(  
            x=second_data["timestamps"],  
            y=second_data["prices"],  
            mode="lines+markers",  
            line_color="blue"  
        )]  
    ).update_layout(  
        title="Taker 秒级价格走势",  
        xaxis_title="时间",  
        yaxis_title="价格",  
        template="plotly_dark"  
    )  

    # Taker成交量图  
    volume_figure = go.Figure()  
    if second_data["timestamps"]:  
        volume_figure.add_traces([  
            go.Bar(  
                x=second_data["timestamps"],  
                y=second_data["buy_volumes"],  
                name="买入",  
                marker_color="green",  
                offsetgroup=0  
            ),  
            go.Bar(  
                x=second_data["timestamps"],  
                y=second_data["sell_volumes"],  
                name="卖出",  
                marker_color="red",  
                offsetgroup=1  
            )  
        ]).update_layout(  
            title="Taker 秒级成交量",  
            barmode="group",  
            template="plotly_dark"  
        )  

    return kline_figure, vwap_figure, depth_rate_figure, price_figure, volume_figure 

if __name__ == "__main__":
    app.run_server(debug=True)