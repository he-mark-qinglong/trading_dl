import dash  
from dash import Dash, html, dcc, Input, Output  
import pandas as pd  
from datetime import datetime  
from utils.chart_manager import ChartManager  
from factors import FactorManager  
from utils.realtime_data_manager import RealtimeDataManager  
from utils.colors import initialize_colors  


class DashVisualizer:  
    def __init__(self, inst_id:str, timeframes = ['1m', '5m', '15m', '1h']):  
        """  
        初始化 DashVisualizer，负责创建 Dash 应用并管理图表更新。  
        """                                                                                                                                                     
        # 初始化因子管理器、颜色配置和实时数据管理器  
        self.factor_manager = FactorManager()  
        self.colors = initialize_colors()  

        self.timeframes = timeframes  #实际回测夏普率的时候发现4h以上的twvwap已经失去意义了。
        self.realtime_manager = RealtimeDataManager(  
            inst_id, self.timeframes, history_days=30000
        )  
        
        # 初始化 ChartManager  
        self.chart_manager = ChartManager(self.colors, self.factor_manager)  
        
        # 初始化 Dash 应用  
        self.app = Dash(__name__)  
        
        # 设置布局  
        self.layout_setted_up = False
        self.setup_layout()

    def setup_layout(self):  
        """  
        设置 Dash 应用的布局，包括图表和更新回调。  
        """  
        self.layout_setted_up = True

        charts = []  

        for tf in self.timeframes:  
            # 获取初始数据  
            dict = self.realtime_manager.get_latest_data_with_factors(tf)
            if dict is None:
                continue
            # signal, trend, market_state
            df, factors = dict['df'], dict['factors']
            if df is not None and not df.empty: 
                # 为每个时间周期创建一个图表容器  
                charts.append(html.Div([  
                    # 悬停数据显示区  
                    html.Div(  
                        id=f'hover-data-{tf}',  
                        style={  
                            'position': 'absolute',  
                            'left': '10px',  
                            'top': '10px',  
                            'backgroundColor': 'rgba(255, 255, 255, 0.8)',  
                            'padding': '10px',  
                            'borderRadius': '5px',  
                            'zIndex': 1000,  
                            'display': 'none'  
                        }  
                    ),  
                    # 图表  
                    dcc.Graph(  
                        id=f'chart-{tf}',  
                        figure=self.chart_manager.create_candlestick_figure(df, factors, tf),  
                        config={'displayModeBar': True}  
                    ),  
                    # 定时器，用于定期更新图表  
                    dcc.Interval(  
                        id=f'interval-{tf}',  
                        interval=120 * 1000,  # 每 30 秒更新一次  
                        n_intervals=0  
                    )  
                ], style={'position': 'relative'}))  

        # 设置应用布局  
        self.app.layout = html.Div(charts)  

        # 为每个时间周期添加更新回调  
        for tf in self.timeframes:  
            self._add_update_callback(tf)  
            self._add_hover_callback(tf)  

    def _add_update_callback(self, timeframe):  
        """  
        为指定时间周期的图表添加更新回调。  
        """  
        @self.app.callback(  
            Output(f'chart-{timeframe}', 'figure'),  
            Input(f'interval-{timeframe}', 'n_intervals')  
        )  
        def update_graph(n):  
            """  
            更新指定时间周期的图表。  
            """  
            # 获取最新数据  
            dict = self.realtime_manager.get_latest_data_with_factors(timeframe)  
            if dict is None:
                return {} 
            
            df, factors, signal = dict['df'], dict['factors'], dict['signal']
            
            def post_data_to_ticker(data):
                import requests
                response = requests.post("http://127.0.0.1:5001/store", json=data)  
                if response.status_code == 200:  
                    #print(f'debug === updated data:{data}')
                    pass
                else:
                    print('post to flask for signal failed')
            # post_data_to_ticker({timeframe:factors['twvwap'].iloc[-1]})
            
            if timeframe == '5m':
                twvwap = factors["twvwap"].iloc[-1]
                l1 = factors["twvwap_lower_band_1"].iloc[-1]
                l2 = factors["twvwap_lower_band_2"].iloc[-1]
                h1 = factors["twvwap_upper_band_1"].iloc[-1]
                h2 = factors["twvwap_upper_band_2"].iloc[-1]
                #print(f'5m: vwap: {twvwap} l1: {l1} l2:{l2} h1:{h1} h2:{h2}')
            if df is not None and not df.empty:  
                # 使用 ChartManager 创建更新后的图表  
                figure = self.chart_manager.create_candlestick_figure(df, factors, timeframe)  
                #self.chart_manager.add_trade_signals(figure, signal)

                return figure
            print(f"No data available for {timeframe} at interval {n}")  
            return {}  

    def _add_hover_callback(self, timeframe):  
        """  
        为指定时间周期的图表添加悬停回调，用于显示光标信息。  
        """  
        @self.app.callback(  
            Output(f'hover-data-{timeframe}', 'children'),  
            Output(f'hover-data-{timeframe}', 'style'),  
            Input(f'chart-{timeframe}', 'hoverData')  
        )  
        def display_hover_data(hover_data):  
            """  
            显示光标悬停时的详细信息。  
            """  
            if hover_data is None or 'points' not in hover_data or len(hover_data['points']) == 0:  
                # 如果没有悬停数据，则隐藏悬停信息框  
                return None, {'display': 'none'}  

            # 提取悬停点的信息  
            point = hover_data['points'][0]  

            # 检查是否存在 'x' 和 'y' 键  
            x = point.get('x', 'N/A')  # 时间戳  
            y = point.get('y', 'N/A')  # 对应的值  
            series_name = point.get('curveNumber', 'Unknown')  # 曲线编号（可以用来区分 OHLC 或指标）  

            # 构造悬停信息内容  
            hover_info = [  
                html.Div(f"Time: {x}"),  
                html.Div(f"Value: {y}"),  
                html.Div(f"Series: {series_name}")  
            ]  

            # 设置悬停信息框的样式  
            hover_style = {  
                'position': 'absolute',  
                'left': '10px',  
                'top': '10px',  
                'backgroundColor': 'rgba(255, 255, 255, 0.8)',  
                'padding': '10px',  
                'borderRadius': '5px',  
                'zIndex': 1000,  
                'display': 'block'  
            }  

            return hover_info, hover_style
    def run(self, port=8051):  
        """  
        启动 Dash 应用。  
        """  
        # 启动实时数据更新  
        self.realtime_manager.start_realtime_updates()  
        try:  
            # 启动 Dash 服务器  
            self.app.run( port=port, debug = True)  
        finally:  
            # 停止实时数据更新  
            self.realtime_manager.stop_realtime_updates()  
 
if __name__ == '__main__':  
    # 创建 DashVisualizer 实例并运行  
    visualizer = DashVisualizer("BTC-USDT-SWAP", ['1m']) #, '1h', '1d']) 
    # visualizer = DashVisualizer("IP-USDT-SWAP", ['1m', '5m', '1h']) 
    port = 8051

    # visualizer = DashVisualizer("ETH-USDT-SWAP")
    # port = 8051
    # visualizer = DashVisualizer("PEPE-USDT-SWAP")
    visualizer.run(port)