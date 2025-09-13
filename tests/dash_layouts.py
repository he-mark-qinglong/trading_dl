
import dash  
from dash import dcc, html  
from dash.dependencies import Output, Input  
import plotly.graph_objs as go  

from tests.okx_DataAggregator import OkxDataAggregator
from factors import FactorManager 
 

class DashLayout:
    def __init__(self, __name__, aggregator:OkxDataAggregator, factor_manager:FactorManager):
        self.aggregator = aggregator
        self.factor_manager = factor_manager
        
        # 4) 构建 Dash 应用 (前端可视化)  
        app = dash.Dash(__name__)  

        app.layout = html.Div([  
            html.H1("OKX 实时盘口与成交单数据 (OOP分层示例)", style={"textAlign": "center"}),  

            # Candlestick K线 + 深度变化率 + VWAP  
            html.Div([  
                # 第一行：K线图和VWAP图  
                html.Div([  
                    # dcc.Graph(id="kline-graph", style={"width": "100%", "display": "inline-block"}),  
                    dcc.Graph(id="vwap-graph", style={"width": "100%", "height":"50%", "display": "inline-block"})  
                ]),  
                # 第二行：深度变化率图和深度波动率图  
                html.Div([  
                    dcc.Graph(id="depth-rate-graph", style={"width": "100%", "display": "inline-block"}),  
                    dcc.Graph(id="depth-volatility-graph", style={"width": "100%", "display": "inline-block"})  
                ]),
            ]),  
            
            html.Div([
                html.Div([  
                    dcc.Graph(id="depth-change-graph", style={"width": "100%", "display": "inline-block"}),    
                ]) 
            ]),

            html.Div([
                html.Div([  
                    dcc.Graph(id="depth-delta-distribution-graph", style={"width": "100%", "display": "inline-block"}),
                ]) 
            ]), 

            # html.Div([  
            #     dcc.Graph(id="factor-graph", style={"width": "100%", "display": "inline-block"})  
            # ]),  

            html.Div([  
                dcc.Graph(id="maker-distribution-graph", style={  
                    "width": "100%",   
                    "height": "600px",  # 设置高度为600px  
                    "display": "inline-block"  
                }  )  
            ]),  

            # 定时刷新  
            dcc.Interval(  
                id="interval-component",  
                interval = 1* 1000,  # 每3秒刷新一次  
                n_intervals=0  
            )  
        ])  
        self.app = app
        

        @app.callback(  
            Output("factor-graph", "figure"),  
            [Input("interval-component", "n_intervals")]  
        )  
        def update_factor_graph(n):  
            """  
            从 FactorManager 中计算因子并可视化  
            """  
            df = self.aggregator.get_episode_data_as_dataframe()  
            if len(df) < self.factor_manager.config.window_size:  
                return go.Figure()  

            # 假设有一个全局的 FactorManager 实例  
            factors, _ = self.factor_manager.calculate_factors(self.aggregator.get_episode_data_as_dataframe())  

            # 创建一个图表，展示因子数据  
            figure = go.Figure()  
            for factor_name, factor_values in factors.items():  
                figure.add_trace(go.Scatter(  
                    x=factor_values.index,  
                    y=factor_values,  
                    mode="lines",  
                    name=factor_name  
                ))  

            figure.update_layout(  
                title="技术因子图表",  
                template="plotly_dark",  
                xaxis_title="时间",  
                yaxis_title="因子值"  
            )  
            return figure  


        @app.callback(  
            Output("maker-distribution-graph", "figure"),  
            [Input("interval-component", "n_intervals")]  
        )  
        def update_maker_distribution_graph(n):  
            """  
            更新 Maker 累计挂单分布图  
            """  
            # 获取累计挂单分布数据  
            cumulative_bids, cumulative_asks = self.aggregator.get_cumulative_order_book(apply_time_weighting=True)  

            # 分别提取价格和累计成交量  
            bid_prices = [entry["price"] for entry in cumulative_bids]  
            bid_volumes = [entry["volume"] for entry in cumulative_bids]  

            ask_prices = [entry["price"] for entry in cumulative_asks]  
            ask_volumes = [entry["volume"] for entry in cumulative_asks]  

            # 创建横向柱状图  
            figure = go.Figure()  

            # 买单分布（绿色柱状图，向左延伸）  
            figure.add_trace(go.Bar(  
                x=[-v for v in bid_volumes],  # 负值向左延伸  
                y=bid_prices,  
                orientation="h",  
                name="买单",  
                marker_color="limegreen",  # 更鲜艳的绿色  
                opacity=1.0  # 不使用透明度  
            ))  

            # 卖单分布（红色柱状图，向右延伸）  
            figure.add_trace(go.Bar(  
                x=ask_volumes,  # 正值向右延伸  
                y=ask_prices,  
                orientation="h",  
                name="卖单",  
                marker_color="crimson",  # 更鲜艳的红色  
                opacity=1.0  # 不使用透明度  
            ))  

            # 更新图表布局  
            figure.update_layout(  
                title="Maker 累计挂单分布",  
                template=None,  # 不使用任何内置模板  
                xaxis_title="累计成交量",  
                yaxis_title="价格",  
                barmode="overlay",  # 保持柱状图重叠  
                plot_bgcolor="black",  # 深色背景  
                paper_bgcolor="black",  # 整个图表背景  
                font=dict(color="white", size=14),  # 白色字体，适配深色背景  
                xaxis=dict(  
                    showgrid=False,  # 移除网格线  
                    zerolinecolor="white",  # 零线颜色为白色  
                    zerolinewidth=1.5  # 增加零线宽度  
                ),  
                yaxis=dict(  
                    showgrid=False,  # 移除网格线  
                    zerolinecolor="white",  # 零线颜色为白色  
                    zerolinewidth=1.5  # 增加零线宽度  
                )  
            )  

            return figure

        @app.callback(  
            Output("depth-change-graph", "figure"),  
            [Input("interval-component", "n_intervals")]  
        )  
        def update_depth_change_graph(n):  
            """  
            更新买卖盘深度变化图  
            """  
            # 获取买卖盘深度变化数据  
            depth_rate_data = self.aggregator.all_episode_data["depth_rate"]  

            # 提取 bid_depth_change 和 ask_depth_change  
            bid_changes = [entry[0] for entry in depth_rate_data]  
            ask_changes = [entry[1] for entry in depth_rate_data]  

            # 创建横向柱状图  
            figure = go.Figure()  

            # 买盘深度变化（绿色柱状图，向左延伸）  
            figure.add_trace(go.Bar(  
                x=[-v for v in bid_changes],  # 负值向左延伸  
                y=list(range(len(bid_changes))),  # 使用索引作为 Y 轴  
                orientation="h",  
                name="买盘深度变化",  
                marker_color="green"  
            ))  

            # 卖盘深度变化（红色柱状图，向右延伸）  
            figure.add_trace(go.Bar(  
                x=ask_changes,  # 正值向右延伸  
                y=list(range(len(ask_changes))),  # 使用索引作为 Y 轴  
                orientation="h",  
                name="卖盘深度变化",  
                marker_color="red"  
            ))  

            # 更新图表布局  
            figure.update_layout(  
                title="买卖盘深度变化",  
                template="plotly_dark",  
                xaxis_title="深度变化",  
                yaxis_title="时间索引",  
                barmode="overlay",  
                xaxis=dict(showgrid=False),  
                yaxis=dict(showgrid=False)  
            )  

            return figure

        from scipy.stats import gaussian_kde
        import numpy as np
        @app.callback(  
            Output("depth-delta-distribution-graph", "figure"),  
            [Input("interval-component", "n_intervals")]  
        )  
        def update_depth_delta_distribution_graph(n):  
            """  
            更新买卖盘深度差值的概率分布图  
            """  
            # 获取买卖盘深度差值数据  
            depth_rate_data = self.aggregator.all_episode_data["depth_rate"]  
            depth_deltas = [entry[2] for entry in depth_rate_data]  

            # 创建概率分布图  
            figure = go.Figure()  

            # 添加直方图  
            figure.add_trace(go.Histogram(  
                x=depth_deltas,  
                name="买卖盘深度差值分布",  
                marker_color="blue",  
                opacity=0.75,  
                histnorm="probability",  # 显示概率分布  
                nbinsx=500,  # 增加直方图的柱子数量  
                hovertemplate="深度差值: %{x}<br>概率: %{y}<extra></extra>"  # 自定义悬停信息  
            ))  

            if len(depth_deltas) > 1:
                # 添加核密度估计（KDE）曲线  
                kde = gaussian_kde(depth_deltas)  
                x_range = np.linspace(min(depth_deltas), max(depth_deltas), 500)  
                y_kde = kde(x_range)  
                figure.add_trace(go.Scatter(  
                    x=x_range,  
                    y=y_kde,  
                    name="KDE 曲线",  
                    line=dict(color="red", width=2),  
                    hovertemplate="深度差值: %{x}<br>密度: %{y}<extra></extra>"  
                ))  

                # 计算统计信息  
                mean = np.mean(depth_deltas)  
                std = np.std(depth_deltas)  

                # 添加统计信息标注  
                figure.add_annotation(  
                    x=mean,  
                    y=max(y_kde),  
                    text=f"均值: {mean:.2f}<br>标准差: {std:.2f}",  
                    showarrow=True,  
                    arrowhead=1,  
                    ax=0,  
                    ay=-40,  
                    font=dict(size=12, color="white")  
                )  

                # 更新图表布局  
                figure.update_layout(  
                    title="买卖盘深度差值概率分布",  
                    template="plotly_dark",  
                    xaxis_title="深度差值",  
                    yaxis_title="概率/密度",  
                    bargap=0.05,  # 减小柱子之间的间隙  
                    showlegend=True,  
                    legend=dict(x=0.8, y=0.9),  
                    hovermode="x unified",  # 统一悬停模式  
                    margin=dict(l=50, r=50, t=80, b=50)  
                )  

                # 更新坐标轴样式  
                figure.update_xaxes(showgrid=True, gridwidth=0.05, gridcolor="rgba(255, 255, 255, 0.1)")  
                figure.update_yaxes(showgrid=True, gridwidth=0.05, gridcolor="rgba(255, 255, 255, 0.1)")  

            return figure

        from scipy.stats import norm 
        from plotly.subplots import make_subplots

        self.peak_history = []
        self.slope_history = [] 

        @app.callback(  
            [  
                Output("vwap-graph", "figure"),  
                Output("depth-rate-graph", "figure"),  
                Output("depth-volatility-graph", "figure")  # 新增输出  
            ],  
            [Input("interval-component", "n_intervals")]  
        )  
        def update_charts(n):  
            """  
            从 self.aggregator 中读取最新的秒/分钟数据，用 Plotly 进行可视化  
            """  
            # 读取分钟数据  
            episode_data = self.aggregator.get_episode_data_as_dataframe()  
            
            # ---------------------------------------------  
            # 计算 VWAP 斜率（差分）  
            if len(episode_data["vwap_smooth"]) > 1:  
                vwap_slopes = np.diff(episode_data["vwap_smooth"])  
            else:  
                vwap_slopes = []  

            # 计算乖离率对应的价格（正、负乖离线）  
            deviation_rates = [ 2]  # 百分比  
            colors = {"red":2, "yellow":5, "blue":10}

            deviation_lines = {rate: {"positive": [], "negative": []} for rate in deviation_rates}  
            for v in episode_data["vwap_smooth"]:  
                for rate in deviation_rates:  
                    deviation_lines[rate]["positive"].append(v * (1 + rate / 100))  
                    deviation_lines[rate]["negative"].append(v * (1 - rate / 100))  

            # 创建 1 行 2 列子图：左侧显示 VWAP 曲线（及乖离率线），右侧显示 VWAP 斜率分布  
            vwap_slope_fig = make_subplots(  
                rows=1, cols=2,  
                column_widths=[0.8, 0.2],  
                subplot_titles=["VWAP 曲线及乖离率", "VWAP Slope 分布"],  
                horizontal_spacing=0.05  
            )  
            
            if len(episode_data.index) > 0:  
                # 左侧：VWAP 曲线 + K线
                vwap_slope_fig.add_trace(  
                    go.Candlestick(  
                        x=episode_data.index,  
                        open=episode_data["open"],  
                        high=episode_data["high"],  
                        low=episode_data["low"],  
                        close=episode_data["close"],  
                        name="K线"  
                    ),  
                    row=1, col=1  
                )  

                vwap_slope_fig.update_layout(  
                    title="1 分钟 K 线图",  
                    template="plotly_dark"  
                )
                #vwap价格
                vwap_slope_fig.add_trace(  
                    go.Scatter(  
                        x=episode_data.index,  
                        y=episode_data["vwap_smooth"],  
                        mode="lines",  
                        line=dict(color="orange"),  
                        name="VWAP_smooth14"  
                    ),  
                    row=1, col=1  
                )  
                #真实价格
                vwap_slope_fig.add_trace(  
                    go.Scatter(  
                        x=episode_data.index,  
                        y=episode_data["market_price"],  
                        mode="lines",  
                        line=dict(color="purple"),  
                        name="market_price"  
                    ),  
                    row=1, col=1  
                ) 
                
                # 添加各个乖离率线  
                for rate in deviation_rates:  
                    # 正乖离  
                    color = None
                    for color_key in colors:
                        if colors[color_key] == rate:
                            color = color_key

                    vwap_slope_fig.add_trace(  
                        go.Scatter(  
                            x=episode_data.index,  
                            y=deviation_lines[rate]["positive"],  
                            mode="lines",  
                            line=dict(dash="dot", color=color),  
                            name=f"+{rate}% Deviation"  
                        ),  
                        row=1, col=1  
                    )  
                    # 负乖离  
                    vwap_slope_fig.add_trace(  
                        go.Scatter(  
                            x=episode_data.index,  
                            y=deviation_lines[rate]["negative"],  
                            mode="lines",  
                            line=dict(dash="dot", color=color_key),  
                            name=f"-{rate}% Deviation"  
                        ),  
                        row=1, col=1  
                    )  
            
            # 右侧：VWAP 斜率的概率分布（含动态密度中心线）
            if len(vwap_slopes) > 1:
                cut_max_previous = 60
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(vwap_slopes)
                    slope_values = np.linspace(min(vwap_slopes), max(vwap_slopes), 100)
                    slope_density = kde(slope_values)
                    
                    # 新增：计算密度峰值位置
                    peak_position = slope_values[np.argmax(slope_density)]

                    # 新增：维护峰值轨迹（保留最近50个观测值）
                    self.peak_history.append(peak_position)
                    self.peak_history = self.peak_history[-min(len(self.peak_history), cut_max_previous):]  # 滑动窗口
                    
                    # 绘制概率密度曲线
                    vwap_slope_fig.add_trace(
                        go.Scatter(
                            x=slope_density,
                            y=slope_values,
                            mode="lines",
                            line=dict(color="blue", width=2),
                            name="瞬时密度分布"
                        ),
                        row=1, col=2
                    )
                    
                    # 新增：指数衰减权重计算  
                    def exp_weights(length, decay=0.9):  
                        """生成指数衰减权重序列"""  
                        weights = [decay**(length-1-i) for i in range(length)]  
                        return np.array(weights)/sum(weights)  

                    # 修改峰值处理部分  
                    if len(self.peak_history) >= 3:  
                        # 计算加权衰减斜率  
                        weights = exp_weights(len(self.peak_history))  
                        x = np.arange(len(self.peak_history))  
                        slope = np.polyfit(x, self.peak_history, 1, w=weights)[0]  
                        
                        # 保存斜率轨迹  
                        self.slope_history.append(slope)  
                        self.slope_history = self.slope_history[-min(len(self.slope_history), cut_max_previous):]  # 滑动窗口

                        # 新增衰减斜率轨迹线  
                        vwap_slope_fig.add_trace(  
                            go.Scatter(  
                                x=np.linspace(0.2*max(slope_density), 0.8*max(slope_density), len(self.slope_history)),  
                                y=np.array(self.peak_history[-len(self.slope_history):]) + np.array(self.slope_history)*np.arange(len(self.slope_history)),  
                                mode="lines",  
                                line=dict(color="#00FF00", width=2),  
                                name="衰减发展斜率",  
                                hovertemplate="预测斜率: %{y:.4f}<extra></extra>"  
                            ),  
                            row=1, col=2  
                        )  
                        
                    # 新增：动态密度中心线
                    vwap_slope_fig.add_trace(
                        go.Scatter(
                            x=np.linspace(0, max(slope_density), len(self.peak_history)),
                            y=self.peak_history,
                            mode="lines+markers",
                            line=dict(color="#FFA500", width=1.5, dash="dot"),
                            marker=dict(size=6, symbol="diamond"),
                            name="密度中心轨迹",
                            hovertemplate="峰值: %{y:.4f}<extra></extra>"
                        ),
                        row=1, col=2
                    )
                    
                    # 绘制直方图（使用透明条显示分布范围）
                    vwap_slope_fig.add_trace(
                        go.Bar(
                            x=vwap_slopes,
                            y=[0]*len(vwap_slopes),
                            marker=dict(
                                color="rgba(100, 149, 237, 0.3)",  # 浅蓝色透明
                                line=dict(width=0)  # 去除边框线
                            ),
                            name="斜率分布直方图",
                            hoverinfo="skip"
                        ),
                        row=1, col=2
                    )
                    
                    # 新增：零轴参考线
                    vwap_slope_fig.add_hline(
                        y=0, 
                        line=dict(color="gray", width=1, dash="dash"),
                        row=1, col=2
                    )
                    
                except Exception as e:
                    print("VWAP斜率可视化异常:", e)

            # 增强布局配置
            vwap_slope_fig.update_layout(
                title="VWAP动态分析 (带密度中心轨迹)",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis2=dict(title="概率密度", showgrid=False),
                yaxis2=dict(
                    title="斜率值",
                    range=[min(vwap_slopes)*1.1 if len(vwap_slopes)>0 else -0.1, 
                        max(vwap_slopes)*1.1 if len(vwap_slopes)>0 else 0.1],
                    zerolinecolor="gray"
                ),
                hovermode="x unified"
            )

            # ---------------------------------------------  
            # 3) 盘口深度pressure图  
            depth_rate_figure = go.Figure()  
            # OBPI length for the last part of the data  
            length_obpi = len(episode_data)  

            if length_obpi > 0:  
                # 绘制 OBPI 和 OBPI EMA  
                depth_rate_figure.add_trace(  
                    go.Scatter(  
                        x=episode_data.index,  
                        y=episode_data["obpi"],  
                        name="OBPI",  
                        line_color="blue"  
                    )  
                )  
                depth_rate_figure.add_trace(  
                    go.Scatter(  
                        x=episode_data.index,  
                        y=episode_data["obpi_ema_fast"],  
                        name="Fast EMA (14)",  
                        line_color="orange"  
                    )  
                )  
                depth_rate_figure.add_trace(  
                    go.Scatter(  
                        x=episode_data.index,  
                        y=episode_data["obpi_ema_slow"],  
                        name="Slow EMA (100)",  
                        line_color="red"  
                    )  
                )  
                
                # 绘制 OBPI Diff 的柱状图，带条件颜色  
                colors = ['green' if diff >= 0 else 'red' for diff in episode_data["obpi_diff"]]  
                depth_rate_figure.add_trace(  
                    go.Bar(  
                        x=episode_data.index,  
                        y=episode_data["obpi_diff"],  
                        name="OBPI Diff",  
                        marker_color=colors,  
                        opacity=0.5  
                    )  
                )  
                
                # 绘制信号线  
                depth_rate_figure.add_trace(  
                    go.Scatter(  
                        x=episode_data.index,  
                        y=episode_data["signal_line"],  
                        name="Signal Line (10)",  
                        line_color="purple",  
                        line=dict(dash='dash')  
                    )  
                )  

            depth_rate_figure.update_layout(  
                title="Order Book Pressure Index (OBPI)",  
                template="plotly_dark"  
            )  

            # ---------------------------------------------  
            # 4) 深度波动率图  
            depth_volatility_figure = go.Figure()  
            if len(episode_data.get("bid_volatility", [])) > 0:  
                depth_volatility_figure.add_trace(  
                    go.Scatter(  
                        x=episode_data.index,  
                        y=episode_data["ask_volatility"],  
                        name="买单深度波动率",  
                        line_color="purple"  
                    )  
                )  
                depth_volatility_figure.add_trace(  
                    go.Scatter(  
                        x=episode_data.index,  
                        y=episode_data["bid_volatility"],  
                        name="卖单深度波动率",  
                        line_color="orange"  
                    )  
                )  
            depth_volatility_figure.update_layout(  
                title="盘口深度波动率",  
                template="plotly_dark"  
            )  

            # 返回4个图：K线图、VWAP+斜率分布图、盘口深度变化率图、盘口深度波动率图  
            return vwap_slope_fig, depth_rate_figure, depth_volatility_figure  