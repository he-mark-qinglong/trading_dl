import os  
import pandas as pd  
import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
from dash import Dash, dcc, html, dash_table  
from dash.dependencies import Input, Output  
from strategy.scaling_strategy import global_data, data_lock  

def plot_avwap_chart(df: pd.DataFrame, output_dir: str = "./charts_avwap") -> go.Figure:  
    """  
    绘制 AVWAP 图表、MACD 指标和 ADX 指标：  
      - 上方子图展示价格与 K 线、AVWAP（高 AVWAP 和低 AVWAP），并在反转位置使用散点图绘制三角形标记；  
      - 同时对 entry 信号保留箭头注释（例如 long_entry、short_entry、long_exit、short_exit）；  
      - 中间子图展示 MACD 指标（MACD 快线、MACD 慢线及 MACD Diff 条形图，正值绿色、负值红色）；  
      - 下方子图展示 ADX 指标及其方向指标 +DI 和 -DI（+DI作为key）。  
    返回生成的 Figure 对象。  
    """  
    # 设置三行子图，分别为：价格图、MACD 图和 ADX 图  
    fig = make_subplots(  
        rows=3, cols=1, shared_xaxes=True,  
        vertical_spacing=0.03, row_heights=[0.7, 0.15, 0.15]  
    )  

    # -------------------------  
    # 第一子图：K 线 + AVWAP及标注  
    # -------------------------  
    fig.add_trace(go.Candlestick(  
        x=df.index,  
        open=df['open'],  
        high=df['high'],  
        low=df['low'],  
        close=df['close'],  
        name='Price',  
        increasing_line_color='#2E8540',  
        decreasing_line_color='#C43836'  
    ), row=1, col=1)  

    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['hiAVWAP'],  
        mode='lines',  
        name='High AVWAP',  
        line=dict(color='rgba(255, 0, 0, 0.5)', width=1)  
    ), row=1, col=1)  

    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['loAVWAP'],  
        mode='lines',  
        name='Low AVWAP',  
        line=dict(color='rgba(0, 255, 0, 0.5)', width=1)  
    ), row=1, col=1)  

    # -------------------------  
    # 增加 5m 的 AVWAP 绘制：以散点图方式显示  
    # -------------------------  
    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['hiAVWAP_5m'],  
        mode='markers',  
        name='High AVWAP 5m',  
        marker=dict(symbol='circle', size=2, color='orange')  
    ), row=1, col=1)  

    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['loAVWAP_5m'],  
        mode='markers',  
        name='Low AVWAP 5m',  
        marker=dict(symbol='circle', size=2, color='blue')  
    ), row=1, col=1)  

    # -------------------------  
    # 增加 15m 的 AVWAP 绘制：以实体线显示，线宽是1m线路宽的 3 倍（即6）  
    # -------------------------  
    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['hiAVWAP_15m'],  
        mode='lines',  
        name='High AVWAP 15m',  
        line=dict(color='darkred', width=3)  
    ), row=1, col=1)  

    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['loAVWAP_15m'],  
        mode='lines',  
        name='Low AVWAP 15m',  
        line=dict(color='darkgreen', width=3)  
    ), row=1, col=1)  

    # 反转标记（向量化处理）  
    mask_up = df['reversal_arrow'].notna() & (df['reversal_arrow'] == 'up')  
    up_x = df.index[mask_up].tolist()  
    up_y = (df.loc[mask_up, 'low'] * 0.999).tolist()  

    mask_down = df['reversal_arrow'].notna() & (df['reversal_arrow'] == 'down')  
    down_x = df.index[mask_down].tolist()  
    down_y = (df.loc[mask_down, 'high'] * 1.001).tolist()  

    if up_x:  
        fig.add_trace(go.Scatter(  
            x=up_x,  
            y=up_y,  
            mode='markers',  
            marker=dict(symbol='triangle-up', size=8, color='green', line=dict(color='green', width=1)),  
            name='Bullish Reversal',  
            hoverinfo='none'  
        ), row=1, col=1)  

    if down_x:  
        fig.add_trace(go.Scatter(  
            x=down_x,  
            y=down_y,  
            mode='markers',  
            marker=dict(symbol='triangle-down', size=8, color='red', line=dict(color='red', width=1)),  
            name='Bearish Reversal',  
            hoverinfo='none'  
        ), row=1, col=1)  

    # 添加入场/出场信号注释  
    annotations = []  
    # long_entry 注解  
    mask = df['entry_signal'] == 'long_entry'  
    for dt, lo_avwap in zip(df.index[mask], df.loc[mask, 'loAVWAP']):  
        annotations.append(dict(  
            x=dt, y=lo_avwap * 0.998,  
            text="L+",  
            showarrow=True, arrowhead=2, arrowsize=1.5,  
            arrowcolor="green", ax=0, ay=30,  
            font=dict(color="green", size=14), xref="x", yref="y1"  
        ))  
    # short_entry 注解  
    mask = df['entry_signal'] == 'short_entry'  
    for dt, hi_avwap in zip(df.index[mask], df.loc[mask, 'hiAVWAP']):  
        annotations.append(dict(  
            x=dt, y=hi_avwap * 1.002,  
            text="S+",  
            showarrow=True, arrowhead=2, arrowsize=1.5,  
            arrowcolor="red", ax=0, ay=-30,  
            font=dict(color="red", size=14), xref="x", yref="y1"  
        ))  
    # long_exit 注解  
    mask = df['entry_signal'] == 'long_exit'  
    for dt, lo_avwap in zip(df.index[mask], df.loc[mask, 'loAVWAP']):  
        annotations.append(dict(  
            x=dt, y=lo_avwap * 0.998,  
            text="L-",  
            showarrow=True, arrowhead=2, arrowsize=1.5,  
            arrowcolor="blue", ax=0, ay=30,  
            font=dict(color="blue", size=14), xref="x", yref="y1"  
        ))  
    # short_exit 注解  
    mask = df['entry_signal'] == 'short_exit'  
    for dt, hi_avwap in zip(df.index[mask], df.loc[mask, 'hiAVWAP']):  
        annotations.append(dict(  
            x=dt, y=hi_avwap * 1.002,  
            text="S-",  
            showarrow=True, arrowhead=2, arrowsize=1.5,  
            arrowcolor="blue", ax=0, ay=-30,  
            font=dict(color="blue", size=14), xref="x", yref="y1"  
        ))  
    if annotations:  
        fig.update_layout(annotations=annotations)  

    # -------------------------  
    # 第二子图：MACD指标  
    # -------------------------  
    # MACD快线：蓝色细线  
    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['macd'],  
        mode='lines',  
        name='MACD Fast',  
        line=dict(color='blue', width=2)  
    ), row=2, col=1)  

    # MACD慢线：粗体橙色  
    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['macd_signal'],  
        mode='lines',  
        name='MACD Signal',  
        line=dict(color='orange', width=3)  
    ), row=2, col=1)  

    # 拆分 MACD Diff 为正负部分，以便分别着色  
    df['macd_diff_pos'] = df['macd_diff'].apply(lambda x: x if x >= 0 else None)  
    df['macd_diff_neg'] = df['macd_diff'].apply(lambda x: x if x < 0 else None)  

    # 正值部分（绿色条）  
    fig.add_trace(go.Bar(  
        x=df.index,  
        y=df['macd_diff_pos'],  
        name='MACD Diff Positive',  
        marker_color='green',  
        opacity=0.5  
    ), row=2, col=1)  
    # 负值部分（红色条）  
    fig.add_trace(go.Bar(  
        x=df.index,  
        y=df['macd_diff_neg'],  
        name='MACD Diff Negative',  
        marker_color='red',  
        opacity=0.5  
    ), row=2, col=1)  

    # 添加零轴参考线  
    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=[0] * len(df),  
        mode='lines',  
        line=dict(color='black', width=1, dash='dot'),  
        showlegend=False  
    ), row=2, col=1)  

    # -------------------------  
    # 第三子图：ADX及+DI、-DI  
    # -------------------------  
    # ADX指标（紫色线）  
    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['adx'],  
        mode='lines',  
        name='ADX',  
        line=dict(color='purple', width=2)  
    ), row=3, col=1)  
    # +DI 指标（绿色线），图例名称为 "+DI"  
    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['+di'],  
        mode='lines',  
        name='+DI',  
        line=dict(color='green', width=2)  
    ), row=3, col=1)  
    # -DI 指标（红色线）  
    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['-di'],  
        mode='lines',  
        name='-DI',  
        line=dict(color='red', width=2)  
    ), row=3, col=1)  

    # -------------------------  
    # 图表整体布局设置  
    # -------------------------  
    fig.update_layout(  
        title="Auto AVWAP Chart, MACD & ADX",  
        xaxis_title="Time",  
        yaxis_title="Price",  
        template="plotly_white",  
        xaxis_rangeslider_visible=False,  
        height=1600,  
        width=80000,  
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)  
    )  
    # 关闭所有子图的范围滑块  
    for i in range(1, 4):  
        fig.update_xaxes(rangeslider_visible=False, row=i, col=1)  

    # 图像保存代码（需要时启用）  
    # os.makedirs(output_dir, exist_ok=True)  
    # filename = f"{df.index[-1].strftime('%y-%m-%d_%H-%M-%S')}_avwap_macd_adx.png"  
    # filepath = os.path.join(output_dir, filename)  
    # try:  
    #     fig.write_image(filepath, scale=2)  
    #     print(f"Chart saved at: {os.path.abspath(filepath)}")  
    # except Exception as e:  
    #     print("保存图表图片失败：", e)  

    return fig  

# 工具函数：根据交易记录返回标记文本（例如 "b+" 或 "s-"）  
def get_trade_label(trade):  
    action = trade.get("action", "")  
    side = trade.get("side", "")  
    if "开仓" in action:  
        return "b+" if side == "long" else "s-"  
    elif "平" in action:  
        return "s+" if "多" in action else "b-"  
    else:  
        return ""  

# 使用 CSV 文件生成静态历史图  
if os.path.exists("btc_5m_avwap_result.csv"):  
    try:  
        df_hist = pd.read_csv("btc_5m_avwap_result.csv", parse_dates=True, index_col=0)  
        show_static_pic = 1  
        if show_static_pic == 1 and not df_hist.empty:  
            # 如有需要可截取最近部分数据：df_hist = df_hist.iloc[-1000:].copy()  
            static_history_fig = plot_avwap_chart(df_hist)  
        else:  
            static_history_fig = go.Figure()  
    except Exception as e:  
        print("读取或绘制历史数据失败：", e)  
        static_history_fig = go.Figure()  
else:  
    print("历史数据文件不存在：", "btc_5m_avwap_result.csv")  
    static_history_fig = go.Figure()  

def create_dash_app():  
    app = Dash(__name__)  
    app.layout = html.Div([  
        html.H2("历史 AVWAP & MACD 图（对比参考 - 动态叠加标记，60秒更新）"),  
        dcc.Graph(id="history-chart", figure=static_history_fig),  
        html.H2("实时资金曲线"),  
        dcc.Graph(id="equity-graph"),  
        html.H2("交易动作记录"),  
        dash_table.DataTable(  
            id="trade-table",  
            columns=[{"name": i, "id": i} for i in ["time", "action", "entry", "price", "margin", "pnl"]],  
            page_size=10  
        ),  
        html.H2("日志记录"),  
        dash_table.DataTable(  
            id="log-table",  
            columns=[{"name": i, "id": i} for i in ["time", "message"]],  
            page_size=10  
        ),  
        # 每10秒更新实时资金曲线、交易与日志  
        dcc.Interval(id="interval-component", interval=10 * 1000, n_intervals=0),  
        # 每60秒更新一次历史图（动态叠加部分）  
        dcc.Interval(id="history-interval", interval=600 * 1000, n_intervals=0)  
    ])  

    # 回调1：更新实时资金曲线及下方交易日志（10秒一次）  
    @app.callback(  
        [Output("equity-graph", "figure"),  
         Output("trade-table", "data"),  
         Output("log-table", "data")],  
        [Input("interval-component", "n_intervals")]  
    )  
    def update_realtime_components(n_intervals):  
        with data_lock:  
            # global_data["equity"] 已经以 DataFrame 形式记录  
            df_equity = global_data["equity"].copy()  
            trades_data = global_data["trades"].copy()  
            logs_data = global_data["logs"].copy()  
        if not df_equity.empty:  
            df_equity["time"] = pd.to_datetime(df_equity["time"])  
            df_equity.set_index("time", inplace=True)  
            fig_live = {  
                "data": [{  
                    "x": df_equity.index,  
                    "y": df_equity["equity"],  
                    "type": "line",  
                    "name": "Equity Curve"  
                }],  
                "layout": {"title": "实时资金曲线"}  
            }  
            latest_time = df_equity.index[-1]  
            fig_live["layout"]["shapes"] = [{  
                "type": "line",  
                "x0": latest_time,  
                "x1": latest_time,  
                "xref": "x",  
                "y0": 0,  
                "y1": 1,  
                "yref": "paper",  
                "line": {"color": "black", "width": 1, "dash": "dot"}  
            }]  
        else:  
            fig_live = {"data": [], "layout": {"title": "实时资金曲线"}}  
        return fig_live, trades_data, logs_data  

    # 回调2：更新历史图（每60秒一次）  
    @app.callback(  
        Output("history-chart", "figure"),  
        [Input("history-interval", "n_intervals")]  
    )  
    def update_history_chart(n_intervals):  
        updated_fig = static_history_fig.to_dict()  
        with data_lock:  
            df_equity = global_data["equity"].copy()  
        if not df_equity.empty:  
            df_equity["time"] = pd.to_datetime(df_equity["time"])  
            df_equity.set_index("time", inplace=True)  
            latest_time = df_equity.index[-1]  
            if "shapes" not in updated_fig["layout"]:  
                updated_fig["layout"]["shapes"] = []  
            updated_fig["layout"]["shapes"].append({  
                "type": "line",  
                "x0": latest_time,  
                "x1": latest_time,  
                "xref": "x",  
                "y0": 0,  
                "y1": 1,  
                "yref": "paper",  
                "line": {"color": "black", "width": 1, "dash": "dot"}  
            })  
        updated_fig["layout"]["annotations"] = updated_fig["layout"].get("annotations", [])  
        with data_lock:  
            for trade in global_data["trades"]:  
                label = get_trade_label(trade)  
                updated_fig["layout"]["annotations"].append({  
                    "x": trade.get("time"),  
                    "y": trade.get("entry"),  
                    "xref": "x",  
                    "yref": "y1",  
                    "text": label,  
                    "showarrow": True,  
                    "arrowhead": 2,  
                    "arrowcolor": "royalblue",  
                    "ax": 0,  
                    "ay": -30,  
                    "font": {"color": "royalblue", "size": 16}  
                })  
        return updated_fig  

    return app  

def run_dash():  
    app = create_dash_app()  
    app.run(debug=True)  

# 修改资金数据更新函数，使用 DataFrame 方式记录资金曲线  
def update_equity(dt, equity):  
    dt = pd.to_datetime(dt)  
    new_row = pd.DataFrame({"time": [dt], "equity": [equity]})  
    with data_lock:  
        global_data["equity"] = pd.concat([global_data["equity"], new_row], ignore_index=True)  

if __name__ == "__main__":  
    run_dash()  