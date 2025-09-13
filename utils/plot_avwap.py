import os  
import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
import pandas as pd  

def plot_avwap_chart(df: pd.DataFrame, output_dir: str = "./charts_avwap") -> go.Figure:  
    """  
    绘制 AVWAP 图表及MACD指标：  
      - 上方子图展示价格与K线、AVWAP（高AVWAP和低AVWAP），并在反转位置使用散点图绘制三角形标记，  
        同时针对 entry 信号保留箭头注释（例如 long_entry、short_entry、long_exit、short_exit）  
      - 下方子图展示MACD因子：MACD、MACD Signal线及MACD Diff直方图  
    返回生成的 Figure 对象，方便后续 Dash 使用或保存图片。  
    """  
    # 创建上下两个子图，第一行为K线图，第二行为MACD图，x轴共享  
    fig = make_subplots(  
        rows=2, cols=1,  
        shared_xaxes=True,  
        vertical_spacing=0.03,  
        row_heights=[0.7, 0.3]  
    )  

    # -------------------  
    # 第一子图：K线+AVWAP  
    # -------------------  
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
        line=dict(color='rgba(255, 0, 0, 0.5)', width=2)  
    ), row=1, col=1)  

    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['loAVWAP'],  
        mode='lines',  
        name='Low AVWAP',  
        line=dict(color='rgba(0, 255, 0, 0.5)', width=2)  
    ), row=1, col=1)  

    # 添加反转标记，使用散点图绘制预先设置好的三角形  
    up_x, up_y, down_x, down_y = [], [], [], []  
    for dt, row in df.iterrows():  
        arrow = row.get('reversal_arrow', None)  
        if pd.notna(arrow):  
            if arrow == 'down':  
                down_x.append(dt)  
                down_y.append(row['high'] * 1.001)  
            elif arrow == 'up':  
                up_x.append(dt)  
                up_y.append(row['low'] * 0.999)  
    if up_x:  
        fig.add_trace(go.Scatter(  
            x=up_x,  
            y=up_y,  
            mode='markers',  
            marker=dict(  
                symbol='triangle-up',  
                size=8,        # 调整样式，确保足够醒目  
                color='green',  
                line=dict(color='green', width=1)  
            ),  
            name='Bullish Reversal',  
            hoverinfo='none'  
        ), row=1, col=1)  
    if down_x:  
        fig.add_trace(go.Scatter(  
            x=down_x,  
            y=down_y,  
            mode='markers',  
            marker=dict(  
                symbol='triangle-down',  
                size=8,  
                color='red',  
                line=dict(color='red', width=1)  
            ),  
            name='Bearish Reversal',  
            hoverinfo='none'  
        ), row=1, col=1)  

    # 移除反转信号的 add_annotation 部分，避免图片渲染问题  

    # 添加入场信号注释（entry 箭头保留）  
    for dt, row in df.iterrows():  
        entry = row.get('entry_signal', None)  
        if pd.notna(entry):  
            if entry == 'long_entry':  
                y_val = row['loAVWAP'] * 0.998  
                text = "Long Entry"  
                entry_color = "green"  
                ay_offset = 30  
            elif entry == 'short_entry':  
                y_val = row['hiAVWAP'] * 1.002  
                text = "Short Entry"  
                entry_color = "red"  
                ay_offset = -30  
            elif entry == 'long_exit':  
                y_val = row['loAVWAP'] * 0.998  
                text = "Long Exit"  
                entry_color = "blue"  
                ay_offset = 30  
            elif entry == 'short_exit':  
                y_val = row['hiAVWAP'] * 1.002  
                text = "Short Exit"  
                entry_color = "blue"  
                ay_offset = -30  
            else:  
                continue  
            fig.add_annotation(  
                x=dt,  
                y=y_val,  
                text=text,  
                showarrow=True,  
                arrowhead=2,  
                arrowsize=1.5,  
                arrowcolor=entry_color,  
                ax=0,  
                ay=ay_offset,  
                font=dict(color=entry_color, size=14),  
                xref="x",  
                yref="y1"  
            )  

    # -------------------  
    # 第二子图：MACD指标  
    # -------------------  
    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['macd'],  
        mode='lines',  
        name='MACD',  
        line=dict(color='blue', width=1)  
    ), row=2, col=1)  

    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=df['macd_signal'],  
        mode='lines',  
        name='MACD Signal',  
        line=dict(color='orange', width=1)  
    ), row=2, col=1)  

    # MACD Diff直方图  
    fig.add_trace(go.Bar(  
        x=df.index,  
        y=df['macd_diff'],  
        name='MACD Diff',  
        marker_color='grey',  
        opacity=0.5  
    ), row=2, col=1)  

    # 添加零线参考  
    fig.add_trace(go.Scatter(  
        x=df.index,  
        y=[0] * len(df),  
        mode='lines',  
        line=dict(color='black', width=1, dash='dot'),  
        showlegend=False  
    ), row=2, col=1)  

    # 总体图表设置  
    fig.update_layout(  
        title="Auto AVWAP Chart & MACD",  
        xaxis_title="Time",  
        yaxis_title="Price",  
        template="plotly_white",  
        xaxis_rangeslider_visible=False,  
        height=900,  
        width=3200,  
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)  
    )  
    fig.update_xaxes(rangeslider_visible=False)  

    # 确保输出路径存在，并保存图片  
    os.makedirs(output_dir, exist_ok=True)  
    filename = f"{df.index[-1].strftime('%y-%m-%d_%H-%M-%S')}_avwap_macd.png"  
    filepath = os.path.join(output_dir, filename)  
    # 保存文件（如有问题可先使用 fig.show() 检查图表）  
    fig.write_image(filepath, scale=2)  
    print(f"AVWAP & MACD chart saved at: {os.path.abspath(filepath)}")  
    return fig  

# 示例用法：  
if __name__ == '__main__':  
    data_file = "btc_5m_avwap_result.csv"  # 请替换为实际数据路径  
    df = pd.read_csv(data_file, parse_dates=True, index_col=0)  
    fig = plot_avwap_chart(df.iloc[-1000:].copy())  

    # 以下为 Dash 展示代码，可按需解注释使用 
    # ''' 
    from dash import Dash, dcc, html  
    app = Dash(__name__)  
    app.layout = html.Div([  
        dcc.Graph(figure=fig)  
    ])  
    app.run_server(debug=True, port=8052)  
    # '''