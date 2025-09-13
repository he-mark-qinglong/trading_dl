import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
import pandas as pd  
import numpy as np  

def volume_profile(df: pd.DataFrame, n_bins: int = 10) -> dict:  
    """成交量分布"""  
    price_range = df['high'].max() - df['low'].min()  
    bin_size = price_range / n_bins  

    volume_profile = pd.Series(0.0, index=range(n_bins))  

    for i in range(len(df)):  
        bin_idx = int((df['close'][i] - df['low'].min()) / bin_size)  
        if bin_idx >= n_bins:  
            bin_idx = n_bins - 1  
        volume_profile[bin_idx] += df['volume'][i]  

    poc_idx = volume_profile.argmax()  
    poc_price = df['low'].min() + poc_idx * bin_size  

    return {  
        'volume_profile': volume_profile,  
        'poc_price': poc_price  
    }  

def plot_dashboard(df: pd.DataFrame):  
    """绘制技术分析仪表盘"""  
    # Volume Profile 计算  
    volume_info = volume_profile(df, n_bins=10)  
    volume_profile_data = volume_info['volume_profile']  
    poc_price = volume_info['poc_price']  

    # 创建子图，左侧为 Volume Profile，右侧为 K 线图  
    fig = make_subplots(  
        rows=1,  
        cols=2,  # 列数为2，一个为VP，一个为K线图  
        column_widths=[0.3, 0.7],  # 设置列宽，左侧 VP 占 30%  
        shared_yaxes=True  # 共享 Y 轴  
    )  

    # 绘制 Volume Profile  
    for i in range(len(volume_profile_data)):  
        price_level = df['low'].min() + i * (df['high'].max() - df['low'].min()) / len(volume_profile_data) + (df['high'].max() - df['low'].min()) / (2 * len(volume_profile_data))  
        
        fig.add_trace(go.Bar(  
            x=[volume_profile_data[i]],  # 成交量  
            y=[price_level],  # 价格作为 Y 轴  
            width=0.1,  # 条形图基本宽度  
            orientation='h',  # 水平方向  
            name='Volume Profile',  
            marker=dict(color='rgba(0, 0, 255, 0.5)')  
        ), row=1, col=1)  

    # 绘制 POC 线  
    fig.add_shape(  
        type="line",  
        x0=0,  
        y0=poc_price,  
        x1=volume_profile_data.max(),  
        y1=poc_price,  
        line=dict(color='rgba(255,0,0,0.8)', width=2, dash='dash'),  
        row=1,  
        col=1  
    )  
    fig.add_annotation(  
        x=volume_profile_data.max(),  
        y=poc_price,  
        text='POC',  
        showarrow=True,  
        arrowhead=2,  
        row=1,  
        col=1  
    )  

    # 绘制 K 线图  
    fig.add_trace(go.Candlestick(  
        x=df.index,  
        open=df['open'],  
        high=df['high'],  
        low=df['low'],  
        close=df['close'],  
        name='Price',  
        increasing_line_color='green',  
        decreasing_line_color='red'  
    ), row=1, col=2)  

    # 更新布局  
    fig.update_layout(title='Technical Analysis Dashboard',  
                      xaxis_title='Volume',  
                      yaxis_title='Price',  
                      xaxis2_title='Time',  # K线图的X轴标题  
                      showlegend=True)  

    # 显示图形  
    fig.show()  

# Sample Data for testing  
data = {  
    'open': [100, 102, 103, 101, 104],  
    'high': [104, 105, 106, 104, 107],  
    'low': [99, 100, 101, 100, 102],  
    'close': [103, 101, 105, 102, 106],  
    'volume': [1000, 1200, 1100, 1050, 1150]  
}  

df = pd.DataFrame(data)  
df.index = pd.date_range(start='2025-03-20', periods=len(df), freq='D')  

# 绘制仪表盘  
plot_dashboard(df)  