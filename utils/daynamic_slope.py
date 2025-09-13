import numpy as np  
import pandas as pd  
import plotly.graph_objects as go  
from plotly.subplots import make_subplots  

def rolling_slope(series: pd.Series, window: int) -> pd.Series:  
    """  
    计算时间序列的滚动窗口斜率（向量化实现）  
    
    参数：  
      series: pd.Series - 时间序列数据，索引需为时间类型  
      window: int - 滚动窗口长度  
    
    返回：  
      pd.Series - 各时间点的斜率值  
    """  
    n = window  
    x = np.arange(n)  # 生成0到n-1的x值  
    
    # 预计算固定值  
    sum_x = x.sum()  
    sum_x2 = (x**2).sum()  
    denom = n * sum_x2 - sum_x**2  
    
    # 计算滚动统计量  
    sum_y = series.rolling(window, min_periods=2).sum()  
    sum_xy = series.rolling(window, min_periods=2).apply(  
        lambda s: np.dot(s, x[-len(s):]),  
        raw=True  
    )  
    
    # 斜率公式  
    slope = (n * sum_xy - sum_x * sum_y) / denom  
    return slope  

def dynamic_window_slope(series: pd.Series, volatility_window: int = 50) -> pd.Series:  
    """  
    根据波动率自动调整窗口长度：  
    - 高波动率时使用小窗口（快速响应）  
    - 低波动率时使用大窗口（过滤噪声）  
    """  
    # 计算波动率（ATR）  
    atr = series.diff().abs().rolling(volatility_window).mean()  
    
    # 动态窗口规则  
    window_sizes = pd.cut(  
        atr,   
        bins=[0, 0.5, 1.0, 2.0, np.inf],  
        labels=[30, 20, 15, 10]  
    ).fillna(20).astype(int)  
    
    # 为每个时点计算对应窗口的斜率  
    slopes = pd.Series(  
        [rolling_slope(series[:i+1], w) for i, w in window_sizes.items()],  
        index=series.index  
    )  
    
    return slopes  

def enhanced_slope(series: pd.Series, window: int) -> pd.DataFrame:  
    """  
    返回包含斜率和趋势强度的DataFrame  
    
    输出列：  
      - slope: 线性回归斜率  
      - strength: R²决定系数（趋势强度）  
    """  
    n = window  
    x = np.arange(n)  
    x_mean = x.mean()  
    # 使用rolling函数时指定min_periods避免起始值问题  
    y_mean = series.rolling(window, min_periods=2).mean()  
    
    # 计算分子：协方差  
    numerator = series.rolling(window, min_periods=2).apply(  
        lambda s: ((x[-len(s):] - x_mean) * (s - s.mean())).sum(),  
        raw=True  
    )  
    
    # 分母：x的平方和  
    denominator = ((x - x_mean)**2).sum()  
    
    # 计算斜率  
    slope = numerator / denominator  
    
    # 计算残差平方和 ss_res  
    ss_res = series.rolling(window, min_periods=2).apply(  
        lambda s: ((s - s.mean())**2).sum(),  
        raw=True  
    )  
    # 总平方和 ss_tot: 注意这里用斜率还原的部分和实际残差的和  
    ss_tot = ss_res + (slope**2) * denominator  
    r_squared = 1 - ss_res / ss_tot  
    
    return pd.DataFrame({'slope': slope, 'strength': r_squared})  

def plot_slope_analysis(series, window=20):  
    """  
    绘制价格曲线、斜率热力图及高对比度趋势标记，背景均设为透明  
    """  
    analysis = enhanced_slope(series, window)  
    print(f"斜率数据统计:\n{analysis['slope'].describe()}")  # 调试输出  
    
    # 创建2行布局：上行为价格曲线及趋势标记，下行为热力图  
    fig = make_subplots(  
        rows=2, cols=1,  
        shared_xaxes=True,  
        vertical_spacing=0.05,  
        row_heights=[0.7, 0.3]  
    )  
    
    # 价格曲线（亮青色）  
    fig.add_trace(go.Scatter(  
        x=series.index, y=series,  
        name='Price',  
        line=dict(color='#00ffff', width=2),  
        marker=dict(color='#00ffff', size=4),  
        hovertemplate="%{x}<br>Price: %{y:.2f}<extra></extra>"  
    ), row=1, col=1)  
    
    # 趋势标记（高对比度），筛选趋势强度较高的点  
    strong_points = analysis[analysis['strength'] > 0.7]  
    fig.add_trace(go.Scatter(  
        x=strong_points.index,  
        y=series.loc[strong_points.index],  
        mode='markers',  
        marker=dict(  
            color=np.where(strong_points['slope'] > 0, '#ff00ff', '#ffff00'),  
            size=12,  
            line=dict(width=2, color='black')  
        ),  
        name='Trend Markers',  
        hovertemplate="%{x}<br>Slope: %{text[0]:.4f}<br>Strength: %{text[1]:.2f}<extra></extra>",  
        text=np.column_stack([strong_points['slope'], strong_points['strength']])  
    ), row=1, col=1)  
    
    # 热力图（红蓝高对比），Y轴固定为空白使其只显示颜色条效果  
    fig.add_trace(go.Heatmap(  
        x=analysis.index,  
        y=[''] * len(analysis),  
        z=[analysis['slope'].values],  
        colorscale=[[0, 'rgba(255,0,0,0.8)'], [1, 'rgba(0,0,255,0.8)']],  
        zmid=0,  
        showscale=True,  
        colorbar=dict(title='Slope'),  
        hoverinfo='x+z'  
    ), row=2, col=1)  
    
    # 全透明背景及高对比度设置  
    fig.update_layout(  
        template=None,  
        paper_bgcolor='rgba(0,0,0,0)',  
        plot_bgcolor='rgba(0,0,0,0)',  
        xaxis=dict(showgrid=False),  
        yaxis=dict(showgrid=False),  
        margin=dict(t=40, b=20),  
        font=dict(color='black')  
    )  
    
    # 调试用辅助线：以平均价格作为基准线  
    fig.add_shape(type="line",  
                  x0=series.index[0], y0=series.mean(),  
                  x1=series.index[-1], y1=series.mean(),  
                  line=dict(color="black", width=3, dash="dot"),  
                  xref='x', yref='y',  
                  row=1, col=1)  
    
    fig.show()  

# 示例：构造测试数据并绘图  
if __name__ == "__main__":  
    # 生成测试数据：模拟价格收盘价序列  
    test_series = pd.Series(  
        data=np.random.randn(1000).cumsum(),  # 随机累积和模拟价格走势  
        index=pd.date_range('2024-01-01', periods=1000, freq='5min'),  
        name='close'  
    )  
    
    # 绘制趋势分析图，传入的是Series类型数据（例如df['close']）  
    plot_slope_analysis(test_series, window=50)  