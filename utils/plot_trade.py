import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
import pandas as pd  
import numpy as np  
import os  

def plot_vpvr(vpvr_dict: dict, output_dir: str = "./charts_vpvr") -> None:  
    """  
    绘制 VPVR 图表并保存  
    :param vpvr_dict: VPVR 数据字典  
    :param output_dir: 输出目录  
    """  
    fig = go.Figure()  

    # 绘制买入成交量  
    fig.add_trace(go.Bar(  
        y=vpvr_dict['Price'],  
        x= -vpvr_dict['Buy_Volume']-vpvr_dict['Sell_Volume'] ,  
        orientation='h',  
        name='Buy v',  
        marker_color='green'  
    ))  

    # 绘制卖出成交量（为负值，位于买入上方）  
    fig.add_trace(go.Bar(  
        y=vpvr_dict['Price'],  
        x=-vpvr_dict['Sell_Volume'],  
        orientation='h',  
        name='Sell v',  
        marker_color='red'  
    ))  

    # 绘制delta成交量  
    fig.add_trace(go.Bar(  
        y=vpvr_dict['Price'],  
        x= -(vpvr_dict['Buy_Volume']-vpvr_dict['Sell_Volume']),  
        orientation='h',  
        name='Buy-Sell v',  
        marker_color='blue'  
    ))  

    # 更新布局  
    fig.update_layout(  
        title='Volume Profile Visible Range (VPVR)',  
        xaxis_title='Volume',  
        yaxis_title='Price',  
        barmode='overlay'  # 重叠模式  
    )  
    
    os.makedirs(output_dir, exist_ok=True)  
    filename = f"{vpvr_dict['twvwap'].index[-1].strftime('%y-%m-%d %H:%M:%S')}.png"   
    filepath = os.path.join(output_dir, filename)  
    fig.write_image(filepath)
    # print(f"VPVR chart saved at: {filepath}")  

def plot_flex_ta_dashboard(  
    df: pd.DataFrame,  
    factors: dict,  
    open_price:float = np.nan,
    stop_loss = np.nan,
    stop_profit = np.nan,
    additional: str = 'original',  
    condition: pd.Series = None,  
    output_dir: str = "./charts",  
    layout_config: dict = None,  
    factor_map: dict = None  
) -> str:  
    """  
    动态生成多因子技术分析仪表盘并保存为图片  
    
    Parameters:  
    -----------  
    df : pd.DataFrame  
        包含OHLCV数据，索引需为时间类型  
    factors : dict  
        技术因子字典，格式为 {因子名称: pd.Series}  
    additional : str  
        用于文件命名标记是原始数据还是实时数据  
    condition : pd.Series, optional  
        条件标记序列，布尔类型  
    output_dir : str  
        输出目录路径  
    layout_config : dict  
        布局配置参数，包含：  
        - main_height: 主图高度比例 (默认0.55)  
        - sub_heights: 子图高度列表 (默认[0.15,0.15,0.15])  
        - colors: 颜色配置字典  
    factor_map : dict  
        因子位置映射表，格式为 {因子名称: 行位置}  
    
    Returns:  
    --------  
    str: 生成的图片文件路径  
    """  

    plot_vpvr(factors)

    # 参数校验与默认值  
    required_cols = ['open', 'high', 'low', 'close']  
    missing = set(required_cols) - set(df.columns)  
    if missing:  
        raise ValueError(f"缺失必要列: {missing}")  
    
    # 合并因子数据  
    for factor_name, factor_series in factors.items():  
        if factor_name not in df.columns and factor_name not in ['Price','Buy_Volume', 'Sell_Volume', 'Total_Volume',]:  
            df[factor_name] = factor_series  
    
    # 布局配置  
    default_layout = {  
        'main_height': 0.55,  
        'sub_heights': [0.15, 0.15, 0.15],  
        'colors': {  
            'bull': '#2E8540',  
            'bear': '#C43836',  
            'condition': 'rgba(255, 144, 14, 0.5)',  
            # VWAP相关指标的特定颜色  
            'twvwap': '#FF0000',                    # 红色  
            'twvwap_std_dev': '#4169E1',            # 皇家蓝  
            'twvwap_upper_band_1': '#FF7F50',       # 珊瑚色  
            'twvwap_lower_band_1': '#6495ED',       # 矢车菊蓝  
            'twvwap_upper_band_2': '#FF6347',       # 番茄色  
            'twvwap_lower_band_2': '#2E8B57',       # 海洋绿  

            # 补充的颜色方案：  
            'twvwap_upper_band_15': '#8A2BE2',      # 紫罗兰色  
            'twvwap_lower_band_15': '#4B0082',      # 靛蓝色  
            'twvwap_upper_band_05': '#00CED1',      # 深蓝绿色  
            'twvwap_lower_band_05': '#20B2AA',      # 浅海蓝  
            'twvwap_upper_band_40': '#FFD700',      # 金色  
            'twvwap_lower_band_40': '#C0C0C0',      # 银色 

            'avwap': '#9932CC',                     # 深紫色  
            'vwap': '#0000FF',                      # 蓝色  
            'twvwap_smooth': '#4CAF90',             # 绿色  
            'twvwap_deviation': '#DA70D6',          # 兰花紫  
            # ADX指标颜色  
            'adx': '#FF9800',                       # 橙色  
            '+di': '#4CAF50',                       # 绿色  
            '-di': '#F44336',                       # 红色  
            # StochRSI指标颜色  
            'stoch_k': '#1E90FF',                   # 道奇蓝  
            'stoch_d': '#FFA500',                    # 橙色  

            'volume_z_score': '#6A5ACD',  
            'zscore_mean': '#FF6347',  
        }  
    }  
    
    if layout_config and 'colors' in layout_config:  
        default_layout['colors'].update(layout_config['colors'])  
        layout_config.pop('colors')  
        
    layout = {**default_layout, **(layout_config or {})}  
    
    # 动态生成子图布局  
    rows = 1 + len(layout['sub_heights'])  
    fig = make_subplots(  
        rows=rows,  
        cols=1,  
        shared_xaxes=True,  
        vertical_spacing=0.03,  
        row_heights=[layout['main_height']] + layout['sub_heights']  
    )  
    
    # 主图绘制  
    _add_main_chart(fig, df, layout['colors'], condition, factors, open_price, stop_loss, stop_profit)
    
    # 动态添加技术指标  
    default_factor_map = {  
        'adx': 2, '+di': 2, '-di': 2,  
        'stoch_k': 3, 'stoch_d': 3,  
        'twvwap_deviation': 4, 'twvwap_std_dev': 4  
    }  
    factor_map = {**default_factor_map, **(factor_map or {})}  
    
    for factor, series in factors.items():  
        row = factor_map.get(factor)  
        if row and row <= rows:  
            _add_factor_trace(fig, df, factor, series, row, layout['colors'])  
    
    # 布局优化  
    _apply_global_layout(fig, df, layout['colors'], stop_loss=stop_loss, stop_profit=stop_profit)  
    
    # 文件保存  
    os.makedirs(output_dir, exist_ok=True)  
    filename = f"{factors['twvwap'].index[-1].strftime('%y-%m-%d %H:%M:%S')}_{additional}.png"   
    filepath = os.path.join(output_dir, filename)  
    fig.write_image(filepath)  
    
    return os.path.abspath(filepath)  
def _add_main_chart(fig, df, colors, condition, factors, open_price=np.nan, stop_loss=np.nan, stop_profit=np.nan):  
    """主图区域绘制"""  
    
    # 提取最后一个信息的时间戳（基于twvwap）  
    latest_time = factors['twvwap'].index[-1]  
    
    # 添加一根垂直虚线，标识最新数据位置  
    fig.add_shape(  
        type="line",  
        x0=latest_time,  
        y0=0,  
        x1=latest_time,  
        y1=1,  
        xref="x",  
        yref="paper",  
        line=dict(  
            color="black",  
            width=2,  
            dash="dash"  
        )  
    )  

    # 蜡烛图  
    fig.add_trace(go.Candlestick(  
        x=df.index,  
        open=df['open'],  
        high=df['high'],  
        low=df['low'],  
        close=df['close'],  
        name='Price',  
        increasing_line_color=colors['bull'],  
        decreasing_line_color=colors['bear']  
    ), row=1, col=1)  
    
    
    # 添加止损止盈线
    if not np.isnan(stop_loss):  
        fig.add_trace(go.Scatter(  
            x=[df.index.min(), df.index.max()],  
            y=[stop_loss, stop_loss],  
            mode='lines',  
            line=dict(color='#FF1493', width=2, dash='dot'),  
            name='Stop Loss'  
        ), row=1, col=1)  
        
        # 右侧标注  
        fig.add_annotation(  
            x=df.index[-1],  
            y=stop_loss,  
            xanchor="left",  
            yanchor="middle",  
            text=f"SL: {stop_loss:.2f}",  
            font=dict(color='#FF1493', size=10),  
            showarrow=False,  
            row=1,  
            col=1  
        )  

    # 添加开仓价
    if not np.isnan(open_price):  
        fig.add_trace(go.Scatter(  
            x=[df.index.min(), df.index.max()],  
            y=[open_price, open_price],  
            mode='lines',  
            line=dict(color='#881493', width=4, dash='dot'),  
            name='Open Price'  
        ), row=1, col=1)  
        
        # 右侧标注  
        fig.add_annotation(  
            x=df.index[-1],  
            y=open_price,  
            xanchor="left",  
            yanchor="middle",  
            text=f"Entry: {open_price:.2f}",  
            font=dict(color='#881493', size=10),  
            showarrow=False,  
            row=1,  
            col=1  
        )  

    if not np.isnan(stop_profit):  
        fig.add_trace(go.Scatter(  
            x=[df.index.min(), df.index.max()],  
            y=[stop_profit, stop_profit],  
            mode='lines',  
            line=dict(color='#00BFFF', width=2, dash='dot'),  
            name='Take Profit'  
        ), row=1, col=1)  
        
        # 右侧标注  
        fig.add_annotation(  
            x=df.index[-1],  
            y=stop_profit,  
            xanchor="left",  
            yanchor="middle",  
            text=f"TP: {stop_profit:.2f}",  
            font=dict(color='#00BFFF', size=10),  
            showarrow=False,  
            row=1,  
            col=1  
        )  
    # 添加VWAP相关指标  
    # 按特定顺序添加，确保视觉上的层次感  
    vwap_layers = [  
        'twvwap_lower_band_2', 'twvwap_upper_band_2',  
        'twvwap_lower_band_1', 'twvwap_upper_band_1',  

        'twvwap_lower_band_05', 'twvwap_upper_band_05',  
        'twvwap_lower_band_15', 'twvwap_upper_band_15',  
        'twvwap_lower_band_40', 'twvwap_upper_band_40', 

        'vwap', 'avwap', 'twvwap_smooth', 'twvwap'  
    ]  
  
    for factor_name in vwap_layers:  
        if factor_name in df.columns:  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=df[factor_name],  
                name=factor_name,  
                line=dict(  
                    color=colors.get(factor_name, '#888888'),  
                    width=2.5 if factor_name in ['twvwap', 'avwap', 'vwap', 'twvwap_smooth'] else 1,  
                    dash='solid'  
                )  
            ), row=1, col=1)  
            # 在主图右侧添加文本注解，标识该指标  
            fig.add_annotation(  
                x=df.index[-1],  
                y=df[factor_name].iloc[-1],  
                xanchor="left",  
                yanchor="middle",  
                text=f"{factor_name}",  
                font=dict(color=colors.get(factor_name, '#888888')),  
                showarrow=False,  
                row=1,  
                col=1  
            )  

    # volume_profile = factors['volume_profile']  
    # poc_price = factors['poc_price']  
    
    # # 绘制 Volume Profile  
    # for i in range(len(volume_profile)):  
    #     price_level = df['low'].min() + i * (df['high'].max() - df['low'].min()) / 10  # 根据箱体大小用价格刻度  
    #     fig.add_trace(go.Bar(  
    #         x=[df.index[-1]],  # 设置X轴的位置  
    #         y=[volume_profile[i]],  # Volume Profile 的高度  
    #         width=0.1,  # 调整宽度  
    #         name='Volume Profile',  
    #         marker_color='rgba(0, 0, 255, 0.5)',  # 背景色  
    #         orientation='v'  # 竖直方向  
    #     ), row=1, col=1)  

    # # 绘制 POC 线  
    # fig.add_shape(  
    #     type="line",  
    #     x0=df.index[0],  
    #     y0=poc_price,  
    #     x1=df.index[-1],  
    #     y1=poc_price,  
    #     line=dict(color='rgba(255,0,0,0.8)', width=2, dash='dash'),   
    #     name='POC'  
    # )  
    # fig.add_annotation(  
    #     x=df.index[-1],  
    #     y=poc_price,  
    #     text='POC',  
    #     showarrow=True,  
    #     arrowhead=2  
    # )  

  
    # 条件标记  
    if condition is not None:  
        valid_condition = condition.reindex(df.index)  
        if valid_condition.any():  
            fig.add_trace(go.Scatter(  
                x=df.index[valid_condition],  
                y=df.loc[valid_condition, 'close'],  
                mode='markers',  
                marker=dict(color=colors['condition'], size=8, symbol='circle'),  
                name='Signal'  
            ), row=1, col=1)  
  
def _add_factor_trace(fig, df, factor, series, row, colors):  
    """动态添加技术指标"""  
    color = colors.get(factor, '#1F77B4')  # 使用默认颜色如果没有特定颜色  
  
    if factor.startswith('stoch'):  
        line_color = colors.get(factor, '#FF7F0E' if 'd' in factor else '#1F77B4')  
        fig.add_trace(go.Scatter(  
            x=df.index,  
            y=series,  
            line=dict(color=line_color, width=1.5),  
            name=factor  
        ), row=row, col=1)  
        if '_k' in factor:  
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=row, col=1)  
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=row, col=1)  
    elif factor in ['adx', '+di', '-di']:  
        fig.add_trace(go.Scatter(  
            x=df.index,  
            y=series,  
            line=dict(color=color, width=1.5),  
            name=factor,  
            hovertemplate=f"<b>{factor}</b><br>时间: %{{x}}<br>值: %{{y:.2f}}<extra></extra>"  
        ), row=row, col=1)  
    elif factor in ['twvwap_deviation', 'twvwap_std_dev']:  
        fig.add_trace(go.Scatter(  
            x=df.index,  
            y=series,  
            line=dict(color=color, width=1.5),  
            name=factor,  
            fill=None  
        ), row=row, col=1)  
    else:  
        fig.add_trace(go.Scatter(  
            x=df.index,  
            y=series,  
            line=dict(color=color, width=1.5),  
            name=factor  
        ), row=row, col=1)  
  
def _apply_global_layout(fig, df, colors, stop_loss, stop_profit):  
    """统一布局样式"""  
    # 计算主图上所有数据的全局最小值和最大值  
    vwap_layers = [  
        'twvwap_lower_band_2', 'twvwap_upper_band_2',  
        'twvwap_lower_band_1', 'twvwap_upper_band_1',  

        'twvwap_lower_band_05', 'twvwap_upper_band_05',  
        'twvwap_lower_band_15', 'twvwap_upper_band_15',  
        'twvwap_lower_band_40', 'twvwap_upper_band_40',  

        'vwap', 'avwap', 'twvwap_smooth', 'twvwap'  
    ]  
    data_mins = [df['low'].min()]  
    data_maxs = [df['high'].max()]  
    for factor in vwap_layers:  
        if factor in df.columns:  
            data_mins.append(df[factor].min())  
            data_maxs.append(df[factor].max())  
    global_low = min([min(data_mins), stop_loss, stop_profit])
    global_high = max([max(data_maxs), stop_loss, stop_profit])  
    price_margin = (global_high - global_low) * 0.05  
    
    fig.update_layout(  
        title=f"Technical Dashboard - {df.index[0]} to {df.index[-1]}",  
        plot_bgcolor='white',  
        paper_bgcolor='white',  
        hovermode='x unified',  
        height=1000,  
        width=1600,  
        showlegend=True,  
        legend=dict(  
            yanchor="top",  
            y=0.99,  
            xanchor="left",  
            x=0.01,  
            bgcolor="rgba(255,255,255,0.8)"  
        ),  
        xaxis_rangeslider_visible=False  
    )  
  
    # 主图Y轴动态范围：根据价格和VWAP指标共同计算  
    fig.update_yaxes(  
        title="Price",  
        range=[global_low - price_margin, global_high + price_margin],  
        row=1, col=1,  
        gridcolor='lightgray'  
    )  
  
    # 设置子图标题和网格  
    sub_titles = {  
        2: "ADX / DI",  
        3: "StochRSI",  
        4: "TWVWAP Deviation"  
    }  
    for i in range(1, len(fig.layout.annotations) + 1):  
        if i in sub_titles:  
            fig.update_yaxes(  
                title=sub_titles[i],  
                row=i, col=1,  
                gridcolor='lightgray'  
            )  
  
    # 美化X轴  
    fig.update_xaxes(  
        gridcolor='lightgray',  
        showspikes=True,  
        spikethickness=1,  
        spikedash="solid",  
        spikecolor="#999999",  
        spikemode="across"  
    )  
  
    # 统一所有子图的样式  
    for i in range(1, 5):  
        if i <= len(fig.data):  
            fig.update_yaxes(showspikes=True, row=i, col=1)  
            fig.update_xaxes(showspikes=True, row=i, col=1)  