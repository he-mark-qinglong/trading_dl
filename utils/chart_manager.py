from plotly.subplots import make_subplots  
import plotly.graph_objects as go  
import numpy as np  
import pandas as pd
from factors import FactorManager
from factors.back_chart_manager_funcs import UnUsedFactorManager

class ChartManager(UnUsedFactorManager):  
    def __init__(self, colors, factor_manager:FactorManager):  
        super().__init__()
        """  
        初始化 ChartManager，负责生成图表。  
        :param colors: 颜色配置字典  
        :param factor_manager: FactorManager 实例，用于计算因子  
        """  
        self.colors = colors  
        self.factor_manager = factor_manager  
        self.bin_count = 50  # 价格分箱数量

    def create_empty_figure(self):  
        """  
        创建一个空图表作为占位。  
        """  
        return {  
            'data': [],  
            'layout': {  
                'xaxis': {'showgrid': False, 'zeroline': False, 'visible': False},  
                'yaxis': {'showgrid': False, 'zeroline': False, 'visible': False},  
                'annotations': [  
                    {  
                        'text': "No Data Available",  
                        'xref': "paper",  
                        'yref': "paper",  
                        'showarrow': False,  
                        'font': {'size': 20, 'color': 'gray'},  
                        'x': 0.5,  
                        'y': 0.5,  
                        'xanchor': 'center',  
                        'yanchor': 'middle'  
                    }  
                ]  
            }  
        }  
    def _add_volume_profile(self, fig, df, factors, row=1, col=2):
        """修复索引问题并增加T-VWAP斜率判断的成交量分布计算"""
        # 生成价格分箱（保留原始索引）
        price_bins = np.linspace(df['low'].min(), df['high'].max(), self.bin_count)
        df['price_bin'] = pd.cut(df['close'], bins=price_bins, labels=price_bins[:-1])
        
        df_len = len(df)  
        # 生成位置索引（关键修复）
        df_copy = df.reset_index(drop=True)  # 创建副本，确保索引是整数序列
        weights = np.ones(df_len)  
        decay_factor_normal = np.exp(-self.factor_manager.config.tau)  
        decay_factor_abnormal = np.exp(-self.factor_manager.config.tau * 2)  
        decay_factors = np.where(self.factor_manager.config.abnormal_flags.values, decay_factor_abnormal, decay_factor_normal)  
        for i in range(1, df_len):  
            weights[:i] *= decay_factors[i-1]  

        # 使用位置索引进行分组计算
        volume_dist = df_copy.groupby('price_bin', observed=False).apply(
            lambda x: np.sum(x['volume'] * weights[x.index.get_level_values(0)])
        ).reset_index(name='volume')
        
        # 计算T-VWAP和Pure VWAP的斜率
        vwap_slope = factors['twvwap_slope']
    
        vwap_values = factors['twvwap']
        vwap_label = 'T-VWAP'
        vwap_color = 'purple'
        
        # 创建水平条形图（保持原有样式）
        fig.add_trace(go.Bar(
            x=volume_dist['volume'],
            y=volume_dist['price_bin'].astype(float),
            orientation='h',
            name='Volume Profile',
            marker_color='rgba(100, 149, 237, 0.6)',
            hoverinfo='y+text',
            hovertext=[f'Vol: {v:.0f}' for v in volume_dist['volume']]
        ), row=row, col=col)
        
        # 添加选择的VWAP线
        fig.add_trace(go.Scatter(
            x=[max(volume_dist['volume'])] * len(vwap_values),
            y=vwap_values,
            mode='lines',
            name=vwap_label,
            line=dict(color=vwap_color, width=2),
            hoverinfo='y+text',
            hovertext=[f'{vwap_label}: {v:.2f}' for v in vwap_values]
        ), row=row, col=col)
        
        # 同步Y轴范围（保持原有逻辑）
        y_min = factors['twvwap'].min()*0.995
        y_max = factors['twvwap'].max()*1.005
        fig.update_yaxes(range=[y_min, y_max], row=row, col=col)

        # 样式调整
        fig.update_xaxes(
            title="Volume",
            row=row, col=col,
            showgrid=False,
            showticklabels=False
        )
        fig.update_yaxes(
            row=row, col=col,
            showgrid=False,
            showticklabels=False,
            scaleanchor='y',  # 同步主图Y轴
            scaleratio=1
        )

    def create_candlestick_figure(self, df, factors, timeframe):
        """创建增强版图表"""
        fig = self._create_subplots()
        
        # 主图区域（左）
        self._add_candlestick(fig, df)
        self._add_bollinger_bands(fig, df, factors)
        #self._add_supertrend(fig, df, factors)
        self._add_weighted_twvwap(fig, df, factors)
        
        # 成交量分布图（右）
        #self._add_volume_profile(fig, df, factors)
        
        # 其他指标区域
        self._add_adx(fig, df, factors)
        self._add_stochrsi(fig, df, factors)
        self._add_volatility_twvwap_deviation(fig, df, factors)
        
        # 风险信号区域  
        # self._add_risk_signals(fig, df, factors)

        # 布局更新
        self._update_layout(fig, timeframe, df)
        return fig


    def _create_subplots(self):  
        """创建包含成交量分布图以及风险信号区域的多列布局，拓展至5行"""  
        return make_subplots(  
            rows=5, cols=2,  
            column_widths=[0.9, 0.1],  
            row_heights=[0.4, 0.2, 0.2, 0.2, 0.2],  
            specs=[  
                [{"secondary_y": True}, {"type": "bar"}],  # 第1行  
                [{"colspan": 2}, None],                    # 第2行  
                [{"colspan": 2}, None],                    # 第3行  
                [{"colspan": 2}, None],                    # 第4行  
                [{"colspan": 2}, None]                     # 第5行（风险信号区域）  
            ],  
            vertical_spacing=0.03,  
            horizontal_spacing=0.03,  
            shared_xaxes=True  
        )  

    def _add_candlestick(self, fig, df):  
        """  
        添加 K 线（蜡烛图）到第一行子图  
        """  
        fig.add_trace(go.Candlestick(  
            x=df.index,  
            open=df['open'],  
            high=df['high'],  
            low=df['low'],  
            close=df['close'],  
            name='OHLC'  
        ), row=1, col=1)  

    def _add_weighted_twvwap(self, fig, df, factors):  
        """  
        添加 TVWAP 及其偏离指标到第一行子图  
        因子中需要包含 'twvwap_smooth' 与 'twvwap'  
        """  
        if all(k in factors for k in ['twvwap_smooth', 'twvwap']):  
            
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['twvwap_lower_band_2'],  
                name='twvwap_lower_band_2',  
                line=dict(color='mediumseagreen', width=1)  
            ), row=1, col=1)  

            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['twvwap_upper_band_1'],  
                name='twvwap_upper_band_1',  
                line=dict(color='coral', width=1)  
            ), row=1, col=1)  
            
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['twvwap'],  
                name='TVWAP',  
                line=dict(color='red', width=2)  
            ), row=1, col=1)  

            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['twvwap_smooth'],  
                name='TVWAP SMOOTH',  
                line=dict(color='purple', width=2)  
            ), row=1, col=1)  
            
            # fig.add_trace(go.Scatter(  
            #     x=df.index,  
            #     y=factors['vwap'],  
            #     name='pure VWAP',  
            #     line=dict(color='blue', width=2)  
            # ), row=1, col=1)  

            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['avwap'],  
                name='pure AVWAP',  
                line=dict(color='green', width=2)  
            ), row=1, col=1) 

            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['twvwap_lower_band_1'],  
                name='twvwap_lower_band_1',  
                line=dict(color='slateblue', width=1)  
            ), row=1, col=1)  
            
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['twvwap_upper_band_2'],  
                name='twvwap_upper_band_2',  
                line=dict(color='tomato', width=1)  
            ), row=1, col=1)  
            

    def _add_volatility_twvwap_deviation(self, fig, df, factors):
        fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['twvwap_deviation'] ,  
                name='twvwap twvwap_smooth dev',  
                line=dict(color='purple', width=1)  
            ), row=4, col=1)  
        
        fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['twvwap_std_dev'],  
                name='twvwap_std_dev',  
                line=dict(color='steelblue', width=2)  
            ), row=4, col=1)    

    def _add_adx(self, fig, df, factors):  
        """  
        添加 ADX 指标到第二行子图，同时显示 +DI 与 -DI  
        """  
        if all(k in factors for k in ['adx', '+di', '-di']):  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['adx'],  
                name='ADX',  
                line=dict(color='#FF9800'),  
                hovertemplate="<b>ADX</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>"  
            ), row=2, col=1)  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['+di'],  
                name='+DI',  
                line=dict(color='#4CAF50'),  
                hovertemplate="<b>+DI</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>"  
            ), row=2, col=1)  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['-di'],  
                name='-DI',  
                line=dict(color='#F44336'),  
                hovertemplate="<b>-DI</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>"  
            ), row=2, col=1)  

    def _add_stochrsi(self, fig, df, factors):  
        """  
        添加 StochRSI 指标到第三行子图  
        """  
    
        if all(k in factors for k in ['stoch_k', 'stoch_d']):  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['stoch_k'],  
                name='Stoch %K',  
                line=dict(color='blue')  # 可以根据需要调整颜色  
            ), row=3, col=1)  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['stoch_d'],  
                name='Stoch %D',  
                line=dict(color='orange')  # 可以根据需要调整颜色  
            ), row=3, col=1)  
            # 添加超买超卖水平线（如果需要）  
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)  # 超买线  
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)  # 超卖线  

    def _update_layout(self, fig, timeframe, df):  
        """  
        更新图表布局和样式：  
         - 根据 K 线数据动态计算纵向（Y 轴）的最小和最大值，并添加适当边距，  
           使图形显示更清晰。  
        """  
        # 动态设置第一行价格区域的范围  
        min_price = df['low'].min()  
        max_price = df['high'].max()  
        margin = (max_price - min_price) * 0.05  # 边距取5%  

        fig.update_layout(  
            height=800,  
            xaxis_rangeslider_visible=False,  
            margin=dict(l=50, r=50, t=50, b=50),  
            showlegend=True,  
            title=f"{timeframe} - Chart View",  
            hovermode='x unified'  
        )  
        # 第一行：主图 - 设置价格的区间范围（双 y 轴中主轴为价格）  
        fig.update_yaxes(  
            title="Price",  
            row=1, col=1,  
            secondary_y=False,  
            range=[min_price - margin, max_price + margin]  
        )  
        # 第一行：副轴通常用于成交量  
        fig.update_yaxes(title="Volume", row=1, col=1, secondary_y=True)  
        
        # 第二行：ADX 指标区域  
        fig.update_yaxes(title="ADX", row=2, col=1)  

        # 第三行：StochRSI 指标区域  
        fig.update_yaxes(title="StochRSI", row=3, col=1)  

        # 第四行：TVWAP偏离指标区域  
        fig.update_yaxes(title="TVWAP Deviation", row=4, col=1)  

        # # 第五行：风险信号区域  
        # fig.update_yaxes(title="Risk Signals", row=5, col=1)  

        # 统一所有子图的 x 与 y 的指示线设置  
        for i in range(1, 6):  
            fig.update_xaxes(row=i, col=1, showspikes=True, spikemode='across')  
            fig.update_yaxes(row=i, col=1, showspikes=True)  

   