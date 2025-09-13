import numpy as np  
import pandas as pd
import plotly.graph_objects as go 

class UnUsedFactorManager:
    def __init__(self):
        pass

    def create_candlestick_figure(self, *args, **kwds):
        # self._add_pivot_points(fig, df, factors)
        # self._add_bollinger_bands(fig, df, factors)
        # self._add_sar(fig, df, factors)

        pass

    def _add_pivot_points(self, fig, df, factors):  
        """  
        添加支撑阻力线（Pivot Points 中的 R1/R2/R3 和 S1/S2/S3）到第一行子图  
        """  
        if all(k in factors for k in ['r1', 'r2', 'r3', 's1', 's2', 's3']):  
            for level, label in [('r1', 'R1'), ('r2', 'R2'), ('r3', 'R3')]:  
                fig.add_trace(go.Scatter(  
                    x=df.index,  
                    y=factors[level],  
                    name=label,  
                    line=dict(color=self.colors['pivot'].get(level, 'blue'), width=1, dash='dot'),  
                    hoverinfo='skip'  
                ), row=1, col=1)  
            for level, label in [('s1', 'S1'), ('s2', 'S2'), ('s3', 'S3')]:  
                fig.add_trace(go.Scatter(  
                    x=df.index,  
                    y=factors[level],  
                    name=label,  
                    line=dict(color=self.colors['pivot'].get(level, 'red'), width=1, dash='dot'),  
                    hoverinfo='skip'  
                ), row=1, col=1)  

    def _add_bollinger_bands(self, fig, df, factors):  
        """  
        添加布林带指标 (BOLL) 到第一行子图  
        """  
        if all(k in factors for k in ['bb_high', 'bb_mid', 'bb_low']):  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['bb_high'],  
                name='BB Upper',  
                line=dict(color='rgba(173, 204, 255, 0.7)', width=1),  
                hoverinfo='skip'  
            ), row=1, col=1)  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['bb_mid'],  
                name='BB Middle',  
                line=dict(color='rgba(173, 204, 255, 0.7)', width=1),  
                hoverinfo='skip'  
            ), row=1, col=1)  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['bb_low'],  
                name='BB Lower',  
                line=dict(color='rgba(173, 204, 255, 0.7)', width=1),  
                fill='tonexty',  
                hoverinfo='skip'  
            ), row=1, col=1)  

    def _add_sar(self, fig, df, factors):  
        """  
        添加 SAR 指标到第一行子图  
        """  
        if 'sar' in factors:  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['sar'],  
                name='SAR',  
                mode='markers',  
                marker=dict(color='rgba(0, 0, 0, 0.6)', size=3, symbol='diamond'),  
                hoverinfo='skip'  
            ), row=1, col=1)  
    def _add_supertrend(self, fig, df, factors):  
        """  
        添加 SuperTrend 指标到第一行子图  
        """  
        if 'supertrend' in factors:  
            # # 绘制 SuperTrend 上轨  
            # fig.add_trace(go.Scatter(  
            #     x=df.index,  
            #     y=factors['supertrend_upper'],  
            #     name='SuperTrend Upper',  
            #     mode='lines',  
            #     line=dict(color='rgba(255, 0, 0, 0.6)', width=1),  
            #     hoverinfo='skip'  
            # ), row=1, col=1)  
            
            # # 绘制 SuperTrend 下轨  
            # fig.add_trace(go.Scatter(  
            #     x=df.index,  
            #     y=factors['supertrend_lower'],  
            #     name='SuperTrend Lower',  
            #     mode='lines',  
            #     line=dict(color='rgba(0, 255, 0, 0.6)', width=1),  
            #     hoverinfo='skip'  
            # ), row=1, col=1)  
            
            # 绘制 SuperTrend 方向标记  
            fig.add_trace(go.Scatter(  
                x=df.index,  
                y=factors['supertrend'],  
                name='SuperTrend',  
                mode='markers',  
                marker=dict(  
                    color=np.where(factors['supertrend_direction'] == 1, 'rgba(0, 255, 0, 0.6)', 'rgba(255, 0, 0, 0.6)'),  
                    size=3,  
                    symbol='diamond'  
                ),  
                hoverinfo='skip'  
            ), row=1, col=1)  
