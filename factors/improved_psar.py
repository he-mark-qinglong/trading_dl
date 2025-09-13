import pandas as pd  
import numpy as np  
from typing import Dict  

class ImprovedPSARIndicator:  
    """改进的 PSAR 指标实现，使用 .iloc 避免警告"""  
    def __init__(  
        self,  
        high: pd.Series,  
        low: pd.Series,  
        close: pd.Series,  
        step: float = 0.02,  
        max_step: float = 0.20,  
    ):  
        self.high = high  
        self.low = low  
        self.close = close  
        self.step = step  
        self.max_step = max_step  
        self._run()  

    def _run(self):  
        """计算 PSAR 值"""  
        up_trend = True  
        af = self.step  
        up_trend_high = self.high.iloc[0]  
        down_trend_low = self.low.iloc[0]  

        # 初始化所有序列  
        self._psar = pd.Series(index=self.close.index, dtype=float)  
        self._psar_up = pd.Series(index=self.close.index, dtype=float)  
        self._psar_down = pd.Series(index=self.close.index, dtype=float)  
        self._psar_up_indicator = pd.Series(index=self.close.index, dtype=float)  
        self._psar_down_indicator = pd.Series(index=self.close.index, dtype=float)  

        self._psar.iloc[0] = down_trend_low  

        for i in range(1, len(self.close)):  
            # 上升趋势  
            if up_trend:  
                self._psar.iloc[i] = self._psar.iloc[i-1] + af * (up_trend_high - self._psar.iloc[i-1])  
                
                if self.low.iloc[i] < self._psar.iloc[i]:  
                    up_trend = False  
                    self._psar.iloc[i] = up_trend_high  
                    self._psar_down.iloc[i] = self._psar.iloc[i]  
                    self._psar_down_indicator.iloc[i] = 1  
                    down_trend_low = self.low.iloc[i]  
                    af = self.step  
                else:  
                    if self.high.iloc[i] > up_trend_high:  
                        up_trend_high = self.high.iloc[i]  
                        af = min(af + self.step, self.max_step)  
                    
                    if self.low.iloc[i-1] < self._psar.iloc[i]:  
                        self._psar.iloc[i] = self.low.iloc[i-1]  
                    
                    self._psar_up.iloc[i] = self._psar.iloc[i]  
                    self._psar_up_indicator.iloc[i] = 1  
            
            # 下降趋势  
            else:  
                self._psar.iloc[i] = self._psar.iloc[i-1] - af * (self._psar.iloc[i-1] - down_trend_low)  
                
                if self.high.iloc[i] > self._psar.iloc[i]:  
                    up_trend = True  
                    self._psar.iloc[i] = down_trend_low  
                    self._psar_up.iloc[i] = self._psar.iloc[i]  
                    self._psar_up_indicator.iloc[i] = 1  
                    up_trend_high = self.high.iloc[i]  
                    af = self.step  
                else:  
                    if self.low.iloc[i] < down_trend_low:  
                        down_trend_low = self.low.iloc[i]  
                        af = min(af + self.step, self.max_step)  
                    
                    if self.high.iloc[i-1] > self._psar.iloc[i]:  
                        self._psar.iloc[i] = self.high.iloc[i-1]  
                    
                    self._psar_down.iloc[i] = self._psar.iloc[i]  
                    self._psar_down_indicator.iloc[i] = 1  

    def psar(self) -> pd.Series:  
        """获取 PSAR 值"""  
        return self._psar  

    def psar_up(self) -> pd.Series:  
        """获取上升趋势的 PSAR 值"""  
        return self._psar_up  

    def psar_down(self) -> pd.Series:  
        """获取下降趋势的 PSAR 值"""  
        return self._psar_down  

    def psar_up_indicator(self) -> pd.Series:  
        """获取上升趋势指示器"""  
        return self._psar_up_indicator  

    def psar_down_indicator(self) -> pd.Series:  
        """获取下降趋势指示器"""  
        return self._psar_down_indicator  

# def parabolic_sar(self, df: pd.DataFrame) -> Dict[str, pd.Series]:  
#     """  
#     计算改进的 SAR 指标  
#     :param df: 包含 high、low、close 列的 DataFrame  
#     :return: 包含各种 SAR 指标的字典  
#     """  
#     sar = ImprovedPSARIndicator(  
#         high=df['high'],  
#         low=df['low'],  
#         close=df['close'],  
#         step=self.config.sar_acceleration,  
#         max_step=self.config.sar_maximum  
#     )  
    
#     return {  
#         'sar': sar.psar(),  
#         'sar_up': sar.psar_up(),  
#         'sar_down': sar.psar_down(),  
#         'sar_up_indicator': sar.psar_up_indicator(),  
#         'sar_down_indicator': sar.psar_down_indicator(),  
#         'sar_norm': self.normalize_factor(sar.psar())  
#     }