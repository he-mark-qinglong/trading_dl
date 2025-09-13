from abc import ABC, abstractmethod  
from typing import Dict, Optional, List, Tuple  
import pandas as pd  

class BaseStrategy(ABC):  
    """策略基类 - 简化版本"""  
    def __init__(self, leverage: int = 10, risk_per_trade: float = 0.02):  
        self.leverage = leverage  # 固定杠杆率  
        self.risk_per_trade = risk_per_trade  
        self.required_factors = []  
        self.current_price = 0.0  
        self.volume = 0.0  
        self.trend = None  
        self.factors = {}  
        self.df = None  

    def setup_data(self, df, factors, trend):  
        """设置策略所需数据"""  
        try:  
            self.df = df  
            self.factors = factors  
            self.trend = trend  
            self.current_price = df['close'].iloc[-1]  
            self.volume = df['volume'].iloc[-1]  
        except Exception as e:  
            self.log_error('setup_data', e, {  
                'df_shape': df.shape if df is not None else None,  
                'factors_keys': list(factors.keys()) if factors is not None else None,  
                'trend': trend  
            })  

    def log_error(self, method_name: str, error: Exception, context: Dict = None):  
        """统一的错误日志处理"""  
        error_msg = (  
            f"{self.__class__.__name__}.{method_name} error:\n"  
            f"Error Type: {type(error).__name__}\n"  
            f"Error Message: {str(error)}\n"  
            f"Context: {context or {}}"  
        )  
        print(error_msg)  
        return error_msg  

    @abstractmethod  
    def check_conditions(self) -> bool:  
        """检查入场条件，子类必须实现"""  
        pass  

    def calculate_tp_sl(self, atr: float, signal_type: str,  
                       sl_multiplier: float = 1.0,  
                       tp_multiplier: float = 1.2) -> Tuple[float, float]:  
        """计算止盈止损价格"""  
        if signal_type == 'buy':  
            stop_loss = self.current_price - (sl_multiplier * atr)  
            take_profit = self.current_price + (tp_multiplier * atr)  
        else:  # sell  
            stop_loss = self.current_price + (sl_multiplier * atr)  
            take_profit = self.current_price - (tp_multiplier * atr)  
            
        return take_profit, stop_loss  

    def calculate_position_size(self, atr: float, balance: float) -> float:  
        """计算仓位大小"""  
        risk_amount = balance * self.risk_per_trade  
        return risk_amount / atr  

    @abstractmethod  
    def generate_signal(self, symbol: str, balance: float) -> Optional[Dict]:  
        """生成交易信号，子类必须实现"""  
        pass  


class UptrendPullbackStrategy(BaseStrategy):  
    def __init__(self, leverage: int = 10, risk_per_trade: float = 0.02):  
        super().__init__(leverage, risk_per_trade)  
        self.required_factors = ['macd', 'macd_signal', 'macd_diff',  
                               'ma_5', 'ma_20', 'pvt', 'obv', 'cci',  
                               'atr']   
    def check_conditions(self) -> bool:  
        try:  
            # 获取数据  
            ma_5 = self.factors['ma_5']  
            ma_20 = self.factors['ma_20']  
            macd_diff = self.factors['macd_diff']  
            macd = self.factors['macd']  
            cci = self.factors['cci']  
            pvt = self.factors['pvt']  
            obv = self.factors['obv']  
            
            # 1. 趋势确认 (提高严格性)  
            trend_confirm = (  
                ma_5.iloc[-1] > ma_20.iloc[-1] and  # 短期均线必须高于中期均线  
                ma_5.iloc[-1] > ma_5.rolling(3).mean().iloc[-1] and  # 短期均线必须向上倾斜  
                macd_diff.iloc[-1] > 0.02 and  # MACD柱状图显著为正  
                macd.iloc[-1] > 0.05  # MACD显著高于零轴  
            )  
            
            # 2. 回调信号 (提高严格性)  
            cci_signal = (  
                (-50 < cci.iloc[-1] < 50) and  # 更严格的CCI中性区域范围  
                cci.iloc[-1] > cci.iloc[-2] and  # CCI必须明确反弹  
                cci.iloc[-1] > cci.iloc[-3] and  # 确保CCI反弹持续至少2个周期  
                (  
                    self.current_price < ma_5.iloc[-1] * 1.005 and  # 价格更接近MA5  
                    self.current_price > ma_5.iloc[-1] * 0.995  # 允许更小的偏差  
                )  
            )  
            
            # 3. 资金流向确认 (提高严格性)  
            pvt_change = (pvt.iloc[-1] - pvt.iloc[-3]) / abs(pvt.iloc[-3])  
            obv_change = (obv.iloc[-1] - obv.iloc[-3]) / abs(obv.iloc[-3])  
            avg_volume = self.df['volume'].rolling(5).mean().iloc[-1]  
            
            flow_strength = (  
                (pvt_change > 0 or obv_change > 0) and  # 资金流必须为正  
                (  
                    self.volume < avg_volume * 1.2 and  # 成交量必须接近均值  
                    self.volume > avg_volume * 0.8  # 添加成交量下限  
                )  
            )  
            
            # 4. 趋势延续信号 (提高严格性)  
            price_higher_low = (  
                self.current_price > self.df['close'].rolling(5).min().iloc[-2] and  # 扩大比较周期  
                self.current_price > self.df['low'].rolling(3).min().iloc[-1]  # 添加近期低点比较  
            )  
            
            macd_higher_low = (  
                macd_diff.iloc[-1] > macd_diff.rolling(5).min().shift(1).iloc[-1]  and  # 扩大MACD比较周期  
                macd.iloc[-1] > macd.iloc[-5:].mean()  # MACD值必须高于过去5周期均值  
            )  
            
            trend_continuation = (  
                price_higher_low and   
                macd_higher_low and  
                cci.iloc[-1] > cci.iloc[-3]  # CCI也需要确认趋势延续  
            )  
            
            # 5. 市场波动率过滤 (保持严格性)  
            atr = self.factors['atr'].iloc[-1]  
            avg_price = (self.df['high'].iloc[-1] + self.df['low'].iloc[-1]) / 2  
            volatility_ok = atr / avg_price < 0.02  # 降低波动率阈值，避免过度波动  
            
            # 组合条件 (更严格的组合)  
            basic_conditions = trend_confirm and volatility_ok  
            
            signal_conditions = (  
                (cci_signal and flow_strength) or  # 回调入场  
                (trend_continuation and flow_strength)  # 趋势延续  
            )  
            
            return basic_conditions and signal_conditions  
            
        except Exception as e:  
            print(f"UptrendPullbackStrategy check_conditions error: {e}")  
            return False  

    def generate_signal(self, symbol: str, balance: float) -> Optional[Dict]:  
        if self.trend != 'uptrend' or not self.check_conditions():  
            return None  
            
        try:  
            atr = self.factors['atr'].iloc[-1]  
            cci = self.factors['cci'].iloc[-1]  
            
            # 使用基类方法计算仓位大小，移除 leverage_ratio 参数  
            size = self.calculate_position_size(atr, balance)  
            
            # 根据CCI位置动态调整风险  
            if -30 < cci < 30:  # 理想的回调位置  
                sl_multiplier = 1.2  
                tp_multiplier = 2.4  # 2.0 * 1.2  
            else:  
                sl_multiplier = 1.5  
                tp_multiplier = 2.7  # 1.8 * 1.5  
            
            # 使用基类方法计算止盈止损，移除 leverage_ratio 参数  
            take_profit, stop_loss = self.calculate_tp_sl(  
                atr, 'buy',  
                sl_multiplier=sl_multiplier,  
                tp_multiplier=tp_multiplier  
            )  
                
            return {  
                'timestamp': self.factors['cci'].index[-1],  
                'signal_type': 'buy',  
                'price': self.current_price,  
                'size': size,  
                'stop_loss': stop_loss,  
                'take_profit': take_profit,  
                'leverage': self.leverage  # 使用固定杠杆率  
            }  
            
        except Exception as e:  
            print(f"UptrendPullbackStrategy generate_signal error: {e}")  
            return None  
  
class DowntrendReboundStrategy(BaseStrategy):  
    def __init__(self, leverage: int = 10, risk_per_trade: float = 0.02):  
        super().__init__(leverage, risk_per_trade)  
        self.required_factors = ['macd', 'macd_signal', 'macd_diff',  
                               'ma_5', 'ma_20', 'pvt', 'obv', 'cci',  
                               'atr']  

    def generate_signal(self, symbol: str, balance: float) -> Optional[Dict]:  
        if self.trend != 'downtrend' or not self.check_conditions():  
            return None  
            
        try:  
            atr = self.factors['atr'].iloc[-1]  
            cci = self.factors['cci'].iloc[-1]  
            
            # 使用基类方法计算仓位大小，移除 leverage_ratio 参数  
            size = self.calculate_position_size(atr, balance)  
            
            # 加密货币市场的动态风险调整  
            if cci < -120:  
                sl_multiplier = 1.2  
                tp_multiplier = 2.4  # 2.0 * 1.2  
            else:  
                sl_multiplier = 1.5  
                tp_multiplier = 2.7  # 1.8 * 1.5  
            
            # 使用基类方法计算止盈止损，移除 leverage_ratio 参数  
            take_profit, stop_loss = self.calculate_tp_sl(  
                atr, 'sell',  
                sl_multiplier=sl_multiplier,  
                tp_multiplier=tp_multiplier  
            )  
                
            return {  
                'timestamp': self.factors['cci'].index[-1],  
                'signal_type': 'sell',  
                'price': self.current_price,  
                'size': size,  
                'stop_loss': stop_loss,  
                'take_profit': take_profit,  
                'leverage': self.leverage  # 使用固定杠杆率  
            }  
            
        except Exception as e:  
            print(f"DowntrendReboundStrategy generate_signal error: {e}")  
            return None 

    def check_conditions(self) -> bool:  
        try:  
            # 获取数据  
            ma_5 = self.factors['ma_5']  
            ma_20 = self.factors['ma_20']  
            macd_diff = self.factors['macd_diff']  
            macd = self.factors['macd']  
            cci = self.factors['cci']  
            pvt = self.factors['pvt']  
            obv = self.factors['obv']  
            
            # 1. 趋势确认 (提高严格性)  
            trend_confirm = (  
                ma_5.iloc[-1] < ma_20.iloc[-1] and  # 短期均线在中期均线下方  
                ma_5.iloc[-1] < ma_5.rolling(3).mean().iloc[-1] and  # 短期均线必须向下  
                macd_diff.iloc[-1] < -0.02 and  # MACD柱状图显著为负  
                macd.iloc[-1] < -0.05  # MACD显著低于零轴  
            )  
            
            # 2. CCI反弹信号 (提高严格性)  
            cci_signal = (  
                cci.iloc[-1] < -100 and  # 更深的超卖阈值  
                cci.iloc[-1] > cci.iloc[-2] and  # CCI开始反弹  
                cci.iloc[-1] > cci.iloc[-3] and  # 确保反弹持续至少2个周期  
                cci.iloc[-1] < -50  # 确保仍然处于较低的区间  
            )  
            
            # 3. 资金流确认 (提高严格性)  
            pvt_change = (pvt.iloc[-1] - pvt.iloc[-3]) / abs(pvt.iloc[-3])  
            obv_change = (obv.iloc[-1] - obv.iloc[-3]) / abs(obv.iloc[-3])  
            
            flow_improve = (  
                pvt_change > -0.01 and  # 更严格的PVT变化阈值  
                obv_change > -0.01 and  # 更严格的OBV变化阈值  
                self.volume > self.df['volume'].rolling(5).mean().iloc[-1]  # 成交量必须大于5周期均值  
            )  
            
            # 4. MACD背离检查 (保持严格性)  
            price_lower_low = (  
                self.current_price < self.df['close'].rolling(5).min().iloc[-2]  # 扩大比较周期  
            )  
            macd_higher_low = (  
                macd_diff.iloc[-1] > macd_diff.rolling(5).min().shift(1).iloc[-1]  # 扩大MACD比较周期  
            )  
            
            potential_divergence = price_lower_low and macd_higher_low  
            
            # 组合条件 (提高严格性)  
            return trend_confirm and (  
                (cci_signal and flow_improve) or  
                (potential_divergence and flow_improve)  
            )  
            
        except Exception as e:  
            print(f"DowntrendReboundStrategy check_conditions error: {e}")  
            return False

class MomentumBreakoutStrategy(BaseStrategy):  
    """动量突破策略 - 适用于高波动市场"""  
    def __init__(self, leverage: int = 10, risk_per_trade: float = 0.02):  
        super().__init__(leverage, risk_per_trade)  
        self.required_factors = ['rsi', 'macd_diff', 'atr', 'atr_pct']  
    def generate_signal(self, symbol: str, balance: float) -> Optional[Dict]:  
        if self.trend != 'volatile' or not self.check_conditions():  
            return None  
            
        try:  
            rsi = self.factors['rsi'].iloc[-1]  
            atr = self.factors['atr'].iloc[-1]  
            avg_volume = self.df['volume'].rolling(20).mean().iloc[-1]  
            
            # 使用基类方法计算仓位大小（考虑高波动性）  
            size = self.calculate_position_size(atr, balance) * 0.7  
            
            # 动量突破的风险参数  
            sl_multiplier = 1.5  
            tp_multiplier = 3.0  
            
            if rsi > 70 and self.volume > avg_volume * 2:  
                signal_type = 'buy'  
            elif rsi < 30 and self.volume > avg_volume * 2:  
                signal_type = 'sell'  
            else:  
                return None  
            
            # 使用基类方法计算止盈止损  
            take_profit, stop_loss = self.calculate_tp_sl(  
                atr, signal_type,  
                sl_multiplier=sl_multiplier,  
                tp_multiplier=tp_multiplier  
            )  
            
            return {  
                'timestamp': self.factors['rsi'].index[-1],  
                'signal_type': signal_type,  
                'price': self.current_price,  
                'size': size,  
                'stop_loss': stop_loss,  
                'take_profit': take_profit,  
                'leverage': self.leverage  # 使用固定杠杆率  
            }  
            
        except Exception as e:  
            print(f"MomentumBreakoutStrategy generate_signal error: {e}")  
            return None
    
    def check_conditions(self) -> bool:  
        # 获取最新指标值  
        rsi = self.factors['rsi'].iloc[-1]  
        macd_diff = self.factors['macd_diff'].iloc[-1]  
        avg_volume = self.df['volume'].rolling(20).mean().iloc[-1]  
        atr_pct = self.factors['atr_pct'].iloc[-1]  
        
        # 检查动量突破条件  
        is_strong_momentum_up = (  
            rsi > 70 and  # 强势RSI  
            macd_diff > 0 and  # MACD柱状图为正  
            self.volume > avg_volume * 2 and  # 成交量显著放大  
            atr_pct > self.factors['atr_pct'].rolling(10).mean().iloc[-1]  # ATR扩大  
        )  
        
        is_strong_momentum_down = (  
            rsi < 30 and  # 弱势RSI  
            macd_diff < 0 and  # MACD柱状图为负  
            self.volume > avg_volume * 2 and  # 成交量显著放大  
            atr_pct > self.factors['atr_pct'].rolling(10).mean().iloc[-1]  # ATR扩大  
        )  
        
        return is_strong_momentum_up or is_strong_momentum_down  

class RangeBreakoutStrategy(BaseStrategy):  
    def __init__(self, leverage: int = 10, risk_per_trade: float = 0.02):  
        super().__init__(leverage, risk_per_trade)  
        self.required_factors = ['bb_high', 'bb_low', 'atr'] 
    """区间突破策略"""  
    def check_conditions(self) -> bool:  
        bb_upper = self.factors['bb_high'].iloc[-1]  
        bb_lower = self.factors['bb_low'].iloc[-1]  
        avg_volume = self.df['volume'].rolling(20).mean().iloc[-1]  
        
        # 检查是否突破布林带且成交量放大  
        is_breakout_up = self.current_price > bb_upper and self.volume > avg_volume * 1.5  
        is_breakout_down = self.current_price < bb_lower and self.volume > avg_volume * 1.5  
        
        return is_breakout_up or is_breakout_down  

    def generate_signal(self, symbol: str, balance: float) -> Optional[Dict]:  
        if self.trend != 'range' or not self.check_conditions():  
            return None  
            
        try:  
            bb_upper = self.factors['bb_high'].iloc[-1]  
            bb_lower = self.factors['bb_low'].iloc[-1]  
            atr = self.factors['atr'].iloc[-1]  
            
            # 使用基类方法计算仓位大小  
            size = self.calculate_position_size(atr, balance)  
            
            # 区间突破的风险参数  
            sl_multiplier = 2.0  
            tp_multiplier = 4.0  # 1:2风险收益比  
            
            if self.current_price > bb_upper:  
                signal_type = 'buy'  
            elif self.current_price < bb_lower:  
                signal_type = 'sell'  
            else:  
                return None  
            
            # 使用基类方法计算止盈止损  
            take_profit, stop_loss = self.calculate_tp_sl(  
                atr, signal_type,  
                sl_multiplier=sl_multiplier,  
                tp_multiplier=tp_multiplier  
            )  
            
            return {  
                'timestamp': self.factors['rsi'].index[-1],  
                'signal_type': signal_type,  
                'price': self.current_price,  
                'size': size,  
                'stop_loss': stop_loss,  
                'take_profit': take_profit,  
                'leverage': self.leverage  # 使用固定杠杆率  
            }  
            
        except Exception as e:  
            print(f"RangeBreakoutStrategy generate_signal error: {e}")  
            return None
        
class AntiMartingaleStrategy(BaseStrategy):  
    """反马丁格尔策略 - 适合Range市场"""  
    def __init__(self, leverage: int = 10, risk_per_trade: float = 0.02):  
        super().__init__(leverage, risk_per_trade)  
        self.base_position = 1.0  
        self.max_levels = 3  
        self.current_level = 0  
        self.required_factors = ['bb_high', 'bb_low', 'bb_mid', 'atr', 'rsi']  

    def check_conditions(self) -> bool:  
        """检查是否在布林带中间区域"""  
        try:  
            bb_upper = self.factors['bb_high'].iloc[-1]  
            bb_lower = self.factors['bb_low'].iloc[-1]  
            
            # 检查是否在布林带中间区域  
            is_middle_range = (self.current_price > bb_lower * 1.2 and  
                             self.current_price < bb_upper * 0.8)  
            
            # 可以添加更多条件  
            return is_middle_range  
            
        except Exception as e:  
            print(f"Check conditions error: {e}")  
            return False  

    def generate_signal(self, symbol: str, balance: float) -> Optional[Dict]:  
        try:  
            if self.trend != 'range' or not self.check_conditions():  
                return None  
                
            atr = self.factors['atr'].iloc[-1]  
            
            # 简化的仓位计算，使用固定增长率  
            if self.current_level < self.max_levels:  
                # 使用固定的增长率，不再考虑杠杆率  
                growth_rate = 1.5  # 可以根据需要调整  
                position_size = self.base_position * (growth_rate ** self.current_level)  
            else:  
                position_size = 0  
                
            signal_type = 'buy' if self.current_price < self.factors['bb_mid'].iloc[-1] else 'sell'  
            
            # 使用基类方法计算止盈止损  
            take_profit, stop_loss = self.calculate_tp_sl(  
                atr, signal_type  
            )  
                
            return {  
                'timestamp': self.factors['rsi'].index[-1],  
                'signal_type': signal_type,  
                'price': self.current_price,  
                'size': position_size,  
                'stop_loss': stop_loss,  
                'take_profit': take_profit,  
                'leverage': self.leverage  # 使用固定杠杆率  
            }  
        except Exception as e:  
            print(f"AntiMartingaleStrategy generate_signal error: {e}")  
            return None    
