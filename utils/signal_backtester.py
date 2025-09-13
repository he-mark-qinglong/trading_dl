import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from utils.plot_trade import plot_flex_ta_dashboard
from factors import FactorManager 
from utils.move_loss_profit import simple_trailing_stop_loss, simple_trailing_take_profit
from factors import VolumeFactors

def shirnk_volume_support(current_df_1m, threshold=3):
    ma_volume = current_df_1m['volume'].rolling(30).mean()

    origin_volume = current_df_1m['volume']
    support_1 = ma_volume.iloc[-1] < origin_volume.iloc[-1] 
    support_2 = ma_volume.iloc[-2] < origin_volume.iloc[-2]
    support_3 = ma_volume.iloc[-3] < origin_volume.iloc[-3]
    support_4 = ma_volume.iloc[-4] >= origin_volume.iloc[-4]

    continuous_up = current_df_1m['close'].iloc[-1] > current_df_1m['close'].iloc[-2] > current_df_1m['close'].iloc[-3]
    continuous_down = current_df_1m['close'].iloc[-1] < current_df_1m['close'].iloc[-2] < current_df_1m['close'].iloc[-3]

    if support_1 + support_2 + support_3 >= threshold:
        return True, continuous_up, continuous_down
    else:
        return False, continuous_up, continuous_down
    
def scaleup_volume_support(current_df_1m, threshold=3):
    ma_volume = current_df_1m['volume'].rolling(30).mean()
    origin_volume = current_df_1m['volume']
    support_1 = ma_volume.iloc[-1] > origin_volume.iloc[-1] 
    support_2 = ma_volume.iloc[-2] > origin_volume.iloc[-2]
    support_3 = ma_volume.iloc[-3] > origin_volume.iloc[-3]
    support_4 = ma_volume.iloc[-4] <= origin_volume.iloc[-4]

    continuous_up = current_df_1m['close'].iloc[-1] > current_df_1m['close'].iloc[-2] > current_df_1m['close'].iloc[-3]
    continuous_down = current_df_1m['close'].iloc[-1] < current_df_1m['close'].iloc[-2] < current_df_1m['close'].iloc[-3]

    if support_1 + support_2 + support_3 >= threshold:
        return True, not continuous_up, not continuous_down
    else:
        return False, not continuous_up, not continuous_down


def fvg_3K_short(current_df, mean_atr):  
    """  
    检测空头的FVG信号。  
    current_df: pd.DataFrame -- 包含K线数据的DataFrame，至少需要最近3根K线数据。  
    """  
    if len(current_df) < 3:  
        raise ValueError("current_df必须包含至少3根K线数据。")  
    
    # 提取最近3根K线的数据  
    k1 = current_df.iloc[-1]  # 当前K线  
    k2 = current_df.iloc[-2]  # 前一根K线  
    k3 = current_df.iloc[-3]  # 前两根K线  
    
    # 空头FVG的条件：前一根K线的最高价 < 当前K线的最低价  
    if k3['high'] < k1['low'] and k2['close'] - k2['open'] < -mean_atr:
        return True  
    return False  


def fvg_3K_long(current_df, mean_atr):  
    """  
    检测多头的FVG信号。  
    current_df: pd.DataFrame -- 包含K线数据的DataFrame，至少需要最近3根K线数据。  
    """  
    if len(current_df) < 3:  
        raise ValueError("current_df必须包含至少3根K线数据。")  
    
    # 提取最近3根K线的数据  
    k1 = current_df.iloc[-1]  # 当前K线  
    k2 = current_df.iloc[-2]  # 前一根K线  
    k3 = current_df.iloc[-3]  # 前两根K线  
    
    # 多头FVG的条件：前一根K线的最低价 > 当前K线的最高价  
    if k3['low'] > k1['high'] and k2['close'] - k2['open'] > mean_atr:
        return True  
    return False  


def recent_fvg_3K(current_df, factors):  
    """  
    检测FVG信号（空头和多头）。  
    current_df: pd.DataFrame -- 包含K线数据的DataFrame，至少需要最近3根K线数据。  
    """  
    mean_atr = factors['atr'].mean()
    short_fvg = fvg_3K_short(current_df, mean_atr)  
    long_fvg = fvg_3K_long(current_df, mean_atr)  

    return {  
        "short_fvg": short_fvg,  
        "long_fvg": long_fvg  
    } 

def find_factor_condition(df, 
                          factors,
                          start_index, 
                          statistics_len,
                          condition_func):  
    """  
    从指定位置向前遍历 DataFrame，  
    每次取区间 [idx, start_index]（包含 idx 和 start_index），计算 factor，  
    当 factor 满足 condition_func 条件时，返回该 idx 及对应的 factor。  

    参数：  
    - df: pandas DataFrame  
    - start_index: 起始位置（整数索引）  
    - condition_func: 回调函数，接收 factor，返回布尔值True/False  
    - compute_factor: 回调函数，接收 df 的子区间（从 idx 到 start_index），返回计算的 factor  

    返回：  
    - (idx, factor) 满足条件时返回，否则返回 (None, None)  
    """ 
    for idx in range(start_index, statistics_len, -int(statistics_len/2)):
        sub_df = df.iloc[idx-statistics_len:idx + 1].copy() 
        
        factors_1m = VolumeFactors().calculate_vpvr(df=sub_df) 
        if condition_func(factors_1m):  
            print('fffffffffffffffffffffffffffffffff is_normal_distribution p_value=0.06')
            VolumeFactors().volume_profile(df=sub_df)['poc_price'], True
            # return df['close'].iloc[idx], True 
    return df['close'].iloc[-1], False  

    
#回归的时候用这里判断开仓方向。
def calculate_weighted_twvwap_slope(twvwaps_1h, twvwaps_1m, use_statistics=True):  
    """  
    计算加权的TWVWAP斜率并基于统计方法判断趋势方向  
    
    参数:  
    factors_1h: 1小时级别因子数据  
    factors: 1分钟级别因子数据  
    use_statistics: 是否使用统计方法确定阈值  
    """  
    
    # 确保数据足够计算  
    if len(twvwaps_1h) < 5 or len(twvwaps_1h) < 12:  
        return False, 0  # 数据不足，返回默认值  
    
    # 1. 正确计算1小时级别斜率 - 使用线性回归  
    window_1h = len(twvwaps_1h)  
    recent_1h = twvwaps_1h.iloc[-window_1h:].values  
    x_1h = np.arange(len(recent_1h))  
    slope_1h = np.polyfit(x_1h, recent_1h, 1)[0] * window_1h  # 标准化斜率  
    
    # 2. 正确计算1分钟级别斜率  
    window_1m = len(twvwaps_1m)  
    recent_1m = twvwaps_1m.iloc[-window_1m:].values  
    x_1m = np.arange(len(recent_1m))  
    slope_1m = np.polyfit(x_1m, recent_1m, 1)[0] * window_1m  # 标准化斜率  
    
    # 3. 计算加权斜率  
    alpha_1h = 0.9
    slope_weighted = alpha_1h * slope_1h + (1 - alpha_1h) * slope_1m  
    # slope_weighted = slope_1h

    # 4. 基于统计方法确定阈值  
    if use_statistics:  
        # 维护历史斜率数据用于动态阈值计算  
        if not hasattr(calculate_weighted_twvwap_slope, 'slope_history'):  
            calculate_weighted_twvwap_slope.slope_history = []  
        
        # 更新历史数据  
        calculate_weighted_twvwap_slope.slope_history.append(slope_weighted)  
        if len(calculate_weighted_twvwap_slope.slope_history) > 60 * 24 * 3:  # 保持历史数据量  
            calculate_weighted_twvwap_slope.slope_history.pop(0)  
        
        # 计算统计阈值 (使用分位数而非固定值)  
        history = np.array(calculate_weighted_twvwap_slope.slope_history)  
        lower_threshold = np.percentile(history, 20)  # 20%分位数作为下跌阈值  
        upper_threshold = np.percentile(history, 80)  # 80%分位数作为上涨阈值  
        
        # 范围判断  
        if slope_weighted < lower_threshold:  
            return -1, slope_weighted  # 下跌趋势  
        elif slope_weighted > upper_threshold:  
            return 1, slope_weighted   # 上涨趋势  
        else:  
            return 0, slope_weighted   # 中性区间  
    else:  
        # 使用固定阈值 (基于经验值，但加入上限判断)  
        if slope_weighted < -10:  
            return -1, slope_weighted  # 下跌趋势  
        elif slope_weighted > 10:         # 增加上涨判断  
            return 1, slope_weighted   # 上涨趋势  
        else:  
            return 0, slope_weighted   # 中性区间  

class Backtester:
    def __init__(self, initial_balance=10000):
        """
        初始化回测器。

        Args:
            initial_balance: 初始资金。
        """
        self.initial_balance = initial_balance

        #self.position == 0: 无仓位, 1: 多头, -1: 空头
        self.position, self.stop_loss_price, self.take_profit_price = 0, 0, 0
        self.tally_hold_time = 0
        self.continuous_lost_times = 0
        self.has_been_crossed_vpvr = False

        self.leverage = 3
        self.last_adjusted_leverrage = None

        #-------------------
        #下跌采用非对称止损。
        self.short_stop_factor = 1.5
        self.short_take_profit_factor = 4.5
        self.risk_reward_ratio_require = 3
        self.max_risk_lost = 0.01
        
        #why:这笔交易以为时间长了以后条件变化了，导致开多之后突然大幅度下跌。本金腰斩。所以需要根据后续的成交量来进行止损调整或者出局的。暂时改为以时间止损位原则。
        '''
        {'timestamp': Timestamp('2024-07-03 22:02:00'), 'leverage': 5, 'type': 'open buy 0', 'price': 60258.0, 'balance': 16321.599999999919}
        {'type': 'is_timeout0 buy take_profit', 'price': 58484.0, 'balance': 7451.5999999999185, 'timestamp': Timestamp('2024-07-04 07:32:00')}
        '''
        self.max_short_hold_time:int = 60*72 #TODO 根据成交量动态调整移动止损，可能是一个更好的选择。

        self.long_stop_factor:float = 1.5
        self.long_take_profit_factor:float = self.short_take_profit_factor * 1.5
        self.max_long_hold_time:int = self.max_short_hold_time

        self.continuus_lost_time = 0
        self.policy = 'regression' 

    def compute_factors_for_window(sefl, df: pd.DataFrame, start_idx: int, statistic_len: int) -> Tuple[pd.Timestamp, Dict[str, pd.Series], pd.DataFrame]:
        """计算单个时间窗口的因子数据"""
        # 根据索引位置获取窗口数据
        window_df = df.iloc[start_idx:start_idx + statistic_len].copy()
        factors, updated_df = FactorManager().calculate_factors(window_df)
        # 返回当前窗口的结束时间戳、因子和更新后的数据
        return (updated_df.index[-1], factors, updated_df)
    
    def setup_regression_tp_sl(self, signal, current_price, factors_1m, move_profit = False):
        if signal < 0:
            if move_profit:
                self.stop_loss_price = current_price+factors_1m['atr'].iloc[-1]
                self.take_profit_price = current_price-factors_1m['atr'].iloc[-1]
            else:
                stop_factor = self.short_stop_factor  
                take_profit_factor = self.short_take_profit_factor / (2 if move_profit else 1)
                # 设置止损 (示例：x倍标准差 + ATR)
                self.stop_loss_price = min(current_price, factors_1m['twvwap_upper_band_1'].iloc[-1]) + stop_factor * factors_1m['atr'].iloc[-1]
                # 设置止盈 (同方向x倍标准差), 做空，需要向上取离得近的数值，所以用max。   更容易够得着
                self.take_profit_price = max(current_price - take_profit_factor * factors_1m['atr'].iloc[-1],
                                                factors_1m['twvwap_lower_band_40'].iloc[-1]) 
        else:
            if move_profit:
                self.stop_loss_price = current_price-factors_1m['atr'].iloc[-1]
                self.take_profit_price = current_price+factors_1m['atr'].iloc[-1]
            else:
                stop_factor = self.long_stop_factor       
                take_profit_factor = self.long_take_profit_factor / (2 if move_profit else 1)
                # 设置止损 (示例：x倍标准差 - ATR), 越min越低。 越max越高。
                self.stop_loss_price = max(current_price, factors_1m['twvwap_lower_band_1'].iloc[-1]) - stop_factor * factors_1m['atr'].iloc[-1]           
                # 设置止盈 (同方向x倍标准差)
                self.take_profit_price = min(current_price + take_profit_factor * factors_1m['atr'].iloc[-1],
                                                factors_1m['twvwap_upper_band_40'].iloc[-1]) 
    def setup_trending_tp_sl(self, signal, current_price, factors_1m, move_profit = False):
        if signal < 0:
            stop_factor = self.long_stop_factor             
            take_profit_factor = self.long_take_profit_factor / (2 if move_profit else 1)
            # 设置止损 (示例：x倍标准差 + ATR)
            self.stop_loss_price = min(current_price, factors_1m['twvwap_upper_band_1'].iloc[-1]) + stop_factor * factors_1m['atr'].iloc[-1]

            # 设置止盈 (同方向x倍标准差), 做空，需要向上取离得近的数值，所以用max。   更容易够得着
            self.take_profit_price = current_price - take_profit_factor * factors_1m['atr'].iloc[-1]
        else:
            stop_factor = self.long_stop_factor             
            take_profit_factor = self.long_take_profit_factor / (2 if move_profit else 1)
            # 设置止损 (示例：2倍标准差 - ATR), 越min越低。 越max越高。
            self.stop_loss_price = max(current_price, factors_1m['twvwap_lower_band_1'].iloc[-1]) - stop_factor * factors_1m['atr'].iloc[-1]
                                    
            # 设置止盈 (同方向x倍标准差)
            self.take_profit_price = current_price + take_profit_factor * factors_1m['atr'].iloc[-1]
            
    def run_backtest(self, df, timeframe, signal_func, factor_manager, 
                     calculate_dynamic_period_func, analyze_price_distribution_func,
                     trailing_stop_loss_func=simple_trailing_stop_loss,
                     trailing_take_profit_func = simple_trailing_take_profit):  
        """
        在单个时间框架上运行回测。

        Args:
            df: 包含价格数据的DataFrame。
            timeframe: 时间框架 (仅用于计算夏普比率的年化因子)。
            signal_func: 交易信号生成函数。
            factor_manager: FactorManager实例。
            calculate_dynamic_period_func: 动态周期计算函数。
            analyze_price_distribution_func: 价格分布分析函数。
            trailing_stop_loss_func: (可选) 移动止损函数。

        Returns:
            一个包含回测结果的字典。
        """

        vol_zscore_1h = None
        if timeframe in ['1m', '3m', '5m', '30m']:
            df_1h_all = df.resample('h').agg({  
                    'open': 'first',      # 每小时的开盘价为该小时第一条记录的开盘价  
                    'high': 'max',        # 每小时的最高价取该小时数据中的最大值  
                    'low': 'min',         # 每小时的最低价取该小时数据中的最小值  
                    'close': 'last',      # 每小时的收盘价为该小时最后一条记录的收盘价  
                    'volume': 'sum'       # 每小时的成交量为该小时数据的总和  
                })  
            
            rolled = df_1h_all['volume'].rolling(24)

            mean = rolled.mean()
            std = rolled.std()
            vol_zscore_1h = abs((df_1h_all['volume'] - mean) / std)
        else:
            return {}
        
        if timeframe in ['1m']:
            df_5m_all = df.resample('5min').agg({  
                    'open': 'first',      # 每小时的开盘价为该小时第一条记录的开盘价  
                    'high': 'max',        # 每小时的最高价取该小时数据中的最大值  
                    'low': 'min',         # 每小时的最低价取该小时数据中的最小值  
                    'close': 'last',      # 每小时的收盘价为该小时最后一条记录的收盘价  
                    'volume': 'sum'       # 每小时的成交量为该小时数据的总和  
                })  
            df_15m_all = df.resample('15min').agg({  
                    'open': 'first',      # 每小时的开盘价为该小时第一条记录的开盘价  
                    'high': 'max',        # 每小时的最高价取该小时数据中的最大值  
                    'low': 'min',         # 每小时的最低价取该小时数据中的最小值  
                    'close': 'last',      # 每小时的收盘价为该小时最后一条记录的收盘价  
                    'volume': 'sum'       # 每小时的成交量为该小时数据的总和  
                })  
            
        
        # 初始化变量
        balance = self.initial_balance
        trades = []  # 用于存储交易记录
        equity_curve = []  # 用于存储权益曲线
        entry_price = 0 #

        open_index = 0

        all_factors_1m, all_caculated_1m_df = FactorManager().calculate_factors(df)
        all_1h_factors, all_1h_df = FactorManager().calculate_factors(df_1h_all)

        adjusted_leverage = self.leverage
        adx_down_condition = None
        adx_up_condition = None
        
        # 循环遍历历史数据
        statistic_len:int = 221
        at_least_1h_factor_len = 50

        i:int = statistic_len * 61
        while i < len(df):
            # 获取当前时间点的数据
            #注意是基于原始的df获得的len，所以要先从原始的df里面拿出index的time才能到计算好的all_caculated_1m_df去索引时间区域
            indexes = df.index[i-statistic_len:i]

            factors_1m = {}
            current_df_1m = all_caculated_1m_df.loc[indexes].iloc[-statistic_len:].copy()
            for key, value in all_factors_1m.items():
                factors_1m[key] = value.loc[indexes].iloc[-statistic_len:].copy()
            factors_1m = FactorManager().update_twvwap(df=current_df_1m, old_factors=factors_1m, not_calc_abnormals = True)
           
            factors_1h = {}
            df_1h_show = all_1h_df.loc[:indexes[-1]].iloc[-at_least_1h_factor_len:].copy()
            for key, value in all_1h_factors.items():
                factors_1h[key] = value.loc[:indexes[-1]].iloc[-at_least_1h_factor_len:].copy()
            factors_1h = FactorManager().update_twvwap(df=df_1h_show, old_factors=factors_1h, not_calc_abnormals = True)
            
            current_price = current_df_1m['close'].iloc[-1]
            current_low = current_price #(current_df_1m['low'].iloc[-1] + current_price) / 2
            current_high = current_price #(current_df_1m['high'].iloc[-1] + current_price) / 2

            atr_1m_til_now = all_factors_1m['atr'].iloc[:i].iloc[-1000:]
            rolled_1m_atr = atr_1m_til_now.rolling(15)
            atr_1m_std = rolled_1m_atr.std()
            atr_1m_mean = rolled_1m_atr.mean()
            
            #########switch trend and regression market type.
            
            inloop_adx_signal = 0
            # 先判断atr异常，要么切换策略，要么还要调整头寸。
            if factors_1m['atr'].iloc[-1] - atr_1m_mean.iloc[-1] >= atr_1m_std.iloc[-1] * 2:
                end_time = current_df_1m.index[-1]
                start_time = all_caculated_1m_df.index[i-200*5]
                tmp_df = df_5m_all.loc[start_time:end_time].copy()
                factors_5m, df_5m = FactorManager().calculate_factors(tmp_df)
                # factors_5m = FactorManager().update_twvwap(df=df_5m, old_factors=factors_5m, not_calc_abnormals = False)

                start_time = all_caculated_1m_df.index[i-200*15]
                tmp_df = df_15m_all.loc[start_time:end_time].copy()
                factors_15m, df_15m = FactorManager().calculate_factors(tmp_df)
                adxs_5m = factors_5m['adx']
                adxs_15m = factors_15m['adx']

                #and factors_15m['adx'].iloc[-1] >= 25
                if factors_1m['is_normal_distribution']:
                    print(f'is_normal_distribution at {current_df_1m.index[-1]}')
                adx_up_condition = factors_5m['+di'].iloc[-1] > factors_5m['-di'].iloc[-1] and factors_15m['+di'].iloc[-1] > factors_15m['-di'].iloc[-1]
                adx_down_condition = factors_5m['+di'].iloc[-1] < factors_5m['-di'].iloc[-1] and factors_15m['+di'].iloc[-1] < factors_15m['-di'].iloc[-1] 
                
                
                #进入了趋势市场需要随时判断的。
                if adxs_5m.iloc[-1] >= 25 and adxs_15m.iloc[-1] >= 25 and not factors_1m['is_normal_distribution']:  
                    if (
                         (factors_5m['stoch_k'].iloc[-1] < 70 and adx_up_condition) or
                           (factors_5m['stoch_k'].iloc[-1] > 30 and adx_down_condition)
                    ):
                        # print(f'{current_df_1m.index[-1] } 5m~15m 级别市场过热了，将会进入单边趋势。 adx_up_condition {adx_up_condition} adx_down_condition {adx_down_condition}')
                        
                        trend_5m_continue = adxs_5m.iloc[-1] > adxs_5m.iloc[-2]
                        if ((self.position > 0 and adx_down_condition) or
                            (self.position < 0 and adx_up_condition)
                            ) and trend_5m_continue:
                            #仓位和趋势相反，并且亏损。  仓位相同不在上面的这个条件中。
                            if self.position * (current_price - self.stop_loss_price) < 0:
                                # balance = self.uncondition_close(entry_price=entry_price,
                                #         current_low=current_low,
                                #         current_high=current_high,
                                #         trades=trades,
                                #         factors=factors_1m,
                                #         current_df_1m=current_df_1m,
                                #         balance=balance)
                            
                                # print('处理了和趋势相反的仓位。')
                                # print(f"adx: factors_5m adx{factors_5m['adx'].iloc[-1]} +di:{factors_5m['+di'].iloc[-1]} -di:{factors_5m['-di'].iloc[-1]}")
                                # print(f"adx: factors_15m adx{factors_15m['adx'].iloc[-1]} +di:{factors_15m['+di'].iloc[-1]} -di:{factors_15m['-di'].iloc[-1]}")

                                # inloop_adx_signal = 1 if adx_up_condition else -1
                                pass
                            else:
                                # （由于仓位并不是不利于我，所以也就相当于在这里重新做了一次开仓--也可以考虑这里是1买2卖的方式，先平掉一部分来保证即使是止损也是赚的）
                                self.setup_trending_tp_sl(1 if self.position > 0 else -1, current_price, factors_1m)
                                #TODO rrr calc and judge if we should close this position.
                                        
                        old_policy = self.policy
                        self.policy = 'trend' 

           
                if (adxs_5m.iloc[-1] <= 20 and adxs_15m.iloc[-1] <= 20) and factors_1m['is_normal_distribution']:
                    if factors_1m['is_normal_distribution'] :
                        print(f'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^is_normal_distribution at {current_df_1m.index[-1]}')
                    # print(f'{current_df_1m.index[-1] } 5m~15m 级别市场恢复常态')
                    
                    if self.position != 0:
                        is_lost = self.position * (current_price - self.stop_loss_price) < 0
                        
                        if is_lost or (adx_up_condition and self.position < 0) or (adx_down_condition and self.position > 0):
                            print('由于趋势结束且亏损，或者新的方向与原来的趋势相反-为了降低进一步回撤的风险，处理反向仓位。')
                            balance = self.uncondition_close(entry_price=entry_price,
                                    current_low=current_low,
                                    current_high=current_high,
                                    trades=trades,
                                    factors=factors_1m,
                                    current_df_1m=current_df_1m,
                                    balance=balance)
                        
                            
                        else:
                            if old_policy != self.policy:  #策略变化，调整头寸和止盈止损。
                                #由于上面没有发现趋势相反，也就是趋势是中性，那就继续持有，做一次止盈止损的调整
                                # （由于仓位并不是不利于我，所以也就相当于在这里重新做了一次开仓--也可以考虑这里是1买2卖的方式，先平掉一部分来保证即使是止损也是赚的）
                                tmp_signal = 1 if self.position > 0 else -1
                                self.setup_regression_tp_sl(tmp_signal, current_price, factors_1m, move_profit=True)
                                if tmp_signal == 1:
                                    self.stop_loss_price = max(current_price - self.long_stop_factor*atr, max(factors_1m['twvwap'].iloc[-1], self.stop_loss_price) ) #多头的止损，向上截断最优。,最多破量价线结束。
                                else:
                                    self.stop_loss_price = max(current_price + self.short_stop_factor*atr, max(factors_1m['twvwap'].iloc[-1], self.stop_loss_price) ) #多头的止损，向上截断最优。,最多破量价线结束。
                                print(f'{current_df_1m.index[-1]} move stop to {self.stop_loss_price}\n')
                                plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(),
                                        factors_1m, 
                                        open_price = entry_price,
                                        stop_loss = self.stop_loss_price,
                                        stop_profit = self.take_profit_price,
                                        additional=f'move_profit_shirnk')
                                
                    if factors_1m['is_normal_distribution']:
                        old_policy = self.policy
                        self.policy = 'regression'
                    if (adxs_5m.iloc[-1] <= 20 and adxs_15m.iloc[-1] <= 20) :
                        print('adx culm down')

            if i % 60 == 0 :
                # self.long_take_profit_factor = (factors_1h['atr'].iloc[-1] / factors_1m['atr'].iloc[-1] )  # 差了三个级别，1m, 5m, 15m, 1h. 也可以后续考虑再把1d的也加进来并且进行加权考虑atr的倍数。
                # self.short_take_profit_factor = self.long_take_profit_factor * 0.9

                till_current_vol_zscore_1h = vol_zscore_1h.loc[:current_df_1m.index[-1]]
                till_current_df_1h = df_1h_all.loc[:current_df_1m.index[-1]]
                # current_1h_vol_z_score = till_current_vol_zscore_1h.iloc[-1]
                mean_of_zscore = till_current_vol_zscore_1h.iloc[-24*3:].mean()
                
                adjusted_leverage, volume_price_type = self.adjust_leverage(mean_of_zscore, till_current_df_1h.iloc[-24*3:].copy())

            if i % (statistic_len*10) == 0:
                left_time = df.index[i-statistic_len]
                right_time = df.index[min(len(df), i+int(statistic_len/2))]
                plot_flex_ta_dashboard(df.loc[left_time:right_time].copy(),
                                    factors_1m, 
                                    open_price = current_high,
                                    stop_loss = df['close'].iloc[-1],
                                    stop_profit = df['close'].iloc[statistic_len],
                                    additional=f'period_{i}')
                        
            # 获取交易信号
            try:
                if inloop_adx_signal == 0:
                    signal_type = simple_twvwap_signal(current_df_1m, factors_1m, calculate_dynamic_period_func, analyze_price_distribution_func)
                    signal, open_vwap_type = signal_type[0], signal_type[1]
                else:
                    signal = inloop_adx_signal
                    open_vwap_type = 0
            except Exception as e:
                print(f'signal_backtester __________{e}')
            
            #trend_direction, current_slope = calculate_weighted_twvwap_slope(factors_1h['twvwap'].iloc[-at_least_1h_factor_len:].copy(), factors_1m['twvwap'])  
            is_1h_continueous, continuous_up_1h, continuous_down_1h = shirnk_volume_support(df_1h_show, threshold=2)
            
            ''''
            ***************************************
            在大周期里找机会，再去小周期里面参与。  大周期没有趋势，就在小周期里做回归，大周期有趋势，就在小周期去做同向趋势-相反的不做。
            '''
            
            if self.policy == 'regression':
                self.short_stop_factor = 1.5
                self.long_stop_factor = 1.5
                self.short_take_profit_factor = 4.5
                self.long_take_profit_factor = self.short_take_profit_factor *1.1
                self.risk_reward_ratio_require = 3

                signal = signal   #回归，有趋势则做反向。不过止损可能要放宽。
            else:  #trend
                self.short_stop_factor = 1.5
                self.long_stop_factor = 1.5
                self.short_take_profit_factor = 6
                self.long_take_profit_factor = self.short_take_profit_factor *1.1
                self.risk_reward_ratio_require = 4

                signal = signal
            
            atr = factors_1m['atr'].mean()
            if self.position > 0:
                at_least_hold_time = 30 
            else:
                at_least_hold_time = 15  #没太大意义，主要是为代码不报错，只有上面两种情况会实际被使用到。

            if self.position != 0 and i % at_least_hold_time == 0:  
                if (self.position > 0 and signal == -2) or (self.position < 0 and signal == 2):
                    print("if (self.position > 0 and signal == -2) or (self.position < 0 and signal == 2)")
                    balance = self.uncondition_close(entry_price=entry_price,
                                        current_low=current_low,
                                        current_high=current_high,
                                        trades=trades,
                                        current_df_1m=current_df_1m,
                                        factors=factors_1m,
                                        balance=balance)
                    print(f'{current_df_1m.index[-1]} 放量, 停止已有仓位==================')
                    
                timeout_condition = i - open_index > int(statistic_len/4 if self.position < 0 else statistic_len)
                is_crossed_vpvr = factors_1m["is_normal_distribution"]
                if is_crossed_vpvr == True and self.has_been_crossed_vpvr == False:
                    self.has_been_crossed_vpvr = True

                #同时更新止盈和止损，但有各自的条件
                #止损仍然是以正态分布形成为准。
                #超时的情况下移动止盈目标，后续在判断是否止盈。满足则离场。  
                if timeout_condition and self.position != 0:
                    if self.position > 0:
                        self.take_profit_price -= atr
                    else:
                        self.take_profit_price += atr
                    open_index = i

                    if self.has_been_crossed_vpvr:
                        if self.position > 0:
                            self.stop_loss_price += atr
                        else:
                            self.take_profit_price -= atr

                if (self.position > 0 and self.hard_stop_vp_poc is not None and self.hard_stop_vp_poc >= current_low) \
                    or (self.position < 0 and self.hard_stop_vp_poc is not None and self.hard_stop_vp_poc <= current_high):
                    print('''if (self.position > 0 and self.hard_stop_vp_poc >= current_low) 
                    or (self.position < 0 and self.hard_stop_vp_poc <= current_high):''')
                    balance = self.uncondition_close(entry_price=entry_price,
                                        current_low=current_low,
                                        current_high=current_high,
                                        trades=trades,
                                        current_df_1m=current_df_1m,
                                        factors=factors_1m,
                                        balance=balance)
                    self.continuous_lost_times += 1

                _, volume_price_type = self.adjust_leverage(mean_of_zscore, till_current_df_1h.iloc[-24*3:].copy())
                        
                # 多头（看多）仓位的减仓条件  
                # 大比例减仓：市场出现明显下跌风险  
                long_big_reduce = ["放量下跌"]  
                # 少量减仓：市场下跌但风险较温和  
                long_small_reduce = ["缩量下跌", "平量下跌"]  
                # 空头（看空）仓位的减仓条件  
                # 大比例减仓：市场出现明显上涨风险  
                short_big_reduce = ["放量上涨"]  
                # 少量减仓：市场上涨但风险较温和  
                short_small_reduce = ["缩量上涨", "平量上涨"]  
                # 其他情况可默认持有仓位  
                continue_hold = ["放量持平", "缩量持平"]  

                fvg_3K = recent_fvg_3K(current_df_1m, factors_1m)
                # 检查止损
                if (self.position > 0 and current_low <= self.stop_loss_price):  # 多头止损
                    if factors_1m['is_normal_distribution'] or short_big_reduce:    
                        self.take_profit_price = current_low - 0.5 * atr
                    
                    elif (self.has_been_crossed_vpvr or fvg_3K['short_fvg']) and (current_low <= factors_1m['twvwap_smooth'].iloc[-1]):
                        if volume_price_type in short_small_reduce:
                            pass
                        else:
                            reduce = self.position / 4
                            if volume_price_type in long_big_reduce:  
                                reduce = self.position / 2
                            elif volume_price_type in short_big_reduce or volume_price_type in long_small_reduce :
                                reduce = self.position / 4 
                            elif volume_price_type in continue_hold:
                                reduce = self.position / 8
                            min_reduction = 2/8  # 这个值根据实际情况设置  
                            reduce = min(max(reduce, min_reduction), self.position)

                            balance += self.profit_of_current(current_low, reduce, entry_price) 
                            self.position -= reduce
                            self.continuous_lost_times = 0
                            
                            trade = {'timestamp': current_df_1m.index[-1], 'type': f'position={self.position} buy stop_loss', 'price': current_high, 'balance': balance}
                            trades.append(trade)

                            print('>>>>>>>>', trade)
                            
                            plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(),
                                            factors_1m, 
                                            open_price = entry_price,
                                            stop_loss = self.stop_loss_price,
                                            stop_profit = self.take_profit_price,
                                            additional=f'stop_loss')
                            if self.position == 0:
                                self.set_to_zero()
                        # balance += self.profit_of_current(current_low, self.position, entry_price) # 假设每次交易1个单位
                        
                        # trade = {'timestamp': current_df_1m.index[-1], 'type': 'buy stop_loss', 'price': current_low, 'balance': balance}
                        # trades.append(trade)
                        # print(trade)

                        # self.set_to_zero()
                        
                        if (current_low - entry_price) < 0:
                            self.continuous_lost_times += 1
                            if self.continuous_lost_times > 1:
                                #亏损了一次，需要跳过下次开仓，拉长统计的区间（因为当前区间的统计失效了)
                                i += int(statistic_len)
                                print(f'lost, skip to half period {i}')
                                if self.continuous_lost_times > 2:
                                    i += int(statistic_len)
                                    self.continuous_lost_times = 0
                                    print(f'lost twice, skip to period {i}')
                #空头止损
                elif (self.position < 0 and current_high >= self.stop_loss_price ):
                    if factors_1m['is_normal_distribution'] or long_big_reduce: 
                        self.take_profit_price = current_high + 0.5 * atr
                    
                    elif (self.has_been_crossed_vpvr or fvg_3K['long_fvg']) and current_low >= factors_1m['twvwap_smooth'].iloc[-1]:# 空头止损
                        if volume_price_type in long_small_reduce:
                            pass
                        else:
                            reduce = self.position / 4
                            if volume_price_type in short_big_reduce:  
                                reduce = self.position/2
                            elif volume_price_type in long_big_reduce or volume_price_type in short_small_reduce:
                                reduce = self.position/4 
                            elif volume_price_type in continue_hold:
                                reduce = self.position/8
                            min_reduction = 2/6  # 这个值根据实际情况设置  
                            reduce = min(max(reduce, min_reduction), self.position)

                            balance += self.profit_of_current(current_low, reduce, entry_price) 
                            self.position -= reduce  #负数减去负数也是正确的。
                            
                            self.continuous_lost_times = 0
                            
                            trade = {'timestamp': current_df_1m.index[-1], 'type': f'position{self.position} sell stop_loss', 'price': current_low, 'balance': balance}
                            trades.append(trade)
                            print('>>>>>>>>', trade)

                            if self.position == 0:
                                self.set_to_zero()
                        # plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(),
                        #                     factors_1m, 
                        #                     open_price = entry_price,
                        #                     stop_loss = self.stop_loss_price,
                        #                     stop_profit = self.take_profit_price,
                        #                     additional=f'stop_loss')
                        
                        # balance += self.profit_of_current(current_high, self.position, entry_price)# 假设每次交易1个单位
                        # trade = {'timestamp': current_df_1m.index[-1], 'type': 'sell stop_loss', 'price': current_high, 'balance': balance}
                        # trades.append(trade)
                        # print(trade)

                        # self.set_to_zero()

                        if (current_high - entry_price) > 0:
                            self.continuous_lost_times += 1
                            if self.continuous_lost_times > 1:
                                #亏损了一次，需要跳过下次开仓，拉长统计的区间（因为当前区间的统计失效了)
                                i += int(statistic_len)
                                print(f'lost, skip to half period {i}')
                                if self.continuous_lost_times > 2:
                                    i += int(statistic_len)
                                    self.continuous_lost_times = 0
                                    print(f'lost twice, skip to period {i}')

                
                # 多头止盈
                if self.position > 0 and current_high >= self.take_profit_price:  # 多头止盈

                    if volume_price_type in short_small_reduce:
                        pass
                    else:
                        reduce = self.position / 8 #未确定的减仓逻辑
                        if volume_price_type in long_big_reduce:  
                            reduce = self.position / 2
                        elif volume_price_type in short_big_reduce or volume_price_type in long_small_reduce :
                            reduce = self.position / 4 
                        elif volume_price_type in continue_hold:
                            reduce = self.position / 8
                        min_reduction = 2/(8 * 2)  # 这个值根据实际情况设置  
                        reduce = min(max(reduce, min_reduction), self.position)

                        # reduce = self.position
                        
                        balance += self.profit_of_current(current_low, reduce, entry_price) 
                        self.position -= reduce
                        self.continuous_lost_times = 0
                        
                        trade = {'timestamp': current_df_1m.index[-1], 'type': f'position={self.position} buy take_profit', 'price': current_high, 'balance': balance}
                        trades.append(trade)

                        print('>>>>>>>>', trade)
                        
                    if self.policy == 'trend' and adx_up_condition and not fvg_3K['short_fvg']:
                        self.setup_trending_tp_sl(1, current_price=current_price, factors_1m=factors_1m, move_profit=True)
                        self.stop_loss_price = max(current_price - self.long_stop_factor*atr, max(factors_1m['twvwap'].iloc[-1], self.stop_loss_price) ) #多头的止损，向上截断最优。,最多破量价线结束。

                        plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(),
                                            factors_1m, 
                                            open_price = entry_price,
                                            stop_loss = self.stop_loss_price,
                                            stop_profit = self.take_profit_price,
                                            additional=f'move_profit')
                        
                        print(f'move profit at time {current_df_1m.index[-1]}\n')
                    # else:
                    #     print(f'policy = {self.policy} adx_up_condition={adx_up_condition} adx_down_condition={adx_down_condition}')
                    #     plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(),
                    #                         factors_1m, 
                    #                         open_price = entry_price,
                    #                         stop_loss = self.stop_loss_price,
                    #                         stop_profit = self.take_profit_price,
                    #                         additional=f'take_profit')
                        
                    #     balance += self.profit_of_current(current_low, self.position, entry_price) 
                    #     self.continuous_lost_times = 0

                    #     trade = {'timestamp': current_df_1m.index[-1], 'type': 'buy take_profit', 'price': current_high, 'balance': balance}
                    #     trades.append(trade)

                    #     print(trade)
                    #     print(f"^^^^^^^^^reason: not match condition === self.policy == 'trend' and adx_up_condition and not fvg_3K['short_fvg']")
                    #     self.set_to_zero()
                #空头严格卡时间。 # 多头止盈
                elif self.position < 0 and (timeout_condition or current_low <= self.take_profit_price):  # 空头止盈
                    if volume_price_type in long_small_reduce:
                        pass
                    else:
                        reduce = self.position / 8  #未确定的减仓逻辑
                        if volume_price_type in short_big_reduce:  
                            reduce = self.position/2
                        elif volume_price_type in long_big_reduce or volume_price_type in short_small_reduce:
                            reduce = self.position/4 
                        elif volume_price_type in continue_hold:
                            reduce = self.position/8
                        min_reduction = 2/(6*2)  # 这个值根据实际情况设置  
                        reduce = min(max(reduce, min_reduction), self.position)
                        
                        # reduce = self.position

                        balance += self.profit_of_current(current_low, reduce, entry_price) 
                        self.position -= reduce  #负数减去负数也是正确的。
                        
                        self.continuous_lost_times = 0
                        
                        trade = {'timestamp': current_df_1m.index[-1], 'type': f'position={self.position} sell take_profit', 'price': current_low, 'balance': balance}
                        trades.append(trade)

                        print('>>>>>>>>', trade)
                        
                    #TODO 考虑平仓一部分以抵消可能回调止损的部分。
                    if not timeout_condition and self.policy == 'trend' and adx_down_condition and not fvg_3K['short_fvg']:
                        self.setup_trending_tp_sl(-1, current_price=current_price, factors_1m=factors_1m, move_profit=True)
                        self.stop_loss_price = min(current_price - self.long_stop_factor*atr,  min(factors_1m['twvwap'].iloc[-1], self.stop_loss_price))  #最多破量价线结束。
                        plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(),
                                            factors_1m, 
                                            open_price = entry_price,
                                            stop_loss = self.stop_loss_price,
                                            stop_profit = self.take_profit_price,
                                            additional=f'move_profit')
                        print(f'move profit at time {current_df_1m.index[-1]}\n')
                    # else:
                    #     print(f'policy = {self.policy} adx_up_condition={adx_up_condition} adx_down_condition={adx_down_condition}')
                    #     plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(),
                    #                         factors_1m, 
                    #                         open_price = entry_price,
                    #                         stop_loss = self.stop_loss_price,
                    #                         stop_profit = self.take_profit_price,
                    #                         additional=f'take_profit')
                        
                    #     balance +=  self.profit_of_current(current_high, self.position, entry_price) 
                    #     self.continuous_lost_times = 0
                        
                    #     trade = {'timestamp': current_df_1m.index[-1], 'type': f'{self.position} sell take_profit', 'price': current_low, 'entry_price': entry_price, 'take_profit_price':self.take_profit_price, 'balance': balance}
                    #     trades.append(trade)

                    #     print(trade)
                    #     print(f"^^^^^^^^^reason: not match condition === not timeout_condition and self.policy == 'trend' and adx_down_condition and not fvg_3K['short_fvg']")

                    #     self.set_to_zero()
            #开仓        
            #TODO 还可以做反抽（在上涨趋势中，如果小周期到了负的两倍标准差，就开多，止损位一个atr单位，止盈为正向的2倍标准差。
            if signal == 1 and self.position == 0 and (
                (self.policy == 'regression' and factors_1m['is_normal_distribution']) or 
                (self.policy == 'trend' and not  factors_1m['is_normal_distribution'])):  # 买入
                
                # poc_price, is_normal_distribution = find_factor_condition(all_caculated_1m_df, all_factors_1m, start_index=i, statistics_len=statistic_len, 
                #                                                       condition_func=lambda fctr:fctr['is_normal_distribution'])
                # if not is_normal_distribution:
                #     poc_price = factors_1m['poc_price']
                poc_price = factors_1m['poc_price']
            
                adx_condition = (self.policy == 'trend' and adx_down_condition == True) 
                poc_condition = poc_price > current_price  #在poc下方做多，虽然有性价比，但是概率而言，难以将它作为初次开仓的止损（支撑）

                
                if ( adx_condition or poc_condition):
                    if self.policy == 'regression':
                        print(f"{current_df_1m.index[-1]} i={i} won't open {signal}, adx={factors_1m['adx'].iloc[-1]}")
                    else:  #trend
                        print(f"{current_df_1m.index[-1]} i={i} won't open {signal}, poc_condition={poc_condition} adx_up_condition={adx_up_condition} adx_down_condition={adx_down_condition} adx={factors_1m['adx'].iloc[-1]}")
                    plot_flex_ta_dashboard(df_1h_show, #df_1h.iloc[:len_of_hour_df+50].copy(),
                                        factors_1h,
                                        open_price = entry_price,
                                        stop_loss=self.stop_loss_price,
                                        stop_profit=self.take_profit_price,
                                        additional=f'no_open_1h_original')
                    i += 1
                    continue
            
                self.leverage = adjusted_leverage
                self.last_adjusted_leverrage = adjusted_leverage

                if self.policy == 'regression':
                    self.setup_regression_tp_sl(signal=signal, current_price=current_low, factors_1m=factors_1m)
                    self.hard_stop_vp_poc = factors_1m['vwap'].iloc[-1] + 3*atr
                else:  #trend
                    self.setup_trending_tp_sl(signal=signal, current_price=current_low, factors_1m=factors_1m)
                    self.hard_stop_vp_poc = None
                

                estimated_rrr = (self.take_profit_price - current_high)/(current_high - self.stop_loss_price)
                loss_percent = abs(current_high-self.stop_loss_price)/current_high
                if estimated_rrr >= self.risk_reward_ratio_require and loss_percent <= self.max_risk_lost: #atr_1m_std.iloc[-1]*1.6:
                    #这里有比较大的优化空间，
                    # 1，开仓可以在本次信号后较低的价格成交以最大程度获得成本优势。
                    # 2.要考虑下一个K线的最低点无法踩到本次收盘价的情况--当做建仓失败。
                    
                    if self.policy == 'regression':
                        percent = (df.iloc[i+1]['low'] - current_high)/current_high
                        if percent <= -0.01/100:  #下一跟K线是跌的，才能买得到.
                            entry_price = current_high + (df.iloc[i+1]['low'] - current_high)/2
                        else:
                            print(f'long: percent={percent} 基于本次收盘价的下一跟K线的开仓价不支持本次开仓，这里简化了收到信号后下次开仓的逻辑')
                            i += 1
                            plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(), 
                                            factors_1m,
                                            open_price = current_high,
                                            stop_loss = self.stop_loss_price,
                                            stop_profit = self.take_profit_price,
                                            additional='+'*8 + f'open failed')
                            continue
                    else:
                        entry_price = current_high
                    self.position = balance * self.leverage/entry_price
                    open_index = i
                    
                    trade = {'timestamp': current_df_1m.index[-1], 'leverage':self.leverage, 'type': f'{self.position} open buy {open_vwap_type} policy={self.policy}', 'price': entry_price, 'balance': balance, }
                    trades.append(trade)
                    print(f'\nrisk loss_percent={loss_percent}\n', trade)
                    plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df) + 50, i+50)].copy(), 
                                           factors_1m,
                                           open_price = entry_price,
                                            stop_loss = self.stop_loss_price,
                                            stop_profit = self.take_profit_price,
                                            additional=f'original_{self.leverage}')
                    plot_flex_ta_dashboard(current_df_1m, factors_1m, 
                                           open_price = entry_price,
                                           additional='realtime')

                    plot_flex_ta_dashboard(df_1h_show, #df_1h_all.iloc[:len_of_hour_df+50].copy(),
                                            factors_1h,
                                            open_price = entry_price,
                                            stop_loss = self.stop_loss_price,
                                            stop_profit = self.take_profit_price,
                                            additional=f'1h_original_{self.leverage}')
                else:
                    print(f'+++++++unopened {signal} because reward/risk ratio {estimated_rrr} not good {current_df_1m.index[-1]} or loss_percent={loss_percent}')
                    plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(), 
                                            factors_1m,
                                            open_price = current_high,
                                            stop_loss = self.stop_loss_price,
                                            stop_profit = self.take_profit_price,
                                            additional='+'*4 +f'not_good')
                    
            #开空
            elif signal == -1 and self.position == 0 and (
                (self.policy == 'regression' and factors_1m['is_normal_distribution']) or 
                (self.policy == 'trend' and not factors_1m['is_normal_distribution'])):  # 卖出

                #regression need to judge the slope trend. 
                #while trend need to juedege +di and -di for the adx direction.
                # poc_price, is_normal_distribution = find_factor_condition(all_caculated_1m_df, all_factors_1m, 
                #                                                           start_index=i, statistics_len=statistic_len, 
                #                                                           condition_func=lambda fctr:fctr['is_normal_distribution'])
                # if not is_normal_distribution:
                #     poc_price = factors_1m['poc_price']
                
                poc_price = factors_1m['poc_price']
                if ((self.policy == 'trend' and adx_up_condition == True)
                    or poc_price < current_price):
                    if self.policy == 'regression':
                        print(f"{current_df_1m.index[-1]} {i} won't open {signal}, adx={factors_1m['adx'].iloc[-1]}")
                    else:  #trend
                        print(f"{current_df_1m.index[-1]} {i} won't open {signal}, adx_up_condition={adx_up_condition} adx_down_condition={adx_down_condition} adx={factors_1m['adx'].iloc[-1]}")

                    plot_flex_ta_dashboard(df_1h_show, #df_1h_all.iloc[:len_of_hour_df+50].copy(),
                                            factors_1h,
                                            open_price = entry_price,
                                            stop_loss=self.stop_loss_price,
                                            stop_profit=self.take_profit_price,
                                            additional=f'no_open_1h_original')
                    i += 1
                    continue
                print(f'close because signal {signal}')
                balance = self.uncondition_close(entry_price=entry_price,
                                        current_low=current_low,
                                        current_high=current_high,
                                        trades=trades,
                                        current_df_1m=current_df_1m,
                                        factors=factors_1m,
                                        balance=balance)
                
                self.leverage = adjusted_leverage
                self.last_adjusted_leverrage = adjusted_leverage

                if self.policy == 'regression':
                    self.setup_regression_tp_sl(signal=signal, current_price=current_high, factors_1m=factors_1m)
                    self.hard_stop_vp_poc = factors_1m['vwap'].iloc[-1] - 3*atr
                else:  #tred
                    self.setup_trending_tp_sl(signal=signal, current_price=current_high, factors_1m=factors_1m)
                    self.hard_stop_vp_poc = None

                estimated_rrr = (self.take_profit_price - current_low)/(current_low - self.stop_loss_price)
                loss_percent = abs(current_low-self.stop_loss_price)/current_low
                if estimated_rrr >= self.risk_reward_ratio_require and loss_percent <= self.max_risk_lost: #atr_1m_std.iloc[-1]*1.6:
                    #这里有比较大的优化空间，
                    # 1，开仓可以在本次信号后较低的价格成交以最大程度获得成本优势。
                    # 2.要考虑下一个K线的最低点无法踩到本次收盘价的情况--当做建仓失败。
                    percent = (df.iloc[i+1]['high'] - current_low)/current_low
                    if self.policy == 'regression':
                        if percent <= 0.01/100:  #下一跟K线是跌的，才能买得到.
                            # entry_price = current_low
                            entry_price = current_low + (df.iloc[i+1]['low'] - current_low)/2
                        else:
                            print(f'short: percent={percent} 基于本次收盘价的下一跟K线的开仓价不支持本次开仓，这里简化了收到信号后下次开仓的逻辑')
                            i += 1

                            plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(), 
                                            factors_1m,
                                            open_price = current_low,
                                            stop_loss = self.stop_loss_price,
                                            stop_profit = self.take_profit_price,
                                            additional='+'*8 + f'open failed')
                            
                            continue
                    else:
                        entry_price = current_low
                    self.position = -(balance * self.leverage/entry_price)
                    open_index = i
                    
                    trade = {'timestamp': current_df_1m.index[-1], 'leverage':self.leverage, 'type': f' {self.position} open sell {open_vwap_type} policy={self.policy}', 'price': entry_price, 'take_profit_price':self.take_profit_price, 'balance': balance, }
                    trades.append(trade)
                    print(f'\nrisk loss_percent={loss_percent}\n', trade)

                    plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df) + 50, i+50)].copy(), 
                                           factors_1m, 
                                           open_price = entry_price,
                                        stop_loss = self.stop_loss_price,
                                        stop_profit = self.take_profit_price,
                                        additional=f'original_{self.leverage}')
                    plot_flex_ta_dashboard(current_df_1m, factors_1m, 
                                           open_price = entry_price,
                                           additional='realtime')
                    plot_flex_ta_dashboard(df_1h_show, #df_1h_all.iloc[:len_of_hour_df+50].copy(),
                                        factors_1h,
                                        open_price = entry_price,
                                        stop_loss = self.stop_loss_price,
                                        stop_profit = self.take_profit_price,
                                        additional=f'1h_original_{self.leverage}')
                else:
                    print(f'-----unopened  {signal} because reward/risk ratio {estimated_rrr} not good{current_df_1m.index[-1]} or loss_percent={loss_percent}')
                    plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(), 
                                           factors_1m,
                                           open_price = current_low,
                                        stop_loss = self.stop_loss_price,
                                        stop_profit = self.take_profit_price,
                                        additional='+'*4 + f'not_good')

            # 计算当前权益
            current_equity = balance +   self.profit_of_current(current_price, self.position, entry_price) 
            equity_curve.append(round(current_equity, 2))

            if current_equity <= 0:
                print(f'{current_df_1m.index[-1]} !!!!!!!!!!!!!!!!!!!爆仓')
                plot_flex_ta_dashboard(df.iloc[i-statistic_len:min(len(df), i+50)].copy(),
                                factors_1m, 
                                open_price = entry_price,
                                stop_loss = self.stop_loss_price,
                                stop_profit = self.take_profit_price,
                                additional=f'exploded')
                break
            
            i += 1
            if self.position != 0:
                self.tally_hold_time += 1
                if (
                    (self.position < 0 and self.tally_hold_time >= self.max_short_hold_time)  #24*12:5m
                    or (self.position > 0 and self.tally_hold_time >= self.max_long_hold_time) 
                    ):
                    print('timeoue')
                    balance = self.uncondition_close(entry_price=entry_price,
                                        current_low=current_low,
                                        current_high=current_high,
                                        trades=trades,
                                        current_df_1m=current_df_1m,
                                        factors=factors_1m,
                                        balance=balance,
                                        is_timeout=True)
                
        # 计算夏普比率 (年化)
        equity_curve = pd.Series(equity_curve)
        returns = equity_curve.pct_change().dropna()
        annualized_sharpe_ratio = np.sqrt(252 * 24 * 60) * returns.mean() / returns.std() if timeframe == '1m' else \
            np.sqrt(365 * 24 * 12) * returns.mean() / returns.std() if timeframe == '5m' else \
            np.sqrt(365 * 24 * 4) * returns.mean() / returns.std() if timeframe == '15m' else \
                np.sqrt(365 * 24) * returns.mean() / returns.std() if timeframe == '1h' else \
                    np.sqrt(265) * returns.mean() / returns.std()

        return {
            'sharpe_ratio': annualized_sharpe_ratio,
            'trades': trades,
            'equity_curve': equity_curve.to_list()
        }
    
    def profit_of_current(self, current_price, reduce_position, entry_price):
        return reduce_position * (current_price - entry_price) - (current_price + entry_price)*0.0002 * abs(reduce_position)
    
    def uncondition_close(self, entry_price, current_low, current_high, trades, current_df_1m, 
                          factors, balance, is_timeout = False):
        if self.position < 0:  # 如果有空头仓位，先平仓,并结算空头盈亏
            balance += self.profit_of_current(current_low, self.position, entry_price)
            type = f"{'is_timeout' if is_timeout else 'reverse'} sell take_profit"
            trade  = {'timestamp': current_df_1m.index[-1], 
                      'type': type,
                      'price': current_low, 'balance': balance}
            trades.append(trade)
            print(trade)
            
            plot_flex_ta_dashboard(current_df_1m,
                                    factors, 
                                    open_price = entry_price,
                                    stop_loss = self.stop_loss_price,
                                    stop_profit = self.take_profit_price,
                                    additional=type)
            
            self.set_to_zero()
        elif self.position > 0:  # 如果有多头仓位，先平仓，并结算多头盈亏
            type = f"{'is_timeout' if is_timeout else 'reverse'} buy take_profit"
            balance += self.profit_of_current(current_high, self.position, entry_price)
            trade = {'timestamp': current_df_1m.index[-1],
                     'type':type, 
                     'price': current_high, 'balance': balance}
            trades.append(trade)
            print(trade)

            plot_flex_ta_dashboard(current_df_1m,
                                    factors, 
                                    open_price = entry_price,
                                    stop_loss = self.stop_loss_price,
                                    stop_profit = self.take_profit_price,
                                    additional=type)
            
            self.set_to_zero()

        return balance

    def set_to_zero(self):
        self.position, self.stop_loss_price, self.take_profit_price = 0, 0, 0
        self.tally_hold_time = 0
        self.has_been_crossed_vpvr = False

    def adjust_leverage(self, mean_of_zscore, till_current_df_1h):  
        # 计算短期和长期移动平均线  
        till_current_df_1h['MA5'] = till_current_df_1h['close'].rolling(window=5).mean()  
        till_current_df_1h['MA20'] = till_current_df_1h['close'].rolling(window=20).mean()  

        # 计算价格及量的变化  
        price_change = till_current_df_1h['close'].iloc[-1] - till_current_df_1h['close'].iloc[-2]  
        volume_change = till_current_df_1h['volume'].iloc[-1] - till_current_df_1h['volume'].iloc[-2]  
        
        # 计算过去N天的平均价格涨幅和平均成交量  
        N = 5  # 自定义周期长度  
        avg_price_change = (till_current_df_1h['close'].iloc[-1] - till_current_df_1h['close'].iloc[-N-1]) / N  
        avg_volume_change = till_current_df_1h['volume'].rolling(window=N).mean().iloc[-1]  
        
        # 初始化量价状态  
        volume_price_type = "未确定"  

        # 判断量价关系及放量情况  
        if price_change > avg_price_change and volume_change > avg_volume_change:  
            volume_price_type = "放量上涨"  
        elif price_change > 0 and volume_change < 0:  
            volume_price_type = "缩量上涨"  
        elif price_change < 0 and volume_change > 0:  
            volume_price_type = "放量下跌"   
        elif price_change < 0 and volume_change < 0:  
            volume_price_type = "缩量下跌"  
        elif price_change == 0 and volume_change > 0:  
            volume_price_type = "放量持平"   
        elif price_change == 0 and volume_change < 0:  
            volume_price_type = "缩量持平"   
        elif price_change > 0 and volume_change == 0:  
            volume_price_type = "平量上涨"  
        elif price_change < 0 and volume_change == 0:  
            volume_price_type = "平量下跌"  

        # 波动率分析（使用真实波动范围或类似指标）  
        atr = till_current_df_1h['close'].rolling(window=14).std()   
        max_volatility = atr.max()  
        min_volatility = atr.min()  

        if max_volatility > min_volatility:  
            normalized_volatility = (atr.iloc[-1] - min_volatility) / (max_volatility - min_volatility)  
        else:  
            normalized_volatility = 0   

        # 按百分比调整杠杆  
        leverage = 5  # 默认杠杆值  

        # 根据量价关系类型和z-score调整杠杆  
        if volume_price_type == "放量上涨":  
            if mean_of_zscore > 0.5:  
                leverage = 3 * (1 - normalized_volatility)  # 强势趋势，加大杠杆  
            elif mean_of_zscore > 0:  
                leverage = 4 * (1 - normalized_volatility)  # 较强的上涨趋势  
            else:  
                leverage = 5 * (1 - normalized_volatility)  # 温和上涨  
            
        elif volume_price_type == "缩量上涨":  
            if mean_of_zscore > 0:  
                leverage = 5 * (1 - normalized_volatility)  # 谨慎加杠杆  
            else:  
                leverage = 4 * (1 - normalized_volatility)  # 保守策略  
            
        elif volume_price_type == "放量下跌":  
            if mean_of_zscore < -0.5:  
                leverage = 3 * (1 - normalized_volatility)  # 明显的下行趋势  
            elif mean_of_zscore < 0:  
                leverage = 4 * (1 - normalized_volatility)  # 较强的下跌趋势  
            else:  
                leverage = 5 * (1 - normalized_volatility)  # 温和下跌  
        
        elif volume_price_type == "缩量下跌":  
            if mean_of_zscore > 0:  
                leverage = 5 * (1 - normalized_volatility)  # 谨慎加杠杆  
            else:  
                leverage = 4 * (1 - normalized_volatility)  # 保守策略  
            
        elif volume_price_type == "放量持平":  
            leverage = 5 * (1 - normalized_volatility)  # 稳妥待机  
            
        elif volume_price_type == "缩量持平":  
            leverage = 5 * (1 - normalized_volatility)  # 继续观望  
            
        elif volume_price_type == "平量上涨":  
            leverage = 5 * (1 - normalized_volatility)  # 在持平时适量加杠杆  
            
        elif volume_price_type == "平量下跌":  
            leverage = 5 * (1 - normalized_volatility)  # 稳妥防守  

        # 确保杠杆在合理范围内（例如，1至10之间）  
        leverage = max(3, min(leverage, 6))  
        print(volume_price_type, 'leverage:',leverage, till_current_df_1h.index[-1], till_current_df_1h['close'].iloc[-1])

        return leverage, volume_price_type 

# 示例：交易信号生成函数
def simple_twvwap_signal(df, factors, calculate_dynamic_period_func, analyze_price_distribution_func):
    """
    基于TWVWAP标准差的简单交易信号生成函数。

    Args:
        df: 包含价格数据的DataFrame。
        factors: 包含TWVWAP相关数据的字典。
        calculate_dynamic_period_func: 动态周期计算函数。
        analyze_price_distribution_func: 价格分布分析函数。

    Returns:
        交易信号 (1: 买入, -1: 卖出, 0: 无操作)。
    """
    
    # 计算动态周期
    period = calculate_dynamic_period_func(df, factors)


    
    # 交易逻辑
    current_price = df['close'].iloc[-1]
    current_low = (df['low'].iloc[-1] + current_price) / 2
    current_high = (df['high'].iloc[-1] + current_price) / 2

    twvwap_avwap_ratio = factors['twvwap'] / factors['avwap']
    twvwap_avwap_ratio_mean = twvwap_avwap_ratio.rolling(20).mean().iloc[-1]
    twvwap_avwap_ratio_std = twvwap_avwap_ratio.rolling(20).std().iloc[-1]
    
    volume_condition, up, down = shirnk_volume_support(df)  # 有价无量是假回踩，有量有价是真反转
    if not volume_condition:# 缩量上涨或者放量下跌，可以做多  由于一直没有信号，所以实际上，可能需要大周期为2根K线的量价为基础，再来小四个级别的周期（这里是1m）来找成交量相反的（但是价格趋势相同）信号进行入场（因为利润是最大化的）
        
        
        #解释：例如做多的时候，current_low在std1的上方，说明价格穿过了std1
        # （最好的时机应该是反穿的时候，避免逆势交易。 上一个收盘价在std1-的下方，新的K线的下半部分上穿std1--也就是站到std1上方)
        if up:
            # price_kicked_band_2 = False
            # for i in range(int(len(df)/10)):
            #     if df['close'].iloc[-1-i] <= factors['twvwap_lower_band_2'].iloc[-1-i]:
            #         price_kicked_band_2 = True
            #         break
            # 第一个条件块：做多 (1,0)
            # upscore_condition = analysis['upscore'] < 0.8
            # price_kicked_band_2_condition = not price_kicked_band_2  # 上涨止损收紧，不希望触摸下轨2
            # current_low_above_lower_band_1 = current_low >= factors['twvwap_lower_band_1'].iloc[-1]  # 当前低价在下轨1之上
            # prev_close_below_lower_band_1 = df['close'].iloc[-2] <= factors['twvwap_lower_band_1'].iloc[-2]  # 上一次收盘在下轨1之下
            twvwap_to_avwap_ratio_1 = factors['twvwap'].iloc[-1] / factors['avwap'].iloc[-1]  # twvwap 和 avwap 的比率
            twvwap_trend_condition_1 = twvwap_to_avwap_ratio_1 >= twvwap_avwap_ratio_mean  # twvwap 在多头趋势中
            macd_positive_1 = factors['macd'].iloc[-1] >= 0  # MACD >= 0
            # rsi_normal = factors['stoch_k'].iloc[-1] <= 90
            current_below_upper_band_1 = current_price < factors['twvwap_upper_band_1'].iloc[-1]
            not_deviate_much = factors['twvwap'].iloc[-1] - factors['vwap'].iloc[-1] <= factors['twvwap_std_dev'].mean()
            not_deviate_much_to_smooth = factors['twvwap'].iloc[-1] - factors['twvwap_smooth'].iloc[-1] <= factors['twvwap_std_dev'].mean()
            if (
                # price_kicked_band_2_condition
                # and upscore_condition
                # and current_low_above_lower_band_1
                # and prev_close_below_lower_band_1
                True
                and twvwap_trend_condition_1
                and macd_positive_1
                # and rsi_normal
                and current_below_upper_band_1
                and not_deviate_much
                and not_deviate_much_to_smooth
            ):
                analysis = analyze_price_distribution_func(df, factors, period=period)
                if analysis['trend'] == 'Strong Uptrend' and up:
                    return (1, 0)  # 做多

            def price_kicked_band_1(factors):
                cdt = False
                for i in range(int(len(df)/10)):
                    if df['close'].iloc[-1-i] <= factors['twvwap_lower_band_1'].iloc[-1-i]:
                        cdt = True
                        break
                return cdt

            # 第二个条件块：做多 (1,1)
            
            twvwap_std_dev_half = 0.5 * factors['twvwap_std_dev'].iloc[-1]  # twvwap 标准差的一半
            current_low_above_twvwap_adjusted = current_low >= factors['twvwap'].iloc[-1] - twvwap_std_dev_half  # 当前低价在 twvwap - 0.5*std_dev 之上
            prev_close_below_twvwap_adjusted = df['close'].iloc[-2] <= factors['twvwap'].iloc[-2] - twvwap_std_dev_half  # 上一次收盘在 twvwap - 0.5*std_dev 之下
            if (
                True
                # and current_low_above_twvwap_adjusted
                # and prev_close_below_twvwap_adjusted
                and twvwap_trend_condition_1
                and macd_positive_1
                # and rsi_normal
                and current_below_upper_band_1
                and not_deviate_much
                and not_deviate_much_to_smooth
                and price_kicked_band_1(factors)
            ):
                analysis = analyze_price_distribution_func(df, factors, period=period)
                if analysis['upscore'] > 0.8:  # 上涨得分 > 0.8
                    return (1, 1)  # 做多,物极必反，超过了0.8的多，可以认为要下跌了。
        elif down:   
        
            #反抽逻辑：本来去下跌趋势，但是短期内K先向上突破了std1+上方，回头的时候，上一个蜡烛在上方，这跟蜡烛图的高点在std+的下方（反向站稳）
            # price_kicked_band_1 = False
            # for i in range(int(len(df)/10)):
            #     if df['close'].iloc[-1-i] >= factors['twvwap_upper_band_1'].iloc[-1-i]:
            #         price_kicked_band_1 = True
            #         break
            # 第一个条件块：做空 (-1, 0)
            # downscore_condition = analysis['downscore'] <= 0.8
            # price_kicked_band_1 = price_kicked_band_1  # 下跌止损大，希望触摸上轨2
            current_high_below_upper_band_1 = current_high <= factors['twvwap_upper_band_1'].iloc[-1]  # 当前高价在上轨1之下
            prev_close_below_upper_band_1 = df['close'].iloc[-2] >= factors['twvwap_upper_band_1'].iloc[-2]  # 上一次收盘在上轨1之上
            
            twvwap_to_avwap_ratio_1 = factors['twvwap'].iloc[-1] / factors['avwap'].iloc[-1]  # twvwap 和 avwap 的比率
            twvwap_trend_condition_short_1 = twvwap_to_avwap_ratio_1 <= twvwap_avwap_ratio_mean  # twvwap 在空头趋势中
            macd_negative_1 = factors['macd'].iloc[-1] <= 0  # MACD <= 0
            rsi_normal = factors['stoch_k'].iloc[-1] >= 10
            current_above_lower_band_1 = current_price > factors['twvwap_lower_band_1'].iloc[-1]
            not_deviate_much = factors['vwap'].iloc[-1] - factors['twvwap'].iloc[-1] <= factors['twvwap_std_dev'].mean()
            not_deviate_much_to_smooth = factors['twvwap_smooth'].iloc[-1] - factors['twvwap'].iloc[-1] <= factors['twvwap_std_dev'].mean()
            if (
                True
                # and price_kicked_band_1
                # and downscore_condition
                # and current_high_below_upper_band_1
                # and prev_close_below_upper_band_1
                and twvwap_trend_condition_short_1
                and macd_negative_1
                # and rsi_normal
                and current_above_lower_band_1
                and not_deviate_much
                and not_deviate_much_to_smooth
            ):
                analysis = analyze_price_distribution_func(df, factors, period=period)
                if analysis['trend'] == 'Strong Downtrend': 
                    return (-1, 0)  # 做空

            def price_kicked_band_1(factors):
                cdt = False
                for i in range(int(len(df)/10)):
                    if df['close'].iloc[-1-i] <= factors['twvwap_upper_band_1'].iloc[-1-i]:
                        cdt = True
                        break
                return cdt
            
            # 第二个条件块：做空 (-1, -1)
            twvwap_std_dev_half = 0.5 * factors['twvwap_std_dev'].iloc[-1]  # twvwap 标准差的一半
            current_high_below_twvwap_adjusted = current_high <= factors['twvwap'].iloc[-1] + twvwap_std_dev_half  # 当前高价在 twvwap + 0.5*std_dev 之下
            prev_close_below_twvwap_adjusted = df['close'].iloc[-2] >= factors['twvwap'].iloc[-2] + twvwap_std_dev_half  # 上一次收盘在 twvwap + 0.5*std_dev 之下

            if (
                True
                # and current_high_below_twvwap_adjusted
                # and prev_close_below_twvwap_adjusted
                # and twvwap_trend_condition_short_1
                and macd_negative_1
                # and rsi_normal
                and current_above_lower_band_1
                and not_deviate_much
                and not_deviate_much_to_smooth
                and price_kicked_band_1(factors)
            ):
                analysis = analyze_price_distribution_func(df, factors, period=period)
                if analysis['downscore'] > 0.8:  # 下跌得分 > 0.8
                    return (-1, 1)  # 做空
                
        diverse_condition, diverse_up, diverse_down = scaleup_volume_support(df)
        analysis = analyze_price_distribution_func(df, factors, period=period)
        if diverse_condition:
            if analysis['trend'] == 'Strong Uptrend':
                if diverse_up:
                    return -2
            if analysis['trend'] == 'Strong Downtrend':
                if diverse_down:
                    return 2
        
        if analysis['upscore'] >= 0.8:
            return (0.5, 0)
        if analysis['downscore'] >= 0.8:
            return (-0.5, 0)
    return (0,0)
