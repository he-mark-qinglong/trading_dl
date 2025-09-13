# 定义开多仓和平仓规则  
long_rules = [  
    {  
        "type": "开仓",  
        "conditions": {  
            "1m": {"trend": "uptrend", "divergence": False},  
            "5m": {"trend": "uptrend", "divergence": False},  
            "15m": {"trend": "uptrend", "divergence": False},  
            "1h": {"trend": "uptrend", "divergence": False},  
            "4h": {"trend": "uptrend", "divergence": False},  
            "1d": {"trend": "uptrend", "divergence": False},  
        },  
        "action": "开多仓",  
        "size": 1.0,  # 100% 仓位  
        "description": "趋势完全一致，无背离，市场方向性极强，50倍杠杆，100%开仓。",  
    },  
    {  
        "type": "开仓",  
        "conditions": {  
            "1m": {"trend": "uptrend", "divergence": False},  
            "5m": {"trend": "uptrend", "divergence": False},  
            "15m": {"trend": "uptrend", "divergence": False},  
            "1h": {"trend": "uptrend", "divergence": False},  
            "4h": {"trend": "uptrend", "divergence": False},  
            "1d": {"trend": "range", "divergence": False},  
        },  
        "action": "开多仓",  
        "size": 1.0,  # 100% 仓位  
        "description": "大周期震荡，小周期趋势一致，市场短期方向明确，50倍杠杆，100%开仓。",  
    },  
    {  
        "type": "平仓",  
        "conditions": {  
            "1m": {"trend": "range", "divergence": False},  
            "5m": {"trend": "range", "divergence": False},  
            "15m": {"trend": "range", "divergence": False},  
            "1h": {"trend": "range", "divergence": False},  
            "4h": {"trend": "range", "divergence": False},  
            "1d": {"trend": "range", "divergence": False},  
        },  
        "action": "全部平仓",  
        "size": 1.0,  # 全部平仓  
        "description": "趋势一致性降低，所有时间框架变为震荡，市场方向不明确，立即平仓锁定利润。",  
    },  
    {  
        "type": "平仓",  
        "conditions": {  
            "1m": {"trend": "downtrend", "divergence": False},  
            "5m": {"trend": "downtrend", "divergence": False},  
            "15m": {"trend": "downtrend", "divergence": False},  
            "1h": {"trend": "downtrend", "divergence": False},  
            "4h": {"trend": "downtrend", "divergence": False},  
            "1d": {"trend": "downtrend", "divergence": False},  
        },  
        "action": "全部平仓",  
        "size": 1.0,  # 全部平仓  
        "description": "趋势反转，所有时间框架变为下降趋势，立即平仓以规避风险。",  
    },  
    {  
        "type": "平仓",  
        "conditions": {  
            "1m": {"trend": "downtrend", "divergence": False},  
            "5m": {"trend": "downtrend", "divergence": False},  
            "15m": {"trend": "downtrend", "divergence": False},  
            "1h": {"trend": "downtrend", "divergence": False},  
            "4h": {"trend": "downtrend", "divergence": False},  
            "1d": {"trend": "range", "divergence": False},  
        },  
        "action": "全部平仓",  
        "size": 1.0,  # 全部平仓  
        "description": "趋势反转，小周期变为下降趋势，大周期仍为震荡，立即平仓以规避风险。",  
    },  
    {  
        "type": "平仓",  
        "conditions": {  
            "1m": {"trend": "range", "divergence": False},  
            "5m": {"trend": "range", "divergence": False},  
            "15m": {"trend": "range", "divergence": False},  
            "1h": {"trend": "range", "divergence": False},  
            "4h": {"trend": "range", "divergence": False},  
            "1d": {"trend": "range", "divergence": False},  
        },  
        "action": "部分平仓",  
        "size": 0.5,  # 部分平仓  
        "description": "趋势一致性降低，小周期变为震荡，部分平仓锁定利润，剩余仓位继续持有。",  
    },  
]  

# 定义开空仓和平仓规则  
short_rules = [  
    {  
        "type": "开仓",  
        "conditions": {  
            "1m": {"trend": "downtrend", "divergence": False},  
            "5m": {"trend": "downtrend", "divergence": False},  
            "15m": {"trend": "downtrend", "divergence": False},  
            "1h": {"trend": "downtrend", "divergence": False},  
            "4h": {"trend": "downtrend", "divergence": False},  
            "1d": {"trend": "downtrend", "divergence": False},  
        },  
        "action": "开空仓",  
        "size": 1.0,  # 100% 仓位  
        "description": "趋势完全一致，无背离，市场方向性极强，50倍杠杆，100%开仓。",  
    },  
    {  
        "type": "开仓",  
        "conditions": {  
            "1m": {"trend": "downtrend", "divergence": False},  
            "5m": {"trend": "downtrend", "divergence": False},  
            "15m": {"trend": "downtrend", "divergence": False},  
            "1h": {"trend": "downtrend", "divergence": False},  
            "4h": {"trend": "downtrend", "divergence": False},  
            "1d": {"trend": "range", "divergence": False},  
        },  
        "action": "开空仓",  
        "size": 1.0,  # 100% 仓位  
        "description": "大周期震荡，小周期趋势一致，市场短期方向明确，50倍杠杆，100%开仓。",  
    },  
    {  
        "type": "平仓",  
        "conditions": {  
            "1m": {"trend": "range", "divergence": False},  
            "5m": {"trend": "range", "divergence": False},  
            "15m": {"trend": "range", "divergence": False},  
            "1h": {"trend": "range", "divergence": False},  
            "4h": {"trend": "range", "divergence": False},  
            "1d": {"trend": "range", "divergence": False},  
        },  
        "action": "全部平仓",  
        "size": 1.0,  # 全部平仓  
        "description": "趋势一致性降低，所有时间框架变为震荡，市场方向不明确，立即平仓锁定利润。",  
    },  
    {  
        "type": "平仓",  
        "conditions": {  
            "1m": {"trend": "uptrend", "divergence": False},  
            "5m": {"trend": "uptrend", "divergence": False},  
            "15m": {"trend": "uptrend", "divergence": False},  
            "1h": {"trend": "uptrend", "divergence": False},  
            "4h": {"trend": "uptrend", "divergence": False},  
            "1d": {"trend": "uptrend", "divergence": False},  
        },  
        "action": "全部平仓",  
        "size": 1.0,  # 全部平仓  
        "description": "趋势反转，所有时间框架变为上升趋势，立即平仓以规避风险。",  
    },  
    {  
        "type": "平仓",  
        "conditions": {  
            "1m": {"trend": "uptrend", "divergence": False},  
            "5m": {"trend": "uptrend", "divergence": False},  
            "15m": {"trend": "uptrend", "divergence": False},  
            "1h": {"trend": "uptrend", "divergence": False},  
            "4h": {"trend": "uptrend", "divergence": False},  
            "1d": {"trend": "range", "divergence": False},  
        },  
        "action": "全部平仓",  
        "size": 1.0,  # 全部平仓  
        "description": "趋势反转，小周期变为上升趋势，大周期仍为震荡，立即平仓以规避风险。",  
    },  
    {  
        "type": "平仓",  
        "conditions": {  
            "1m": {"trend": "range", "divergence": False},  
            "5m": {"trend": "range", "divergence": False},  
            "15m": {"trend": "range", "divergence": False},  
            "1h": {"trend": "range", "divergence": False},  
            "4h": {"trend": "range", "divergence": False},  
            "1d": {"trend": "range", "divergence": False},  
        },  
        "action": "部分平仓",  
        "size": 0.5,  # 部分平仓  
        "description": "趋势一致性降低，小周期变为震荡，部分平仓锁定利润，剩余仓位继续持有。",  
    },  
]  

import numpy as np
import pandas as pd

def find_last_abnormal_volume(abnormal_flags, current_index):
    """
    从当前索引向前回溯，找到最近的异常成交量发生的索引。

    Args:
        abnormal_flags: 布尔型Series，标记异常成交量。
        current_index: 当前索引（通常是DataFrame的最后一个索引）。

    Returns:
        最近的异常成交量发生的索引。如果没有找到，返回None。
    """
    # 确保abnormal_flags是Series
    if not isinstance(abnormal_flags, pd.Series):
        raise TypeError("abnormal_flags must be a pandas Series")

    # 确保current_index在abnormal_flags的索引范围内
    if current_index not in abnormal_flags.index:
        raise ValueError("current_index is out of range for abnormal_flags")

    # 将current_index转换为abnormal_flags索引中的位置
    current_position = abnormal_flags.index.get_loc(current_index)

    # 从当前位置向前回溯
    for i in range(current_position - 1, -1, -1):
        if abnormal_flags.iloc[i]:
            return abnormal_flags.index[i]
    return None  # 如果没有找到异常成交量

def calculate_dynamic_period(df, factors, default_period=100):
    """
    根据异常成交量动态计算统计周期。

    Args:
        df: 包含价格数据 (如 'close', 'open') 的DataFrame。
        factors: 包含TWVWAP相关数据和abnormal_flags的字典。
        default_period: 默认的统计周期。

    Returns:
        一个整数，表示动态计算的统计周期。
    """
    current_index = df.index[-1]
    last_abnormal_index = find_last_abnormal_volume(factors['abnormal_flags'], current_index)

    if last_abnormal_index is None:
        return default_period # 如果没有找到异常成交量，使用默认周期
    else:
        # 计算从最近异常成交量到当前索引的周期数
        period = len(df.loc[last_abnormal_index:current_index]) -1 # -1 为了去除重复计算的异常成交量
        return min(period + 50, len(df))
    
def analyze_price_distribution(df, factors, period=100):
    """
    分析价格在TWVWAP标准差带内的分布情况，判断趋势类型。

    Args:
        df: 包含价格数据 (如 'close', 'open') 的DataFrame。
        factors: 包含TWVWAP相关数据的字典，包括：
            'twvwap': TWVWAP序列
            'twvwap_std_dev': TWVWAP标准差序列
            'twvwap_upper_band_1': TWVWAP 1倍标准差上轨序列
            'twvwap_lower_band_1': TWVWAP 1倍标准差下轨序列
            'twvwap_upper_band_2': TWVWAP 2倍标准差上轨序列
            'twvwap_lower_band_2': TWVWAP 2倍标准差下轨序列
        period: 统计周期。

    Returns:
        一个字典，包含每个标准差区间内的频率，以及趋势判断结果。
    """

    if len(df) < period:
        return None  # 数据不足

    # 确保df和factors的索引对齐
    if not df.index.equals(factors['twvwap'].index):
        raise ValueError("df and factors must have the same index")

    # 获取最近period个周期的数据
    recent_df = df.iloc[-period:]

    recent_factors = {}
    for key, value in factors.items():
        if key in ['twvwap_upper_band_2', 'twvwap_upper_band_1', 'twvwap',
                    'twvwap_lower_band_1', 'twvwap_lower_band_2']:
            recent_factors[key] = value.iloc[-period:]
    
    # 统计价格分布 (直接使用factors中的数据)
    above_2std = (recent_df['close'] > recent_factors['twvwap_upper_band_2']).sum()
    between_1std_2std_up = ((recent_df['close'] <= recent_factors['twvwap_upper_band_2']) & (recent_df['close'] > recent_factors['twvwap_upper_band_1'])).sum()
    between_0std_1std_up = ((recent_df['close'] <= recent_factors['twvwap_upper_band_1']) & (recent_df['close'] > recent_factors['twvwap'])).sum()
    between_0std_1std_down = ((recent_df['close'] <= recent_factors['twvwap']) & (recent_df['close'] > recent_factors['twvwap_lower_band_1'])).sum()
    between_1std_2std_down = ((recent_df['close'] <= recent_factors['twvwap_lower_band_1']) & (recent_df['close'] > recent_factors['twvwap_lower_band_2'])).sum()
    below_2std = (recent_df['close'] <= recent_factors['twvwap_lower_band_2']).sum()

    total_count = len(recent_df)

    # 计算频率
    distribution = {
        'above_2std': above_2std / total_count,
        'between_1std_2std_up': between_1std_2std_up / total_count,
        'between_0std_1std_up': between_0std_1std_up / total_count,
        'between_0std_1std_down': between_0std_1std_down / total_count,
        'between_1std_2std_down': between_1std_2std_down / total_count,
        'below_2std': below_2std / total_count,
    }

    # 趋势判断 5m
    strong_uptrend_threshold = 0.7  # 强上升趋势阈值
    weak_uptrend_threshold = 0.55  # 弱上升趋势阈值
    strong_downtrend_threshold = 0.7  # 强下降趋势阈值
    weak_downtrend_threshold = 0.55  # 弱下降趋势阈值

    # # 趋势判断 1m
    strong_uptrend_threshold = 0.65  # 强上升趋势阈值
    weak_uptrend_threshold = 0.6  # 弱上升趋势阈值
    strong_downtrend_threshold = 0.65  # 强下降趋势阈值
    weak_downtrend_threshold = 0.6  # 弱下降趋势阈值

    upscore = distribution['above_2std'] + distribution['between_1std_2std_up'] + distribution['between_0std_1std_up']
    downscore =  distribution['below_2std'] + distribution['between_1std_2std_down'] + distribution['between_0std_1std_down'] 
    if upscore > strong_uptrend_threshold:
        trend = 'Strong Uptrend'
    elif upscore > weak_uptrend_threshold and distribution['between_0std_1std_down'] < 0.3:
        trend = 'Weak Uptrend'  # 弱上升趋势
    elif downscore > strong_downtrend_threshold:
        trend = 'Strong Downtrend'
    elif downscore > weak_downtrend_threshold and distribution['between_0std_1std_up'] < 0.3:
        trend = 'Weak Downtrend'  # 弱下降趋势
    elif distribution['between_0std_1std_up'] + distribution['between_0std_1std_down'] > 0.6:
        trend = 'ranging'
    else:
        trend = 'Undefined'  # 不明确的趋势

    distribution['trend'] = trend
    distribution['upscore'] = upscore
    distribution['downscore'] = downscore

    return distribution