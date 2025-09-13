
# 移动止损函数  
def simple_trailing_stop_loss(signal, df, factors, position, entry_price, current_stop_loss):  
    """  
    一个简单的移动止损函数。  

    Args:  
        df: 包含价格数据的DataFrame。  
        factors: 包含TWVWAP相关数据的字典。  
        position: 当前仓位 (1: 多头, -1: 空头)。  
        entry_price: 开仓价格。  
        current_stop_loss: 当前止损价格。  

    Returns:  
        新的止损价格。  
    """  

    twvwap = factors['twvwap'].iloc[-1]  
    twvwap_std_dev = factors['twvwap_std_dev'].iloc[-1]  
    current_price = df['close'].iloc[-1]  

    # 计算当前的标准差倍数 n  
    if position == 1:  # 多头  
        if current_price >= twvwap:  # 价格高于TWVWAP  
            n = (current_price - twvwap) / twvwap_std_dev  # 计算当前偏离的标准差倍数  
            if n >= 1:  # 只有当n大于等于1时才移动止损  
                #若果市场还处于弱势的多头，可以放大止损。如果市场趋势已经弱势的空头，则收紧止损。
                b = 1 
                if signal == 0.5:
                    b = 1  #若果市场还处于弱势的多头，可以放大止损。
                elif signal == -0.5:
                    b = 0 #如果市场趋势已经弱势的空头，则收紧止损。
                else:
                    b = 0.5 #中性
                new_stop_loss = twvwap + (n - b) * twvwap_std_dev  # 移动止损到TWVWAP + (n - 0.5) * twvwap_std_dev  
                return new_stop_loss  

    elif position == -1:  # 空头  
        if current_price <= twvwap:  # 价格低于TWVWAP  
            n = (twvwap - current_price) / twvwap_std_dev  # 计算当前偏离的标准差倍数  
            if n >= 1:  # 只有当n大于等于1时才移动止损  
                #若果市场还处于弱势的空头，可以放大止损。如果市场趋势已经弱势的多头，则收紧止损。
                b = 1 
                
                if signal == 0.5:
                    b = 0.25
                elif signal == -0.5:
                    b = 0.1
                else:
                    b = 0.5 #中性

                new_stop_loss = twvwap - (n - b) * twvwap_std_dev  # 移动止损到TWVWAP - (n - 0.5) * twvwap_std_dev  
                return new_stop_loss  

    return current_stop_loss  # 否则保持止损不变

# 移动止损函数  
def simple_trailing_take_profit(signal, df, factors, position, entry_price, current_take_profit):  
    """  
    一个简单的移动止损函数。  

    Args:  
        df: 包含价格数据的DataFrame。  
        factors: 包含TWVWAP相关数据的字典。  
        position: 当前仓位 (1: 多头, -1: 空头)。  
        entry_price: 开仓价格。  
        current_stop_loss: 当前止损价格。  

    Returns:  
        新的止损价格。  
    """  

    twvwap = factors['twvwap'].iloc[-1]  
    twvwap_std_dev = factors['twvwap_std_dev'].iloc[-1]  
    current_price = df['close'].iloc[-1]  

    # 计算当前的标准差倍数 n  
    if position == 1:  # 多头  
        if current_price >= current_take_profit:  # 价格高于old  
            n = (current_price - twvwap) / twvwap_std_dev  # 计算当前偏离的标准差倍数  
            if n >= 1:  # 只有当n大于等于1时才移动止损  
                b = 0.5 
                if signal == 0.5:
                    b = 0.5  #若果市场还处于弱势的多头，可以放大止盈。
                elif signal == -0.5:
                    b = 0 #如果市场趋势已经弱势的空头，则收紧止赢。
                else:
                    b = 0.25 #中性
                current_take_profit = twvwap + (n + b) * twvwap_std_dev  # 移动止损到TWVWAP + (n - 0.5) * twvwap_std_dev  
                return current_take_profit  + factors['atr'].iloc[-1]

    elif position == -1:  # 空头  
        if current_price <= current_take_profit:  # 价格低于TWVWAP  
            n = (twvwap - current_price) / twvwap_std_dev  # 计算当前偏离的标准差倍数  
            if n >= 1:  # 只有当n大于等于1时才移动止损  
                b = 0.5 
                
                if signal == 0.5:
                    b = 0  #若果市场还处于弱势的多头，则收紧止赢。
                elif signal == -0.5:
                    b = 0.5 #如果市场趋势已经弱势的空头，可以放大止盈。
                else:
                    b = 0.25 #中性
                current_take_profit = twvwap - (n + b) * twvwap_std_dev  # 移动止损到TWVWAP - (n - 0.5) * twvwap_std_dev  
                return current_take_profit + factors['atr'].iloc[-1]

    return current_take_profit  # 否则保持止损不变