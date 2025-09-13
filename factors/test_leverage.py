import numpy as np  

def calculate_profit(initial_capital, win_rate, profit_pct, loss_pct, trades, simulations=1000):  
    results = []  
    
    for _ in range(simulations):  
        capital = initial_capital  
        trades_sequence = np.random.choice([1, -1], size=trades, p=[win_rate, 1-win_rate])  
        
        for trade in trades_sequence:  
            if trade == 1:  # 盈利  
                capital *= (1 + profit_pct/100)  
            else:  # 亏损  
                capital *= (1 - loss_pct/100)  
        
        results.append(capital)  
    
    return np.mean(results), np.std(results)  

# 设置参数  
initial_capital = 100  
win_rate = 0.55  
profit_pct = 2  
loss_pct = 1  

# 计算100次和1000次交易的结果  
trades_scenarios = [100, 1000]  

for trades in trades_scenarios:  
    mean_result, std_result = calculate_profit(  
        initial_capital, win_rate, profit_pct, loss_pct, trades  
    )  
    
    print(f"\n交易{trades}次的统计结果:")  
    print(f"平均最终资金: {mean_result:.2f}")  
    print(f"收益率: {((mean_result/initial_capital - 1) * 100):.2f}%")  
    print(f"标准差: {std_result:.2f}")