import numpy as np  

# 假设以下模块用于获取历史数据（此处为示例；请按你实际情况调整）  
from utils.history_data import DataManager, HistoricalDataLoader  
from factors.factors import weighted_atr
if __name__ == "__main__":  
    # 参数设置  
    symbol = "BTC-USDT-SWAP"  
    timeframe = '1m'  
    # 这里以10日数据为例，1分钟K线，10天内的数据  
    length = 60 * 24 * 10  

    # 获取历史数据  
    df = HistoricalDataLoader('binance').fetch_historical_data(symbol, timeframe, DataManager('data'), 50000)  
    df = df.iloc[-length:].copy()  

    # 计算ATR因子  
    from factors.factors import VolatilityFactors, FactorConfig  
    vf = VolatilityFactors(FactorConfig())  
    atr_factors = vf.atr(df)  

    # 打印ATR指标的部分数据预览  
    print("ATR指标:")  
    print("ATR绝对值：")  
    print(atr_factors['atr'].tail())  
    print("ATR百分比：")  
    print(atr_factors['atr_pct'].tail())  

    # 假设使用ATR历史数据，实际中可以根据计算ATR因子获得历史序列：  
    # 这里尝试直接利用atr因子获取ATR序列，若数据不足，则用随机数据模拟  
    atr_hist = atr_factors['atr'].dropna().values[-100:]  # 保留最新100个周期  

    # 计算融合后的ATR  
    combined_atr, atr_large, atr_small, alpha = weighted_atr(atr_hist)  
    print(f"大周期ATR: {atr_large:.4f}")  
    print(f"小周期ATR: {atr_small:.4f}")  
    print(f"自适应权重α: {alpha:.2f}")  
    print(f"融合后的ATR: {combined_atr:.4f}")  