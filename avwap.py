#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Auto AVWAP (Anchored-VWAP) Python版本  

原始 Pine Script 作者：Electrified (electrifiedtrading)  
改写者：ChatGPT  

说明：  
  1. 计算 AVWAP、ATR、反转信号、入场信号等  
  2. 加入 MACD 计算，由外部 TrendFactors 类完成，并将结果放进 output DataFrame 中  
  3. 最后输出保存到 CSV 文件中  

License: MPL 2.0 https://mozilla.org/MPL/2.0/  
"""  

import numpy as np  
import pandas as pd  
import concurrent.futures  
from scipy.stats import shapiro  
import ta  # 需要安装 ta 库： pip install ta  

##############################################  
# 计算辅助指标函数（省略部分函数）  
##############################################  
def sma(series, window):  
    return series.rolling(window=window, min_periods=window).mean()  

def ema(series, window):  
    return series.ewm(span=window, adjust=False).mean()  

def wma(series, window):  
    weights = np.arange(1, window + 1)  
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)  

def vwma(series, volume, window):  
    num = (series * volume).rolling(window).sum()  
    den = volume.rolling(window).sum()  
    return num / den  

def vawma(series, volume, window):  
    def _vawma(x):  
        prices = x[:, 0]  
        vols = x[:, 1]  
        weights = np.arange(1, len(prices) + 1)  
        weighted_vols = vols * weights[::-1]  
        if weighted_vols.sum() == 0:  
            return np.nan  
        return (prices * weighted_vols).sum() / weighted_vols.sum()  
    data = pd.concat([series, volume], axis=1)  
    return data.rolling(window=window).apply(_vawma, raw=True)  

def getMA(series, volume, mode, window):  
    mode = mode.upper()  
    if mode == "WMA":  
        return wma(series, window)  
    elif mode == "EMA":  
        return ema(series, window)  
    elif mode == "VWMA":  
        return vwma(series, volume, window)  
    elif mode == "VAWMA":  
        return vawma(series, volume, window)  
    else:  
        return sma(series, window)  

def compute_rsi(series, period):  
    delta = series.diff()  
    up = delta.clip(lower=0)  
    down = -delta.clip(upper=0)  
    roll_up = up.ewm(com=(period - 1), adjust=False).mean()  
    roll_down = down.ewm(com=(period - 1), adjust=False).mean()  
    rs = roll_up / roll_down  
    return 100 - (100 / (1 + rs))  

def compute_stochastic(rsi_series, length):  
    lowest = rsi_series.rolling(window=length, min_periods=length).min()  
    highest = rsi_series.rolling(window=length, min_periods=length).max()  
    return 100 * (rsi_series - lowest) / (highest - lowest)  

def compute_atr(df, period=14):  
    high = df["high"]  
    low = df["low"]  
    close = df["close"]  
    tr1 = high - low  
    tr2 = (high - close.shift(1)).abs()  
    tr3 = (low - close.shift(1)).abs()  
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)  
    return tr.rolling(window=period, min_periods=1).mean()  

##############################################  
# 参数优化及AVWAP计算函数（部分代码，如 _optimize_parameters 等，省略部分细节）  
##############################################  
def _evaluate_candidate(args):  
    tau, window, seg_df, n, epsilon = args  
    tvwap = np.empty(n)  
    cum_pv = 0.0  
    cum_vol = 0.0  
    decay_factor = np.exp(-tau)  
    if 'hlc3' in seg_df.columns:  
        typical = seg_df['hlc3']  
    else:  
        typical = seg_df['close']  #(seg_df['high'] + seg_df['low'] + seg_df['close']) / 3  
    volume = seg_df['volume']  
    for i in range(n):  
        cum_pv *= decay_factor  
        cum_vol *= decay_factor  
        cum_pv += typical.iloc[i] * volume.iloc[i]  
        cum_vol += volume.iloc[i]  
        tvwap[i] = cum_pv / (cum_vol + epsilon)  
    try:  
        score = shapiro(pd.Series(tvwap).dropna()).statistic  
    except Exception:  
        score = -np.inf  
    return score, tau, window  

def _optimize_parameters(df, tau_candidates=np.linspace(0.001, 0.005, 5),  
                         window_candidates=range(14, 29, 5)):  
    best_score = -np.inf  
    best_tau = tau_candidates[0]  
    best_window = window_candidates[0]  
    epsilon = 1e-8  
    n = len(df)  
    candidate_args = []  
    for tau in tau_candidates:  
        for window in window_candidates:  
            candidate_args.append((tau, window, df, n, epsilon))  
    with concurrent.futures.ThreadPoolExecutor() as executor:  
        results = executor.map(_evaluate_candidate, candidate_args)  
    for score, tau, window in results:  
        if score > best_score:  
            best_score = score  
            best_tau = tau  
            best_window = window  
    return best_tau, best_window  

def compute_auto_avwap(df, useHiLow=True, useOpen=True, k_mode="WMA",  
                        smoothK=4, smoothD=4, lengthRSI=64, lengthStoch=48,  
                        lowerBand=20, upperBand=80, lowerReversal=20, upperReversal=80,  
                        decay=0.01, atr_period=14, atr_multiplier=1.5):  
    original_index = df.index.copy()  
    df = df.copy()  
    if 'open' not in df.columns:  
        df['open'] = df['close']  
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3  
    src = df['hlc3']  
    df['rsi'] = compute_rsi(src, period=lengthRSI)  
    df['stoch'] = compute_stochastic(df['rsi'], length=lengthStoch)  
    df['k'] = getMA(df['stoch'], df['volume'], k_mode, smoothK)  
    df['d'] = sma(df['k'], smoothD)  
    df['k'] = df['k'].bfill()  
    df['d'] = df['d'].bfill()  

    n = len(df)  
    hiAVWAP_list = [np.nan] * n  
    loAVWAP_list = [np.nan] * n  

    phi = df.loc[df.index[0], 'high']  
    plo = df.loc[df.index[0], 'low']  
    hi_val = df.loc[df.index[0], 'high']  
    lo_val = df.loc[df.index[0], 'low']  
    state = 0  

    hiAVWAP_s = 0.0  
    loAVWAP_s = 0.0  
    hiAVWAP_v = 0.0  
    loAVWAP_v = 0.0  

    current_tau = decay  
    decay_factor = np.exp(-current_tau)  
    seg_start = 0  

    for i, idx in enumerate(df.index):  
        hiAVWAP_s *= decay_factor  
        hiAVWAP_v *= decay_factor  
        loAVWAP_s *= decay_factor  
        loAVWAP_v *= decay_factor  

        high_val = df.loc[idx, 'high'] 
        low_val = (df.loc[idx, 'low'] + df.loc[idx, 'close']) / 2  
        vol = df.loc[idx, 'volume']  
        vwapHi = high_val if useHiLow else df.loc[idx, 'hlc3']  
        vwapLo = low_val if useHiLow else df.loc[idx, 'hlc3']  
        d_val = df.loc[idx, 'd']  
        k_val = df.loc[idx, 'k']  

        if (d_val < lowerBand) or (high_val > phi):  
            phi = high_val  
        if (d_val > upperBand) or (low_val < plo):  
            plo = low_val  

        if high_val > hi_val:  
            hi_val = high_val  
            hiAVWAP_s = 0.0  
            hiAVWAP_v = 0.0  
        if low_val < lo_val:  
            lo_val = low_val  
            loAVWAP_s = 0.0  
            loAVWAP_v = 0.0  

        hiAVWAP_s += vwapHi * vol  
        loAVWAP_s += vwapLo * vol  
        hiAVWAP_v += vol  
        loAVWAP_v += vol  

        if state != -1 and d_val < lowerBand:  
            state = -1  
        elif state != 1 and d_val > upperBand:  
            state = 1  

        if ((hi_val > phi and state == 1 and k_val < d_val and k_val < lowerReversal) or  
            (lo_val < plo and state == -1 and k_val > d_val and k_val > upperReversal)) and i < n - 1:  
            seg_df = df.iloc[seg_start:i+1]  
            best_tau, best_window = _optimize_parameters(seg_df)  
            current_tau = best_tau  
            decay_factor = np.exp(-current_tau)  
            if (hi_val > phi and state == 1 and k_val < d_val and k_val < lowerReversal):  
                hi_val = phi  
                hiAVWAP_s = 0.0  
                hiAVWAP_v = 0.0  
            if (lo_val < plo and state == -1 and k_val > d_val and k_val > upperReversal):  
                lo_val = plo  
                loAVWAP_s = 0.0  
                loAVWAP_v = 0.0  
            seg_start = i + 1  

        hi_avwap = hiAVWAP_s / hiAVWAP_v if hiAVWAP_v != 0 else np.nan  
        lo_avwap = loAVWAP_s / loAVWAP_v if loAVWAP_v != 0 else np.nan  
        hiAVWAP_list[i] = hi_avwap  
        loAVWAP_list[i] = lo_avwap  

    df['hiAVWAP'] = hiAVWAP_list  
    df['loAVWAP'] = loAVWAP_list  


    atr = compute_atr(df, period=atr_period)  
    df['atr'] = atr  
    df['avg_atr'] = df['atr'].rolling(window=14).mean().fillna(df['atr'])  

    # 增加 gap 条件：只有当 |resistance - support| >= 2 * avg_atr 时才认为有效  
    df['valid_gap'] = (abs(df['hiAVWAP'] - df['hiAVWAP']) >= 1.5 * df['avg_atr'])  

    # 反转信号：在满足 valid_gap 条件的前提下  
    df['reversal_arrow'] = None  
    df.loc[(df['high'] > (df['hiAVWAP'] + atr_multiplier * df['avg_atr'])) &  
           (df['valid_gap']), 'reversal_arrow'] = 'up'  
    df.loc[(df['low'] < (df['loAVWAP'] - atr_multiplier * df['avg_atr'])) &  
           (df['valid_gap']), 'reversal_arrow'] = 'down'  

    # 假设原始数据为 df_original，且其 index 为 DatetimeIndex 且包含 'close' 列  
    higher_tf_minutes = 60  # 例如 15 分钟  
    resample_rule = f"{higher_tf_minutes}min"  

    df_resampled = df['close'].resample(resample_rule).agg(  
            open=('first'),  
            high=('max'),  
            low=('min'),  
            close=('last')  
        )  

    from factors.factors import TrendFactors, FactorConfig  
    tf = TrendFactors(FactorConfig())  
    # 使用多线程同时计算 MACD 和 ADX  
    with concurrent.futures.ThreadPoolExecutor() as executor:  
        future_macd = executor.submit(tf.macd, df_resampled)  
        future_adx = executor.submit(tf.adx, df_resampled)  
        macd_result = future_macd.result()  
        adx_result = future_adx.result()  

    for col, series in macd_result.items():  
        df[col] = series.reindex(df.index, method='ffill')  
    for col, series in adx_result.items():  
        df[col] = series.reindex(df.index, method='ffill')    


    # 根据反转信号生成入场信号，不过仅在有效 gap 条件下作数  
    entry_signals = [None] * n  
    reversal_counts = [0] * n  
    last_direction = None  
    entry_placed = False  
    reversal_count = 0  

    for i, idx in enumerate(df.index):  
        # 增加 gap 条件判断：仅当当前 bar gap 合法时才参与信号计算  
        curr_valid_gap = df.loc[idx, 'valid_gap']  
        curr_reversal = df.loc[idx, 'reversal_arrow']  
        if curr_reversal is not None and curr_valid_gap:  
            last_direction = curr_reversal  
            reversal_count += 1  
            entry_placed = False  
        reversal_counts[i] = reversal_count 
        # print(df.loc[idx, 'macd_diff'])
        if not pd.isna(df.loc[idx, 'macd_diff']):
            macd_long = df.loc[idx, 'macd_diff'] >= 0 and df.loc[idx, 'macd'] >= 0
            macd_shot = df.loc[idx, 'macd_diff'] < 0 and df.loc[idx, 'macd'] < 0
            # TODO 在[-2和2]之间的时候根据斜率达到一定程度来决定趋势是哪一边。
            if last_direction == 'up' and not entry_placed and curr_valid_gap:  
                if df.loc[idx, 'close'] <= df.loc[idx, 'loAVWAP']:  
                    entry_signals[i] = 'long_entry'  if macd_long else 'short_exit'
                    entry_placed = True  
            elif last_direction == 'down' and not entry_placed and curr_valid_gap:  
                if  df.loc[idx, 'close'] >= df.loc[idx, 'hiAVWAP']:  
                    entry_signals[i] = 'short_entry'  if not macd_shot else 'long_exit'
                    entry_placed = True 
        # else:
        #     print(f'idx at {idx} nan MACD')

    df['entry_signal'] = entry_signals  
    df['reversal_count'] = reversal_counts  
        
    df.index = original_index  
    return df

##############################################  
# 主函数：加载数据、计算 AVWAP、MACD，并输出结果  
##############################################  
if __name__ == '__main__':  
    from utils.history_data import DataManager, HistoricalDataLoader  
    from strategy.scaling_strategy import backtest_scaling_strategy_with_capital  
    from strategy.dash_app import run_dash
    from utils.plot_avwap import plot_avwap_chart

    import os
    if True or os.environ.get("WERKZEUG_RUN_MAIN") == "true":  
        symbols = ["BTC-USDT-SWAP"]  
        timeframes = ['1m']  

        length = 60*24*10
        for symbol in symbols:  
            print(f"\nFetching data for {symbol}")  
            for timeframe in timeframes:  
                df = HistoricalDataLoader('binance').fetch_historical_data(symbol, timeframe, DataManager('data'), 50000)  
                df = df.iloc[-length:].copy()  

                # 计算 AVWAP 及其它因子  
                result = compute_auto_avwap(df, useHiLow=True, useOpen=True, k_mode="WMA",  
                                            smoothK=4, smoothD=4, lengthRSI=64, lengthStoch=48,  
                                            lowerBand=10, upperBand=90, lowerReversal=10, upperReversal=90,  
                                            decay=0.001, atr_period=14, atr_multiplier=1.5) 
                
                # 针对 5 分钟数据  
                resample_rule_5m = "5min"  
                df_5m = result[['open', 'high', 'low', 'close', 'volume']].resample(resample_rule_5m).agg({  
                    "open": "first",  
                    "high": "max",  
                    "low": "min",  
                    "close": "last",  
                    "volume": "sum"  
                })  
                # 这里需要确保 df_5m 的时间序列和原始数据对齐（通常 resample 后 index 是 DateTimeIndex）  
                result_5m = compute_auto_avwap(df_5m, useHiLow=True, useOpen=True, k_mode="WMA",  
                                            smoothK=4, smoothD=4, lengthRSI=64, lengthStoch=48,  
                                            lowerBand=20, upperBand=80, lowerReversal=20, upperReversal=80,  
                                            decay=0.01, atr_period=14, atr_multiplier=1.5) 

                # 将 5m 计算好的 VWAP 系列 reindex（向前填充）到 1m 数据上  
                result["hiAVWAP_5m"] = result_5m["hiAVWAP"].reindex(result.index, method="ffill")  
                result["loAVWAP_5m"] = result_5m["loAVWAP"].reindex(result.index, method="ffill")  

                # 针对 15 分钟数据  
                resample_rule_15m = "15min"  
                df_15m = result[['open', 'high', 'low', 'close', 'volume']].resample(resample_rule_15m).agg({  
                    "open": "first",  
                    "high": "max",  
                    "low": "min",  
                    "close": "last",  
                    "volume": "sum"  
                })  
                result_15m = compute_auto_avwap(df_15m, useHiLow=True, useOpen=True, k_mode="WMA",  
                                            smoothK=4, smoothD=4, lengthRSI=64, lengthStoch=48,  
                                            lowerBand=25, upperBand=75, lowerReversal=25, upperReversal=75,  
                                            decay=0.03, atr_period=14, atr_multiplier=1.5) 
                result["hiAVWAP_15m"] = result_15m["hiAVWAP"].reindex(result.index, method="ffill")  
                result["loAVWAP_15m"] = result_15m["loAVWAP"].reindex(result.index, method="ffill") 
                
                result.to_csv("btc_5m_avwap_result.csv") 

                data_file = "btc_5m_avwap_result.csv"  # 请替换为实际数据路径  
                result = pd.read_csv(data_file, parse_dates=True, index_col=0)  

                result = result.iloc[1000:].copy()
                length -= 1000
                # plot_avwap_chart(result.iloc[-1000:].copy()) 

                from strategy.scaling_strategy import run_backtest  
                from strategy.dash_app import run_dash 
                import os, threading

                if True:
                    run_on_dash = True
                    if run_on_dash:
                        # 仅在真正运行的进程中启动回测线程，避免 debug 模式下自动重载时重复启动  
                        if os.environ.get("WERKZEUG_RUN_MAIN") == "true":  
                            backtest_thread = threading.Thread(target=run_backtest, args=(length,), daemon=True)  
                            backtest_thread.start()  

                        # 启动 Dash 应用（主线程）  
                        # run_dash(length)  
                        run_dash()
                    else:
                        initial_capital = 1000  
                        report = backtest_scaling_strategy_with_capital(result, initial_capital=initial_capital, leverage=10, fee_rate=0.0005)  
                        print("回测结果：")  
                        print(report)  