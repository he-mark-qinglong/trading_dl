import pandas as pd  
import numpy as np  
from typing import Dict , List


def is_market_open(timestamp: pd.Timestamp) -> bool:  
    """判断当前时间是否在各大交易所开盘的1分钟内"""  
    
    # 定义各大交易所的开盘时间（UTC时间）  
    # 纽交所（NYSE）：美国东部时间 9:30 AM  
    nyse_open = timestamp.replace(hour=14, minute=30, second=0, microsecond=0)  # UTC时间  

    # 港交所（HKEX）：香港时间 9:30 AM  
    hkex_open = timestamp.replace(hour=1, minute=30, second=0, microsecond=0)  # UTC时间  

    # 伦敦交易所（LSE）：英国时间 8:00 AM  
    lse_open = timestamp.replace(hour=8, minute=0, second=0, microsecond=0)  # UTC时间  

    # 东京交易所：背景时间 17:00 AM  
    tokyo_open = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)  # UTC时间 

    # 检查当前时间是否在任何一个交易所开盘的1分钟内  
    n_minuts = 30
    if (nyse_open <= timestamp < nyse_open + pd.Timedelta(minutes=n_minuts) or  
        hkex_open <= timestamp < hkex_open + pd.Timedelta(minutes=n_minuts) or  
        tokyo_open <= timestamp < tokyo_open + pd.Timedelta(minutes=n_minuts) or
        lse_open <= timestamp < lse_open + pd.Timedelta(minutes=n_minuts)):  
        return False  # 在开盘n_minuts分钟内，不交易  
    
    return True  # 其他时间可以交易
import numpy as np  
import pandas as pd  
from typing import Dict  

class TVWAPStrategy:  
    """
    flowchart TD
    A[开始] --> B{市场是否开盘?}
    B -->|是| C[获取因子数据]
    B -->|否| D[返回 0.0]
    
    C --> E{判断 TVWAP 相对 VWAP 的位置}
    E -->|TVWAP > VWAP| F{判断开仓条件}
    E -->|TVWAP < VWAP| G{判断开仓条件}
    
    F -->|close < TVWAP| H[生成正信号（做多）]
    F -->|close >= TVWAP| I[返回 0.0]
    
    G -->|close > TVWAP| J[生成负信号（做空）]
    G -->|close <= TVWAP| K[返回 0.0]
    
    H --> L[计算信号强度]
    J --> L
    
    L --> M{根据 ADX 和布林带宽度调整风险敞口}
    M --> N[结合价格斜率、动量和相对位置]
    
    N --> O[波动率调整信号强度]
    O --> P[综合信号]
    
    P --> Q[返回信号]
    D --> Q
    """
    def __init__(self, df, factors: Dict[str, pd.Series], theta_low: pd.Series, theta_high: pd.Series):  
        self.factors = factors  
        self.theta_low = theta_low  
        self.theta_high = theta_high  
        self.df = df
        # 新增价格关系缓存  
        self.price_relation_cache = {}  
        self.momentum_cache = {}  
        self.volatility_cache = {}  

    def calculate_price_momentum(self, timestamp: pd.Timestamp, window: int = 5) -> float:  
        """计算价格动量"""  
        if timestamp in self.momentum_cache:  
            return self.momentum_cache[timestamp]  
        
        try:  
            close_prices = self.df['close'].loc[timestamp-pd.Timedelta(days=window):timestamp]  
            
            # 使用线性回归斜率作为动量指标  
            x = np.arange(len(close_prices))  
            slope, _ = np.polyfit(x, close_prices.values, 1)  
            
            # 动量强度分级  
            momentum = np.sign(slope)  
            
            self.momentum_cache[timestamp] = momentum  
            return momentum  
        except Exception:  
            return 0  

    def calculate_price_relations(self, timestamp: pd.Timestamp) -> Dict[str, float]:  
        """计算价格间的关系"""  
        if timestamp in self.price_relation_cache:  
            return self.price_relation_cache[timestamp]  
        
        try:  
            vwap = self.factors['vwap'].loc[timestamp]  
            tvwap = self.factors['tvwap'].loc[timestamp]  
            close = self.df['close'].loc[timestamp]  
            
            # 计算价格间的相对偏离  
            relations = {  
                'vwap_tvwap_diff': (tvwap - vwap) / vwap,  
                'close_vwap_diff': (close - vwap) / vwap,  
                'close_tvwap_diff': (close - tvwap) / tvwap  
            }  
            
            self.price_relation_cache[timestamp] = relations  
            return relations  
        except Exception:  
            return {  
                'vwap_tvwap_diff': 0,  
                'close_vwap_diff': 0,  
                'close_tvwap_diff': 0  
            }  

    def generate_signal(self, timestamp: pd.Timestamp,   
                    positions: List[Dict],   
                    adx_slope_threshold: float = 5.25,  
                    adx_threshold: float = 25,  
                    bb_squeeze_threshold: float = 1.33) -> float:  
        """生成交易信号（-1到1之间的浮点数）"""  
        try:  
            # 检查市场是否开盘  
            if is_market_open(timestamp):  
                return 0.0  # 市场开盘时返回 0  

            # 获取因子数据  
            vol = self.factors['tvwap_deviation'].loc[timestamp]  
            slope = self.factors['tvwap_slope'].loc[timestamp]  
            adx = self.factors['adx'].loc[timestamp]  
            bb_width = self.factors['bb_width'].loc[timestamp]  
            close = self.df['close'].loc[timestamp]  
            vwap = self.factors['vwap'].loc[timestamp]  
            tvwap = self.factors['tvwap'].loc[timestamp]  

            # 判断 TVWAP 相对 VWAP 的位置  
            if tvwap > vwap:  # 上升趋势  
                if close < tvwap:  # 开仓条件  
                    signal = 1.0  # 生成正信号（做多）  
                else:  
                    return 0.0  # 不开仓  
            elif tvwap < vwap:  # 下降趋势  
                if close > tvwap:  # 开仓条件  
                    signal = -1.0  # 生成负信号（做空）  
                else:  
                    return 0.0  # 不开仓  
            else:  
                return 0.0  # 无信号  

            # 计算信号强度  
            exposure = 1.0  # 初始风险敞口  

            # 根据 ADX 和布林带宽度调整风险敞口  
            if adx >= adx_threshold:  
                if vol * slope > 0:  # 趋势强劲且量价方向一致  
                    exposure *= 1.5  # 增加风险敞口  
                else:  
                    exposure *= 0.5  # 减小风险敞口  
            if bb_width >= bb_squeeze_threshold:  
                if adx >= adx_threshold:  # 波动性高且趋势强劲  
                    exposure *= 1.0  # 保持风险敞口  
                else:  
                    exposure *= 0.5  # 减小风险敞口  

            # 新的信号生成逻辑  
            signal_components = [  
                exposure * np.sign(slope),  # 原有斜率信号  
                exposure * self.calculate_price_momentum(timestamp),  # 价格动量信号  
                exposure * np.sign(self.calculate_price_relations(timestamp)['close_tvwap_diff'])  # 价格相对位置信号  
            ]  

            # 波动率调整  
            volatility = self.factors['hist_vol'].loc[timestamp]  
            if volatility > 0.05:  # 高波动率  
                signal_components = [s * 0.5 for s in signal_components]  
            elif volatility < 0.01:  # 低波动率  
                signal_components = [s * 1.5 for s in signal_components]  

            # 综合信号  
            signal = np.mean(signal_components)  

            # 信号约束  
            return np.clip(signal, -1, 1)  
        except KeyError:  
            return 0.0

class BacktestEngine:  
    """
    flowchart TD
    A[开始] --> B{是否有持仓?}
    B -->|有持仓| C{是否满足平仓条件?}
    B -->|无持仓| D[保持空仓]
    
    C -->|满足平仓| E[计算收益]
    C -->|不满足平仓| F[保持当前仓位]
    
    E --> G[清空仓位]
    F --> G
    
    D --> H[更新收益为 0]
    
    G --> I[更新净值]
    H --> I
    
    I --> J[下一个时间点]
    """
    def __init__(self, data: pd.DataFrame, strategy, factors, initial_capital=1e6):  
        self.data = data  
        self.factors = factors
        self.strategy = strategy  
        self.initial_capital = initial_capital  
        
        self.positions = []  # 记录所有开仓  

    def run(self) -> pd.DataFrame:  
        """执行回测"""  
        df = self.data.copy()  
        df['position'] = 0.0  
        df['returns'] = 0.0  
        df['equity'] = float(self.initial_capital)  
        
        for i in range(50, len(df)):  
            current_time = df.index[i]  
            prev_time = df.index[i-1]  
            
            # 使用前一时刻的因子生成信号  
            signal = self.strategy.generate_signal(prev_time, self.positions)  
            
            # 获取 VWAP 和 TVWAP  
            vwap = self.factors['vwap'].loc[current_time]  
            tvwap = self.factors['tvwap'].loc[current_time]  
            close = df.loc[current_time, 'close']  

            # 开仓逻辑  
            if signal != 0 and len(self.positions) == 0:  
                self.positions.append({  
                    'direction': np.sign(signal),  
                    'price': close,  
                    'time': current_time  
                })  
                df.loc[current_time, 'position'] = signal  
            
            # 平仓逻辑  
            if self.positions and len(self.positions) > 0:  
                last_position = self.positions[-1]  
                
                # 上升趋势做多的平仓条件  
                if (last_position['direction'] > 0 and     
                    abs(close - tvwap) / tvwap < 0.001):  
                    
                    # 计算收益  
                    price_change = close / last_position['price'] - 1  
                    df.loc[current_time, 'returns'] = last_position['direction'] * price_change  
                    
                    # 清空仓位  
                    self.positions.clear()  
                    df.loc[current_time, 'position'] = 0  
                
                # 下降趋势做空的平仓条件  
                elif (last_position['direction'] < 0 and   
                    abs(close - tvwap) / tvwap < 0.001):  
                    
                    # 计算收益  
                    price_change = close / last_position['price'] - 1  
                    df.loc[current_time, 'returns'] = last_position['direction'] * price_change  
                    
                    # 清空仓位  
                    self.positions.clear()  
                    df.loc[current_time, 'position'] = 0  
                else:  
                    # 保持原有仓位  
                    df.loc[current_time, 'position'] = last_position['direction']  
                    df.loc[current_time, 'returns'] = 0  
            
            else:  
                df.loc[current_time, 'position'] = 0  
                df.loc[current_time, 'returns'] = 0  
            
            # 更新净值  
            df.loc[current_time, 'equity'] = df.loc[prev_time, 'equity'] * (1 + df.loc[current_time, 'returns'])  
        
        return df[['position', 'returns', 'equity']]
    
    def evaluate(self, df: pd.DataFrame) -> dict:  
        """绩效评估"""  
        returns = df['returns']  
        equity = df['equity']  
        
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0  
        max_drawdown = (equity / equity.cummax() - 1).min()  
        calmar = returns.mean() / abs(max_drawdown) if max_drawdown !=0 else 0  
        
        return {  
            "sharpe": sharpe,  
            "max_drawdown": max_drawdown,  
            "calmar": calmar,  
            "total_return": equity.iloc[-1] / self.initial_capital - 1  
        }
    

from scipy.optimize import differential_evolution  
    

from factors import FactorManager, FactorConfig  
# from strategies import TVWAPStrategy  
# from backtest import BacktestEngine  
import numpy as np  
import pandas as pd  


import numpy as np  
import pandas as pd  

def weighted_quantile(values, weights, quantile):  
    """计算加权分位数"""  
    # 将值和权重组合在一起  
    sorted_indices = np.argsort(values)  
    sorted_values = values[sorted_indices]  
    sorted_weights = weights[sorted_indices]  
    
    # 计算累积权重  
    cumulative_weights = np.cumsum(sorted_weights)  
    cumulative_weights /= cumulative_weights[-1]  # 归一化  
    
    # 找到分位数对应的索引  
    quantile_index = np.searchsorted(cumulative_weights, quantile)  
    
    return sorted_values[quantile_index]  

def dynamic_quantile(series: pd.Series, quantile: float,   
                      warmup_period: int = 50,   
                      decay_factor: float = 0.4) -> pd.Series:  
    """  
    修复广播错误的动态分位数实现  
    特征：固定长度滚动窗口 + 指数衰减权重  
    """  
    q_values = []  
    
    for i in range(len(series)):  
        if i < warmup_period:  
            # 初始阶段使用简单分位数  
            q = series[:i+1].quantile(quantile)  
        else:  
            # 固定窗口长度（提高计算效率）  
            window_start = i - warmup_period + 1  
            window_data = series[window_start:i+1]  
            
            # 生成指数衰减权重（从新到旧衰减）  
            weights = np.array([np.exp(-decay_factor) ** (warmup_period - j - 1)   
                                for j in range(len(window_data))])  # 使用当前窗口长度  
            
            # 归一化权重  
            weights /= weights.sum()  
            
            # 加权分位数计算  
            q = weighted_quantile(window_data.values, weights, quantile)  # 使用自定义的加权分位数计算  
            
        q_values.append(q)  
    
    return pd.Series(q_values, index=series.index)  # 返回结果
    
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

def analyze_trend_factors(df, factors, window=15, quantile_range=[0.7, 0.9]):  
    """  
    分析趋势因子与后续价格走势的关系  
    
    Parameters:  
    - df: 原始价格DataFrame  
    - factors: 因子字典，包含 'adx', 'adx_slope', 'bb_width'  
    - window: 用于计算后续价格变化的窗口期  
    - quantile_range: 用于筛选因子阈值的分位数范围  
    
    Returns:  
    - 趋势因子与后续价格走势的统计结果  
    """  
    # 计算后续价格变化率  
    df['future_return'] = df['close'].pct_change(window).shift(-window)  
    
    # 合并因子数据  
    combined_df = pd.DataFrame({  
        'adx': factors['adx'],  
        'adx_slope': factors['adx_slope'],  
        'bb_width': factors['bb_width'],  
        'future_return': df['future_return']  
    }).dropna()  
    
    # 分析结果存储  
    results = {}  
    best_thresholds = {}  # 存储最佳阈值  
    
    # ADX 分析  
    for adx_quantile in np.linspace(0.8, 0.95, 6):  
        adx_threshold = max(combined_df['adx'].quantile(adx_quantile), 25)  
        adx_subset = combined_df[combined_df['adx'] >= adx_threshold]  
        
        results[f'adx_threshold_{adx_quantile:.2f}'] = {  
            'mean_future_return': adx_subset['future_return'].mean(),  
            'std_future_return': adx_subset['future_return'].std(),  
            'positive_ratio': (adx_subset['future_return'] > 0).mean()  
        }  
        
        # 记录最佳阈值  
        if 'best_adx' not in best_thresholds or results[f'adx_threshold_{adx_quantile:.2f}']['mean_future_return'] > best_thresholds['best_adx']['mean_future_return']:  
            best_thresholds['best_adx'] = {  
                'threshold': float(adx_threshold),  # 确保是标量  
                'mean_future_return': results[f'adx_threshold_{adx_quantile:.2f}']['mean_future_return']  
            }  
    
    # ADX Slope 分析  
    for slope_quantile in np.linspace(0.7, 0.95, 6):  
        slope_threshold = combined_df['adx_slope'].quantile(slope_quantile)  
        slope_subset = combined_df[combined_df['adx_slope'] >= slope_threshold]  
        
        results[f'adx_slope_threshold_{slope_quantile:.2f}'] = {  
            'mean_future_return': slope_subset['future_return'].mean(),  
            'std_future_return': slope_subset['future_return'].std(),  
            'positive_ratio': (slope_subset['future_return'] > 0).mean()  
        }  
        
        # 记录最佳阈值  
        if 'best_adx_slope' not in best_thresholds or results[f'adx_slope_threshold_{slope_quantile:.2f}']['mean_future_return'] > best_thresholds['best_adx_slope']['mean_future_return']:  
            best_thresholds['best_adx_slope'] = {  
                'threshold': float(slope_threshold),  # 确保是标量  
                'mean_future_return': results[f'adx_slope_threshold_{slope_quantile:.2f}']['mean_future_return']  
            }  
    
    # BB Squeeze 分析  
    for bb_quantile in np.linspace(0.1, 0.3, 6):  
        bb_threshold = combined_df['bb_width'].rolling(15).quantile(bb_quantile).iloc[-1]  # 取最后一个值  
        bb_subset = combined_df[combined_df['bb_width'] < bb_threshold]  
        
        results[f'bb_squeeze_threshold_{bb_quantile:.2f}'] = {  
            'mean_future_return': bb_subset['future_return'].mean(),  
            'std_future_return': bb_subset['future_return'].std(),  
            'positive_ratio': (bb_subset['future_return'] > 0).mean()  
        }  
        
        # 记录最佳阈值  
        if 'best_bb' not in best_thresholds or results[f'bb_squeeze_threshold_{bb_quantile:.2f}']['mean_future_return'] > best_thresholds['best_bb']['mean_future_return']:  
            best_thresholds['best_bb'] = {  
                'threshold': float(bb_threshold),  # 确保是标量  
                'mean_future_return': results[f'bb_squeeze_threshold_{bb_quantile:.2f}']['mean_future_return']  
            }  
    
    # 组合条件分析  
    combined_results = {}  
    for adx_quantile in [0.8, 0.85, 0.9]:  
        for slope_quantile in [0.8, 0.85, 0.9]:  
            for bb_quantile in [0.1, 0.15, 0.2]:  
                adx_threshold = combined_df['adx'].quantile(adx_quantile)  
                slope_threshold = combined_df['adx_slope'].quantile(slope_quantile)  
                bb_threshold = combined_df['bb_width'].rolling(15).quantile(bb_quantile).iloc[-1]  # 取最后一个值  
                
                combined_subset = combined_df[  
                    (combined_df['adx'] >= adx_threshold) &   
                    (combined_df['adx_slope'] >= slope_threshold) &   
                    (combined_df['bb_width'] < bb_threshold)  
                ]  
                
                key = f'combined_{adx_quantile}_{slope_quantile}_{bb_quantile}'  
                combined_results[key] = {  
                    'mean_future_return': combined_subset['future_return'].mean(),  
                    'std_future_return': combined_subset['future_return'].std(),  
                    'positive_ratio': (combined_subset['future_return'] > 0).mean(),  
                    'sample_size': len(combined_subset)  
                }  
    
    # 可视化  
    plt.figure(figsize=(15, 5))  
    
    # 单因子未来收益分布  
    plt.subplot(131)  
    for key, value in results.items():  
        plt.bar(key, value['mean_future_return'],   
                yerr=value['std_future_return'],   
                capsize=5)  
    plt.title('Single Factor Future Return')  
    plt.xticks(rotation=45, ha='right')  
    plt.ylabel('Mean Future Return')  
    
    # 组合因子未来收益分布  
    plt.subplot(132)  
    combined_means = [v['mean_future_return'] for v in combined_results.values()]  
    combined_stds = [v['std_future_return'] for v in combined_results.values()]  
    plt.bar(range(len(combined_means)), combined_means, yerr=combined_stds, capsize=5)  
    plt.title('Combined Factors Future Return')  
    plt.ylabel('Mean Future Return')  
    
    # 组合因子样本量  
    plt.subplot(133)  
    sample_sizes = [v['sample_size'] for v in combined_results.values()]  
    plt.bar(range(len(sample_sizes)), sample_sizes)  
    plt.title('Combined Factors Sample Size')  
    plt.ylabel('Sample Count')  
    
    plt.tight_layout()  
    plt.show()  
    
    # 打印最佳组合结果  
    best_combined = max(combined_results.items(), key=lambda x: x[1]['mean_future_return'])  
    print("Best Combined Condition:", best_combined)  
    
    # 打印最佳阈值  
    print("Best Thresholds:")  
    for key, value in best_thresholds.items():  
        print(f"{key}: {value['threshold']:.4f}, Mean Future Return: {value['mean_future_return']:.4f}")  
    
    return results, combined_results  



def search_df_risk(df: pd.DataFrame):  
    # 生成因子  
    config = FactorConfig()

    # 因子管理器计算因子  
    factor_manager = FactorManager(config)  
    factors, df_filtered = factor_manager.calculate_factors(df) 

    # results, combined_results = analyze_trend_factors(df, factors) 

    theta_low = dynamic_quantile(factors['tvwap_deviation'], quantile=0.15)  
    theta_high = dynamic_quantile(factors['tvwap_deviation'], quantile=0.85) 
    # 初始化策略  
    strategy = TVWAPStrategy(df, factors, theta_low, theta_high)  
    
    # 运行回测  
    engine = BacktestEngine(df_filtered, strategy, factors)  
    results = engine.run()  
    
    # 评估结果  
    metrics = engine.evaluate(results)  
    print(f"夏普比率: {metrics['sharpe']:.2f}")

from utils import DataManager, HistoricalDataLoader
def main():  
    # 初始化数据加载器  
    data_manager = DataManager()  
    loader = HistoricalDataLoader('okx')  

    # 定义要获取的交易对和时间框架    
    symbols = ['TRUMP-USDT-SWAP', 'SOL-USDT-SWAP', 'BTC-USDT-SWAP', 'XRP-USDT-SWAP',]  
    timeframes = ['1m', '3m', '5m'] #['1h', '4h', '1d']  

    # 获取并保存数据  
    for symbol in symbols:  
        print(f"\nFetching data for {symbol}")  
        for timeframe in timeframes:  
            df = loader.fetch_historical_data(symbol, timeframe, data_manager)  
            if not df.empty:  
                print(f"Fetched and updated {timeframe} data for {symbol}, total rows: {len(df)}") 
                # print('df.index=', df.index[-1])
                # start_time = '2025-02-14 11:00:00'  
                # end_time = '2025-02-14 11:15:00'  

                end_time = '2025-02-14 10:00:00'  #11点的时候程成交量很大。市场单边运行

                # 截取时间片  
                # df = df.loc[:end_time].copy()  
                search_df_risk(df)

                print('--------------------------') 
            else:  
                print(f"No new data for {symbol} {timeframe}")  


if __name__ == "__main__":
    main()