from typing import Tuple, Dict  
import os
import torch  
import pandas as pd  
import numpy as np  
from factors import FactorManager  
from utils import HistoricalDataLoader, DataManager  
from utils.transformer_trend import TrendDetector
from utils.transformer_trend import classify_trend_from_factors  
from config import trend_detect_config

# keys_to_keep = [
#     # 'obv', 'obv_ma', 'obv_slope', 'obv_norm', 'obv_ma_norm', 
#     'vwap', 'vwap_deviation', 

#     #'vwap_norm', 'vwap_dev_norm', 
    
#     'twvwap', 'twvwap_deviation', 'twvwap_smooth', 'twvwap_slope',
#     'twvwap_std_dev', 'twvwap_upper_band_1', 'twvwap_lower_band_1',
#     'twvwap_upper_band_2', 'twvwap_lower_band_2',

#     # 'cmf', 'cmf_norm', 
#     # 'cmf_smooth', 'force_index', 'force_index_norm', 
#     # 'force_index_smooth', 'pvt', 

#     #'rsi', 
#     #'rsi_norm', 'rsi_smooth', 
    
#     # 'stoch_k', 'stoch_d',

#     #'stoch_k_norm', 'stoch_d_norm', 
    
#     # 'kdj_k', 'kdj_d', 'kdj_j', 
#     # 'kdj_k_norm', 
#     # 'kdj_d_norm', 
#     # 'kdj_j_norm', 

#     # 'cci', 'cci_norm', 'cci_smooth', 'roc', 
#     # 'roc_norm', 'roc_smooth', 'wr', 'wr_norm', 
#     # 'wr_smooth', 'ma_5', 'ma_5_norm', 'ma_5_slope', 
#     # 'ma_10', 'ma_10_norm', 'ma_10_slope', 'ma_20', 
#     # 'ma_20_norm', 'ma_20_slope', 

#     # 'ma_60', 
#     # 'ma_60_norm', 
#     # 'ma_60_slope', 
#     # 'ema_5', 
#     # 'ema_5_norm', 'ema_10', 'ema_10_norm', 
#     # 'ema_20', 'ema_20_norm', 'ema_60', 
#     # 'ema_60_norm',
#     'macd', 'macd_signal', 'macd_diff', 
#     # 'macd_norm', 
#     # 'macd_signal_norm', 
#     # 'macd_diff_norm', 
#     'adx', '+di', '-di', 
#     # 'supertrend', 'supertrend_direction', 
#     # 'supertrend_upper', 'supertrend_lower', 'supertrend_norm', 
#     # 'sar', 

#     # 'sar_up', 'sar_down', 
#     # 'sar_up_indicator', 'sar_down_indicator', 
#     # 'sar_norm', 'bb_high', 'bb_mid', 'bb_low', 'bb_width', 'bb_high_norm', 
#     # 'bb_width_norm', 'atr', 'atr_norm', 'atr_pct', 'kc_high', 'kc_mid', 
#     # 'kc_low', 'kc_width', 'kc_width_norm', 'hist_vol', 'hist_vol_norm', 
#     # 'parkinson_vol', 'parkinson_vol_norm', 'daily_range', 'avg_range', 
#     # 'daily_range_norm', 'avg_range_norm', 'pivot', 
#     # 'r1', 'r2', 'r3', 's1', 's2', 's3', 'pivot_norm', 
#     # 'high_fractal', 'low_fractal', 'zigzag', 'zigzag_norm', 
#     # 'support_resistance_levels', 'price_distribution'
# ]

keys_to_keep = [
    'vwap', 'vwap_deviation', 
    
    'twvwap', 'twvwap_deviation', 'twvwap_smooth', 'twvwap_slope',
    'twvwap_std_dev', 'twvwap_upper_band_1', 'twvwap_lower_band_1',
    'twvwap_upper_band_2', 'twvwap_lower_band_2',

    'macd', 'macd_signal', 'macd_diff', 
]

def filter_factors(factors: Dict[str, pd.Series], keys_to_keep: list) -> np.ndarray:  
    """  
    过滤指定的因子，并返回时间序列数据  
    :param factors: 原始因子字典，key 为因子名称，value 为 pd.Series  
    :param keys_to_keep: 要保留的因子 key 列表  
    :return: 过滤后的因子值数组 (shape: sequence_length x num_features)  
    """  
    # print('factors[keys_to_keep[0]]:', factors[keys_to_keep[0]])
    filtered_values = np.array([  
        [factors[key].iloc[i] for key in keys_to_keep if key in factors]  
        for i in range(len(factors[keys_to_keep[0]]))  # 遍历时间序列  
    ], dtype=np.float32)  
    return filtered_values 


def generate_training_data_with_dynamic_factors(  
    df: pd.DataFrame,  
    timeframe: str,  
    factor_manager: FactorManager,  
    sequence_length: int = 60,  
    future_window_size: int = 6,  # 未来窗口的大小  
    history_buffer: int = 60,  # 滑动窗口因子所需的最小历史数据长度  
    start_position: int = 0,  # 从第几个时间点开始生成数据  
    keys_to_keep: list = None  # 要保留的因子名称  
) -> Tuple[torch.Tensor, torch.Tensor]:  
    """  
    生成批量训练数据，动态计算因子，预测未来趋势  
    :param df: 原始数据框  
    :param timeframe: 时间框架  
    :param factor_manager: 用于计算因子的管理器  
    :param sequence_length: 当前时间窗口的长度  
    :param future_window_size: 未来窗口的长度  
    :param history_buffer: 滑动窗口因子所需的最小历史数据长度  
    :param start_position: 从数据的第几个时间点开始生成数据  
    :param keys_to_keep: 要保留的因子名称  
    :return: 训练数据和标签，分别为 PyTorch 张量  
    """  
    train_data = []  
    train_labels = []  

    # 一次性计算所有因子  
    factors, _ = factor_manager.calculate_factors(df)  
    assert len(factors['twvwap'] == len(df)), f"len(factors['twvwap']:{len(factors['twvwap'])} len(df):{len(df)}"
    # 检查因子是否正常  
    for key in keys_to_keep:  
        if key not in factors:  
            raise ValueError(f"Missing key in factors: {key}")  
        if factors[key].isnull().any():  
            print(f"Warning: Key {key} contains NaN values")  
        if np.isinf(factors[key]).any():  
            print(f"Warning: Key {key} contains Inf values")  

    # 起点和终点的计算  
    if len(factors["twvwap"])+start_position < history_buffer:  
        raise ValueError(f"len factors:{len(factors['twvwap'])} start_position must be >= history_buffer ({history_buffer}) to ensure sufficient historical data.")  

    end_index = len(factors["twvwap"]) - future_window_size 

    assert start_position+sequence_length <= end_index, f"end_index must be greater than start_position to generate training data. {start_position} >= {end_index}"
    # 遍历数据，从 start_position 开始，正序索引  
    for i in range(start_position+sequence_length, end_index):  # 正序遍历  
        # 当前时间窗口的数据 (sequence_length, num_features)  
        try:
            # 假设以下变量已经定义：  
            # factors: 包含多个因子的字典，每个因子是一个 pandas.Series  
            # keys_to_keep: 需要保留的因子键列表  
            # sequence_length: 时间序列长度  
            # i: 当前索引  

            # 初始化一个空的列表，用于存储每个因子的值  
            factor_values = []  

            # 遍历需要保留的因子键  
            for key in keys_to_keep:  
                # 提取当前因子在指定时间范围内的值  
                values = factors[key].iloc[i - sequence_length:i].values  
                # 检查提取的值是否符合预期的维度 (sequence_length,)  
                assert values.shape == (sequence_length,), f"Unexpected shape for factor '{key}': {values.shape}, expected ({sequence_length}), values{values}, factors[key].iloc[i - sequence_length:i] {factors[key].iloc[i - sequence_length:i]}"  
                factor_values.append(values)  

            # 将因子值列表转换为 NumPy 数组  
            factor_values = np.array(factor_values)  
            # 检查转换后的数组是否符合预期的维度 (num_features, sequence_length)  
            assert factor_values.shape == (len(keys_to_keep), sequence_length), f"Unexpected shape after stacking: {factor_values.shape}, expected ({len(keys_to_keep)}, {sequence_length})"  

            # 转置为 (sequence_length, num_features)  
            factor_values = factor_values.T  
            # 检查转置后的数组是否符合预期的维度 (sequence_length, num_features)  
            assert factor_values.shape == (sequence_length, len(keys_to_keep)), f"Unexpected shape after transpose: {factor_values.shape}, expected ({sequence_length}, {len(keys_to_keep)})"
            # 检查形状是否符合预期  
            expected_shape = (sequence_length, len(keys_to_keep))  
            assert factor_values.shape == expected_shape, (  
                f"factor_values shape mismatch: expected {expected_shape}, "  
                f"but got {factor_values.shape} at index {i}. "  
                f"Check keys_to_keep: {keys_to_keep} and sequence_length: {sequence_length}"  
            )  
            
        except Exception as e:
            print(f'generate_training_data_with_dynamic_factors-sequence_length:{e}')

        try:
            # 使用未来窗口的因子生成趋势标签  
            current_price = df['close'].iloc[i]
            label = classify_trend_from_factors({key: factors[key].iloc[i+1:i + future_window_size] for key in keys_to_keep}, df[i+1:i + future_window_size], current_price)  
        except Exception as e:
            print(f'generate_training_data_with_dynamic_factors-regroup keys:{e}')

        # 添加到训练数据和标签中  
        train_data.append(factor_values)  
        train_labels.append(label) 

    # 转换为 PyTorch 张量  
    train_data = np.array(train_data, dtype=np.float32)  # 先转换为 numpy.ndarray  
    train_data = torch.tensor(train_data, dtype=torch.float32)  # 再转换为 PyTorch 张量  
    train_labels = torch.tensor(train_labels, dtype=torch.long)  # 标签直接转换  

            # 检查训练数据的形状  
    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")  
    assert train_data.dim() == 3, f"train_data must be a 3D tensor (batch_size, sequence_length, num_features) not {train_data.dim()}"  
    assert train_labels.dim() == 2, f"train_labels must be a 1D tensor (batch_size, laebl) which is not:{train_labels.dim()} "  

    return train_data, train_labels 

def train_symbol_timeframe(symbol, timeframe, data_manager: DataManager, data_loader: HistoricalDataLoader):
    df = data_loader.fetch_historical_data(symbol, timeframe, data_manager, 5000) 
    # df = data_manager.load_data(symbol, timeframe)
    df = df.iloc[0:len(df)-2] 
    if not df.empty:  
        # 初始化因子管理器  
        factor_manager = FactorManager()  

        # 动态生成训练数据  
        train_data, train_labels = generate_training_data_with_dynamic_factors(  
            df, timeframe, factor_manager, sequence_length=160, start_position=0,
            keys_to_keep = keys_to_keep 
        )  

        # 输出训练数据大小  
        print(f"Generated training data for {symbol} {timeframe}:")  
        # return train_data, train_labels  
        if train_data is None or train_labels is None:  
            print("No training data available.")  
            return  

        # 初始化趋势检测器  
        num_features = train_data.shape[-1]  # 因子数量  
        num_classes = 4  # uptrend, range, downtrend, volatile  
        trend_detector = TrendDetector(trend_detect_config, num_features=num_features, num_classes=num_classes)  
        # 加载模型权重  
        # trend_detector.model.load_state_dict(torch.load( f"models/{symbol}_{timeframe}_model_weights.pth", map_location=torch.device('cpu') )) 
        # 训练模型  
        trend_detector.train(symbol, timeframe, train_data, train_labels, epochs=1000)
        # 检查并创建目录  
        os.makedirs(os.path.dirname(f'models/'), exist_ok=True)
        save_path = f"models/{symbol}_{timeframe}_model_weights.pth"
        # 保存模型权重  
        torch.save(trend_detector.model.state_dict(), save_path)  
        print(f"Model saved to {save_path}")  
    else:  
        print(f"No new data for {symbol} {timeframe}")  

from concurrent.futures import ProcessPoolExecutor 
def main_train_data(symbols=['BTC-USDT-SWAP']):  
    """  
    主训练逻辑  
    """  
    # 初始化数据加载器  
    data_manager = DataManager()  
    data_loader = HistoricalDataLoader('binance')  

    # 定义要获取的交易对和时间框架  
    timeframes = ['1m', '5m', '15m', '1h']  
    timeframes = ['5m']
    # 创建一个全局进程池  
    with ProcessPoolExecutor() as executor:  
        futures = []  
        for symbol in symbols:  
            for timeframe in timeframes:  
                # 提交任务到进程池  
                futures.append(  
                    executor.submit(train_symbol_timeframe, symbol, timeframe, data_manager, data_loader)  
                )  

        # 等待所有任务完成  
        for future in futures:  
            try:  
                future.result()  # 捕获潜在的异常  
            except Exception as e:  
                print(f"Error in task: {e}")  

def generate_latest_data_with_dynamic_factors(  
    df: pd.DataFrame,  
    factor_manager: FactorManager,  
    sequence_length: int = 60,  
    keys_to_keep: list = None  
) -> torch.Tensor:  
    """  
    从最新的 df 数据生成用于预测的单条因子数据  
    :param df: 原始数据框  
    :param factor_manager: 用于计算因子的管理器  
    :param sequence_length: 时间窗口长度  
    :param keys_to_keep: 要保留的因子名称  
    :return: 单条预测数据，形状为 (1, sequence_length, num_features)  
    """  
    # 动态计算因子  
    factors, _ = factor_manager.calculate_factors(df)  

    # 检查因子是否正常  
    for key in keys_to_keep:  
        if key not in factors:  
            raise ValueError(f"Missing key in factors: {key}")  
        if factors[key].isnull().any():  
            print(f"Warning: Key {key} contains NaN values")  
        if np.isinf(factors[key]).any():  
            print(f"Warning: Key {key} contains Inf values")  

    # 确保有足够的历史数据  
    if len(factors[keys_to_keep[0]]) < sequence_length:  
        raise ValueError(f"Not enough historical data to generate a sequence of length {sequence_length}. "  
                         f"Available: {len(factors[keys_to_keep[0]])}, Required: {sequence_length}")  

    # 提取最新时间窗口的数据  
    factor_values = []  
    for key in keys_to_keep:  
        # 提取当前因子在最新时间窗口的值  
        values = factors[key].iloc[-sequence_length:].values  
        # 检查提取的值是否符合预期的维度  
        assert values.shape == (sequence_length,), f"Unexpected shape for factor '{key}': {values.shape}"  
        factor_values.append(values)  

    # 将因子值列表转换为 NumPy 数组  
    factor_values = np.array(factor_values)  
    # 转置为 (sequence_length, num_features)  
    factor_values = factor_values.T  
    # 检查形状是否符合预期  
    expected_shape = (sequence_length, len(keys_to_keep))  
    assert factor_values.shape == expected_shape, (  
        f"factor_values shape mismatch: expected {expected_shape}, "  
        f"but got {factor_values.shape}. Check keys_to_keep: {keys_to_keep} and sequence_length: {sequence_length}"  
    )  

    # 转换为 PyTorch 张量，并增加 batch 维度  
    latest_data = torch.tensor(factor_values, dtype=torch.float32).unsqueeze(0)  # (1, sequence_length, num_features)  

    # 检查最终数据形状  
    assert latest_data.dim() == 3, f"latest_data must be a 3D tensor (1, sequence_length, num_features), not {latest_data.dim()}"  

    return latest_data, factors 

def example_real_time_predict(symbols = ['BTC-USDT-SWAP']):
    data_manager = DataManager()  
    loader = HistoricalDataLoader('binance')  
    
    # 定义要获取的交易对和时间框架  
    timeframes = ['1m', '5m', '15m', '1h', '4h']  
    timeframes = ['5m']  
    # 获取并保存数据  
    for symbol in symbols:  
        #print(f"\nFetching data for {symbol}")  
        for timeframe in timeframes:  
            df = loader.fetch_historical_data(symbol, timeframe, data_manager, 1000)  
            df = df.iloc[-500:].copy()
            if not df.empty:  
                #print(f"Fetched and updated {timeframe} data for {symbol}, total rows: {len(df)}")  

                # 初始化因子管理器  
                factor_manager = FactorManager()  

                sequence_length = 160  
                prediction_data, factors = generate_latest_data_with_dynamic_factors(  
                    df=df,  
                    factor_manager=factor_manager,  
                    sequence_length=sequence_length,
                    keys_to_keep=keys_to_keep  
                )

                trend_detector = TrendDetector(trend_detect_config, num_features=prediction_data.shape[-1], num_classes=4)  

                # 加载模型权重  
                trend_detector.model.load_state_dict(torch.load( f"models/{symbol}_{timeframe}_model_weights.pth", map_location=torch.device('cpu') ))  
                trend_detector.model = trend_detector.model.to("cuda")
                trend_detector.model.eval()  

                # 使用模型进行预测  
                predicted_class, confidence, max_down_pct, max_up_pct = trend_detector.predict(prediction_data)  

                is_divergence = trend_detector.detect_weighted_divergence(factors=factors, df=df, market_trend=predicted_class)

                print(f">>>>>>>>>>>{timeframe}-Predicted trend: {predicted_class}, Confidence: {confidence:.2f}, is_divergence:{is_divergence},  max_down_pct, max_up_pct: { (max_down_pct+1)*df['close'].iloc[-1], (max_up_pct+1)*df['close'].iloc[-1]}")

if __name__ == "__main__":  

    # symbols = ['BTC-USDT-SWAP', 'SOL-USDT-SWAP']
    symbols = ['BTC-USDT-SWAP'] 
    main_train_data(symbols=symbols) 

    #BTC:【15m:0.17, 30m:0.34, 1h:0.11, 4h:<0.1, 1d:0.15】
    #SOL:[1m:0.05, 3m:0.39, 5m:0.5, 15m:0.25~0.7, 30m:0.4, 1h:0.43, 4h:<0.01, 1d:0.03]
    import time

    times = 15
    while True:
        example_real_time_predict(symbols=symbols)
        time.sleep(60)
        times -= 1
        if times < 0:
            times = 15
            main_train_data(symbols=symbols) 