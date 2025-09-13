import ccxt
import pandas as pd
import talib
import time
from config import ExchangeConfig
# 1. 初始化交易所
def initialize_exchange(api_key, api_secret, password):
    exchange = ccxt.okx({
        'apiKey': api_key,
        'secret': api_secret,
        'password': password,
        'enableRateLimit': True
    })
    exchange.set_sandbox_mode(True)  # 开启模拟交易模式（测试环境）
    return exchange

# 2. 获取实时数据
def fetch_data(exchange, symbol, timeframe, limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# 3. 计算技术指标
def calculate_indicators(data):
    data['MA5'] = talib.SMA(data['close'], timeperiod=5)
    data['MA20'] = talib.SMA(data['close'], timeperiod=20)
    data['ADX'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
    data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    return data

# 4. 判断趋势方向
def determine_trend(data):
    if data['MA5'].iloc[-1] > data['MA20'].iloc[-1] and data['ADX'].iloc[-1] > 25:
        return 'up'  # 上升趋势
    elif data['MA5'].iloc[-1] < data['MA20'].iloc[-1] and data['ADX'].iloc[-1] > 25:
        return 'down'  # 下降趋势
    else:
        return 'neutral'  # 无趋势

# 5. 生成交易信号（多时间框架过滤）
def generate_signal(data_1m, data_5m, data_15m):
    trend_5m = determine_trend(data_5m)
    trend_15m = determine_trend(data_15m)
    
    # 确保5分钟和15分钟趋势一致
    if trend_5m == trend_15m and trend_5m != 'neutral':
        if trend_5m == 'up' and data_1m['MA5'].iloc[-1] > data_1m['MA20'].iloc[-1] and data_1m['ADX'].iloc[-1] > 25:
            return 'buy'  # 开多
        elif trend_5m == 'down' and data_1m['MA5'].iloc[-1] < data_1m['MA20'].iloc[-1] and data_1m['ADX'].iloc[-1] > 25:
            return 'sell'  # 开空
    return 'hold'  # 无信号

# 6. 下单函数（合约交易）
def place_order(exchange, symbol, signal, amount, leverage):
    try:
        # 设置杠杆
        exchange.set_leverage(leverage, symbol)
        
        if signal == 'buy':
            # 开多
            order = exchange.create_order(symbol, 'market', 'buy', amount, params={'posSide': 'long'})
        elif signal == 'sell':
            # 开空
            order = exchange.create_order(symbol, 'market', 'sell', amount, params={'posSide': 'short'})
        print(f"Order placed: {order}")
        return order
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

# 7. 主程序
def main():
    
    # 配置API密钥
    api_key = ExchangeConfig.api_key
    api_secret = ExchangeConfig.secret_key
    password = ExchangeConfig.passphrase
    
    # 初始化交易所
    exchange = initialize_exchange(api_key, api_secret, password)
    symbol = 'BTC/USDT:USDT'  # 合约交易对
    amount = 1  # 合约张数
    leverage = 10  # 杠杆倍数

    while True:
        try:
            # 获取多时间框架数据
            data_1m = fetch_data(exchange, symbol, '1m')
            data_5m = fetch_data(exchange, symbol, '5m')
            data_15m = fetch_data(exchange, symbol, '15m')
            
            # 计算技术指标
            data_1m = calculate_indicators(data_1m)
            data_5m = calculate_indicators(data_5m)
            data_15m = calculate_indicators(data_15m)
            
            # 生成交易信号
            signal = generate_signal(data_1m, data_5m, data_15m)
            print(f"Signal: {signal}")
            
            # 执行交易
            if signal in ['buy', 'sell']:
                place_order(exchange, symbol, signal, amount, leverage)
            
            # 等待下一次循环
            time.sleep(60)  # 每分钟运行一次
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()