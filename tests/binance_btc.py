import ccxt
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timezone

def convert_kline_to_dict(kline):
    """将Binance K线数据转换为目标格式"""
    return {
        "open": float(kline[1]),
        "high": float(kline[2]),
        "low": float(kline[3]),
        "close": float(kline[4]),
        "volume": float(kline[5]),
        "timestamp": kline[0]
    }

def fetch_binance_klines(symbol='BTC/USDT', start_date='2023-01-01', end_date='2024-12-31'):
    exchange = ccxt.binance({'enableRateLimit': True})
    timeframe = '1m'
    
    # 时间转换
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    all_klines = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            klines = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_ts,
                limit=1000
            )
            
            if not klines:
                break
                
            all_klines.extend(klines)
            current_ts = klines[-1][0] + 60000  # 下一分钟
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    return [convert_kline_to_dict(k) for k in all_klines if k[0] <= end_ts]

def save_to_parquet(data, filename):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_parquet(
        filename,
        engine='pyarrow',
        compression='snappy',
        partition_cols=None,
        coerce_timestamps='ms',
        allow_truncated_timestamps=False
    )

if __name__ == "__main__":
    print("开始获取BTC历史数据...")
    btc_data = fetch_binance_klines()
    print(f"共获取到 {len(btc_data)} 条数据")
    
    output_file = "BTC_2023-2024.parquet"
    save_to_parquet(btc_data, output_file)
    print(f"数据已保存到 {output_file}")