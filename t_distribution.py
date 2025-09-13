# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.stats as stats
# import yfinance as yf

# # 获取价格数据
# ticker = "AAPL"
# data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
# prices = data['Close']

# # 计算对数收益率
# returns = np.log(prices / prices.shift(1)).dropna()

# # 计算均值和标准差
# mu, sigma = returns.mean(), returns.std()

# # 拟合t分布
# df, loc, scale = stats.t.fit(returns)

# # 生成正态分布和t分布的PDF
# x = np.linspace(returns.min(), returns.max(), 1000)
# normal_pdf = stats.norm.pdf(x, mu, sigma)
# t_pdf = stats.t.pdf(x, df, loc, scale)

# # 绘制K线图
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

# # K线图
# ax1.plot(prices.index, prices, label='Price', color='black')
# ax1.set_title(f'{ticker} Price Chart')
# ax1.set_ylabel('Price')
# ax1.legend()

# # 收益率分布图
# ax2.hist(returns, bins=50, density=True, alpha=0.6, color='blue', label='Returns')
# ax2.plot(x, normal_pdf, 'r-', label='Normal Distribution')
# ax2.plot(x, t_pdf, 'g-', label='t-Distribution')
# ax2.set_title('Returns Distribution')
# ax2.set_xlabel('Returns')
# ax2.set_ylabel('Density')
# ax2.legend()

# plt.tight_layout()
# plt.show()


import requests  
import requests  
import pandas as pd  
import time  

def fetch_ls_ratio_history(symbol='BTCUSDT', period='5m', start_time=None, end_time=None):  
    """  
    批量抓取 Binance Futures BTCUSDT 多空比历史数据.  
    
    参数:  
      symbol: 合约交易对, 默认 'BTCUSDT'  
      period: 数据周期 如 '1m', '5m', '15m', '30m', '1h' 等  
      start_time: 开始时间，毫秒时间戳  
      end_time: 结束时间，毫秒时间戳  

    返回:  
      Pandas DataFrame，包含历史多空比数据，同时增加了时间格式化字段  
    """  
    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"  
    limit = 500  # Binance接口规定一次最多返回500条数据  
    all_data = []  
    
    # 如果未指定结束时间，默认为当前时间  
    if end_time is None:  
        end_time = int(time.time() * 1000)  
    # 如果未指定开始时间，可以设定一个默认时间，比如过去一天  
    if start_time is None:  
        start_time = end_time - 24 * 60 * 60 * 1000  # 最近24小时  
    
    current_start_time = start_time  
    while True:  
        params = {  
            "symbol": symbol,  
            "period": period,  
            "limit": limit,  
            "startTime": current_start_time,  
            "endTime": end_time  
        }  
        try:  
            response = requests.get(url, params=params, timeout=5)  
            response.raise_for_status()  
            data = response.json()  
        except Exception as e:  
            print("请求出错：", e)  
            break  

        if not data:  
            break  

        all_data.extend(data)  
        # Binance的返回数据中每条记录都有'timestamp'  
        last_timestamp = data[-1]['timestamp']  
        # 检查是否已到结束时间  
        if last_timestamp >= end_time:  
            break  
        
        # 更新下一次的起始时间，避免重复  
        current_start_time = last_timestamp + 1  
        # 睡眠一下，防止请求过快被限制  
        time.sleep(0.1)  
    
    # 整理成 DataFrame  
    df = pd.DataFrame(all_data)  
    if not df.empty:  
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')  
    return df  

if __name__ == "__main__":  
    import datetime
    # 示例：获取最近一周的多空比历史数据  
    end_time = int(time.time() * 1000)  
    start_time = end_time - 1 * 24 * 60 * 60 * 1000  # 最近七天  
    df = fetch_ls_ratio_history(symbol='BTCUSDT', period='5m', start_time=start_time, end_time=end_time)  
    
    print("BTC 多空比数据:")  
    def t2s_time(timestamp):
        return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') 
    for j in range(len(df)):
        print(t2s_time(df["timestamp"].iloc[j]/1000), df.iloc[j])  