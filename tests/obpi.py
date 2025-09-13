import numpy as np  
import pandas as pd  
import datetime  
import matplotlib.pyplot as plt  

class OrderBookPressureIndex:  
    def __init__(self, fast_ema_length=14, slow_ema_length=100, signal_length=10):  
        self.fast_ema_length = fast_ema_length  # 快线长度  
        self.slow_ema_length = slow_ema_length  # 慢线长度  
        self.signal_length = signal_length  # 信号线长度  
        self.obpi_data = []  
        self.obpi_ema_fast = 0  # 快速 EMA  
        self.obpi_ema_slow = 0  # 慢速 EMA  
        self.signal_line = 0  # 信号线  

    def update_obpi(self, bids, asks):  
        """  
        更新 OBPI 数据并计算其值。  
        :param bids: list，包含 (price, volume) 元组  
        :param asks: list，包含 (price, volume) 元组  
        """  
        now = datetime.datetime.now()  

        # 计算买入和卖出压力  
        buy_pressure = sum(volume for price, volume in bids)  
        sell_pressure = sum(volume for price, volume in asks)  

        # 计算当前 OBPI 值  
        current_obpi = buy_pressure - sell_pressure  

        # 计算快速和慢速 EMA  
        self.obpi_ema_fast = self.calculate_ema(current_obpi, self.fast_ema_length)  
        self.obpi_ema_slow = self.calculate_ema(current_obpi, self.slow_ema_length)  

        # 计算 OBPI Diff  
        obpi_diff = self.obpi_ema_fast - self.obpi_ema_slow  

        # 计算信号线  
        self.signal_line = self.calculate_ema(obpi_diff, self.signal_length)  

        # 更新 OBPI 数据  
        self.obpi_data.append({  
            "timestamp": now,  
            "obpi": current_obpi,  
            "obpi_ema_fast": self.obpi_ema_fast,  
            "obpi_ema_slow": self.obpi_ema_slow,  
            "obpi_diff": obpi_diff,  
            "signal_line": self.signal_line,  
        })  

    def calculate_ema(self, current_obpi, length):  
        """  
        计算给定长度的 EMA 值。  
        :param current_obpi: 当前 OBPI 值  
        :param length: EMA 的长度  
        :return: EMA 值  
        """  
        if len(self.obpi_data) >= length:  
            previous_ema = self.obpi_data[-length]["obpi_ema_fast"] if length == self.fast_ema_length else self.obpi_data[-length]["obpi_ema_slow"]  
            alpha = 2 / (length + 1)  
            return (current_obpi - previous_ema) * alpha + previous_ema  
        else:  
            return current_obpi  # 如果数据不足，返回当前值  

    def get_obpi_data(self):  
        """  
        获取当前的 OBPI 数据，包含 timestamp、obpi、obpi_ema、signal_line 和 obpi_diff。  
        :return: list 当前的 OBPI 数据  
        """  
        return self.obpi_data  

def plot_obpi(obpi_data):  
    timestamps = [data["timestamp"] for data in obpi_data]  
    obpi_values = [data["obpi"] for data in obpi_data]  
    obpi_ema_fast_values = [data["obpi_ema_fast"] for data in obpi_data]  
    obpi_ema_slow_values = [data["obpi_ema_slow"] for data in obpi_data]  
    obpi_diff_values = [data["obpi_diff"] for data in obpi_data]  
    signal_line_values = [data["signal_line"] for data in obpi_data]  

    plt.figure(figsize=(14, 7))  

    # 绘制 OBPI 和其快速/慢速 EMA  
    plt.plot(timestamps, obpi_values, label='OBPI', color='blue', linewidth=1.5)  
    plt.plot(timestamps, obpi_ema_fast_values, label='Fast EMA (14)', color='orange', linewidth=1.5)  
    plt.plot(timestamps, obpi_ema_slow_values, label='Slow EMA (100)', color='red', linewidth=1.5)  

    # 绘制 OBPI Diff 的柱状图  
    colors = ['green' if diff >= 0 else 'red' for diff in obpi_diff_values]  
    plt.bar(timestamps, obpi_diff_values, color=colors, alpha=0.6, label='OBPI Diff', width=0.01)  

    # 绘制信号线  
    plt.plot(timestamps, signal_line_values, label='Signal Line (10)', color='purple', linestyle='--', linewidth=1.5)  

    plt.title('Order Book Pressure Index (OBPI) and Diff')  
    plt.xlabel('Timestamp')  
    plt.ylabel('OBPI Value')  
    plt.legend()  
    plt.xticks(rotation=45)  

    plt.tight_layout()  
    plt.show()  

# 使用示例  
if __name__ == "__main__":  
    obpi_calculator = OrderBookPressureIndex(fast_ema_length=14, slow_ema_length=100, signal_length=10)  

    # 模拟传入一些 bids 和 asks 数据  
    for _ in range(500):  # 生成 50 个数据点  
        bids = [(100 + np.random.uniform(-1, 1), np.random.randint(1, 10)) for _ in range(3)]  
        asks = [(100 + np.random.uniform(-1, 1), np.random.randint(1, 10)) for _ in range(3)]  

        # 更新 OBPI 数据  
        obpi_calculator.update_obpi(bids, asks)  

    # 获取 OBPI 数据并绘图  
    obpi_data = obpi_calculator.get_obpi_data()  
    plot_obpi(obpi_data)  