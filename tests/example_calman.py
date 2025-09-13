# -*- coding: utf-8 -*-
"""
简单的卡尔曼滤波示例：
 - 对单市场的经调整VWAP数据进行平滑
 - 采用一维状态，状态转移模型 x[k] = x[k-1]
 - 观测模型：z[k] = x[k] + 随机噪声
"""

import numpy as np
import matplotlib.pyplot as plt

def kalman_filter(vwap_data, Q=1e-5, R=0.001, initial_value=None, initial_uncertainty=1.0):
    """
    vwap_data: 原始VWAP数据 (1D array or list)
    Q: 过程噪声协方差 (代表系统变化的不确定性)
    R: 观测噪声协方差 (代表测量误差)
    initial_value: 初始状态估计，若为None则使用第一个数据
    initial_uncertainty: 初始估计的不确定性
    """
    n = len(vwap_data)
    filtered_values = np.zeros(n)

    # 初始状态
    x = vwap_data[0] if initial_value is None else initial_value
    P = initial_uncertainty

    for k in range(n):
        # 预测：因为模型假设恒定状态，所以预测值就是上一状态
        x_prior = x
        P_prior = P + Q  # 预测协方差

        # 当前观测
        z = vwap_data[k]
        
        # 计算卡尔曼增益
        K = P_prior / (P_prior + R)
        
        # 更新状态：将预测值与实际测量值进行融合
        x = x_prior + K * (z - x_prior)
        # 更新估计协方差
        P = (1 - K) * P_prior
        
        # 保存滤波后的结果
        filtered_values[k] = x

    return filtered_values

if __name__ == "__main__":
    # 模拟一些经调整后的VWAP数据：真实趋势 + 随机噪声
    np.random.seed(42)
    t = np.linspace(0, 10, 100)  # 例如10个单位时间内100个数据点
    true_trend = 100 + 0.5 * t  # 假设真实趋势线
    noise = np.random.normal(0, 0.5, size=t.shape)  # 随机噪声
    vwap_data = true_trend + noise

    # 应用卡尔曼滤波
    filtered_vwap = kalman_filter(vwap_data, Q=1e-3, R=0.25)

    # 设置支持中文的字体  
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体  
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


    # 绘制结果对比
    plt.figure(figsize=(10, 5))
    plt.plot(t, vwap_data, label="原始VWAP数据", marker="o", linestyle="--")
    plt.plot(t, filtered_vwap, label="滤波后VWAP（估计趋势）", linewidth=2)
    plt.plot(t, true_trend, label="真实趋势", linewidth=2, linestyle=":")
    plt.xlabel("时间")
    plt.ylabel("VWAP")
    plt.legend()
    plt.title("卡尔曼滤波平滑经调整后VWAP数据")
    plt.show()

    # 结论:
    # 通过卡尔曼滤波，可以更平滑地提取出数据中的趋势，
    # 同时减少观测噪声的干扰，从而有助于我们更准确地判断市场状态