import numpy as np  
import matplotlib.pyplot as plt  

# 假设 depth_deltas 是历史买卖盘深度差值数据  
depth_deltas = np.random.normal(0, 100, 1000)  # 示例数据，均值为 0，标准差为 100  

# 计算均值和标准差  
mean = np.mean(depth_deltas)  
std = np.std(depth_deltas)  

# 计算乖离率  
bias_ratios = (depth_deltas - mean) / std  

# 设置支持中文的字体  
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体  
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

wanted_bias = 1.1
# 绘制乖离率分布图  
plt.figure(figsize=(10, 6))  
plt.hist(bias_ratios, bins=50, density=True, alpha=0.7, color='blue', label='乖离率分布')  
plt.axvline(x=wanted_bias, color='red', linestyle='--', label=f'{wanted_bias} 倍标准差')  
plt.axvline(x=-wanted_bias, color='red', linestyle='--')  
plt.title('乖离率分布图')  
plt.xlabel('乖离率')  
plt.ylabel('密度')  
plt.legend()  
plt.grid(True)  
plt.show() 