# 创建新环境
conda create -n trading python=3.10

# 激活环境
conda activate trading

# 安装基础依赖
pip install numpy pandas ccxt dash plotly gymnasium

# 安装技术分析库
pip install ta

# 安装机器学习相关
pip install torch scikit-learn

# 安装其他工具
pip install python-dateutil