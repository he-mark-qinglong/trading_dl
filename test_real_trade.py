from envs.trading_core import TradingCore
from envs.exchange_manager import ExchangeManager  
from config import TradingEnvConfig, ExchangeConfig  

# 初始化配置  
trading_config = TradingEnvConfig(  
    symbol="PEPE-USDT-SWAP",  
    leverage=50,  
    initial_balance=3.3,  
    commission_rate=0.0005,  
    max_position=100  
)  
exchange_config = ExchangeConfig(   
)  
# 初始化交易模块  
exchange_manager = ExchangeManager(trading_config, exchange_config)  
trading_core = TradingCore(trading_config, exchange_manager)  

# 获取当前市场价格（假设从交易所获取）  
ticker = exchange_manager.exchange.fetch_ticker(trading_config.symbol)  
current_price = ticker['last']  # 获取最新成交价  

# 打印账户余额  
account_balance = exchange_manager.get_account_balance()  
print(f"账户余额: {account_balance} USDT")  

# 打印当前持仓信息  
positions = exchange_manager.get_positions()  
print(f"当前持仓: {positions}")  

# 示例：开仓 0.01 BTC  
trade_size = 0.1  # 买入 0.01 BTC  
success = trading_core.execute_trade(trade_size, current_price)  

if success:  
    print("交易成功！")  
else:  
    print("交易失败！")  

import time 
time.sleep(30)
success = trading_core.close_position()
if success:  
    print("平仓交易成功！")  
else:  
    print("平仓交易失败！")  
