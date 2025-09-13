import gymnasium as gym  
from gymnasium import spaces  
import numpy as np  
import pandas as pd  
from typing import Dict, Tuple, Optional  

from .trading_core import TradingCore  
from .exchange_manager import ExchangeManager  
from config import TradingEnvConfig, ExchangeConfig  
from factors import DataManager  

class ContractTradingEnv(gym.Env):  
    def __init__(self, envContext):  
        print(f'envContext=============={envContext}')  
        trading_config, exchange_config = envContext['trading_config'], envContext['exchange_config']

        
        super().__init__()  
        self.config = trading_config  
        
        

        # 初始化交易所和交易核心  
        self.exchange_manager = ExchangeManager(trading_config, exchange_config)  
        self.trading_core = TradingCore(trading_config, self.exchange_manager)  
        
        # 初始化因子管理器  
        self.data_manager = DataManager(trading_config.symbol)  
        
        #先缓存一次全量数据集保持本地有数据可以计算各类因子，以及滑动窗口
        temp_size = self.config.window_size
        self.config.window_size = 60*60
        self._get_data()  
        self.config.window_size = temp_size

        # 设置动作空间  
        self.action_space = spaces.Box(  
            low=-1, high=1, shape=(1,), dtype=np.float32  
        )  
        
        # 设置观察空间  
        self.observation_space = self._setup_observation_space()  
        
    def _setup_observation_space(self) -> spaces.Box:  
        """设置观察空间"""  
        # 获取样本数据计算状态维度  
        sample_data = self._get_data()  
        if sample_data is not None:  
            self.data_manager.update_data(sample_data)  
            state = self._calculate_state()  
            state_dim = len(state)  
        else:  
            state_dim = 100  # 默认维度  
            
        return spaces.Box(  
            low=-np.inf,  
            high=np.inf,  
            shape=(state_dim,),  
            dtype=np.float32  
        )  
        
    def _get_data(self) -> Optional[pd.DataFrame]:  
        """获取市场数据"""  
        try:  
            # 获取更多的历史数据  
            limit = max(100, self.config.window_size * 3)  # 确保有足够的数据  
            ohlcv = self.exchange_manager.exchange.fetch_ohlcv(  
                self.config.symbol,  
                self.config.timeframe,  
                limit=limit  
            )  
            
            df = pd.DataFrame(  
                ohlcv,  
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']  
            )  
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  
            df.set_index('timestamp', inplace=True)  
            
            return df  
        except Exception as e:  
            print(f"获取数据失败: {e}")  
            return None 
            
    def _calculate_state(self) -> np.ndarray:  
        """计算当前状态"""  
        df = self._get_data()  
        if df is None:  
            return np.zeros(self.observation_space.shape)  
            
        try:  
            # 更新数据管理器  
            self.data_manager.update_data(df)  
            
            # 获取所有时间周期的因子  
            factors = []  
            for timeframe in ['3m', '5m', '15m']:  
                timeframe_factors = self.data_manager.get_factors(timeframe)  
                if timeframe_factors:  
                    latest_factors = np.array([v.iloc[-1] for v in timeframe_factors.values()])  
                    factors.append(latest_factors)  
                    
            # 添加位置信息  
            position_info = np.array([  
                self.trading_core.position,  
                self.trading_core.balance / self.config.initial_balance - 1,  
                self.trading_core.unrealized_pnl / self.config.initial_balance  
            ])  
            
            return np.concatenate([*factors, position_info])  
            
        except Exception as e:  
            print(f"计算状态失败: {e}")  
            return np.zeros(self.observation_space.shape)  
            
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:  
        """执行交易步骤"""  
        action = float(action[0])  
        
        df = self._get_data()  
        if df is None:  
            return self._calculate_state(), 0, True, False, {}  
            
        current_price = df['close'].iloc[-1]  
        target_position = action * self.config.max_position  
        trade_size = target_position - self.trading_core.position  
        
        if trade_size != 0:  
            self.trading_core.execute_trade(trade_size, current_price)  
            
        # 更新未实现盈亏  
        if self.trading_core.position != 0:  
            self.trading_core.unrealized_pnl = (  
                self.trading_core.position *   
                (current_price - self.trading_core.entry_price) *  
                self.config.leverage  
            )  
            
        new_state = self._calculate_state()  
        reward = self.trading_core.calculate_reward(action)  
        
        # 计算保证金率  
        margin_ratio = (  
            (self.trading_core.balance + self.trading_core.unrealized_pnl) /  
            (abs(self.trading_core.position) * current_price)  
            if self.trading_core.position != 0 else 1  
        )  
        
        # 检查是否触及止损  
        done = (  
            self.trading_core.balance <= self.config.initial_balance * 0.2 or  
            margin_ratio <= 0.2  
        )  
        
        info = {  
            'balance': self.trading_core.balance,  
            'position': self.trading_core.position,  
            'unrealized_pnl': self.trading_core.unrealized_pnl,  
            'trade_size': trade_size,  
            'current_price': current_price,  
            'margin_ratio': margin_ratio  
        }  
        
        return new_state, reward, done, False, info  
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:  
        """重置环境"""  
        super().reset(seed=seed)  
        
        try:  
            balance = self.exchange_manager.exchange.fetch_balance()  
            self.trading_core.balance = float(balance.get('USDT', {}).get('free', 0))  
        except Exception as e:  
            print(f"获取余额失败: {e}")  
            self.trading_core.balance = self.config.initial_balance  
            
        self.trading_core.position = 0.0  
        self.trading_core.entry_price = 0.0  
        self.trading_core.unrealized_pnl = 0.0  
        
        initial_state = self._calculate_state()  
        info = {  
            'balance': self.trading_core.balance,  
            'position': self.trading_core.position,  
            'unrealized_pnl': self.trading_core.unrealized_pnl,  
            'margin_ratio': 1.0  
        }  
        
        return initial_state, info