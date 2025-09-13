from typing import Dict, Tuple, Optional  
import numpy as np  
import pandas as pd  
import time
from config import TradingEnvConfig  
from .exchange_manager import ExchangeManager  

class TradingCore:  
    def __init__(self, trading_config: TradingEnvConfig, exchange_manager: ExchangeManager):  
        self.config = trading_config  
        self.exchange = exchange_manager  
        self.balance = self.exchange.get_account_balance()  # 实时获取账户余额  
        self.position = 0.0  
        self.entry_price = 0.0  
        self.unrealized_pnl = 0.0  

        self.stop_loss_pct = 0.02  # 2%止损  
        self.trailing_stop_pct = 0.03  # 3%追踪止损  
        self.max_position = trading_config.max_position  
        self.trades = []  # 存储交易记录  

    def log_trade(self, trade_type: str, price: float, size: float):  
        """  
        记录交易信息。  
        :param trade_type: 交易类型（'buy', 'sell', 'stop_loss', 'take_profit'）  
        :param price: 交易价格  
        :param size: 交易量  
        """  
        trade = {  
            'timestamp': time.time(),  
            'price': price,  
            'type': trade_type,  
            'size': size  
        }  
        self.trades.append(trade)  
        print(f"记录交易: {trade}")  

    def get_trade_logs(self):  
        """  
        获取所有交易记录。  
        """  
        return self.trades  
    
    def close_position(self) -> bool:  
        """平掉当前持仓"""  
        try:  
            # 获取当前持仓信息  
            positions = self.exchange.get_positions()  
            for pos in positions:  
                if pos['contracts'] > 0:  # 如果有持仓  
                    side = 'buy' if pos['side'] == 'short' else 'sell'  # 平掉反向持仓  
                    self.exchange.exchange.create_order(  
                        symbol=self.config.symbol,  
                        type='market',  
                        side=side,  
                        amount=pos['contracts'],  # 平掉所有持仓  
                        params={  
                            'tdMode': 'cross',  
                            'posSide': pos['side'],  
                            'reduceOnly': True  # 仅用于减少持仓  
                        }  
                    )  
                    print(f"成功平仓: {pos['contracts']} {pos['side']} 持仓")  
            # 更新状态  
            self.position = 0.0  
            self.entry_price = 0.0  
            self.unrealized_pnl = 0.0  
            return True  
        except Exception as e:  
            print(f"平仓失败: {e}")  
            return False  

    def execute_trade(self, trade_size: float, current_price: float) -> bool:  
        """执行交易"""  
        # 更新账户余额  
        self.balance = self.exchange.get_account_balance()  

        if not self._check_trade_limits(trade_size, current_price):  
            return False  

        try:  
            # 调整最小交易量  
            if 0 < abs(trade_size) < self.exchange.min_amount:  
                trade_size = self.exchange.min_amount if trade_size > 0 else -self.exchange.min_amount  

            side = 'buy' if trade_size > 0 else 'sell'  
            pos_side = 'long' if trade_size > 0 else 'short'  

            if not self._check_and_close_opposite_position(pos_side):  
                return False  

            order = self.exchange.exchange.create_order(  
                symbol=self.config.symbol,  
                type='market',  
                side=side,  
                amount=abs(trade_size),  
                params={  
                    'tdMode': 'cross',  
                    'posSide': pos_side,  
                    'leverage': self.config.leverage,  
                }  
            )  

            # 更新状态  
            self._update_after_trade(trade_size, current_price)  
            return True  

        except Exception as e:  
            print(f"交易执行失败: {e}")  
            return False  
        
    def calculate_all_factors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算并验证所有因子"""
        factors = super().calculate_all_factors(df)
        
        # 验证因子
        validation_results = self.validator.batch_validate_factors(factors, df['close'])
        
        # 存储结果
        self.factor_results = {
            name: {
                'data': factor,
                'validation': validation_results[name]
            }
            for name, factor in factors.items()
        }
        
        return factors
    
    def _check_trade_limits(self, trade_size: float, current_price: float) -> bool:  
        """检查交易限制"""  
        if 0 < abs(trade_size) < self.exchange.min_amount:  
            print(f"交易量太小: {abs(trade_size)} < {self.exchange.min_amount}")  
            return False  
            
        contract_value = abs(trade_size) * current_price  
        required_margin = contract_value / self.config.leverage  
        
        if required_margin > self.balance:  
            print(f"保证金不足: 需要 ${required_margin:.2f}，可用 ${self.balance:.2f}")  
            return False  
            
        return True  
    
    def _check_and_close_opposite_position(self, target_pos_side: str) -> bool:  
        """检查并平掉反向持仓"""  
        try:  
            positions = self.exchange.exchange.fetch_positions([self.config.symbol])  
            for pos in positions:  
                if pos['contracts'] > 0 and pos['side'] != target_pos_side:  
                    self.exchange.exchange.create_order(  
                        symbol=self.config.symbol,  
                        type='market',  
                        side='buy' if pos['side'] == 'short' else 'sell',  
                        amount=pos['contracts'],  
                        params={  
                            'tdMode': 'cross',  
                            'posSide': pos['side'],  
                            'reduceOnly': True  
                        }  
                    )  
            return True  
        except Exception as e:  
            print(f"平仓失败: {e}")  
            return False  
    
    def _update_after_trade(self, trade_size: float, current_price: float):  
        """交易后更新状态"""  
        contract_value = abs(trade_size) * current_price  
        cost = contract_value * self.config.commission_rate  
        self.balance -= cost  
        self.position += trade_size  
        self.entry_price = current_price  
        
    def calculate_reward(self, action: float) -> float:  
        """计算奖励"""  
        returns = self.unrealized_pnl / self.config.initial_balance  
        cost = abs(action - self.position) * self.config.commission_rate  
        reward = returns - cost  
        
        if abs(self.position) > self.config.max_position:  
            reward -= abs(self.position - self.config.max_position)  
        
        return reward