from typing import Dict, List  
import pandas as pd  

def get_latest_price(exchange, symbol):  
    """  
    获取交易品种的最新价格  
    :param exchange: ccxt 交易所实例  
    :param symbol: 交易对（如 'BTC/USDT'）  
    :return: 最新价格  
    """  
    ticker = exchange.fetch_ticker(symbol)  
    return ticker['last']
def get_account_balance(exchange, currency='USDT'):  
    """  
    获取账户余额  
    :param exchange: ccxt 交易所实例  
    :param currency: 计价货币（如 USDT）  
    :return: 账户余额  
    """  
    balance = exchange.fetch_balance()  
    return balance['total'].get(currency, 0)

class DynamicMultiTimeframeStrategy:  
        def __init__(self, leverage: int = 50, risk_per_trade: float = 0.02):  
            self.signals = []  
            self.position_size = 0  
              
            self.leverage = leverage  
            self.risk_per_trade = risk_per_trade  

        def calculate_position_size(self, atr: float, current_price: float, balance: float, symbol: str) -> float:  
            """  
            基于 ATR 计算仓位大小  
            """  
            #假设pepe，最低开仓1手为100万股pepe原始合约股。
            min_position_size = 1_000_000  #后续根据symbol来获取或者本地保存后进行字典查询。
            # 风险敞口（账户的 2%）  
            risk_amount = balance * self.risk_per_trade
            
            # 使用 1.5 倍 ATR 作为止损距离  
            stop_loss_distance = 2.5 * atr  
            
            # 计算止损比例  
            stop_loss_percentage = stop_loss_distance / current_price  
            
            # 考虑杠杆的实际风险比例  
            leveraged_risk = stop_loss_percentage * self.leverage  
            
            # 计算安全开仓金额  
            position_size = risk_amount / leveraged_risk  

            # 检查是否满足最低开仓单位  
            if position_size < min_position_size * current_price:  
                return 0  # 无法开仓  

            # 返回最终仓位大小（向下取整到最小单位的倍数）  
            return position_size // (min_position_size * current_price) * min_position_size

        def generate_signal(self, trend: str, factors: Dict[str, pd.Series], current_price: float, symbol: str, balance) -> Dict:  
            """  
            根据趋势和因子生成交易信号  
            :param trend: 当前市场趋势（'uptrend', 'downtrend', 'range'）  
            :param factors: 因子字典  
            :param current_price: 当前价格  
            :param symbol: 交易对（如 'BTC/USDT'）  
            :param exchange: ccxt 交易所实例  
            :return: 交易信号  
            """  
            rsi = factors['rsi'].iloc[-1]  
            atr = factors['atr'].iloc[-1]
            bollinger_lower = factors['bb_low'].iloc[-1]
            bollinger_middle = factors['bb_mid'].iloc[-1]
            bollinger_higher = factors['bb_high'].iloc[-1]

            up_boll_middle = bollinger_middle*1.02
            down_boll_middle = bollinger_middle * 0.98
            #print(f'trend:{trend}, rsi:{rsi}, down_boll_middle:{down_boll_middle} current_price:{current_price}, up_boll_middle:{up_boll_middle}')
            if trend == 'uptrend' and rsi < 55 and current_price < up_boll_middle:  
                size = self.calculate_position_size(atr=atr, current_price=current_price,
                                                    symbol=symbol, balance=balance) 
                return {  
                    'timestamp': factors['rsi'].index[-1],  
                    'signal_type': 'buy',  
                    'price': current_price,  
                    'size': size  
                }  
            elif trend == 'downtrend' and rsi > 45 and current_price > down_boll_middle:  
                size = self.calculate_position_size(atr=atr, current_price=current_price,
                                                    symbol=symbol, balance=balance)  
                return {  
                    'timestamp': factors['rsi'].index[-1],  
                    'signal_type': 'sell',  
                    'price': current_price,  
                    'size': size  
                }  
            # 震荡市场：布林带下轨开多，中轨平仓；布林带上轨开空，中轨平仓  
            elif trend == 'range':  
                # 开多信号：接近下轨  
                if current_price < bollinger_lower:  
                    size = self.calculate_position_size(atr=atr, current_price=current_price,
                                                        symbol=symbol, balance=balance)
                    return {  
                        'timestamp': factors['rsi'].index[-1],  
                        'signal_type': 'buy',  
                        'price': current_price,  
                        'size': size,  
                        'target_price': bollinger_middle  # 设置平仓目标为中轨  
                    }  
                # 开空信号：接近上轨  
                elif current_price > bollinger_higher:  
                    size = self.calculate_position_size(atr=atr, current_price=current_price,
                                                        symbol=symbol, balance=balance)  
                    return {  
                        'timestamp': factors['rsi'].index[-1],  
                        'signal_type': 'sell',  
                        'price': current_price,  
                        'size': size,  
                        'target_price': bollinger_middle  # 设置平仓目标为中轨  
                    }  

            #print(f'trend:{trend}  rsi:{rsi} bollinger_lower:{bollinger_lower} current_price:{current_price}  bollinger_higher:{bollinger_higher}')
            return None

        def process_data(self, trend: str, factors: Dict[str, pd.Series], current_price: float, symbol: str, balance):  
            """  
            处理因子数据并生成交易信号  
            :param trend: 当前市场趋势  
            :param factors: 因子字典  
            :param current_price: 当前价格  
            :param symbol: 交易对（如 'BTC/USDT'）  
            :param exchange: ccxt 交易所实例  
            """  
            signal = self.generate_signal(trend, factors, current_price, symbol, balance)  
            if signal and signal not in self.signals:  
                self.signals.append(signal)  
                self.position_size += signal['size']  
                # print(f"生成交易信号: {signal}")

        def get_signal(self): 
            if len(self.signals) == 0:
                return None
            return self.signals[-1]