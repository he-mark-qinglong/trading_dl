import random
from tests.exchange_manager import ExchangeManager
from typing import Optional, Dict, List, Tuple
import threading 

class DeviatedPrice:
    low:float = None
    high:float = None
    def __init__(self, low, high):
        self.low = low
        self.high = high

class DualPositionTrader:  
    def __init__(self, exchange_manager: ExchangeManager,  
                 take_profit: float, stop_loss: float,  
                 fixed_usdt_amount: float, taker_fee: float, maker_fee=0.0002):  
        self.exchange_manager = exchange_manager  
        self.take_profit = take_profit  # 止盈百分比（例如0.3%）  
        self.stop_loss = stop_loss  # 止损百分比  
        self.fixed_usdt_amount = fixed_usdt_amount  # 固定开仓金额（例如0.5 USDT）  
        self.taker_fee = taker_fee  # 吃单手续费（例如0.05%）  
        self.leverage = exchange_manager.config.leverage  
        self.maker_fee = maker_fee

        self.hedge_target_profit = take_profit / 2

        self.not_trade = True

        self.lock = threading.Lock()
        self.tvwap:float = None
        self.atr_value:float = None

        self.high_possiblility_distribution_regression_price:float = None

        self.five_minutes_supertrend = None  #价格之上，趋势向下。反之亦然

    # def set_take_profit_vwap_price(self, long_minus_short_delta, vwap, low_possible_deviation1: DeviatedPrice, low_possible_deviation2:DeviatedPrice, low_possible_deviation3:DeviatedPrice):
    #     """
    #     夹缝规则，如果价格偏离量价一定回归概率区间，就在那附近根据盘口的方向来挂单，
    #     1.逻辑1，如果显示多单单多，我们就补充空单。 需要看效果，   逻辑是，大周期的斜率< 0,且在高概率回顾的位置。
    #     2.逻辑2，与1相反，如果空单多，我们就补充更多空单主推上涨。  逻辑是趋势，原则是更大周期的vwap斜率>0，且在高概率回归的位置。
    #     3.逻辑3，不管方向，两头挂单，  逻辑是震荡市场，原则是更大周期vwap斜率概率上在高概率区间，且数值逼近0轴。
    #     """
    #     self.tvwap = vwap
    #     self.high_possiblility_distribution_regression_price = high_possiblility_distribution_regression_price
        
    #     if long_minus_short_delta > 0:
    #         open_price = high_possiblility_distribution_regression_price
    #     else:
    #         open_price = high_possiblility_distribution_regression_price
        
    #     self.execute_trade(long_minus_short_delta, 0, )
    def update_tvwap(self, tvwap, atr_value):
        with self.lock:  # 确保线程安全 
            self.tvwap = tvwap
            self.atr_value = atr_value

    def execute_trade(self, long_minus_short_delta: float, multiplyer: float, depth_price: float):  
        """  
        根据深度变化执行开仓交易，并设置止盈和止损  
        """  
        try:  
            if self.not_trade:
                print(f"目前不开仓trend:{'up' if depth_price >= self.five_minutes_supertrend else 'down'} price:{depth_price} supertrend:{self.five_minutes_supertrend}", )
                return 
            if self.five_minutes_supertrend is None:
                return

            short_large_timeframe_vwap_and_depth_distribution = True
            if long_minus_short_delta > 0 and short_large_timeframe_vwap_and_depth_distribution:
                #self.set_take_profit_and_stop_loss_for_existing_position()  #考虑平仓

                #print("目前只做空")
                #return  #只做空。
                pass
            elif long_minus_short_delta < 0 and not short_large_timeframe_vwap_and_depth_distribution:
                #print("目前只做多")
                #return   #只做多--空不做
                pass
        
            # 根据固定金额计算开仓量  
            # amount = abs(long_minus_short_delta)/amount_dividend * self.fixed_usdt_amount * random.uniform(1, 1.4)  
            amount = multiplyer * self.fixed_usdt_amount * random.uniform(1, 1.4) 
            # 确定仓位方向  
            """
            订单方向（side）：描述的是当前订单的行为（买入或卖出）。  
            仓位方向（posSide）：描述的是订单所针对的仓位类型（多头或空头）。单笔止损止盈基于仓位（开仓多还是空）类型。
            side	posSide	含义
            buy	    long	开多仓
            buy	    short	平空仓/卖出空头
            sell	long	平多仓/卖出多头
            sell	short	开空仓
            """
            
            
            side = "buy" if long_minus_short_delta > 0 else "sell"   
            pos_side = "long" if side == "buy" else "short"  # 明确指定仓位方向
            
            self.cancel_reverse_unopened_positions(pos_side, long_minus_short_delta, depth_price)
            # print(f"准备开仓: {side} {pos_side}, 开仓金额: {self.fixed_usdt_amount} USDT,",
            #       f"开仓量: {amount:.4f} {self.exchange_manager.config.symbol}, 挂单价格:{depth_price}")  
            
            # 开仓  
            self.exchange_manager.exchange.create_limit_order(
                symbol=self.exchange_manager.config.symbol, 
                side=side,
                price=depth_price,
                amount=amount,
                params={
                    "reduceOnly": False,  # 允许开新仓
                    "posSide": pos_side  # 明确指定仓位方向 
                })
            # order = self.exchange_manager.exchange.create_order_with_take_profit_and_stop_loss(
            #     symbol=self.exchange_manager.config.symbol,  
            #     type="limit",
            #     price=depth_price,  
            #     #takeProfit=self.calc_take_profit(side=pos_side, entry_price=depth_price),
            #     # stopLoss=self.calc_stop_loss(side=pos_side, entry_price=depth_price),
            #     side=side,  
            #     amount=amount,  
            #     params={  
            #         "reduceOnly": False,  # 允许开新仓
            #         "posSide": pos_side  # 明确指定仓位方向 
            #     }  )
            
            #趋势策略下：当新的信号出现，需要suppress contrary positions(include cancelling unopened positions)
            #self.adjust_reverse_position(pos_side, long_minus_short_delta, depth_price)
            
            #对冲策略下，不需要考虑这个，只考虑整体盈利后平仓（或者回归正态分布中心的时候、又或者）
            
            # print(f"开仓成功: {order}")  
            #self.set_take_profit_and_stop_loss_for_existing_position()  
            
        except Exception as e:  
            print(f"开仓交易执行失败: {e}")  

    #hedge strategy:-----------------------
    def cancel_open_orders(self):  
        """  
        取消所有未完成的开仓订单  
        """  
        try:  
            # 获取当前未完成的订单  
            open_orders = self.exchange_manager.exchange.fetch_open_orders(self.exchange_manager.config.symbol)  
            
            # 遍历并取消每个未完成的订单  
            for order in open_orders:  
                if order['status'] == 'open':  # 确保订单是未完成状态  
                    self.exchange_manager.exchange.cancel_order(order['id'], self.exchange_manager.config.symbol)  
                    print(f"取消未完成订单: {order['id']}, 方向: {order['side']}, 数量: {order['amount']}")  
            
            print("所有未完成的开仓订单已取消")  
        except Exception as e:  
            print(f"取消未完成订单失败: {e}")  
            
    def real_time_hedgeing(self, five_minutes_supertrend):
        self.five_minutes_supertrend = five_minutes_supertrend
        
        #如果要在这里交易的话，应该是价格偏离tvwap一个atr之后才开仓，同时，结合trend选择仓位调整。
        market_price = self.get_market_price()
        with self.lock:  # 确保线程安全 
            #后续加个atr范围，也就是表示趋势属于震荡
            if self.not_trade or self.five_minutes_supertrend is None or self.atr_value is None:
                return
            atr = self.atr_value
            if self.tvwap > self.five_minutes_supertrend+atr:  #uptrend, will return to 1m tvwap
                long_multi = 0.3
                short_multi = 0.7
            elif self.tvwap < self.five_minutes_supertrend+atr: #downtrend, will return
                long_multi = 0.7
                short_multi = 0.3
            else:  #算是range
                long_multi = 0.5
                short_multi = 0.5
                
            if market_price > self.tvwap + self.atr_value*3:
                #uppon tvwap:open_short
                self.execute_trade(-1, short_multi, market_price)
            elif market_price < self.tvwap - self.atr_value*3:
                #below tvwap:open_long
                self.execute_trade(1, long_multi, market_price)
            else:
                #not deviat much, no much profit, hold or skip.
                if self.should_close_at_tvwap(market_price, self.tvwap):
                    self.hedge_close_positions(self.tvwap)

                    '''
                    1. 为什么需要取消未完成的开仓订单？
                    避免新仓位的建立：

                    如果未完成的开仓订单在平仓后被触发，会导致新的仓位被建立，可能与当前的策略目标相冲突。
                    锁定利润：

                    达到目标利润后，策略的目标是锁定收益。如果未完成的开仓订单被触发，可能会导致新的风险敞口，影响整体收益。
                    减少资金占用：

                    未完成的开仓订单会占用保证金或资金，取消这些订单可以释放资金，用于其他交易或策略。
                    简化逻辑：

                    在平仓后，取消未完成的开仓订单可以让策略逻辑更加清晰，避免复杂的状态管理。
                    '''
                    self.cancel_open_orders()

    print_interval = 10
    print_counter = 0
    def is_profit_enough(self, market_price: float) -> bool:  
        """  
        判断当前持仓的利润是否足够，考虑手续费和杠杆  
        :param market_price: 当前市场价格  
        :return: True 如果利润足够，False 否则  
        """  
        try:  
            # 获取当前持仓  
            positions = self.exchange_manager.exchange.fetch_positions([self.exchange_manager.config.symbol])  
            
            total_unrealized_pnl = 0  # 总未实现盈亏  
            total_margin = 0  # 总保证金  
            total_fees = 0  # 总手续费  

            for position in positions:  
                if position['entryPrice'] is None: #只是挂单，没有开仓。
                    continue
                entry_price = float(position['entryPrice'])  # 开仓价格  
                amount = float(position['contracts'])  # 仓位数量  
                pos_side = position['side']  # 仓位方向（long 或 short）  

                if amount > 0:  
                    # 计算未实现盈亏  
                    if pos_side == "long":  
                        unrealized_pnl = (market_price - entry_price) * amount  
                    else:  # short  
                        unrealized_pnl = (entry_price - market_price) * amount  

                    # 累计未实现盈亏  
                    total_unrealized_pnl += unrealized_pnl  

                    # 计算保证金（假设每个仓位的杠杆相同）  
                    margin = amount * entry_price / self.leverage  
                    total_margin += margin  

                    # 计算手续费（开仓和平仓各一次）  
                    fees = 2 * amount * entry_price * self.maker_fee  
                    total_fees += fees  

            # 计算净利润（扣除手续费）  
            net_profit = total_unrealized_pnl - total_fees  

            # 计算利润率  
            if total_margin > 0:  
                profit_ratio = net_profit / total_margin  
            else:  
                profit_ratio = 0  

            # 判断是否达到目标利润率  
            self.print_counter += 1
            if self.print_counter > self.print_interval:
                print(f"当前净利润率: {profit_ratio:.2%}, 目标利润率: {self.hedge_target_profit:.2%}") 
                self.print_counter = 0

            return profit_ratio >= self.hedge_target_profit  
        except Exception as e:  
            print(f"判断利润是否足够失败: {e}")  
            return False 
    def should_close_at_tvwap(self, market_price: float, tvwap_price: float) -> bool:  
        """  
        判断当前市场价格是否等于 TVWAP 价格，如果是则平仓  
        :param market_price: 当前市场价格  
        :param tvwap_price: 当前 TVWAP 价格  
        :return: True 如果市场价格等于 TVWAP 价格，False 否则  
        """  
        try:  
            # 获取当前持仓  
            positions = self.exchange_manager.exchange.fetch_positions([self.exchange_manager.config.symbol])  

            total_entry_price = 0  # 总开仓价格  
            total_amount = 0  # 总仓位数量  

            for position in positions:  
                if position['entryPrice'] is None:  # 只是挂单，没有开仓  
                    continue  
                entry_price = float(position['entryPrice'])  # 开仓价格  
                amount = float(position['contracts'])  # 仓位数量  

                # 累计开仓价格和仓位数量  
                total_entry_price += entry_price * amount  
                total_amount += amount  

            # 计算平均开仓价格  
            if total_amount > 0:  
                avg_entry_price = total_entry_price / total_amount  
            else:  
                avg_entry_price = 0  
                return False

            # 判断市场价格是否等于 TVWAP 价格  
            if abs(market_price - tvwap_price) < self.atr_value/2:  # 允许微小误差  
                print(f">>>>>>>市场价格 {market_price} 等于 TVWAP 价格 {tvwap_price}，触发平仓")  
                return True  
            else:  
                print(f">>>>>>>市场价格 {market_price} 不等于 TVWAP 价格 {tvwap_price}，继续持仓")  
                return False  
        except Exception as e:  
            print(f">>>>>>判断是否平仓失败: {e}")  
            return False 
    def close_all_positions_with_reduce_only(self, market_price: float, pos_side: str, amount: float):  
        """  
        使用 reduceOnly 参数平掉指定方向的仓位  
        :param market_price: 当前市场价格  
        :param pos_side: 仓位方向（long 或 short）  
        :param amount: 平仓数量  
        """  
        try:  
            exit_side = "sell" if pos_side == "long" else "buy"  
            # adjusted_price = market_price * (1 + 0.001) if pos_side == "long" else market_price * (1 - 0.001)  
            adjusted_price = market_price
            # 创建 reduceOnly 限价单  
            self.exchange_manager.exchange.create_reduce_only_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="limit",  
                price=adjusted_price,  
                side=exit_side,  
                amount=amount,  
                params={  
                    "reduceOnly": True,  
                    "posSide": pos_side,  
                }  
            )  
            print(f"{pos_side} 仓位平仓订单已提交，价格: {adjusted_price}, 数量: {amount}")  
        except Exception as e:  
            print(f"平仓失败: {e}")  

    def hedge_close_positions(self, market_price: float):  
        """  
        对冲平仓：同时平掉多头和空头仓位  
        :param market_price: 当前市场价格  
        """  
        try:  
            # 获取当前持仓  
            positions = self.exchange_manager.exchange.fetch_positions([self.exchange_manager.config.symbol])  
            
            # 分别统计多头和空头的总仓位  
            long_amount = sum([float(position['contracts']) for position in positions if position['side'] == "long"])  
            short_amount = sum([float(position['contracts']) for position in positions if position['side'] == "short"])  
            
            # 创建线程  
            threads = []  
            if long_amount > 0:  
                long_thread = threading.Thread(target=self.close_all_positions_with_reduce_only, args=(market_price, "long", long_amount))  
                threads.append(long_thread)  
            
            if short_amount > 0:  
                short_thread = threading.Thread(target=self.close_all_positions_with_reduce_only, args=(market_price, "short", short_amount))  
                threads.append(short_thread)  
            
            # 启动线程  
            for thread in threads:  
                thread.start()  
            
            # 等待所有线程完成  
            for thread in threads:  
                thread.join()  
            
            print("多头和空头仓位平仓操作已完成")  
        except Exception as e:  
            print(f"对冲平仓失败: {e}")

    #hedge strategy end-------------------------------

    def cancel_reverse_unopened_positions(self, good_pos_side: str, long_minus_short_delta: float, depth_price: float):
        # 获取未成交的订单  
        open_orders = self.exchange_manager.exchange.fetch_open_orders(self.exchange_manager.config.symbol)  
        for order in open_orders:  
            if order['side'] != good_pos_side:  
                self.exchange_manager.exchange.cancel_order(id=order['id'], symbol=self.exchange_manager.config.symbol)  
                print(f"取消未开仓的反向仓位订单: {order['id']}")  
                
    
            
    def set_take_profit_and_stop_loss_for_existing_position(self, care_side='both', shirnk_stop_profit=False):  
        """  
        为已经开仓的仓位设置整体止盈和止损（区分多空方向）  
        :param care_side: 关注的仓位方向，'both'（多空都关注）、'long'（仅多仓）、'short'（仅空仓）  
        :param shirnk_stop_profit: 是否缩小止盈  
        """  
        try:  
            # 获取当前持仓  
            positions = self.exchange_manager.exchange.fetch_positions([self.exchange_manager.config.symbol])  

            # 初始化多空方向的整体开仓价格和仓位数量  
            long_entry_price_total = 0  # 多仓总开仓价格  
            long_amount_total = 0  # 多仓总数量  
            short_entry_price_total = 0  # 空仓总开仓价格  
            short_amount_total = 0  # 空仓总数量  

            # 计算多空方向的整体开仓价格和仓位数量  
            for position in positions:  
                if position['entryPrice'] is None:  # 是否已经开仓了  
                    continue  
                pos_side = position['side']  # 仓位方向（long 或 short）  
                entry_price = float(position['entryPrice'])  # 开仓价格  
                amount = float(position['contracts'])  # 仓位数量  

                if pos_side == 'long':  
                    long_entry_price_total += entry_price * amount  
                    long_amount_total += amount  
                elif pos_side == 'short':  
                    short_entry_price_total += entry_price * amount  
                    short_amount_total += amount  

            # 计算多空方向的平均开仓价格  
            long_avg_entry_price = long_entry_price_total / long_amount_total if long_amount_total > 0 else 0  
            short_avg_entry_price = short_entry_price_total / short_amount_total if short_amount_total > 0 else 0  

            self.cancel_existing_stop_loss_take_profit_orders('long')
            self.cancel_existing_stop_loss_take_profit_orders('short')
            # 为多空方向的整体持仓设置止盈和止损  
            if care_side == 'both' or care_side == 'long':  
                if long_amount_total > 0:  
                    print(f"为多仓设置止盈和止损，平均开仓价格: {long_avg_entry_price:.2f}, 数量: {long_amount_total:.4f}")  
                    self.set_overall_take_profit(pos_side='long', amount=long_amount_total, entry_price=long_avg_entry_price, shirnk_stop_profit=shirnk_stop_profit)  
                    self.set_overall_stop_loss(pos_side='long', amount=long_amount_total, entry_price=long_avg_entry_price, shirnk_stop_profit=False)  

            if care_side == 'both' or care_side == 'short':  
                if short_amount_total > 0:  
                    print(f"为空仓设置止盈和止损，平均开仓价格: {short_avg_entry_price:.2f}, 数量: {short_amount_total:.4f}")  
                    self.set_overall_take_profit(pos_side='short', amount=short_amount_total, entry_price=short_avg_entry_price, shirnk_stop_profit=shirnk_stop_profit)  
                    self.set_overall_stop_loss(pos_side='short', amount=short_amount_total, entry_price=short_avg_entry_price, shirnk_stop_profit=False)  

        except Exception as e:  
            print(f"为已开仓仓位设置止盈止损失败: {e}")
    
    def cancel_existing_stop_loss_take_profit_orders(self, pos_side: str):  
        """  
        取消之前设置的止损止盈订单  
        :param pos_side: 仓位方向，'long' 或 'short'  
        """  
        try:  
            # 获取所有未成交的订单  
            open_orders = self.exchange_manager.exchange.fetch_open_orders(symbol=self.exchange_manager.config.symbol)  

            # 筛选出需要取消的止损止盈订单  
            for order in open_orders:  
                # 检查订单是否为止损止盈订单  
                if order['info'].get('posSide') == pos_side and order['type'] in ['stop', 'take_profit']:  
                    # 取消订单  
                    self.exchange_manager.exchange.cancel_order(order['id'], symbol=self.exchange_manager.config.symbol)  
                    print(f"已取消订单: {order['id']}, 类型: {order['type']}, 方向: {pos_side}")  

        except Exception as e:  
            print(f"取消止损止盈订单失败: {e}")  
            
    def set_overall_take_profit(self, pos_side: str, amount: float, entry_price: float, shirnk_stop_profit=False):  
        """设置整体止盈订单"""  
        try:  
            overall_take_profit_price = entry_price * (1 + self.take_profit / (200 if shirnk_stop_profit else 100)) \
                if pos_side == "long" else entry_price * (1 - self.take_profit / (200 if shirnk_stop_profit else 100))  
            overall_take_profit_price = self.tvwap+self.atr_value if pos_side == "long" else self.tvwap - self.atr_value
            # 挂整体止盈单  
            exit_side = "sell" if pos_side == "long" else "buy"  
            # self.exchange_manager.exchange.set_take_profit_and_stop_loss_params(
            #     symbol=self.exchange_manager.config.symbol,  
            #     type="limit",  
            #     side=exit_side,  
            #     amount=amount, 
            #       )
            order = self.exchange_manager.exchange.create_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="limit",  
                side=exit_side,  
                amount=amount,  
                price=overall_take_profit_price,  
                params={  
                    "posSide": pos_side,  
                    "reduceOnly": False  # 允许整体止盈  
                }  
            )  
            print(f"++++整体止盈订单已挂单: {order}")  
        except Exception as e:  
            print(f"++++xxxx设置整体止盈失败: {e}")  

    def set_overall_stop_loss(self, pos_side: str, amount: float, entry_price: float, shirnk_stop_profit=False):  
        """设置整体止损订单"""  
        try:  
            overall_stop_loss_price = entry_price * (1 - self.stop_loss / (200 if shirnk_stop_profit else 100))\
                  if pos_side == "long" else entry_price * (1 + self.stop_loss / (200 if shirnk_stop_profit else 100))  
            # overall_stop_loss_price=self.calc_stop_loss(pos_side=pos_side, entry_price=entry_price)

            # 挂整体止损单  
            exit_side = "sell" if pos_side == "long" else "buy"  
            order = self.exchange_manager.exchange.create_stop_loss_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="stop_limit",  # 使用市价止损单="stop"  
                side=exit_side,  
                amount=amount,  
                stopLossPrice=overall_stop_loss_price,  
                params={
                    "posSide":pos_side,
                    },
            )  
            print(f"----整体止损订单已挂单: {order}")  
        except Exception as e:  
            print(f"----xxxx设置整体止损失败: {e}")

    def get_market_price(self) -> float:  
        """获取市场价格"""  
        try:  
            ticker = self.exchange_manager.exchange.fetch_ticker(self.exchange_manager.config.symbol)
            # print('ticker', ticker)  
            return float(ticker['last'])  
        except Exception as e:  
            print(f"获取市场价格失败: {e}")  
            return 0.0  

    def get_average_entry_price(self, pos_side: str) -> Optional[float]:  
        """获取开仓价格"""  
        try:  
            same_side_entry_prices = []
            positions = self.exchange_manager.exchange.fetch_positions([self.exchange_manager.config.symbol])  
            for position in positions:  
                if position['side'] == pos_side:  
                     same_side_entry_prices.append(float(position['entryPrice']))
            return mean(same_side_entry_prices) if len(same_side_entry_prices) > 0 else None
        except Exception as e:  
            print(f"获取开仓价格失败: {e}")  
            return None  
        
    def calc_take_profit(self, side, entry_price):
        effective_take_profit = self.take_profit * random.uniform(1, 1.2) + (self.maker_fee * 2 * 100 / self.leverage)  
        if side == "long":  
            take_profit_price = entry_price * (1 + effective_take_profit / 100)  
        else:  
            take_profit_price = entry_price * (1 - effective_take_profit / 100)  
        # print(f"entry_price={entry_price}, take_profit_price={take_profit_price}, pos_side={side}")
        return take_profit_price
    
    def calc_stop_loss(self, side, entry_price):
        effective_stop_loss = self.stop_loss - (self.maker_fee * 2 * 100 / self.leverage)  
        if side == "long":  
            stop_loss_price = entry_price * (1 - effective_stop_loss / 100)  
        else:  
            stop_loss_price = entry_price * (1 + effective_stop_loss / 100) 
        # print(f"entry_price={entry_price}, stop_loss_price={stop_loss_price}, pos_side={side}")
        return stop_loss_price
    
    def set_take_profit(self, pos_side: str, amount: float, entry_price: float):  
        """设置止盈订单"""  
        try:  
            take_profit_price = self.calc_take_profit(pos_side=pos_side, entry_price=entry_price)
            # 挂止盈单  
            exit_side = "sell" if pos_side == "long" else "buy"  
            order = self.exchange_manager.exchange.create_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="limit",  
                side=exit_side,  
                amount=amount,  
                price=take_profit_price,  
                params={  
                    "posSide": pos_side  
                }  
            )  
            print(f"止盈订单已挂单: {order}")  
        except Exception as e:  
            print(f"设置止盈失败: {e}")  

    def set_stop_loss(self, pos_side: str, amount: float, entry_price: float):  
        """设置止损订单"""  
        try:  
            # 挂止损单  
            stopLossPrice=self.calc_stop_loss(pos_side=pos_side, entry_price=entry_price)
            exit_side = "sell" if pos_side == "long" else "buy"  
            order = self.exchange_manager.exchange.create_stop_loss_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="stop_limit",  # 使用市价止损单="stop"  
                side=exit_side,  
                amount=amount,  
                stopLossPrice=stopLossPrice,
                params={  
                    "posSide": pos_side
                }  
            )  
            print(f"止损订单已挂单: {order}")  
        except Exception as e:  
            print(f"设置止损失败: {e} stop_loss_price={stopLossPrice}")  

# 使用示例  
# 假设我们已经有了 exchange_manager 的实例  
# trader = DualPositionTrader(exchange_manager, take_profit=0.3, stop_loss=40, fixed_usdt_amount=0.5, taker_fee=0.0005)  