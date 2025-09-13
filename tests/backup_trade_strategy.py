def adjust_reverse_position(self, good_pos_side: str, long_minus_short_delta: float, depth_price: float):  
        """  
        调整反向仓位：  
        1. 取消未开仓的反向仓位。  
        2. 缩小已开仓反向仓位的止盈和止损。  
        3. 如果反向仓位已经超过盈利或损失的一半，则平仓。  
        """  
        try: 
            # 获取当前持仓  
            positions = self.exchange_manager.exchange.fetch_positions([self.exchange_manager.config.symbol])  
            #先统计反方向是否整体盈利
            prices = []
            amounts = []
            for position in positions:  
                # 1. 取消未开仓的反向仓位  
                if position['entryPrice'] is None: #是否已经开仓了 
                    continue
                # 检查是否为反向仓位  
                if position['side'] == good_pos_side:  
                    continue  

                # 获取仓位信息  
                entry_price = float(position['entryPrice'])  
                  
                amount = float(position['contracts'])  
                unrealized_pnl = float(position['unrealizedPnl'])  

                prices.append(entry_price)
                amounts.append(amount)

            import numpy as np
            average_entry_price = np.mean(prices)
            
            # 2. 平仓条件：如果反向仓位已经超过盈利平衡或损失的一半（因为新的方向反了，不抗风险就平掉，抗风险就继续扛单，这个条件不止损） or 无条件平仓 
            # current_price = self.get_market_price()
            profit_delta = depth_price - average_entry_price
            profit_delta = 0 + profit_delta if position['side'] == "long" else -profit_delta
            
            can_close_all = False
            if profit_delta >= (self.taker_fee+self.maker_fee + self.take_profit/1.2)/self.leverage:
                can_close_all = True
            #再根据如果是盈利的，全部平仓掉。
            if can_close_all:
                # 平仓  
                # 如果持仓是 多仓（long），则平仓方向为 卖出（sell）；如果持仓是 空仓（short），则平仓方向为 买入（buy）。
                total_amount = np.sum(amounts)
                exit_side = "sell" if position['side'] == "long" else "buy"  
                self.exchange_manager.exchange.create_reduce_only_order(
                    symbol=self.exchange_manager.config.symbol,  
                    type="limit",
                    # type="market",
                    price=depth_price,  
                    side=exit_side,  
                    amount=total_amount,  
                    params={
                        "posSide": 'long'if exit_side == 'sell' else 'short',
                    }  # 仅平仓  
                    )
                print(f"平仓反向仓位 { position['side']}, 数量: {amount:.4f}, 价格: {depth_price:.2f}")  

            # 3. 缩小已开仓反向仓位的止盈和止损 
            if abs(long_minus_short_delta) > 4 * 3000:   #eth 3000 is normal
                # self.set_take_profit_and_stop_loss_for_existing_position(care_side='long' if good_pos_side == 'short' else 'short', shirnk_stop_profit=True) 
                print(f"突破了delta的极限值{long_minus_short_delta}，缩小反向仓位 {good_pos_side} 的止盈和止损") 

        except Exception as e:  
            print(f"调整反向仓位失败: {e}")  