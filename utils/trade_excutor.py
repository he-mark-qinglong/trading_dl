from abc import ABC, abstractmethod  
from typing import Dict, Any
from .risk_manager import RiskManager
import pandas as pd
import random, time
from exchange_manager import ExchangeManager

class TradeExecutor(ABC):  
    @abstractmethod  
    def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:  
        """  
        执行交易  
        :param signal: 交易信号  
        :return: 交易执行结果  
        """  
        pass 


class RealTradeExecutor(TradeExecutor):  
    def __init__(self, exchange_manager: ExchangeManager, risk_manager: RiskManager):  
        """  
        实现真实交易的执行器  
        :param exchange_manager: 交易所管理器  
        :param risk_manager: 风险管理器  
        """  
        self.exchange_manager = exchange_manager  
        self.risk_manager = risk_manager  
        self.trade_log = []  # 真实交易日志  

    def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:  
        """  
        真实交易执行  
        :param signal: 交易信号  
        :return: 交易执行结果  
        """  
        if signal["signal"] is None:  
            return {"status": "no_action", "details": "No valid signal to execute"}  

        try:  
            # 获取交易方向  
            side = signal["signal"].split("_")[1]  # "open_long" -> "long"  

            # 执行开仓或平仓操作  
            if signal["signal"].startswith("open"):  
                return self.open_position(side, signal["position_size"])  
            elif signal["signal"].startswith("close"):  
                return self.close_position(side, signal["position_size"])  
            else:  
                return {"status": "error", "details": "Unknown signal"}  
        except Exception as e:  
            return {"status": "error", "details": str(e)}  

    def open_position(self, side: str, target_position_size: float, steps: int = 5):  
        """  
        逐步建仓  
        :param side: 建仓方向 ("long" 或 "short")  
        :param target_position_size: 目标仓位大小  
        :param steps: 建仓的步数  
        """  
        try:  
            # 计算每次建仓的数量  
            step_size = target_position_size / steps  
            pos_side = "long" if side == "long" else "short"  

            for i in range(steps):  
                # 随机浮动金额（模拟逐步建仓的随机性）  
                amount = step_size * random.uniform(0.8, 1.2)  

                # 获取市场价格  
                market_price = self.exchange_manager.exchange.fetch_ticker(self.exchange_manager.config.symbol)['last']  

                # 创建订单  
                order = self.exchange_manager.exchange.create_order(  
                    symbol=self.exchange_manager.config.symbol,  
                    type="market",  
                    side="buy" if side == "long" else "sell",  
                    amount=amount,  
                    params={"posSide": pos_side}  
                )  

                print(f"第 {i + 1} 次建仓成功: {order}")  

                # 设置止盈和止损  
                self.set_take_profit(pos_side, amount, market_price)  
                self.set_stop_loss(pos_side, amount, market_price)  

                # 等待一定时间再进行下一次建仓  
                time.sleep(60 / steps)  

            print(f"逐步建仓完成: {side}, 总目标仓位: {target_position_size:.4f}")  
        except Exception as e:  
            print(f"建仓失败: {e}")  

    def close_position(self, side: str, current_position_size: float):  
        """  
        平仓操作  
        :param side: 平仓方向 ("long" 或 "short")  
        :param current_position_size: 当前持仓大小  
        """  
        try:  
            pos_side = "long" if side == "long" else "short"  
            exit_side = "sell" if side == "long" else "buy"  

            # 创建平仓订单  
            order = self.exchange_manager.exchange.create_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="market",  
                side=exit_side,  
                amount=current_position_size,  
                params={"posSide": pos_side}  
            )  

            print(f"平仓成功: {order}")  
        except Exception as e:  
            print(f"平仓失败: {e}")  

    def set_take_profit(self, pos_side: str, amount: float, entry_price: float):  
        """设置止盈订单"""  
        try:  
            effective_take_profit = self.risk_manager.take_profit + (self.risk_manager.taker_fee * 2 * 100 / self.risk_manager.leverage)  
            if pos_side == "long":  
                take_profit_price = entry_price * (1 + effective_take_profit / 100)  
            else:  
                take_profit_price = entry_price * (1 - effective_take_profit / 100)  

            exit_side = "sell" if pos_side == "long" else "buy"  
            order = self.exchange_manager.exchange.create_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="limit",  
                side=exit_side,  
                amount=amount,  
                price=take_profit_price,  
                params={"posSide": pos_side}  
            )  
            print(f"止盈订单已挂单: {order}")  
        except Exception as e:  
            print(f"设置止盈失败: {e}")  

    def set_stop_loss(self, pos_side: str, amount: float, entry_price: float):  
        """设置止损订单"""  
        try:  
            effective_stop_loss = self.risk_manager.stop_loss - (self.risk_manager.taker_fee * 2 * 100 / self.risk_manager.leverage)  
            if pos_side == "long":  
                stop_loss_price = entry_price * (1 - effective_stop_loss / 100)  
            else:  
                stop_loss_price = entry_price * (1 + effective_stop_loss / 100)  

            exit_side = "sell" if pos_side == "long" else "buy"  
            order = self.exchange_manager.exchange.create_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="stopMarket",  
                side=exit_side,  
                amount=amount,  
                params={"posSide": pos_side, "stopPx": stop_loss_price}  
            )  
            print(f"止损订单已挂单: {order}")  
        except Exception as e:  
            print(f"设置止损失败: {e}")


import json  
import websocket  
import threading  
import time  
from typing import Optional, Dict  

class Position:  
    def __init__(self):  
        self.direction = None  # 持仓方向 ("long" 或 "short")  
        self.size = 0.0  # 持仓量  
        self.entry_price = 0.0  # 开仓价格  
        self.stop_loss_price = 0.0  # 止损价格  
        self.take_profit_price = 0.0  # 止盈价格  
        
class OKXWebSocketTrader:  
    def __init__(self, exchange_manager, risk_manager, target_position_size: float):  
        """  
        OKX WebSocket 驱动的交易模块  
        :param exchange_manager: 交易所管理器  
        :param risk_manager: 风险管理器  
        :param target_position_size: 目标仓位大小  
        """  
        self.exchange_manager = exchange_manager  
        self.risk_manager = risk_manager  
        self.target_position_size = target_position_size  # 总目标仓位  
        self.position = Position()  # 本地维护的仓位信息  
        self.running = True  # 控制 WebSocket 的运行状态  
        self.ws = None  # WebSocket 客户端  
        self.remaining_size = target_position_size / 3  # 每次建仓的目标仓位（1/3）  
        self.partial_size = self.remaining_size / 50  # 每次建仓的最小单位（1/50）  

    def start(self):  
        """  
        启动 WebSocket 实时监控  
        """  
        threading.Thread(target=self._run_websocket).start()  

    def stop(self):  
        """  
        停止 WebSocket 实时监控  
        """  
        self.running = False  
        if self.ws:  
            self.ws.close()  

    def _run_websocket(self):  
        """  
        WebSocket 实时监控逻辑  
        """  
        url = "wss://ws.okx.com:8443/ws/v5/public"  # OKX WebSocket 地址  
        self.ws = websocket.WebSocketApp(  
            url,  
            on_message=self._on_message,  
            on_error=self._on_error,  
            on_close=self._on_close  
        )  
        self.ws.on_open = self._on_open  
        self.ws.run_forever()  

    def _on_open(self, ws):  
        """  
        WebSocket 连接成功时的回调  
        """  
        print("WebSocket 连接成功")  
        # 订阅市场数据（例如 ticker 数据）  
        params = {  
            "op": "subscribe",  
            "args": [  
                {"channel": "tickers", "instId": self.exchange_manager.config.symbol}  
            ]  
        }  
        ws.send(json.dumps(params))  

    def _on_message(self, ws, message):  
        """  
        WebSocket 接收到消息时的回调  
        """  
        data = json.loads(message)  
        if "data" in data:  
            market_price = float(data["data"][0]["last"])  # 获取最新价格  
            self._process_market_data(market_price)  

    def _on_error(self, ws, error):  
        """  
        WebSocket 出现错误时的回调  
        """  
        print(f"WebSocket 错误: {error}")  

    def _on_close(self, ws, close_status_code, close_msg):  
        """  
        WebSocket 关闭时的回调  
        """  
        print("WebSocket 连接关闭")  

    def _process_market_data(self, market_price: float):  
        """  
        处理实时市场数据  
        """  
        try:  
            # 检查是否满足建仓条件  
            if self.remaining_size > 0:  # 仅在还有剩余仓位时建仓  
                if self.is_best_entry_point(market_price):  
                    print(f"满足建仓条件，当前价格: {market_price}")  
                    self.open_position("long", self.partial_size)  # 每次建仓一小部分  

            # 检查是否触发止损或止盈  
            self.check_stop_conditions(market_price)  
        except Exception as e:  
            print(f"处理市场数据失败: {e}")  

    def is_best_entry_point(self, market_price: float) -> bool:  
        """  
        判断是否是最佳建仓点（示例逻辑）  
        """  
        # 示例：简单判断价格是否低于某个阈值  
        # 实际可以使用 RSI、布林带等因子  
        return market_price < 100  # 示例条件  

    def open_position(self, side: str, amount: float):  
        """  
        执行建仓操作  
        """  
        try:  
            # 创建订单  
            order = self.exchange_manager.exchange.create_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="market",  
                side="buy" if side == "long" else "sell",  
                amount=amount,  
                params={"posSide": side}  
            )  

            # 更新仓位信息  
            self.position.size += amount  
            self.remaining_size -= amount  
            print(f"建仓成功: {order}")  
        except Exception as e:  
            print(f"建仓失败: {e}")  

    def check_stop_conditions(self, market_price: float):  
        """  
        检查是否触发止损或止盈  
        """  
        try:  
            # 检查止损  
            if self.position.direction == "long" and market_price <= self.position.stop_loss_price:  
                print("触发止损，平仓")  
                self.close_position("long")  
            elif self.position.direction == "short" and market_price >= self.position.stop_loss_price:  
                print("触发止损，平仓")  
                self.close_position("short")  

            # 检查止盈  
            if self.position.direction == "long" and market_price >= self.position.take_profit_price:  
                print("触发止盈，平仓")  
                self.close_position("long")  
            elif self.position.direction == "short" and market_price <= self.position.take_profit_price:  
                print("触发止盈，平仓")  
                self.close_position("short")  
        except Exception as e:  
            print(f"检查止损止盈失败: {e}")  

    def close_position(self, side: str):  
        """  
        平仓操作  
        """  
        try:  
            # 创建平仓订单  
            order = self.exchange_manager.exchange.create_order(  
                symbol=self.exchange_manager.config.symbol,  
                type="market",  
                side="sell" if side == "long" else "buy",  
                amount=self.position.size,  
                params={"posSide": side}  
            )  

            # 清空仓位信息  
            self.position = Position()  
            print(f"平仓成功: {order}")  
        except Exception as e:  
            print(f"平仓失败: {e}")