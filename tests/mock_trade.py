from dataclasses import dataclass  
from datetime import datetime  
from enum import Enum  
import math  
from typing import Dict, Union, Literal, Optional  

@dataclass  
class Position:  
    """仓位数据结构"""  
    """仓位数据结构"""  
    entry_price: float = 0.0
    position_size: float = 0.0
    tp_price: float = 0.0
    sl_price: float = 0.0
    opened_at: datetime = None
    current_pnl: float = 0.0  
    current_margin_requirement: float = 0.0  
    last_price: float = 0.0  

    def update_state(self, current_price: float, leverage: float):  
        """更新仓位状态"""  
        self.last_price = current_price  
        self.current_margin_requirement = (self.position_size * current_price) / leverage  
        # 计算未实现盈亏  
        self.current_pnl = (current_price - self.entry_price) * self.position_size  

    def get_status_description(self, side: str) -> str:  
        """获取仓位状态描述"""  
        return (  
            f"{side}仓位: "  
            f"数量={self.position_size:.4f}, "  
            f"开仓价={self.entry_price:.4f}, "  
            f"当前价={self.last_price:.4f}, "  
            f"未实现盈亏={self.current_pnl:.2f}, "  
            f"保证金占用={self.current_margin_requirement:.2f}"  
        )
    
    def is_active(self) -> bool:  
        return self.position_size > 0  
    
    @classmethod  
    def create_empty(cls) -> 'Position':  
        """创建空仓位"""  
        return cls(  
            entry_price=0.0,  
            position_size=0.0,  
            tp_price=0.0,  
            sl_price=0.0,  
            opened_at=datetime.now(),  
            current_pnl=0.0,  
            current_margin_requirement=0.0,  
            last_price=0.0  
        )
@dataclass  
class MarginState:  
    """保证金状态跟踪"""  
    available_margin: float = 0.0  
    total_pnl: float = 0.0  
    total_margin_requirement: float = 0.0  
    last_fee: float = 0.0  
    effective_leverage: float = 0.0  
    margin_ratio: float = 0.0  
class MarginManager:  
    """保证金管理器"""  
    def __init__(self, initial_equity: float, leverage: float, maintenance_margin: float, fee_rate: float):  
        self.initial_equity = initial_equity  
        self.current_margin = initial_equity  
        self.leverage = leverage  
        self.maintenance_margin = maintenance_margin  
        self.fee_rate = fee_rate  
        self.peak_equity = initial_equity  
        
        # 状态跟踪  
        self.state = MarginState()  
        
    def calculate_fee(self, contract_value: float) -> float:  
        self.state.last_fee = max(contract_value * self.fee_rate, 0.01)  
        return self.state.last_fee  

    def check_margin_requirement(self, positions: Dict[str, Position], current_price: float) -> bool:  
        self.state.total_pnl = self._calculate_total_pnl(positions, current_price)  
        self.state.available_margin = self.current_margin + self.state.total_pnl  
        self.state.total_margin_requirement = self._calculate_margin_requirement(positions, current_price)  
        
        # 当没有仓位时，保证金比例设为最大值  
        if self.state.total_margin_requirement == 0:  
            self.state.margin_ratio = float('inf')  # 表示无限大的保证金比例  
            return True  # 没有仓位时总是满足保证金要求  
        
        # 有仓位时正常计算保证金比例  
        self.state.margin_ratio = self.state.available_margin / (self.state.total_margin_requirement * self.maintenance_margin)  
        return self.state.margin_ratio >= 1  

    def _calculate_total_pnl(self, positions: Dict[str, Position], current_price: float) -> float:  
        """计算总未实现盈亏"""  
        total = 0  
        for side, pos in positions.items():  
            if pos.is_active():  
                multiplier = 1 if side == "LONG" else -1  
                total += (current_price - pos.entry_price) * pos.position_size * multiplier  
        self.state.total_pnl = total  
        return total  
    
    def _calculate_margin_requirement(self, positions: Dict[str, Position], current_price: float) -> float:  
        total = 0  
        has_active_positions = False  
        for pos in positions.values():  
            if pos.is_active():  
                has_active_positions = True  
                pos.update_state(current_price, self.leverage)  
                total += pos.current_margin_requirement  
        
        self.state.total_margin_requirement = total  
        return total  

    def get_status_description(self) -> str:  
        """获取保证金状态描述"""  
        if self.state.total_margin_requirement == 0:  
            return "无仓位"  
        else:  
            return (f"保证金比例: {self.state.margin_ratio:.2f} | "  
                   f"可用保证金: {self.state.available_margin:.2f} | "  
                   f"所需保证金: {self.state.total_margin_requirement * self.maintenance_margin:.2f}")  
        
    def reset_peak_equity(self, current_equity: float) -> None:  
        """重置峰值权益"""  
        self.peak_equity = current_equity  

@dataclass  
class RiskState:  
    """风险状态跟踪"""  
    current_return: float = 0.0  
    drawdown: float = 0.0  
    last_tp_price: float = 0.0  
    last_sl_price: float = 0.0  
    close_reason: str = ""  
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH  

class RiskManager:  
    """风险管理器"""  
    def __init__(self, take_profit_pct: float, stop_loss_pct: float,  
                 total_profit_target: float, trailing_stop_activation: float,  
                 trailing_stop_pct: float, total_stop_loss_pct: float):  
        self.take_profit_pct = take_profit_pct  
        self.stop_loss_pct = stop_loss_pct  
        self.total_profit_target = total_profit_target  
        self.trailing_stop_activation = trailing_stop_activation  
        self.trailing_stop_pct = trailing_stop_pct  
        self.total_stop_loss_pct = total_stop_loss_pct  
        
        # 状态跟踪  
        self.state = RiskState()  

    def calculate_tp_sl_prices(self, entry_price: float, is_long: bool) -> tuple[float, float]:  
        if is_long:  
            self.state.last_tp_price = entry_price * (1 + self.take_profit_pct)  
            self.state.last_sl_price = entry_price * (1 - self.stop_loss_pct)  
        else:  
            self.state.last_tp_price = entry_price * (1 - self.take_profit_pct)  
            self.state.last_sl_price = entry_price * (1 + self.stop_loss_pct)  
        return self.state.last_tp_price, self.state.last_sl_price  

    def should_close_all(self, current_equity: float, initial_equity: float, peak_equity: float) -> tuple[bool, str]:  
        # 更新状态  
        self.state.current_return = (current_equity - initial_equity) / initial_equity  
        self.state.drawdown = (peak_equity - current_equity) / peak_equity  
        
        # 更新风险等级  
        if self.state.current_return < 0:  
            self.state.risk_level = "HIGH"  
        elif self.state.drawdown > self.trailing_stop_pct / 2:  
            self.state.risk_level = "MEDIUM"  
        else:  
            self.state.risk_level = "LOW"  

        # 检查各种平仓条件  
        if current_equity <= initial_equity * (1 - self.total_stop_loss_pct):  
            self.state.close_reason = "触发总止损"  
            return True, self.state.close_reason  

        if self.state.current_return >= self.total_profit_target:  
            self.state.close_reason = "达到目标收益"  
            return True, self.state.close_reason  

        if self.state.current_return >= self.trailing_stop_activation:  
            if self.state.drawdown >= self.trailing_stop_pct:  
                self.state.close_reason = "触发移动止损"  
                return True, self.state.close_reason  

        self.state.close_reason = ""  
        return False, ""
    

class LeveragedMockTrader:  
    """  
    杠杆交易模拟器  
    整合保证金管理和风险管理功能  
    """  
    def __init__(self, initial_equity=10000.0, fee_rate=0.0004,  
                 take_profit_pct=0.10, stop_loss_pct=0.30,  
                 leverage=50, maintenance_margin=0.005,  
                 total_profit_target=0.30,  
                 trailing_stop_activation=0.10,  
                 trailing_stop_pct=0.03,  
                 total_stop_loss_pct=0.30):  
        
        self._validate_init_params(locals())  
        
        self.margin_manager = MarginManager(  
            initial_equity=initial_equity,  
            leverage=leverage,  
            maintenance_margin=maintenance_margin,  
            fee_rate=fee_rate  
        )  
        
        self.risk_manager = RiskManager(  
            take_profit_pct=take_profit_pct,  
            stop_loss_pct=stop_loss_pct,  
            total_profit_target=total_profit_target,  
            trailing_stop_activation=trailing_stop_activation,  
            trailing_stop_pct=trailing_stop_pct,  
            total_stop_loss_pct=total_stop_loss_pct  
        )  
        
        self.positions = {  
            "LONG": Position(),  
            "SHORT": Position()  
        }  

    def on_price_tick(self, newest_price: float, newest_tradable_volume: float) -> None:  
        """处理价格更新"""  
        # 如果没有活跃仓位，直接返回  
        if not self._has_active_positions():  
            return  
            
        # 1.更新所有活跃仓位的状态  
        self._update_positions_state(newest_price)  
        
        # 2.计算当前权益并检查是否需要全部平仓  
        current_equity = self._calculate_total_equity(newest_price)  
        should_close, reason = self.risk_manager.should_close_all(  
            current_equity,  
            self.margin_manager.initial_equity,  
            self.margin_manager.peak_equity  
        )  
        
        if should_close:  
            self._close_all_positions(newest_price, reason)  
            return  
        
        # 3. 检查个别仓位的止盈止损  
        closed_positions = []  
        for side, position in self.positions.items():  
            if position.is_active():  
                if self._should_close_position(side, position, newest_price):  
                    self._close_position(side, newest_price)  
                    closed_positions.append(side)  
        
        # 清理已平仓的仓位状态  
        for side in closed_positions:  
            self.positions[side] = Position.create_empty()  
        
        # 4. 更新峰值权益（只在有活跃仓位时更新）  
        if self._has_active_positions() and current_equity > self.margin_manager.peak_equity:  
            self.margin_manager.peak_equity = current_equity  
        
        # 5. 检查保证金要求（只在有活跃仓位时检查）  
        if self._has_active_positions():  
            if not self.margin_manager.check_margin_requirement(self.positions, newest_price):  
                self._close_all_positions(newest_price, "触发强制平仓")  

    def _update_positions_state(self, current_price: float) -> None:  
        """更新所有活跃仓位的状态"""  
        for side, position in self.positions.items():  
            if position.is_active():  
                position.update_state(current_price, self.margin_manager.leverage)  

    def _has_active_positions(self) -> bool:  
        """检查是否有活跃仓位"""  
        return any(pos.is_active() for pos in self.positions.values())  

    def on_signal(self, signal: Literal["BUY", "SELL"], strength: float, last_price: float) -> None:  
        """  
        处理交易信号  
        
        Args:  
            signal: 交易信号方向 ("BUY" or "SELL")  
            # strength: 信号强度 (0-100)  
            last_price: 当前价格  
        """  
        # 1. 基础参数验证  
        assert signal in ["BUY", "SELL"], f"无效的交易信号: {signal}"  
        assert last_price > 0, f"无效的价格: {last_price}"  
        
        # 2. 信号强度验证和映射  
        assert isinstance(strength, (int, float)), f"信号强度类型错误: {type(strength)}"  
        # assert 0 <= strength <= 100, f"无效的信号强度: {strength}"  
        
        # 计算映射后的信号强度  
        mapped_strength = 0.1 + 0.4 / (1 + math.exp(-strength / 100))  
        assert 0.1 <= mapped_strength <= 0.5, f"信号强度映射异常: {mapped_strength}"  
        
        # 记录信号相关信息  
        self.last_signal = {  
            "original_strength": strength,  
            "mapped_strength": mapped_strength,  
            "signal": signal,  
            "price": last_price,  
            "timestamp": datetime.now()  
        }  
        
        # 3. 处理交易逻辑  
        position_side = "LONG" if signal == "BUY" else "SHORT"  
        position = self.positions[position_side]  
        
        if position.is_active():  
            # 已有仓位，检查止盈止损  
            if self._should_close_position(position_side, position, last_price):  
                self._close_position(position_side, last_price)  
        else:  
            # 开新仓位，传递映射后的强度值  
            self._open_position(position_side, mapped_strength, last_price)  
        
        # 4. 打印状态  
        self._print_signal_info()  
        self._print_status(last_price)  

    def _print_signal_info(self):  
        """打印信号相关信息"""  
        if hasattr(self, 'last_signal'):  
            print("\n=== 信号信息 ===")  
            print(f"方向: {self.last_signal['signal']}")  
            print(f"原始强度: {self.last_signal['original_strength']:.2f}")  
            print(f"映射强度: {self.last_signal['mapped_strength']:.4f}")  
            print(f"价格: {self.last_signal['price']:.2f}") 

    def _validate_init_params(self, params: dict) -> None:  
        """验证初始化参数"""  
        validations = {  
            'initial_equity': lambda x: x > 0,  
            'fee_rate': lambda x: 0 < x < 1,  
            'take_profit_pct': lambda x: 0 < x < 1,  
            'stop_loss_pct': lambda x: 0 < x < 1,  
            'leverage': lambda x: x > 0,  
            'maintenance_margin': lambda x: 0 < x < 1,  
            'total_profit_target': lambda x: 0 < x < 1,  
            'trailing_stop_pct': lambda x: 0 < x < params['trailing_stop_activation'] < 1,  
            'total_stop_loss_pct': lambda x: 0 < x < 1  
        }  
        
        for param_name, validator in validations.items():  
            value = params[param_name]  
            assert validator(value), f"参数 {param_name} 验证失败: {value}"  

    def _calculate_total_equity(self, current_price: float) -> float:  
        """计算当前总权益"""  
        unrealized_pnl = self.margin_manager._calculate_total_pnl(self.positions, current_price)  
        return self.margin_manager.current_margin + unrealized_pnl  

    def _should_close_position(self, side: str, position: Position, current_price: float) -> bool:  
        """检查是否应该平掉指定仓位"""  
        if side == "LONG":  
            return current_price >= position.tp_price or current_price <= position.sl_price  
        else:  
            return current_price <= position.tp_price or current_price >= position.sl_price  

    def _open_position(self, side: str, strength: float, price: float) -> None:  
        """开仓操作"""  
        
        # 1. 计算仓位大小  
        assert 0.0 < strength <= 1.0, f'strength not in range(0,1]'
        max_contract_value = self.margin_manager.current_margin * self.margin_manager.leverage  
        contract_value = max_contract_value * strength  
        position_size = contract_value / price  

        # 2. 计算保证金相关数据  
        initial_margin = contract_value / self.margin_manager.leverage  # 初始保证金  
        maintenance_margin = contract_value * self.margin_manager.maintenance_margin  # 维持保证金  
        fee = self.margin_manager.calculate_fee(contract_value)  # 手续费  
        effective_leverage = round(contract_value / initial_margin, 1)  # 实际使用的杠杆  

        # 3. 计算止盈止损价格  
        tp_price, sl_price = self.risk_manager.calculate_tp_sl_prices(price, side == "LONG")  

        # 4. 验证保证金是否足够  
        if initial_margin + fee > self.margin_manager.current_margin:  
            print(f"保证金不足，需要 {initial_margin + fee:.2f}，当前可用 {self.margin_manager.current_margin:.2f}")  
            return  

        # 添加风险检查  
        risk_checks = {  
            "保证金充足": initial_margin + fee <= self.margin_manager.current_margin,  
            "杠杆合规": effective_leverage <= self.margin_manager.leverage,  
            "仓位规模合理": 0.0 < strength <= 1.0,  
            "价格有效": price > 0  
        }  
        
        if not all(risk_checks.values()):  
            print("\n=== 风险检查失败 ===")  
            for check, result in risk_checks.items():  
                if not result:  
                    print(f"❌ {check}")  
            return  
        
        # 5. 创建新仓位  
        self.positions[side] = Position(  
            entry_price=price,  
            position_size=position_size,  
            tp_price=tp_price,  
            sl_price=sl_price,  
            opened_at=datetime.now()  
        )  

        # 6. 扣除手续费  
        self.margin_manager.current_margin -= fee  

        # 7. 打印详细信息  
        liquidation_price = self._calculate_liquidation_price(side, price, position_size, maintenance_margin)  
        print("\n=== 开仓详情 ===")  
        print(f"方向: {side}")  
        print(f"价格: {price:.4f}")  
        print(f"数量: {position_size:.4f}")  
        
        print("\n--- 合约信息 ---")  
        print(f"合约价值: {contract_value:.2f} USDT")  
        print(f"实际杠杆: {effective_leverage:.2f}x")  
        print(f"占用资金比例: {(initial_margin/self.margin_manager.current_margin*100):.1f}%")  
        
        print("\n--- 保证金信息 ---")  
        print(f"初始保证金: {initial_margin:.2f} USDT")  
        print(f"维持保证金: {maintenance_margin:.2f} USDT")  
        print(f"手续费: {fee:.2f} USDT")  
        print(f"剩余保证金: {self.margin_manager.current_margin:.2f} USDT")  
        print(f"可用保证金: {(self.margin_manager.current_margin - initial_margin):.2f} USDT")  
        
        print("\n--- 风险信息 ---")  
        print(f"止盈价格: {tp_price:.4f} ({abs((tp_price-price)/price*100):.1f}%)")  
        print(f"止损价格: {sl_price:.4f} ({abs((sl_price-price)/price*100):.1f}%)")  
        print(f"预估强平价格: {liquidation_price:.4f} ({abs((liquidation_price-price)/price*100):.1f}%)")  

    def _calculate_liquidation_price(self, side: str, entry_price: float,   
                                position_size: float, maintenance_margin: float) -> float:  
        """计算预估强平价格"""  
        # 计算从当前保证金到维持保证金的价格变动  
        margin_diff = self.margin_manager.current_margin - maintenance_margin  
        price_change = margin_diff / position_size  
        
        # 多仓向下强平，空仓向上强平  
        if side == "LONG":  
            return entry_price - price_change  
        else:  
            return entry_price + price_change 
 
    def _close_position(self, side: str, price: float) -> None:  
        """平仓操作"""  
        position = self.positions[side]  
        if not position.is_active():  
            return  
            
        # 计算平仓收益  
        multiplier = 1 if side == "LONG" else -1  
        pnl = (price - position.entry_price) * position.position_size * multiplier  
        
        # 计算并扣除手续费  
        contract_value = position.position_size * price  
        fee = self.margin_manager.calculate_fee(contract_value)  
        
        # 更新保证金  
        self.margin_manager.current_margin += pnl - fee  
        
        # 重置仓位状态  
        self.positions[side] = Position.create_empty()  

        # 打印平仓信息  
        print(f"\n=== 平仓信息 ===")  
        print(f"方向: {side}")  
        print(f"平仓价格: {price:.4f}")  
        print(f"开仓价格: {position.entry_price:.4f}")  
        print(f"仓位大小: {position.position_size:.4f}")  
        print(f"收益: {pnl:.2f}")  
        print(f"手续费: {fee:.2f}")  
        print(f"净收益: {(pnl - fee):.2f}")  

    def _calculate_total_equity(self, current_price: float) -> float:  
        """计算当前总权益"""  
        total_pnl = 0.0  
        for side, position in self.positions.items():  
            if position.is_active():  
                multiplier = 1 if side == "LONG" else -1  
                total_pnl += (current_price - position.entry_price) * position.position_size * multiplier 
        return self.margin_manager.current_margin + total_pnl  
    
    def _close_all_positions(self, price: float, reason: str) -> None:  
        """平掉所有仓位"""  
        print(f"\n=== 全部平仓 ({reason}) ===")  
        
        # 记录平仓前的状态  
        pre_close_equity = self._calculate_total_equity(price)  #for print

        for side, position in self.positions.items():  
            if position.is_active():  
                self._close_position(side, price)  
        
        # 确保所有仓位状态都被重置  
        for side in self.positions:  
            self.positions[side] = Position.create_empty()  
        
        # 重置峰值权益为当前权益  
        current_equity = self._calculate_total_equity(price)  
        self.margin_manager.reset_peak_equity(current_equity)  
        
        print(f"平仓前权益: {pre_close_equity:.2f}")  
        print(f"平仓后权益: {current_equity:.2f}")  
        print(f"峰值权益已重置为: {current_equity:.2f}")  

    def _print_status(self, current_price: float):  
        """打印当前状态"""  
        # 更新所有仓位状态  
        for side, pos in self.positions.items():  
            if pos.is_active():  
                pos.update_state(current_price, self.margin_manager.leverage)  
        
        # 计算总权益  
        total_pnl = sum(pos.current_pnl for pos in self.positions.values() if pos.is_active())  
        total_equity = self.margin_manager.current_margin + total_pnl  
        
        # 计算收益率  
        profit_pct = (total_equity - self.margin_manager.initial_equity) / self.margin_manager.initial_equity * 100  

        print("\n当前状态:")  
        print(f"保证金: {self.margin_manager.current_margin:.2f}")  
        print(f"总权益: {total_equity:.2f}")  
        print(f"收益率: {profit_pct:.2f}%")  
        
        # 打印活跃仓位信息  
        for side, pos in self.positions.items():  
            if pos.is_active():  
                print(pos.get_status_description(side)) 