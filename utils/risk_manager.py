class RiskManager:  
    def __init__(self, account_balance: float, max_risk_per_trade: float = 0.01):  
        """  
        初始化风险管理器  
        :param account_balance: 当前账户余额  
        :param max_risk_per_trade: 单笔交易的最大风险比例（如 0.01 表示 1%）  
        """  
        self.account_balance = account_balance  
        self.max_risk_per_trade = max_risk_per_trade  

    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> float:  
        """  
        根据账户余额和风险限额计算仓位大小  
        :param entry_price: 开仓价格  
        :param stop_loss_price: 止损价格  
        :return: 仓位大小  
        """  
        # 每笔交易的最大风险金额  
        max_risk_amount = self.account_balance * self.max_risk_per_trade  

        # 每单位的风险（开仓价格与止损价格的差值）  
        risk_per_unit = abs(entry_price - stop_loss_price)  

        # 如果风险为 0，返回 0（避免除以 0 的错误）  
        if risk_per_unit == 0:  
            return 0  

        # 计算仓位大小  
        position_size = max_risk_amount / risk_per_unit  
        return position_size  

    def calculate_stop_loss(self, entry_price: float, risk_tolerance: float) -> float:  
        """  
        根据风险容忍度计算止损价格  
        :param entry_price: 开仓价格  
        :param risk_tolerance: 风险容忍度（如 0.02 表示 2%）  
        :return: 止损价格  
        """  
        stop_loss_price = entry_price * (1 - risk_tolerance)  
        return stop_loss_price  

    def calculate_take_profit(self, entry_price: float, reward_risk_ratio: float) -> float:  
        """  
        根据目标收益风险比计算止盈价格  
        :param entry_price: 开仓价格  
        :param reward_risk_ratio: 收益风险比（如 2 表示 2:1）  
        :return: 止盈价格  
        """  
        stop_loss_price = self.calculate_stop_loss(entry_price, self.max_risk_per_trade)  
        risk_per_unit = abs(entry_price - stop_loss_price)  
        take_profit_price = entry_price + (risk_per_unit * reward_risk_ratio)  
        return take_profit_price