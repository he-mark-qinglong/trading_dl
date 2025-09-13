import os  
import time  
import threading  
import pandas as pd  
import numpy as np  

# 全局数据共享（资金曲线、交易记录、日志）  
global_data = {  
    "equity": pd.DataFrame(columns=["time", "equity"]),  
    "trades": [],  
    "logs": []  
}  
data_lock = threading.Lock()  

def record_log(dt, message):  
    with data_lock:  
        global_data["logs"].append({"time": dt, "message": message})  
    print(f'{dt} {message}')  

def update_equity(dt, equity):  
    dt = pd.to_datetime(dt)  
    new_row = pd.DataFrame({"time": [dt], "equity": [equity]})  
    with data_lock:  
        global_data["equity"] = pd.concat([global_data["equity"], new_row], ignore_index=True)  

def update_trade(trade_dict):  
    with data_lock:  
        global_data["trades"].append(trade_dict)  
    print(f'trade: {trade_dict}')  

def calculate_trade_pnl(trade, price, fee_rate, leverage):  
    trade_margin = trade["margin"]  
    fee_trade = trade_margin * fee_rate * leverage  
    if trade["side"] == "long":  
        pnl = trade_margin * leverage * ((price / trade["entry"]) - 1) - fee_trade  
    else:  
        pnl = trade_margin * leverage * (1 - (price / trade["entry"])) - fee_trade  
    return pnl  

def exit_order(trade, price, fee_rate, leverage, dt, action_label):  
    """独立平仓单个订单"""  
    pnl = calculate_trade_pnl(trade, price, fee_rate, leverage)  
    update_trade({  
        "time": dt,  
        "action": action_label,  
        "side": trade["side"],  
        "entry": trade["entry"],  
        "price": price,  
        "margin": trade["margin"],  
        "pnl": pnl,  
        "stop_loss": trade.get("stop_loss", None)  
    })  
    return pnl  

def backtest_scaling_strategy_with_capital(  
    df: pd.DataFrame,  
    initial_capital=10000,  
    tolerance=0.004,  
    leverage=10,  
    fee_rate=0.0005,  
    slippage=0.01,  
    adx_entry_threshold_ratio=0.1,  
    noadx_entry_threshold_ratio=4.3,  
    exit_threshold_ratio=4.30,  
    escape_threshold_ratio=150,  
    open_reciprocal=10,  
    max_holding_ratio=0.1  
):  
    trades = []  # 持仓列表，每个订单为字典格式，包含 "time", "side", "entry", "atr", "margin" 以及后续的 "stop_loss"  
    capital = initial_capital  
    equity_trace = []  
    pending_signal = None  

    # 当仓位首次建立时记录本次开仓之前的本金  
    position_base = None  

    df["prev_close"] = df["close"].shift(1)  

    for dt, row in df.iterrows():  
        price = row["close"]  
        current_macd = row.get("macd", None)  
        current_atr = row.get("avg_atr", None)  # 用于进场信号时确定仓位参数  
        signal = row.get("entry_signal", None)  

        # 计算整体持仓的浮盈（全仓加总）  
        floating_pnl = sum(calculate_trade_pnl(trade, price, fee_rate, leverage) for trade in trades)  
        equity = capital + floating_pnl  
        used_margin = sum(t["margin"] for t in trades)  
        available_capital = equity - used_margin  
        equity_trace.append(equity)  
        update_equity(dt, equity)  

        # 如果有新仓位建立，记录本次建仓前本金（position_base）  
        if trades and position_base is None:  
            # 记录开仓前本金（这里取当前资金，也可取 equity）  
            position_base = capital  

        # 逐个订单管理——独立的止盈止损逻辑（移动止盈跟单）  
        orders_to_exit = []  
        for trade in trades.copy():  
            # 对每个订单，若未设置止损，则初始止损等于开仓价  
            if "stop_loss" not in trade:  
                trade["stop_loss"] = trade["entry"]  
            if trade["side"] == "long":  
                # 当盈利达到 4.3×入场ATR时，将止损调整到 入场价 + 1×ATR  
                if price >= trade["entry"] + 6 * trade["atr"]:  
                    new_sl = trade["entry"] + 1.7 * trade["atr"]  
                    if new_sl > trade["stop_loss"]:  
                        trade["stop_loss"] = new_sl  
                        record_log(dt, f"多单调整止损: 入场价 {trade['entry']:.2f}, 当前价 {price:.2f}, 新止损 {new_sl:.2f}")  
                # 若当前价格下穿止损，则触发平仓该订单  
                if price <= trade["stop_loss"]:  
                    orders_to_exit.append(trade)  
            elif trade["side"] == "short":  
                # 对空单：如果盈利达到 4.3×ATR（即入场价 - price >= 4.3×atr），将止损调整到 入场价 - 1×ATR  
                if price <= trade["entry"] - 6 * trade["atr"]:  
                    new_sl = trade["entry"] - 1.7 * trade["atr"]  
                    if new_sl < trade["stop_loss"]:  
                        trade["stop_loss"] = new_sl  
                        record_log(dt, f"空单调整止损: 入场价 {trade['entry']:.2f}, 当前价 {price:.2f}, 新止损 {new_sl:.2f}")  
                # 若当前价格上穿止损，则触发平仓该订单  
                if price >= trade["stop_loss"]:  
                    orders_to_exit.append(trade)  
        # 个别订单止盈/止损平仓处理  
        for trade in orders_to_exit:  
            pnl_change = exit_order(trade, price, fee_rate, leverage, dt, "个别止盈止损平仓")  
            capital += pnl_change  
            trades.remove(trade)  
            record_log(dt, f"订单平仓: 方向 {trade['side']}, 入场 {trade['entry']:.2f}, 止损 {trade.get('stop_loss'):.2f}, 当前价 {price:.2f}, 盈亏 {pnl_change:.2f}")  
            # 如果全部订单平仓，则重置仓位基准  
            if len(trades) == 0:  
                position_base = None  

        # --------------------- 进场逻辑（示例） ---------------------  
        # 该部分保持原有逻辑，示例中仅在有 entry_signal 时开仓  
        # 假设信号 "long_entry" 和 "short_entry" 分别代表多单和空单进场的信号  
        if signal in ["long_entry", "short_entry"]:  
            new_side = "long" if signal == "long_entry" else "short"  
            # 计算仓位分配：此处简单使用 available_capital / open_reciprocal 进行开仓额度限制  
            base_trade_margin = equity / open_reciprocal  
            if available_capital < base_trade_margin:  
                record_log(dt, f"进场资金不足: 可用 {available_capital:.2f}, 需要 {base_trade_margin:.2f}")  
            else:  
                trade_margin = base_trade_margin  # 可根据实际策略调整  
                # 此处用到当前 ATR 作为入场时的 atr 参数（存入订单中，用于后续止盈止损判断）  
                order = {  
                    "time": dt,  
                    "side": new_side,  
                    "entry": price,  
                    "atr": current_atr if current_atr is not None else 0,  
                    "margin": trade_margin  
                }  
                trades.append(order)  
                update_trade({  
                    "time": dt,  
                    "action": "开仓",  
                    "side": new_side,  
                    "entry": price,  
                    "margin": trade_margin,  
                    "atr": order["atr"]  
                })  
                record_log(dt, f"进场开仓: {new_side}, 当前价 {price:.2f}, 开仓额度 {trade_margin:.2f}")  
        # -------------------------------------------------------------  

        # 此处可以添加其他风险控制逻辑（例如整体止损、部分平仓等），目前主要关注单订单独立止盈止损。  

    # 最后平仓处理（强制平掉剩余持仓）  
    final_price = df["close"].iloc[-1]  
    remaining_pnl = sum(calculate_trade_pnl(t, final_price, fee_rate, leverage) for t in trades)  
    final_capital = capital + remaining_pnl  

    # 计算最大权益回撤  
    peak_equity = -np.inf  
    max_drawdown_equity_pct = 0  
    for eq in equity_trace:  
        if eq > peak_equity:  
            peak_equity = eq  
        else:  
            dd = (peak_equity - eq) / peak_equity * 100  
            if dd > max_drawdown_equity_pct:  
                max_drawdown_equity_pct = dd  

    report = {  
        "initial_capital": initial_capital,  
        "final_capital": final_capital,  
        "open_trades": len(trades),  
        "total_pnl": final_capital - initial_capital,  
        "max_drawdown_equity_pct": max_drawdown_equity_pct,  
        "final_price": final_price,  
        "leverage": leverage,  
        "trades": trades  
    }  
    return report  

def run_backtest(length=1000):  
    data_file = "btc_5m_avwap_result.csv"  
    if not os.path.exists(data_file):  
        record_log(pd.Timestamp.now(), f"数据文件未找到: {data_file}")  
        return  
    df = pd.read_csv(data_file, parse_dates=True, index_col=0)  
    report = backtest_scaling_strategy_with_capital(df[-length:].copy())  
    record_log(pd.Timestamp.now(), f"回测结果: {report}")  

if __name__ == "__main__":  
    run_backtest()  
