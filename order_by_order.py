
import os  
import time  
import threading  
import pandas as pd  
import numpy as np  

# 全局数据共享（资金曲线、交易记录、日志）  
global_data = {  
    "equity": pd.DataFrame(columns=["time", "equity"]),   # 使用 DataFrame 记录资金曲线  
    "trades": [],   # 每笔交易记录 {"time", "action", "entry", "price", "margin", "pnl"}  
    "logs": []      # 日志记录 {"time", "message"}  
}  
# 线程锁，确保多线程读写安全  
data_lock = threading.Lock()  

def record_log(dt, message):  
    with data_lock:  
        global_data["logs"].append({"time": dt, "message": message})  
    print(f'{dt} {message}')  

def update_equity(dt, equity):  
    # 确保 dt 为 pandas.Timestamp 类型  
    dt = pd.to_datetime(dt)  
    new_row = pd.DataFrame({"time": [dt], "equity": [equity]})  
    with data_lock:  
        global_data["equity"] = pd.concat([global_data["equity"], new_row], ignore_index=True)  

def update_trade(trade_dict):  
    with data_lock:  
        global_data["trades"].append(trade_dict)  
    print(f'trade:{trade_dict}')  


# --------------------- 提取平仓逻辑 --------------------- #  
def calculate_trade_pnl(trade, price, fee_rate, leverage):  
    """  
    计算单笔交易的盈亏。  
    参数:  
      trade：字典，必须包含 "entry"，"margin"，"side"（"long" 或 "short"）  
      price：当前价格  
      fee_rate, leverage: 费用和杠杆  
    返回：该交易的盈亏 pnl（扣除手续费）  
    """  
    trade_margin = trade["margin"]  
    fee_trade = trade_margin * fee_rate * leverage  
    if trade["side"] == "long":  
        pnl = trade_margin * leverage * ((price / trade["entry"]) - 1) - fee_trade  
    else:  
        pnl = trade_margin * leverage * (1 - (price / trade["entry"])) - fee_trade  
    return pnl  


def exit_trades(trades, price, fee_rate, leverage, dt, action_label, partial=False, partial_ratio=0.5):  
    """  
    对传入的仓位列表执行平仓操作。  
    参数:  
      trades: 仓位列表（列表中每一项均为交易字典）  
      price: 平仓价格  
      fee_rate, leverage：费用及杠杆  
      dt: 当前时间，用于记录日志  
      action_label: 平仓操作标签，例如 "止损全平(0.8%)" 或 "部分止盈平仓(6%)"  
      partial: 是否做部分平仓操作。False 表示全平，True 表示平部分仓 (按 partial_ratio)  
      partial_ratio: 部分平仓比例，缺省 0.5 表示平一半仓位；如果全平，则应传入1  
    返回: 累计平仓盈亏  
    平仓后，对应仓位在 trades 中会相应减少，若平掉全部则会从列表中移除。  
    """  
    total_exit_pnl = 0  
    # 复制一份原有仓位列表，平仓时统一修改 trades 列表  
    trades_to_process = trades.copy()  
    for trade in trades_to_process:  
        ratio = partial_ratio if partial else 1  # 全平时 ratio 为1  
        # 计算待平仓的仓位金额  
        exit_margin = trade["margin"] * ratio  
        # 计算当前平仓盈亏（仅针对待平部分）  
        fee_trade = exit_margin * fee_rate * leverage  
        if trade["side"] == "long":  
            pnl = exit_margin * leverage * ((price / trade["entry"]) - 1) - fee_trade  
        else:  
            pnl = exit_margin * leverage * (1 - (price / trade["entry"])) - fee_trade  
        total_exit_pnl += pnl  
        update_trade({  
            "time": dt,  
            "action": action_label,  
            "side": trade["side"],  
            "entry": trade["entry"],  
            "price": price,  
            "exit_margin": exit_margin,  
            "pnl": pnl  
        })  
        # 调整仓位：如果是部分平仓则减少 margin，否则清除该仓位  
        trade["margin"] -= exit_margin  
        if trade["margin"] <= 1e-8:  
            trades.remove(trade)  
    return total_exit_pnl  

##############################################  
# 回测策略函数（含平仓出场逻辑示例）  
##############################################  
def backtest_scaling_strategy_with_capital(  
    df: pd.DataFrame,  
    initial_capital=10000,  
    tolerance=0.004,  
    leverage=10,  
    fee_rate=0.0005,    # 单边手续费  
    slippage=0.01,      # 滑点（单位USDT）  
    adx_entry_threshold_ratio = 0.1,  # 进场门槛倍率（ATR）  
    noadx_entry_threshold_ratio = 3,
    exit_threshold_ratio=30,   # 止盈门槛倍率（ATR）  
    escape_threshold_ratio=150,  # 逃离门槛倍率（ATR）  
    open_reciprocal=1000,       # 开仓资金参数  %
    max_holding_ratio = 0.1, #最大持仓水平
):  
    """  
    策略说明：  
      1. 允许持续进场与加仓，不超过风险上限。  
      2. 当价格达到止盈水平或收到 exit 信号时全平对应持仓。  
      3. MACD过滤、Escape Logic、以及部分平仓均内置。  

    遍历每行数据时，会调用 update_equity()/record_log()/update_trade()  
    实时更新全局数据供 Dash 展示。  
    """  
    
    trades = []     # 当前持仓记录，每项为 {"time", "side", "entry", "atr", "margin"}  
    capital = initial_capital  
    equity_trace = []  
    pending_signal = None  

    # 新增变量：记录本次开仓前的本金。当首次开仓时记录，此后不再更新，  
    # 用于判断持仓利润是相对于此时本金的多少  
    position_base = None  
    max_floating_pnl = 0

    # 计算上一个收盘价  
    df["prev_close"] = df["close"].shift(1)  
    
    for dt, row in df.iterrows():  
        price = row["close"]  
        current_macd = row.get("macd", None)  
        current_atr = row.get("avg_atr", None)  # 使用平滑ATR  
        signal = row.get("entry_signal", None)  
        
        # 计算当前持仓的浮动盈亏  
        floating_pnl = 0  
        for trade in trades:  
            trade_margin = trade["margin"]  
            fee_trade = trade_margin * fee_rate * leverage  
            if trade["side"] == "long":  
                pnl = trade_margin * leverage * ((price / trade["entry"]) - 1) - fee_trade  
            else:  
                pnl = trade_margin * leverage * (1 - (price / trade["entry"])) - fee_trade  
            floating_pnl += pnl  

        equity = capital + floating_pnl  
        used_margin = sum(t["margin"] for t in trades)  
        available_capital = equity - used_margin  
        equity_trace.append(equity)  

        # 先更新资金曲线数据，现改为用 DataFrame 记录  
        update_equity(dt, equity)  
        
        # 计算当前持仓的浮动盈亏（floating_pnl），并更新 equity、used_margin 等数据  
        floating_pnl = 0  
        for trade in trades:  
            trade_margin = trade["margin"]  
            fee_trade = trade_margin * fee_rate * leverage  
            if trade["side"] == "long":  
                pnl = trade_margin * leverage * ((price / trade["entry"]) - 1) - fee_trade  
            else:  
                pnl = trade_margin * leverage * (1 - (price / trade["entry"])) - fee_trade  
            floating_pnl += pnl  

        equity = capital + floating_pnl  
        used_margin = sum(t["margin"] for t in trades)  
        available_capital = equity - used_margin  

        update_equity(dt, equity)  

        # === 止损逻辑：若浮动亏损达到当前权益的0.8%，则全平仓位 ===  
        max_loss_percent = 0.008
        if trades and floating_pnl <= -equity * max_loss_percent:  
            msg = f"{dt} 止损条件触发：浮亏 {floating_pnl:.2f} 达到权益{max_loss_percent*100}% ({-equity*max_loss_percent:.2f})，全平仓位"  
            record_log(dt, msg)  
            total_exit_pnl = 0  
            for t in trades:  
                fee_trade = t["margin"] * fee_rate * leverage  
                if t["side"] == "long":  
                    pnl = t["margin"] * leverage * ((price / t["entry"]) - 1) - fee_trade  
                else:  
                    pnl = t["margin"] * leverage * (1 - (price / t["entry"])) - fee_trade  
                total_exit_pnl += pnl  
                update_trade({  
                    "time": dt,  
                    "action": f"止损全平({max_loss_percent*100}%)",  
                    "side": t["side"],  
                    "entry": t["entry"],  
                    "price": price,  
                    "margin": t["margin"],  
                    "pnl": pnl  
                })  
            capital += total_exit_pnl  
            trades.clear()  
            position_base = None   # 全平后重置本次开仓前本金  
            max_floating_pnl = 0  # 重置  
            continue  

        # 计算当前持仓的浮动盈亏（floating_pnl），并更新 equity、used_margin 等数据  
        floating_pnl = 0  
        for trade in trades:  
            trade_margin = trade["margin"]  
            fee_trade = trade_margin * fee_rate * leverage  
            if trade["side"] == "long":  
                pnl = trade_margin * leverage * ((price / trade["entry"]) - 1) - fee_trade  
            else:  
                pnl = trade_margin * leverage * (1 - (price / trade["entry"])) - fee_trade  
            floating_pnl += pnl  

        equity = capital + floating_pnl  
        used_margin = sum(t["margin"] for t in trades)  
        available_capital = equity - used_margin  

        update_equity(dt, equity)  

        # 当持仓刚刚建立时（即 trades 非空且 position_base 未设置），记录 position_base  
        if trades and (position_base is None):  
            position_base = capital  # 或 equity，此处取 capital作为开仓前本金  

        # 更新历史最大浮盈  
        if floating_pnl > max_floating_pnl:  
            max_floating_pnl = floating_pnl  

        # ------------------ 新增移动止盈逻辑 ------------------  
        # 只有当持仓已产生利润达到本次开仓前本金的 0.4%时（即 max_floating_pnl >= position_base*0.004），  
        # 才判断当浮盈回撤超过最大浮盈50%时触发全平。  
        if trades and (position_base is not None) and (max_floating_pnl >= position_base * 0.004) and (floating_pnl < 0.5 * max_floating_pnl):  
            msg = (f"{dt} 移动止盈触发: 本次开仓前本金 {position_base:.2f}，最大浮盈 {max_floating_pnl:.2f}, 当前浮盈 {floating_pnl:.2f}，"  
                   f"回撤超过50%，全平仓")  
            record_log(dt, msg)  
            pnl_change = exit_trades(trades, price, fee_rate, leverage, dt, "浮盈回撤全平", partial=False)  
            capital += pnl_change  
            trades.clear()  
            # 清仓后重置 position_base 与 max_floating_pnl  
            position_base = None   # 全平后重置本次开仓前本金  
            max_floating_pnl = 0  
            continue  
        # -------------------------------------------------------  

        # 止损逻辑：若浮亏达到权益的 0.8% 时全平  
        max_loss_percent = 0.008  
        if trades and floating_pnl <= -equity * max_loss_percent:  
            msg = f"{dt} 止损触发：浮亏 {floating_pnl:.2f} 达到权益{max_loss_percent*100}%，全平仓位"  
            record_log(dt, msg)  
            pnl_change = exit_trades(trades, price, fee_rate, leverage, dt, f"止损全平({max_loss_percent*100}%)", partial=False)  
            capital += pnl_change  
            trades.clear()  
            position_base = None   # 全平后重置本次开仓前本金  
            max_floating_pnl = 0    # 重置最大浮盈  
            continue  
        # =====================================================================  


        # === 止损逻辑：若浮动亏损达到当前权益的0.2%，则全平仓位 ===  
        # 止损逻辑通常不考虑趋势方向，目的是防止亏损扩大。  
        if trades and floating_pnl <= -equity * 0.002:  
            msg = f"{dt} 止损条件触发：浮亏 {floating_pnl:.2f} 达到权益0.2% ({-equity*0.002:.2f})，全平仓位"  
            record_log(dt, msg)  
            total_exit_pnl = 0  
            for t in trades:  
                fee_trade = t["margin"] * fee_rate * leverage  
                if t["side"] == "long":  
                    pnl = t["margin"] * leverage * ((price / t["entry"]) - 1) - fee_trade  
                else:  
                    pnl = t["margin"] * leverage * (1 - (price / t["entry"])) - fee_trade  
                total_exit_pnl += pnl  
                update_trade({  
                    "time": dt,  
                    "action": "止损全平(0.2%)",  
                    "side": t["side"],  
                    "entry": t["entry"],  
                    "price": price,  
                    "margin": t["margin"],  
                    "pnl": pnl  
                })  
            capital += total_exit_pnl  
            trades.clear()  
            position_base = None   # 全平后重置本次开仓前本金  
            max_floating_pnl = 0  # 重置  
            continue  

        # === 止盈逻辑：当浮动盈亏达到当前权益的0.6%时触发 ===  
        if trades and floating_pnl >= equity * 0.06:  
            # 判断市场趋势是否有利，从而决定止盈平仓比例  
            favorable = False  
            long_trades = [t for t in trades if t["side"] == "long"]  
            short_trades = [t for t in trades if t["side"] == "short"]  

            if long_trades and (current_macd > 0 and row["adx"] > 25) and row['+di'] > row['-di']:  
                favorable = True  
            if short_trades and (current_macd < 0 and row["adx"] > 25) and row['+di'] < row['-di']:  
                favorable = True  

            if favorable:  
                msg = (f"{dt} 部分止盈条件触发：趋势有利（MACD {current_macd:.2f}, ADX {row['adx']:.2f}, "  
                       f"DI+ {row['+di']:.2f} vs DI- {row['-di']:.2f}），浮盈 {floating_pnl:.2f} 达到权益6% "  
                       f"({equity*0.06:.2f})，平半仓以保本")  
                record_log(dt, msg)  
                total_exit_pnl = 0  
                for trade in trades:  
                    # 平仓一半仓位  
                    reduce_margin = trade["margin"] / 2  
                    fee_trade = reduce_margin * fee_rate * leverage  
                    if trade["side"] == "long":  
                        pnl = reduce_margin * leverage * ((price / trade["entry"]) - 1) - fee_trade  
                    else:  
                        pnl = reduce_margin * leverage * (1 - (price / trade["entry"])) - fee_trade  
                    total_exit_pnl += pnl  
                    update_trade({  
                        "time": dt,  
                        "action": "部分止盈平仓(6%)",  
                        "side": trade["side"],  
                        "entry": trade["entry"],  
                        "price": price,  
                        "reduce_margin": reduce_margin,  
                        "pnl": pnl  
                    })  
                    trade["margin"] -= reduce_margin  
                capital += total_exit_pnl  
                # 移除 margin 接近0的仓位  
                trades = [t for t in trades if t["margin"] > 0]  
            else:  
                msg = f"{dt} 止盈条件触发：浮盈 {floating_pnl:.2f} 达到权益6% ({equity*0.06:.2f})且指标不利，全平仓位"  
                record_log(dt, msg)  
                total_exit_pnl = 0  
                for t in trades:  
                    fee_trade = t["margin"] * fee_rate * leverage  
                    if t["side"] == "long":  
                        pnl = t["margin"] * leverage * ((price / t["entry"]) - 1) - fee_trade  
                    else:  
                        pnl = t["margin"] * leverage * (1 - (price / t["entry"])) - fee_trade  
                    total_exit_pnl += pnl  
                    update_trade({  
                        "time": dt,  
                        "action": "止盈全平(6%)",  
                        "side": t["side"],  
                        "entry": t["entry"],  
                        "price": price,  
                        "margin": t["margin"],  
                        "pnl": pnl  
                    })  
                capital += total_exit_pnl  
                trades.clear()  
                position_base = None   # 全平后重置本次开仓前本金  
                max_floating_pnl = 0  # 重置  
                continue  
        

        # MACD过滤：仅允许趋势方向进场  
        if signal in ["long_entry", "short_entry"] and current_macd is not None:  
            if current_macd > 0 and signal == "short_entry":  
                short_trades = [t for t in trades if t["side"] == "short"]  
                if short_trades:  
                    pnl_short = sum(  
                        t["margin"] * leverage * (1 - (price / t["entry"])) - (t["margin"] * fee_rate * leverage)  
                        for t in short_trades  
                    )  
                    if current_atr and pnl_short >= 1 * current_atr:  
                        msg = f"{dt} MACD {current_macd:.2f} > 0; 空单累计盈亏 {pnl_short:.2f}>=阈值，全平空单"  
                        record_log(dt, msg)  
                        for t in short_trades:  
                            fee_trade = t["margin"] * fee_rate * leverage  
                            pnl = t["margin"] * leverage * (1 - (price / t["entry"])) - fee_trade  
                            capital += pnl  
                            update_trade({  
                                "time": dt,  
                                "action": "平空",  
                                "entry": t["entry"],  
                                "price": price,  
                                "margin": t["margin"],  
                                "pnl": pnl  
                            })  
                        trades = [t for t in trades if t["side"] != "short"]  
                signal = None  

            elif current_macd <= 0 and signal == "long_entry":  
                long_trades = [t for t in trades if t["side"] == "long"]  
                if long_trades:  
                    pnl_long = sum(  
                        t["margin"] * leverage * ((price / t["entry"]) - 1) - (t["margin"] * fee_rate * leverage)  
                        for t in long_trades  
                    )  
                    if current_atr and pnl_long >= 1 * current_atr:  
                        msg = f"{dt} MACD {current_macd:.2f} <= 0; 多单累计盈亏 {pnl_long:.2f}>=阈值，全平多单"  
                        record_log(dt, msg)  
                        for t in long_trades:  
                            fee_trade = t["margin"] * fee_rate * leverage  
                            pnl = t["margin"] * leverage * ((price / t["entry"]) - 1) - fee_trade  
                            capital += pnl  
                            update_trade({  
                                "time": dt,  
                                "action": "平多",  
                                "entry": t["entry"],  
                                "price": price,  
                                "margin": t["margin"],  
                                "pnl": pnl  
                            })  
                        trades = [t for t in trades if t["side"] != "long"]  
                signal = None  

        # -------------------------------  
        # 风险控制逻辑：部分平仓  
        # allowed_margin = equity * max_holding_ratio * 1.2 - used_margin  
        # if allowed_margin <= 0:  
        #     reduction_ratio = 0.5  
        #     msg = f"{dt} 风险控制：持仓加浮盈占比 {(used_margin + floating_pnl)/equity*100:.2f}%，触发减仓，每笔交易减半。"  
        #     record_log(dt, msg)  
        #     for trade in trades:  
        #         reduce_margin = trade["margin"] * reduction_ratio  
        #         fee_trade = reduce_margin * fee_rate * leverage  
        #         if trade["side"] == "long":  
        #             realized = reduce_margin * leverage * ((price / trade["entry"]) - 1) - fee_trade  
        #         else:  
        #             realized = reduce_margin * leverage * (1 - (price / trade["entry"])) - fee_trade  
        #         capital += realized  
        #         update_trade({  
        #             "time": dt,  
        #             "action": "部分平仓",  
        #             "side": trade["side"],  
        #             "entry": trade["entry"],  
        #             "price": price,  
        #             "reduce_margin": reduce_margin,  
        #             "realized": realized  
        #         })  
        #         trade["margin"] -= reduce_margin  
        #     used_margin = sum(t["margin"] for t in trades)  
        #     available_capital = equity - used_margin  

        # ----- 出场信号处理 -----  
        if signal in ["long_exit", "short_exit"]:  
            if signal == "long_exit":  
                long_trades = [t for t in trades if t["side"] == "long"]  
                if long_trades:  
                    pnl_total = 0  
                    for t in long_trades:  
                        fee_trade = t["margin"] * fee_rate * leverage  
                        pnl = t["margin"] * leverage * ((price / t["entry"]) - 1) - fee_trade  
                        pnl_total += pnl  
                        update_trade({  
                            "time": dt,  
                            "action": "平多(Exit)",  
                            "entry": t["entry"],  
                            "price": price,  
                            "margin": t["margin"],  
                            "pnl": pnl  
                        })  
                    msg = f"{dt} long_exit 信号触发：全平多单，累计盈亏 {pnl_total:.2f}"  
                    record_log(dt, msg)  
                    capital += pnl_total  
                    trades = [t for t in trades if t["side"] != "long"]  
            elif signal == "short_exit":  
                short_trades = [t for t in trades if t["side"] == "short"]  
                if short_trades:  
                    pnl_total = 0  
                    for t in short_trades:  
                        fee_trade = t["margin"] * fee_rate * leverage  
                        pnl = t["margin"] * leverage * (1 - (price / t["entry"])) - fee_trade  
                        pnl_total += pnl  
                        update_trade({  
                            "time": dt,  
                            "action": "平空(Exit)",  
                            "entry": t["entry"],  
                            "price": price,  
                            "margin": t["margin"],  
                            "pnl": pnl  
                        })  
                    msg = f"{dt} short_exit 信号触发：全平空单，累计盈亏 {pnl_total:.2f}"  
                    record_log(dt, msg)  
                    capital += pnl_total  
                    trades = [t for t in trades if t["side"] != "short"]  
            signal = None  
        if row['adx'] > 25:
            entry_atr_threshold_ratio = adx_entry_threshold_ratio
        elif row['adx'] < 20:
            entry_atr_threshold_ratio = noadx_entry_threshold_ratio
        else:
            entry_atr_threshold_ratio = (noadx_entry_threshold_ratio + adx_entry_threshold_ratio)/2

        # ----- 进场信号处理（仅 long_entry / short_entry） -----  
        if signal in ["long_entry", "short_entry"]:  
            new_side = "long" if signal == "long_entry" else "short"  
            current_atr_raw = row.get("atr", None)  
            dummy_fee = (equity / open_reciprocal) * fee_rate * leverage  
            if current_atr_raw is not None:  
                # 对于多单用 low AVWAP；空单用 high AVWAP  
                baseline = row["hiAVWAP"] if new_side == "short" else row["loAVWAP"]  
                pending_signal = {  
                    "side": new_side,  
                    "baseline": baseline,  
                    "atr": entry_atr_threshold_ratio * current_atr_raw,  
                    "signal_time": dt  
                }  

        # 如果当前没有持仓、没有 pending_signal，并且本行无任何信号，则直接跳过后续较多计算  
        if (not trades) and (pd.isna(signal)) and (pending_signal is None):  
            continue  
        else:  
            pass  

        # ----- 持续进场逻辑 -----  
        if pending_signal and price < row['prev_close']:  
            side = pending_signal["side"]  
            baseline = pending_signal["baseline"]  
            atr_pending = pending_signal["atr"]  
            condition_met = False  
            if side == "short" and price >= baseline + entry_atr_threshold_ratio * atr_pending:  
                condition_met = True  
            elif side == "long" and price <= baseline - entry_atr_threshold_ratio * atr_pending:  
                condition_met = True  

            if condition_met:  
                base_trade_margin = equity / open_reciprocal  
                allowed_margin = equity * 0.20 - used_margin  
                if allowed_margin <= 0:  
                    record_log(dt, f"{dt} 持仓比例 {(used_margin/equity)*100:.2f}% 达到风险上限，拒绝持续进场 {side}")  
                    pending_signal = None  
                else:  
                    trade_margin = min(base_trade_margin, allowed_margin)  
                    if available_capital < trade_margin:  
                        reverse_side = "short" if side == "long" else "long"  
                        reverse_trades = [t for t in trades if t["side"] == reverse_side]  
                        removed_trades = []  
                        for rt in reverse_trades:  
                            fee_trade_rt = rt["margin"] * fee_rate * leverage  
                            if rt["side"] == "long":  
                                pnl_rt = rt["margin"] * leverage * ((price / rt["entry"]) - 1) - fee_trade_rt  
                            else:  
                                pnl_rt = rt["margin"] * leverage * (1 - (price / rt["entry"])) - fee_trade_rt  
                            if pnl_rt > 0:  
                                record_log(dt, f"{dt} 平仓反向仓位 {rt['side']}: 入场价={rt['entry']}, 当前价={price}, 盈利 {pnl_rt:.2f}，释放仓位 {rt['margin']:.2f}")  
                                capital += pnl_rt  
                                removed_trades.append(rt)  
                        for rt in removed_trades:  
                            trades.remove(rt)  
                        used_margin = sum(t["margin"] for t in trades)  
                        available_capital = (capital + floating_pnl) - used_margin  

                    if available_capital < trade_margin:  
                        record_log(dt, f"{dt} 持续进场信号 {side} 资金不足 (可用: {available_capital:.2f}, 需要: {trade_margin:.2f})")  
                    else:  
                        trades.append({  
                            "time": dt,  
                            "side": side,  
                            "entry": price,  
                            "atr": atr_pending,  
                            "margin": trade_margin  
                        })  
                        used_margin = sum(t["margin"] for t in trades)  
                        position_ratio = used_margin / equity if equity != 0 else 0  
                        record_log(dt, f"{dt} 持续进场 {side}: 当前价={price}, 进场额度={trade_margin:.2f}, 持仓比重 {(position_ratio*100):.2f}%")  
                        update_trade({  
                            "time": dt,  
                            "action": "开仓",  
                            "side": side,  
                            "entry": price,  
                            "margin": trade_margin  
                        })  
            else:  
                reversal_arrow = row['reversal_arrow']  
                if pd.notna(reversal_arrow) and (  
                     (reversal_arrow == 'down' and  pending_signal["side"] == 'long')  
                     or (reversal_arrow == 'up' and  pending_signal["side"] == 'short')  
                ):  
                    pending_signal = None  

        # ----- 平仓出场逻辑（标准止盈 / 整体止盈 / 风险控制） -----  
        direction = row.get("reversal_arrow", None)  
        escape_executed = False  
        if not escape_executed and current_atr is not None and trades:  
            long_trades = [t for t in trades if t["side"] == "long"]  
            short_trades = [t for t in trades if t["side"] == "short"]  

            # 标准止盈逻辑  
            if long_trades:  
                total_margin = sum(t["margin"] for t in long_trades)  
                avg_entry_long = sum(t["margin"] * t["entry"] for t in long_trades) / total_margin  
                if price >= avg_entry_long + exit_threshold_ratio * current_atr:  
                    pnl_total = 0  
                    for t in long_trades:  
                        fee_trade = t["margin"] * fee_rate * leverage  
                        pnl = t["margin"] * leverage * ((price / t["entry"]) - 1) - fee_trade  
                        pnl_total += pnl  
                        update_trade({  
                            "time": dt,  
                            "action": "止盈平多",  
                            "entry": t["entry"],  
                            "price": price,  
                            "margin": t["margin"],  
                            "pnl": pnl  
                        })  
                    record_log(dt, f"{dt} 多头止盈：加权均价 {avg_entry_long:.2f}, 当前价 {price}, 盈亏 {pnl_total:.2f}，全平多单")  
                    capital += pnl_total  
                    trades = [t for t in trades if t["side"] != "long"]  

            if short_trades:  
                total_margin = sum(t["margin"] for t in short_trades)  
                avg_entry_short = sum(t["margin"] * t["entry"] for t in short_trades) / total_margin  
                if price <= avg_entry_short - exit_threshold_ratio * current_atr:  
                    pnl_total = 0  
                    for t in short_trades:  
                        fee_trade = t["margin"] * fee_rate * leverage  
                        pnl = t["margin"] * leverage * (1 - (price / t["entry"])) - fee_trade  
                        pnl_total += pnl  
                        update_trade({  
                            "time": dt,  
                            "action": "止盈平空",  
                            "entry": t["entry"],  
                            "price": price,  
                            "margin": t["margin"],  
                            "pnl": pnl  
                        })  
                    record_log(dt, f"{dt} 空头止盈：加权均价 {avg_entry_short:.2f}, 当前价 {price}, 盈亏 {pnl_total:.2f}，全平空单")  
                    capital += pnl_total  
                    trades = [t for t in trades if t["side"] != "short"]  

        # 整体止盈逻辑：若浮动盈利达到使用保证金的预设比例（例如20%），则平掉一半仓位以锁定部分盈利  
        if trades and floating_pnl >= max_holding_ratio * used_margin and floating_pnl > price * 2/100:  
            record_log(dt, f"{dt} 整体止盈：浮盈 {floating_pnl:.2f} >= {max_holding_ratio*100:.0f}% 持仓额度，平半仓锁定盈利")  
            total_exit_pnl = 0  
            for t in trades:  
                # 计算本次平仓涉及的仓位金额（平一半）  
                reduce_margin = t["margin"] / 2  
                fee_trade = reduce_margin * fee_rate * leverage  
                if t["side"] == "long":  
                    pnl = reduce_margin * leverage * ((price / t["entry"]) - 1) - fee_trade  
                else:  
                    pnl = reduce_margin * leverage * (1 - (price / t["entry"])) - fee_trade  
                total_exit_pnl += pnl  
                update_trade({  
                    "time": dt,  
                    "action": "整体半止盈",  
                    "side": t["side"],  
                    "entry": t["entry"],  
                    "price": price,  
                    "reduce_margin": reduce_margin,  
                    "pnl": pnl  
                })  
                # 剩余持仓减半  
                t["margin"] -= reduce_margin  
            capital += total_exit_pnl  
            # 如果仓位金额减为0则从持仓中清除  
            trades = [t for t in trades if t["margin"] > 0]  

            # 新增止损逻辑：使用 escape_threshold_ratio  
            # 对于多单，若价格低于平均入场价 - escape_threshold_ratio * current_atr，则触发止损  
            if long_trades:  
                total_margin_long = sum(t["margin"] for t in long_trades)  
                avg_entry_long = sum(t["margin"] * t["entry"] for t in long_trades) / total_margin_long  
                if price <= avg_entry_long - escape_threshold_ratio * current_atr:  
                    pnl_total = 0  
                    for t in long_trades:  
                        fee_trade = t["margin"] * fee_rate * leverage  
                        pnl = t["margin"] * leverage * ((price / t["entry"]) - 1) - fee_trade  
                        pnl_total += pnl  
                        update_trade({  
                            "time": dt,  
                            "action": "止损平多",  
                            "entry": t["entry"],  
                            "price": price,  
                            "margin": t["margin"],  
                            "pnl": pnl  
                        })  
                    record_log(dt, f"{dt} 多单止损：加权均价 {avg_entry_long:.2f}, 当前价 {price}, 盈亏 {pnl_total:.2f}，全平多单")  
                    capital += pnl_total  
                    trades = [t for t in trades if t["side"] != "long"]  

            # 对于空单，若价格高于平均入场价 + escape_threshold_ratio * current_atr，则触发止损  
            if short_trades:  
                total_margin_short = sum(t["margin"] for t in short_trades)  
                avg_entry_short = sum(t["margin"] * t["entry"] for t in short_trades) / total_margin_short  
                if price >= avg_entry_short + escape_threshold_ratio * current_atr:  
                    pnl_total = 0  
                    for t in short_trades:  
                        fee_trade = t["margin"] * fee_rate * leverage  
                        pnl = t["margin"] * leverage * (1 - (price / t["entry"])) - fee_trade  
                        pnl_total += pnl  
                        update_trade({  
                            "time": dt,  
                            "action": "止损平空",  
                            "entry": t["entry"],  
                            "price": price,  
                            "margin": t["margin"],  
                            "pnl": pnl  
                        })  
                    record_log(dt, f"{dt} 空单止损：加权均价 {avg_entry_short:.2f}, 当前价 {price}, 盈亏 {pnl_total:.2f}，全平空单")  
                    capital += pnl_total  
                    trades = [t for t in trades if t["side"] != "short"]  

        # ----- 平仓出场逻辑（标准止盈 / 整体止盈 / 风险控制） -----  
        direction = row.get("reversal_arrow", None)  
        escape_executed = False  
        if not escape_executed and current_atr is not None and trades:  
            long_trades = [t for t in trades if t["side"] == "long"]  
            short_trades = [t for t in trades if t["side"] == "short"]  

            if long_trades:  
                total_margin = sum(t["margin"] for t in long_trades)  
                avg_entry_long = sum(t["margin"] * t["entry"] for t in long_trades) / total_margin  
                if price >= avg_entry_long + exit_threshold_ratio * current_atr:  
                    pnl_total = 0  
                    for t in long_trades:  
                        fee_trade = t["margin"] * fee_rate * leverage  
                        pnl = t["margin"] * leverage * ((price / t["entry"]) - 1) - fee_trade  
                        pnl_total += pnl  
                        update_trade({  
                            "time": dt,  
                            "action": "止盈平多",  
                            "entry": t["entry"],  
                            "price": price,  
                            "margin": t["margin"],  
                            "pnl": pnl  
                        })  
                    record_log(dt, f"{dt} 多头止盈：加权均价 {avg_entry_long:.2f}, 当前价 {price}, 盈亏 {pnl_total:.2f}，全平多单")  
                    capital += pnl_total  
                    trades = [t for t in trades if t["side"] != "long"]  

            if short_trades:  
                total_margin = sum(t["margin"] for t in short_trades)  
                avg_entry_short = sum(t["margin"] * t["entry"] for t in short_trades) / total_margin  
                if price <= avg_entry_short - exit_threshold_ratio * current_atr:  
                    pnl_total = 0  
                    for t in short_trades:  
                        fee_trade = t["margin"] * fee_rate * leverage  
                        pnl = t["margin"] * leverage * (1 - (price / t["entry"])) - fee_trade  
                        pnl_total += pnl  
                        update_trade({  
                            "time": dt,  
                            "action": "止盈平空",  
                            "entry": t["entry"],  
                            "price": price,  
                            "margin": t["margin"],  
                            "pnl": pnl  
                        })  
                    record_log(dt, f"{dt} 空头止盈：加权均价 {avg_entry_short:.2f}, 当前价 {price}, 盈亏 {pnl_total:.2f}，全平空单")  
                    capital += pnl_total  
                    trades = [t for t in trades if t["side"] != "short"]  

        used_margin = sum(t["margin"] for t in trades)  
        # time.sleep(0.5)  # 模拟计算延时，便于观察实时更新  

    # 最后平仓处理  
    final_price = df["close"].iloc[-1]  
    remaining_pnl = 0  
    for t in trades:  
        fee_trade = t["margin"] * fee_rate * leverage  
        if t["side"] == "long":  
            gross_profit = t["margin"] * leverage * ((final_price / t["entry"]) - 1)  
        else:  
            gross_profit = t["margin"] * leverage * (1 - (final_price / t["entry"]))  
        remaining_pnl += (gross_profit - fee_trade)  
    final_capital = capital + remaining_pnl  

    # 计算最大权益回撤  
    peak_equity = -np.inf  
    max_drawdown_equity_pct = 0  
    for eq in equity_trace:  
        if eq > peak_equity:  
            peak_equity = eq  
        else:  
            drawdown_pct = (peak_equity - eq) / peak_equity * 100  
            if drawdown_pct > max_drawdown_equity_pct:  
                max_drawdown_equity_pct = drawdown_pct  

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
    record_log(pd.Timestamp.now(), f"回测结束: {report}")  
    return report  

##############################################  
# 封装的回测入口函数  
##############################################  
def run_backtest():  
    data_file = "btc_5m_avwap_result.csv"  # 请替换为实际数据文件路径  
    if not os.path.exists(data_file):  
        record_log(pd.Timestamp.now(), f"数据文件未找到: {data_file}")  
        return  
    df = pd.read_csv(data_file, parse_dates=True, index_col=0)  
    report = backtest_scaling_strategy_with_capital(df)  
    record_log(pd.Timestamp.now(), f"回测结果: {report}")  

if __name__ == "__main__":  
    run_backtest()  