import os  
import threading  
from strategy.scaling_strategy import run_backtest  
from strategy.dash_app import run_dash  

# 全局变量，确保回测只启动一次  
backtest_started = False   

def main():  
    global backtest_started  

    # 仅在真正运行的进程中启动回测线程，并确保不是在调试模式下启动  
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" and not backtest_started:  
        backtest_started = True  # 设置标记为已启动  
        backtest_thread = threading.Thread(target=run_backtest, daemon=True)  
        backtest_thread.start()  

    # 启动 Dash 应用（主线程）  
    run_dash()  

if __name__ == '__main__':  
    main()  