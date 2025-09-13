import threading  
from tests.okx_DataAggregator import OkxDataAggregator
from factors import FactorManager 
from tests.binance_websocket_client import BinanceDataGateway

from tests.dash_layouts import DashLayout

if __name__ == "__main__":

    # inst_id="BERA-USDT-SWAP"
    # inst_id="FARTCOIN-USDT-SWAP"
    # inst_id="AIXBT-USDT-SWAP"
    inst_id="BTCUSDT"
    # inst_id="PEPE-USDT-SWAP"
    # inst_id="ETH-USDT-SWAP"
    factor_manager = FactorManager()

    # 1) 初始化数据聚合器  
    from tests.exchange_manager import TradingEnvConfig,ExchangeConfig, ExchangeManager, BinanceExchangeConfig
    # 配置交易对和杠杆
    trading_config = TradingEnvConfig(symbol=inst_id,  leverage=25)
    # 初始化交易所管理器
    exchange_manager = ExchangeManager(trading_config, BinanceExchangeConfig())

    from tests.dual_positionTrader import DualPositionTrader
    percentage_every_position = 0.01 # * 一笔占用本金1%
    best_psychology_stop_loss = 5
    f'0.15 != {trading_config.leverage} * {best_psychology_stop_loss}'
    eth_min_amount = 0.005 #under leverage 100
    eth_min_amount = 3.4 #bera  （0.3*3.4 > 19trump最低要求1单位）
    trader = DualPositionTrader(exchange_manager,
                                take_profit=0.3, 
                                stop_loss=best_psychology_stop_loss, 
                                fixed_usdt_amount=eth_min_amount,
                                taker_fee=0.0005, maker_fee=0.0002) 
    aggregator = OkxDataAggregator(trader=trader)  
    threading.Thread(target=aggregator.episode_task, daemon=True).start()  

    ws_client = BinanceDataGateway(aggregator, 
                                   symbol=inst_id, 
                                   limit=0, interval=1) 
    
    threading.Thread(target=ws_client.run_forever, daemon=True).start()  

    dash = DashLayout(__name__, aggregator, factor_manager)
    dash.app.run(debug=True)
    