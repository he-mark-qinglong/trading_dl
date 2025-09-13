from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

class BinanceExchangeConfig:
    api_key:str = "Cy4lhqGCE4rg9OvhxCDE7PRuxPQIfE6mWyqpk3S8tPw2yauX3bip9Z5gpZy63YYv"
    secret_key:str = "nKhvh8Vfw08YPyOSQ4iu9xWUPuJI97c3TnfJyuaqc812IbIYOf87gxxxT7xC7ONe"

client = Client(BinanceExchangeConfig.api_key, BinanceExchangeConfig.secret_key)

# get market depth
depth = client.get_order_book(symbol='BTCUSDT', limit=20)

print(depth["bids"])
print(depth['asks'])

# get all symbol prices
prices = client.get_all_tickers()
btc = client.get_ticker(symbol='BTCUSDT')
print('btc=======', btc)
# withdraw 100 ETH
# check docs for assumptions around withdrawals
# from binance.exceptions import BinanceAPIException
# try:
#     result = client.withdraw(
#         asset='ETH',
#         address='<eth_address>',
#         amount=100)
# except BinanceAPIException as e:
#     print(e)
# else:
#     print("Success")

# # fetch list of withdrawals
# withdraws = client.get_withdraw_history()

# # fetch list of ETH withdrawals
# eth_withdraws = client.get_withdraw_history(coin='ETH')

# get a deposit address for BTC
address = client.get_deposit_address(coin='BTC')

# get historical kline data from any date range

# fetch 1 minute klines for the last day up until now
klines = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1MINUTE, "30 day ago UTC")
print(f'len of 1m 30days klines={len(klines)}')
# fetch 30 minute klines for the last month of 2017
klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")

# fetch weekly klines since it listed
klines = client.get_historical_klines("NEOBTC", Client.KLINE_INTERVAL_1WEEK, "1 Jan, 2017")
# print('klines', klines)
# create order through websockets
# order_ws = client.create_order( symbol="BTCUSDT", side=client.SIDE_BUY, type=client.ORDER_TYPE_LIMIT_MAKER, quantity=0.1, price=83777)