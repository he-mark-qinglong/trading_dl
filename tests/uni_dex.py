import requests  

class UniswapDataFetcher:  
    def __init__(self):  
        # Uniswap V3 的 The Graph 子图 URL  
        self.graph_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"  

    def fetch_pair_data(self, token0: str, token1: str):  
        """  
        查询交易对的市场数据  
        :param token0: 交易对中的第一个代币地址（如 LUMO）  
        :param token1: 交易对中的第二个代币地址（如 USDT 或 WETH）  
        """  
        query = """  
        {  
            pools(where: {token0: "%s", token1: "%s"}) {  
                id  
                token0 {  
                    symbol  
                }  
                token1 {  
                    symbol  
                }  
                feeTier  
                liquidity  
                volumeUSD  
                sqrtPrice  
            }  
        }  
        """ % (token0, token1)  

        response = requests.post(self.graph_url, json={"query": query})  
        if response.status_code == 200:  
            data = response.json()  
            print(data)
            return data["data"]["pools"]  
        else:  
            print(f"查询失败，状态码: {response.status_code}")  
            return None  


if __name__ == "__main__":  
    # 初始化数据获取器  
    fetcher = UniswapDataFetcher()  

    # 查询 LUMO-USDT 交易对的数据（需要代币合约地址）  
    lumo_address = "4FkNq8RcCYg4ZGDWh14scJ7ej3m5vMjYTcWoJVkupump"  # 替换为 LUMO 的合约地址  
    usdt_address = "0xdAC17F958D2ee523a2206206994597C13D831ec7"  # USDT 合约地址  

    pair_data = fetcher.fetch_pair_data(lumo_address, usdt_address)  
    if pair_data:  
        for pool in pair_data:  
            print(f"交易对: {pool['token0']['symbol']}/{pool['token1']['symbol']}")  
            print(f"手续费等级: {pool['feeTier']}")  
            print(f"流动性: {pool['liquidity']}")  
            print(f"交易量（USD）: {pool['volumeUSD']}")  
            print(f"价格（sqrtPrice）: {pool['sqrtPrice']}")  
            print("-" * 30)  
    else:  
        print("未找到 LUMO 的交易对数据，请检查代币地址或流动性池是否存在。")