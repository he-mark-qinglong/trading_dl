import ccxt  
from typing import Optional  
from config import ExchangeConfig  

def init_exchange(exchange_config: Optional[ExchangeConfig] = None):  
    """初始化交易所接口"""  
    config = {  
        'enableRateLimit': True,  
        'options': {  
            'defaultType': 'swap',  
        }  
    }  
    
    if exchange_config:  
        config.update({  
            'apiKey': exchange_config.api_key,  
            'secret': exchange_config.secret_key,  
            'password': exchange_config.passphrase  
        })  
    
    try:  
        exchange = ccxt.okx(config)  
        return exchange  
    except Exception as e:  
        print(f"Error initializing exchange: {e}")  
        return None