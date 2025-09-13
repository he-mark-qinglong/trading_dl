# from .contract_env import ContractTradingEnv  

# __all__ = ['ContractTradingEnv']  

# 注册环境（可选，如果需要使用gym.make()）  
try:  
    import gymnasium as gym  
    gym.register(  
        id='ContractTrading-v0',  
        entry_point='envs.contract_env:ContractTradingEnv',  
        kwargs={  
            'config': None,  
            'exchange_config': None  
        }  
    )  
except Exception as e:  
    print(f"Warning: Could not register environment: {e}")