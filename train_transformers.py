import gymnasium as gym  
import numpy as np  
import os  
import ray  
from ray import train  
from ray.rllib.algorithms.ppo import PPO  
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2  
from ray.rllib.models import ModelCatalog  
from ray.rllib.utils.annotations import override  
import torch  
import ccxt
import torch.nn as nn  
from typing import Dict, List  

# 注意力机制模块  
class AttentionBlock(nn.Module):  
    def __init__(self, embed_dim, num_heads=2, dropout=0.1):  
        super().__init__()  
        self.attention = nn.MultiheadAttention(  
            embed_dim,   
            num_heads=num_heads,   
            dropout=dropout,   
            batch_first=True  
        )  
        self.norm1 = nn.LayerNorm(embed_dim)  
        self.norm2 = nn.LayerNorm(embed_dim)  
        self.feed_forward = nn.Sequential(  
            nn.Linear(embed_dim, embed_dim * 4),  
            nn.ReLU(),  
            nn.Linear(embed_dim * 4, embed_dim)  
        )  
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x):  
        # 自注意力层  
        attn_output, _ = self.attention(x, x, x)  
        x = self.norm1(x + self.dropout(attn_output))  
        
        # 前馈网络  
        ff_output = self.feed_forward(x)  
        x = self.norm2(x + self.dropout(ff_output))  
        return x  

# 自定义Transformer模型  
class TransformerModel(TorchModelV2, nn.Module):  
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):  
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)  
        nn.Module.__init__(self)  

        self.input_dim = int(np.product(obs_space.shape))  
        self.embed_dim = 32  
        self.seq_len = 8  
        
        # 输入嵌入  
        self.embedding = nn.Sequential(  
            nn.Linear(self.input_dim, self.embed_dim),  
            nn.ReLU()  
        )  
        
        # Transformer块  
        self.transformer_block = AttentionBlock(  
            embed_dim=self.embed_dim,  
            num_heads=2  
        )  
        
        # 策略头（动作输出）  
        self.policy_head = nn.Sequential(  
            nn.Linear(self.embed_dim, 64),  
            nn.ReLU(),  
            nn.Linear(64, num_outputs)  
        )  
        
        # 价值头  
        self.value_head = nn.Sequential(  
            nn.Linear(self.embed_dim, 64),  
            nn.ReLU(),  
            nn.Linear(64, 1)  
        )  

        # 初始化  
        self._cur_value = None  

    @override(TorchModelV2)  
    def forward(self, input_dict, state, seq_lens):  
        # 处理输入  
        x = input_dict["obs"].float()  
        
        # 嵌入  
        x = self.embedding(x)  
        
        # 添加批次维度和序列长度维度  
        if len(x.shape) == 2:  
            x = x.unsqueeze(1)  
        
        # 复制到序列长度  
        x = x.repeat(1, self.seq_len, 1)  
        
        # Transformer处理  
        x = self.transformer_block(x)  
        
        # 取最后一个时间步的输出  
        x = x[:, -1, :]  
        
        # 策略输出  
        policy_out = self.policy_head(x)  
        
        # 存储价值预测  
        self._cur_value = self.value_head(x).squeeze(1)  
        
        return policy_out, state  

    @override(TorchModelV2)  
    def value_function(self):  
        return self._cur_value  

from config.config import TradingEnvConfig, ExchangeConfig  
from envs.contract_env import ContractTradingEnv

# 检查账户状态的函数也稍作修改  
def check_account_status(exchange):  
    try:  
        trading_env_config = TradingEnvConfig(  
            symbol="PEPE/USDT:USDT",  
            timeframe="1m",  
            initial_balance=10000.0,  
            max_position=1,  
            leverage=10
        )
        # 获取账户余额  
        balance = exchange.fetch_balance()  
        print("\n=== 账户余额 ===")  
        print(f"USDT 余额: {balance.get('USDT', {}).get('total', 0)}")  
        print(f"可用 USDT: {balance.get('USDT', {}).get('free', 0)}")  
        print(f"已用 USDT: {balance.get('USDT', {}).get('used', 0)}")  

        trading_env_config.initial_balance = balance.get('USDT', {}).get('free', 0)
        
        # 获取持仓信息  
        positions = exchange.fetch_positions(['PEPE/USDT:USDT'])  
        print("\n=== 持仓信息 ===")  
        for position in positions:  
            print(f"交易对: {position['symbol']}")  
            print(f"持仓数量: {position['contracts']}")  
            print(f"持仓方向: {position['side']}")  
            print(f"杠杆倍数: {position['leverage']}")  
            print(f"保证金模式: {position['marginMode']}")  

        # 获取账户配置  
        try:  
            account_config = exchange.privateGetAccountConfig()  
            print("\n=== 账户配置 ===")  
            print(f"持仓模式: {account_config.get('data', [{}])[0].get('posMode')}")  
        except Exception as e:  
            print(f"获取账户配置失败: {e}")  

        return trading_env_config
    except Exception as e:  
        print(f"检查账户状态时出错: {e}")  

def set_account_leverage(exchange, symbol, leverage):  
    print("\n=== 开始设置交易账户 ===")  
    
    # 1. 设置持仓模式为单向持仓  
    try:  
        exchange.privatePostAccountSetPositionMode({  
            'posMode': 'long_short_mode'  
        })  
        print("持仓模式设置成功")  
    except Exception as e:  
        print(f"设置持仓模式失败: {e}")  
    
    # 2. 设置杠杆倍数 - 修改这部分  
    try:  
        # 只指定保证金模式，不设置 posSide  
        exchange.set_leverage(leverage, symbol, params={  
            'mgnMode': 'cross'  # 只设置全仓模式  
        })  
        print(f"杠杆倍数设置为 {leverage}x")  
    except Exception as e:  
        print(f"设置杠杆失败: {e}")  
        if hasattr(e, 'response'):  
            print(f"详细错误: {e.response}")  

# 训练配置  
def get_training_config():  
     # 配置环境  
    # 交易所配置（如果有API密钥）  
    exchange_config = ExchangeConfig(   
    )  

    exchange = ccxt.okx({  
        'enableRateLimit': True,  
        'apiKey': exchange_config.api_key,  
        'secret': exchange_config.secret_key,  
        'password': exchange_config.passphrase,  
        'options': {  
            'defaultType': 'swap',  
        }  
    }) 
    
    # 检查账户状态  
    print("正在检查账户状态...")  
    trading_env_config = check_account_status(exchange)  

    # 如果需要设置杠杆倍数  
    set_account_leverage(exchange, 'PEPE/USDT:USDT', trading_env_config.leverage)
    
    # PPO配置  
    config = {  
        "env": ContractTradingEnv,  
        "env_config": {  
            "trading_config": trading_env_config,  
            "exchange_config": exchange_config  
        },
        "num_workers": 1,  
        "framework": "torch",  
        "model": {  
            "custom_model": "transformer_model",  
        },  
        # "train_batch_size": 4000,  
        # "sgd_minibatch_size": 128,  

        "num_sgd_iter": 30,  
        "lr": 3e-4,  
        "gamma": 0.99,  
        "lambda": 0.95,  
        "clip_param": 0.2,  
        "entropy_coeff": 0.01,  
        "enable_rl_module_and_learner": False,
        'sample_timeout_s':20,

        # # 启用 Replay Buffer  
        # "replay_buffer_config": {  
        #     "type": "MultiAgentReplayBuffer",  # 使用多智能体经验回放缓冲区（适用于单智能体也可以）  
        #     "capacity": 100000,  # Replay Buffer 的容量（存储的样本数量）  
        #     "storage_unit": "timesteps",  # 存储单位为时间步  
        # },  

        # 数据采样和回放相关的配置  
        "batch_mode": "truncate_episodes",  # 使用截断的时间步来构建批次  
        "rollout_fragment_length": 10,  # 每次 rollout 的片段长度  
        "train_batch_size": 10,  # 每次训练的 batch 大小  
        "minibatch_size":5,
        "learning_starts": 10,  # 在开始训练前，至少收集 10000 个时间步的数据  
        "timesteps_per_iteration": 1000,  # 每次迭代的时间步数  
    }  
    return config

def main():     
    config = get_training_config() 
    # 初始化Ray  
    ray.init()  
    
    # 注册自定义模型  
    ModelCatalog.register_custom_model("transformer_model", TransformerModel)  
    
    # 创建PPO训练器   
    trainer = PPO(config=config)  

    # 训练循环  
    num_iterations = 100  
    for i in range(num_iterations):  
        result = trainer.train()['env_runners']  
        
        # 打印训练信息  
        print(f"Iteration {i + 1}")  
        print(f"Episode Reward Mean: {result['episode_reward_mean']}")  
        print(f"Episode Length Mean: {result['episode_len_mean']}")  
        print(f"time:{result['sampler_perf']}")
        
        # 每10轮评估一次  
        if (i + 1) % 10 == 0:  
            # eval_env = gym.make("CartPole-v1") 
            eval_env = ContractTradingEnv(config['env_config'])
            state = eval_env.reset()[0]  
            done = False  
            total_reward = 0  
            
            while not done:  
                action = trainer.compute_single_action(state)  
                state, reward, done, truncated, info = eval_env.step(action)  
                total_reward += reward  
            
            print(f"Evaluation Reward: {total_reward}")  
    
    # 保存模型  
    checkpoint_dir = trainer.save()  
    print(f"Model saved at: {checkpoint_dir}")  
    
    # 清理  
    ray.shutdown()  

if __name__ == "__main__":  
    main()