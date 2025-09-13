# validators/factor_validator.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationConfig:
    """因子验证配置"""
    min_correlation: float = 0.1  # 最小相关性要求
    min_stability: float = 0.6    # 最小稳定性要求
    future_windows: List[int] = (1, 3, 5)  # 未来收益计算窗口
    stability_window: int = 20    # 稳定性计算窗口

@dataclass
class ValidationResult:
    """因子验证结果"""
    factor_name: str
    correlations: Dict[int, float]  # 不同时间窗口的相关性
    stability: float
    turnover: float  # 因子换手率
    ic_decay: Dict[int, float]  # 信息系数衰减
    is_valid: bool

class FactorValidator:
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.validation_results: Dict[str, ValidationResult] = {}

    def calculate_future_returns(self, prices: pd.Series, window: int) -> pd.Series:
        """计算未来收益率"""
        return prices.shift(-window) / prices - 1

    def calculate_correlation(self, factor: pd.Series, returns: pd.Series) -> float:
        """计算因子与收益的相关性"""
        # 使用rank相关性，减少异常值影响
        return factor.rank().corr(returns.rank())

    def calculate_stability(self, factor: pd.Series) -> float:
        """计算因子稳定性"""
        # 使用自相关性作为稳定性指标
        return factor.rolling(self.config.stability_window).corr(
            factor.shift(1)
        ).mean()

    def calculate_turnover(self, factor: pd.Series) -> float:
        """计算因子换手率"""
        return abs(factor - factor.shift(1)).mean() / abs(factor).mean()

    def calculate_ic_decay(self, factor: pd.Series, prices: pd.Series) -> Dict[int, float]:
        """计算信息系数衰减"""
        ic_decay = {}
        for window in self.config.future_windows:
            returns = self.calculate_future_returns(prices, window)
            ic = self.calculate_correlation(factor, returns)
            ic_decay[window] = ic
        return ic_decay

    def validate_factor(self, factor_name: str, factor: pd.Series, prices: pd.Series) -> ValidationResult:
        """验证单个因子"""
        # 确保数据对齐
        factor = factor.copy()
        prices = prices.copy()
        common_index = factor.index.intersection(prices.index)
        factor = factor[common_index]
        prices = prices[common_index]

        # 计算各项指标
        correlations = {}
        for window in self.config.future_windows:
            returns = self.calculate_future_returns(prices, window)
            correlations[window] = self.calculate_correlation(factor, returns)

        stability = self.calculate_stability(factor)
        turnover = self.calculate_turnover(factor)
        ic_decay = self.calculate_ic_decay(factor, prices)

        # 判断因子是否有效
        is_valid = (
            abs(correlations[1]) > self.config.min_correlation and
            stability > self.config.min_stability
        )

        result = ValidationResult(
            factor_name=factor_name,
            correlations=correlations,
            stability=stability,
            turnover=turnover,
            ic_decay=ic_decay,
            is_valid=is_valid
        )

        self.validation_results[factor_name] = result
        return result

    def batch_validate_factors(self, factors: Dict[str, pd.Series], prices: pd.Series) -> Dict[str, ValidationResult]:
        """批量验证多个因子"""
        results = {}
        for factor_name, factor_series in factors.items():
            try:
                results[factor_name] = self.validate_factor(factor_name, factor_series, prices)
            except Exception as e:
                print(f"验证因子 {factor_name} 时出错: {e}")
        return results

    def get_validation_summary(self) -> pd.DataFrame:
        """获取验证结果摘要"""
        summary_data = []
        for factor_name, result in self.validation_results.items():
            summary_data.append({
                'factor_name': factor_name,
                'correlation_1d': result.correlations[1],
                'correlation_3d': result.correlations[3],
                'correlation_5d': result.correlations[5],
                'stability': result.stability,
                'turnover': result.turnover,
                'is_valid': result.is_valid
            })
        return pd.DataFrame(summary_data)

    def plot_validation_results(self, factor_name: str) -> None:
        """绘制因子验证结果"""
        try:
            import matplotlib.pyplot as plt
            result = self.validation_results[factor_name]

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Factor Validation Results: {factor_name}')

            # IC衰减图
            windows = list(result.ic_decay.keys())
            ic_values = list(result.ic_decay.values())
            axes[0, 0].plot(windows, ic_values, marker='o')
            axes[0, 0].set_title('IC Decay')
            axes[0, 0].set_xlabel('Forward Window')
            axes[0, 0].set_ylabel('Information Coefficient')

            # 相关性图
            windows = list(result.correlations.keys())
            corr_values = list(result.correlations.values())
            axes[0, 1].plot(windows, corr_values, marker='o')
            axes[0, 1].set_title('Return Correlations')
            axes[0, 1].set_xlabel('Forward Window')
            axes[0, 1].set_ylabel('Correlation')

            # 稳定性和换手率
            metrics = ['Stability', 'Turnover']
            values = [result.stability, result.turnover]
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_title('Stability & Turnover')
            axes[1, 0].set_ylim(0, 1)

            # 有效性状态
            axes[1, 1].text(0.5, 0.5, f'Valid: {result.is_valid}',
                          ha='center', va='center', fontsize=20)
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"绘制验证结果时出错: {e}")