"""
时间序列分解模块
基于statsmodels实现STL分解（趋势、季节性、残差）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDecomposer:
    """时间序列分解器类"""
    
    def __init__(self):
        """初始化分解器"""
        self.decomposition_results = {}
        
    def stl_decomposition(self, 
                         data: pd.Series,
                         seasonal: int = 7,
                         trend: Optional[int] = None,
                         robust: bool = False) -> Dict:
        """
        STL分解（Season and Trend decomposition using Loess）
        
        Args:
            data: 时间序列数据
            seasonal: 季节性周期长度
            trend: 趋势平滑参数
            robust: 是否使用鲁棒性STL
            
        Returns:
            包含趋势、季节性、残差的字典
        """
        logger.info(f"执行STL分解，季节性周期: {seasonal}")
        
        # 确保数据是时间序列
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是DatetimeIndex")
        
        # 执行STL分解
        stl = STL(data, seasonal=seasonal, trend=trend, robust=robust)
        result = stl.fit()
        
        # 保存结果
        decomposition_dict = {
            'original': data,
            'trend': result.trend,
            'seasonal': result.seasonal,
            'resid': result.resid,
            'fitted': result.trend + result.seasonal,
            'stl_object': result
        }
        
        self.decomposition_results['stl'] = decomposition_dict
        
        logger.info("STL分解完成")
        return decomposition_dict
    
    def classical_decomposition(self, 
                              data: pd.Series,
                              model: str = 'additive',
                              period: Optional[int] = None) -> Dict:
        """
        经典时间序列分解
        
        Args:
            data: 时间序列数据
            model: 分解模型 ('additive', 'multiplicative')
            period: 季节性周期
            
        Returns:
            包含趋势、季节性、残差的字典
        """
        logger.info(f"执行经典分解，模型: {model}")
        
        # 执行经典分解
        result = seasonal_decompose(data, model=model, period=period)
        
        # 保存结果
        decomposition_dict = {
            'original': data,
            'trend': result.trend,
            'seasonal': result.seasonal,
            'resid': result.resid,
            'fitted': result.trend + result.seasonal if model == 'additive' 
                     else result.trend * result.seasonal,
            'decompose_object': result
        }
        
        self.decomposition_results['classical'] = decomposition_dict
        
        logger.info("经典分解完成")
        return decomposition_dict
    
    def analyze_stationarity(self, data: pd.Series) -> Dict:
        """
        平稳性分析
        
        Args:
            data: 时间序列数据
            
        Returns:
            平稳性检验结果
        """
        logger.info("执行平稳性分析")
        
        # ADF检验
        adf_result = adfuller(data.dropna())
        
        # 整理结果
        stationarity_result = {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05,
            'interpretation': '平稳' if adf_result[1] < 0.05 else '非平稳'
        }
        
        logger.info(f"ADF检验结果: {stationarity_result['interpretation']}")
        logger.info(f"p值: {stationarity_result['p_value']:.4f}")
        
        return stationarity_result
    
    def detect_anomalies_from_residuals(self, 
                                      residuals: pd.Series,
                                      method: str = 'iqr',
                                      threshold: float = 2.0) -> np.ndarray:
        """
        基于残差的异常检测
        
        Args:
            residuals: 残差序列
            method: 检测方法 ('iqr', 'zscore', 'threshold')
            threshold: 阈值
            
        Returns:
            异常值索引
        """
        logger.info(f"基于残差的异常检测，方法: {method}")
        
        # 去除NaN值
        clean_residuals = residuals.dropna()
        
        if method == 'iqr':
            Q1 = clean_residuals.quantile(0.25)
            Q3 = clean_residuals.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            anomalies = (clean_residuals < lower_bound) | (clean_residuals > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((clean_residuals - clean_residuals.mean()) / clean_residuals.std())
            anomalies = z_scores > threshold
            
        elif method == 'threshold':
            anomalies = np.abs(clean_residuals) > threshold
            
        else:
            raise ValueError(f"不支持的异常检测方法: {method}")
        
        anomaly_indices = clean_residuals[anomalies].index
        logger.info(f"检测到{len(anomaly_indices)}个异常值")
        
        return anomaly_indices
    
    def seasonal_strength(self, decomposition_result: Dict) -> float:
        """
        计算季节性强度
        
        Args:
            decomposition_result: 分解结果
            
        Returns:
            季节性强度 (0-1)
        """
        seasonal = decomposition_result['seasonal']
        resid = decomposition_result['resid']
        
        # 计算季节性强度
        seasonal_var = np.var(seasonal.dropna())
        residual_var = np.var(resid.dropna())
        
        if residual_var == 0:
            return 1.0
        
        strength = seasonal_var / (seasonal_var + residual_var)
        return min(1.0, max(0.0, strength))
    
    def trend_strength(self, decomposition_result: Dict) -> float:
        """
        计算趋势强度
        
        Args:
            decomposition_result: 分解结果
            
        Returns:
            趋势强度 (0-1)
        """
        trend = decomposition_result['trend']
        resid = decomposition_result['resid']
        
        # 计算趋势强度
        trend_var = np.var(trend.dropna())
        residual_var = np.var(resid.dropna())
        
        if residual_var == 0:
            return 1.0
        
        strength = trend_var / (trend_var + residual_var)
        return min(1.0, max(0.0, strength))
    
    def plot_decomposition(self, 
                          decomposition_result: Dict,
                          figsize: Tuple[int, int] = (12, 10),
                          title: str = "时间序列分解结果") -> plt.Figure:
        """
        绘制分解结果
        
        Args:
            decomposition_result: 分解结果
            figsize: 图形大小
            title: 图形标题
            
        Returns:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # 原始数据
        axes[0].plot(decomposition_result['original'], color='blue', linewidth=1)
        axes[0].set_title('原始数据')
        axes[0].grid(True, alpha=0.3)
        
        # 趋势
        axes[1].plot(decomposition_result['trend'], color='green', linewidth=1)
        axes[1].set_title('趋势')
        axes[1].grid(True, alpha=0.3)
        
        # 季节性
        axes[2].plot(decomposition_result['seasonal'], color='orange', linewidth=1)
        axes[2].set_title('季节性')
        axes[2].grid(True, alpha=0.3)
        
        # 残差
        axes[3].plot(decomposition_result['resid'], color='red', linewidth=1)
        axes[3].set_title('残差')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residual_analysis(self, 
                             residuals: pd.Series,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        残差分析图
        
        Args:
            residuals: 残差序列
            figsize: 图形大小
            
        Returns:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('残差分析', fontsize=16)
        
        # 残差时间序列图
        axes[0, 0].plot(residuals, color='red', linewidth=1)
        axes[0, 0].set_title('残差时间序列')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 残差直方图
        axes[0, 1].hist(residuals.dropna(), bins=30, density=True, alpha=0.7, color='red')
        axes[0, 1].set_title('残差分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q图')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 自相关图
        try:
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(residuals.dropna(), ax=axes[1, 1], lags=20)
            axes[1, 1].set_title('残差自相关')
        except:
            axes[1, 1].text(0.5, 0.5, '无法绘制自相关图', ha='center', va='center')
        
        plt.tight_layout()
        return fig
    
    def forecast_with_decomposition(self, 
                                  decomposition_result: Dict,
                                  periods: int = 30) -> pd.Series:
        """
        基于分解结果的预测
        
        Args:
            decomposition_result: 分解结果
            periods: 预测期数
            
        Returns:
            预测结果
        """
        logger.info(f"基于分解结果预测未来{periods}期")
        
        trend = decomposition_result['trend']
        seasonal = decomposition_result['seasonal']
        
        # 提取季节性模式
        seasonal_pattern = seasonal.dropna()
        seasonal_length = len(seasonal_pattern)
        
        # 生成未来日期
        last_date = trend.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        
        # 趋势外推（简单线性外推）
        trend_clean = trend.dropna()
        if len(trend_clean) >= 2:
            trend_slope = (trend_clean.iloc[-1] - trend_clean.iloc[-2])
            future_trend = [trend_clean.iloc[-1] + trend_slope * (i + 1) for i in range(periods)]
        else:
            future_trend = [trend_clean.iloc[-1]] * periods
        
        # 季节性循环
        future_seasonal = []
        for i in range(periods):
            seasonal_index = i % seasonal_length
            if seasonal_index < len(seasonal_pattern):
                future_seasonal.append(seasonal_pattern.iloc[seasonal_index])
            else:
                future_seasonal.append(0)
        
        # 合并预测
        forecast = pd.Series(
            [t + s for t, s in zip(future_trend, future_seasonal)],
            index=future_dates
        )
        
        return forecast
    
    def summary_statistics(self, decomposition_result: Dict) -> Dict:
        """
        计算分解结果的统计摘要
        
        Args:
            decomposition_result: 分解结果
            
        Returns:
            统计摘要
        """
        original = decomposition_result['original']
        trend = decomposition_result['trend']
        seasonal = decomposition_result['seasonal']
        resid = decomposition_result['resid']
        
        summary = {
            'original_stats': {
                'mean': original.mean(),
                'std': original.std(),
                'min': original.min(),
                'max': original.max()
            },
            'trend_stats': {
                'mean': trend.mean(),
                'std': trend.std(),
                'min': trend.min(),
                'max': trend.max()
            },
            'seasonal_strength': self.seasonal_strength(decomposition_result),
            'trend_strength': self.trend_strength(decomposition_result),
            'residual_stats': {
                'mean': resid.mean(),
                'std': resid.std(),
                'min': resid.min(),
                'max': resid.max()
            },
            'variance_explained': {
                'trend': np.var(trend.dropna()) / np.var(original.dropna()),
                'seasonal': np.var(seasonal.dropna()) / np.var(original.dropna()),
                'residual': np.var(resid.dropna()) / np.var(original.dropna())
            }
        }
        
        return summary

if __name__ == "__main__":
    # 测试代码
    decomposer = TimeSeriesDecomposer()
    
    # 生成测试数据
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    
    # 创建具有趋势和季节性的时间序列
    trend = np.linspace(0, 10, 365)
    seasonal = 3 * np.sin(2 * np.pi * np.arange(365) / 7)  # 周季节性
    noise = np.random.normal(0, 0.5, 365)
    
    ts_data = pd.Series(trend + seasonal + noise, index=dates)
    
    print("时间序列数据:")
    print(ts_data.head())
    
    # STL分解
    stl_result = decomposer.stl_decomposition(ts_data, seasonal=7)
    print(f"\nSTL分解完成")
    
    # 计算统计摘要
    summary = decomposer.summary_statistics(stl_result)
    print(f"季节性强度: {summary['seasonal_strength']:.4f}")
    print(f"趋势强度: {summary['trend_strength']:.4f}")
    
    # 异常检测
    anomalies = decomposer.detect_anomalies_from_residuals(stl_result['resid'])
    print(f"检测到{len(anomalies)}个异常值")
    
    # 平稳性分析
    stationarity = decomposer.analyze_stationarity(ts_data)
    print(f"平稳性检验: {stationarity['interpretation']}")
    
    # 预测
    forecast = decomposer.forecast_with_decomposition(stl_result, periods=30)
    print(f"未来30天预测均值: {forecast.mean():.2f}")
    
    # 绘制分解结果
    fig = decomposer.plot_decomposition(stl_result)
    # plt.show()  # 如果需要显示图形，取消注释
    
    print("\n时间序列分解测试完成") 