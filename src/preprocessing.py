"""
数据预处理模块
负责数据清洗、缺失值插补、单位转换等预处理任务
"""

import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理器类"""
    
    def __init__(self):
        """初始化预处理器"""
        self.scalers = {}
        
    def handle_missing_values(self, 
                            data: Union[pd.DataFrame, xr.Dataset], 
                            method: str = 'interpolate',
                            **kwargs) -> Union[pd.DataFrame, xr.Dataset]:
        """
        处理缺失值
        
        Args:
            data: 输入数据
            method: 处理方法 ('interpolate', 'forward_fill', 'backward_fill', 'mean', 'median')
            **kwargs: 额外参数
            
        Returns:
            处理后的数据
        """
        if isinstance(data, pd.DataFrame):
            return self._handle_missing_dataframe(data, method, **kwargs)
        elif isinstance(data, xr.Dataset):
            return self._handle_missing_xarray(data, method, **kwargs)
        else:
            raise ValueError("不支持的数据类型")
    
    def _handle_missing_dataframe(self, 
                                df: pd.DataFrame, 
                                method: str,
                                **kwargs) -> pd.DataFrame:
        """处理DataFrame中的缺失值"""
        df_processed = df.copy()
        
        logger.info(f"原始数据缺失值统计: {df.isnull().sum().sum()}")
        
        if method == 'interpolate':
            # 时间序列插值
            df_processed = df_processed.interpolate(method='time', limit_direction='both')
        elif method == 'forward_fill':
            df_processed = df_processed.fillna(method='ffill')
        elif method == 'backward_fill':
            df_processed = df_processed.fillna(method='bfill')
        elif method == 'mean':
            df_processed = df_processed.fillna(df_processed.mean())
        elif method == 'median':
            df_processed = df_processed.fillna(df_processed.median())
        else:
            raise ValueError(f"不支持的处理方法: {method}")
        
        logger.info(f"处理后缺失值统计: {df_processed.isnull().sum().sum()}")
        return df_processed
    
    def _handle_missing_xarray(self, 
                             ds: xr.Dataset, 
                             method: str,
                             **kwargs) -> xr.Dataset:
        """处理xarray Dataset中的缺失值"""
        ds_processed = ds.copy()
        
        if method == 'interpolate':
            # 时间维度插值
            ds_processed = ds_processed.interpolate_na(dim='time', method='linear')
        elif method == 'forward_fill':
            ds_processed = ds_processed.fillna(method='ffill')
        elif method == 'backward_fill':
            ds_processed = ds_processed.fillna(method='bfill')
        else:
            # 对于其他方法，转换为numpy数组处理
            for var in ds_processed.data_vars:
                if method == 'mean':
                    ds_processed[var] = ds_processed[var].fillna(ds_processed[var].mean())
                elif method == 'median':
                    ds_processed[var] = ds_processed[var].fillna(ds_processed[var].median())
        
        return ds_processed
    
    def convert_temperature_units(self, 
                                temperature_data: Union[pd.Series, xr.DataArray],
                                from_unit: str = 'kelvin',
                                to_unit: str = 'celsius') -> Union[pd.Series, xr.DataArray]:
        """
        温度单位转换
        
        Args:
            temperature_data: 温度数据
            from_unit: 原始单位 ('kelvin', 'celsius', 'fahrenheit')
            to_unit: 目标单位 ('kelvin', 'celsius', 'fahrenheit')
            
        Returns:
            转换后的温度数据
        """
        logger.info(f"温度单位转换: {from_unit} -> {to_unit}")
        
        # 先转换为摄氏度
        if from_unit == 'kelvin':
            celsius_data = temperature_data - 273.15
        elif from_unit == 'fahrenheit':
            celsius_data = (temperature_data - 32) * 5/9
        else:  # celsius
            celsius_data = temperature_data
        
        # 再转换为目标单位
        if to_unit == 'kelvin':
            return celsius_data + 273.15
        elif to_unit == 'fahrenheit':
            return celsius_data * 9/5 + 32
        else:  # celsius
            return celsius_data
    
    def normalize_data(self, 
                      data: Union[pd.DataFrame, np.ndarray],
                      method: str = 'standardize',
                      feature_range: Tuple[float, float] = (0, 1)) -> Union[pd.DataFrame, np.ndarray]:
        """
        数据标准化/归一化
        
        Args:
            data: 输入数据
            method: 标准化方法 ('standardize', 'minmax')
            feature_range: MinMax归一化的范围
            
        Returns:
            标准化后的数据
        """
        logger.info(f"数据标准化: {method}")
        
        if method == 'standardize':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        if isinstance(data, pd.DataFrame):
            # 为DataFrame处理
            scaled_data = pd.DataFrame(
                scaler.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            self.scalers[id(data)] = scaler
            return scaled_data
        else:
            # 为numpy数组处理
            scaled_data = scaler.fit_transform(data)
            self.scalers[id(data)] = scaler
            return scaled_data
    
    def detect_outliers(self, 
                       data: Union[pd.Series, np.ndarray],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> np.ndarray:
        """
        异常值检测
        
        Args:
            data: 输入数据
            method: 检测方法 ('iqr', 'zscore', 'modified_zscore')
            threshold: 阈值
            
        Returns:
            异常值索引数组
        """
        logger.info(f"异常值检测: {method}")
        
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            outliers = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
            
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        outlier_indices = np.where(outliers)[0]
        logger.info(f"检测到{len(outlier_indices)}个异常值")
        
        return outlier_indices
    
    def smooth_time_series(self, 
                          data: Union[pd.Series, xr.DataArray],
                          method: str = 'rolling_mean',
                          window: int = 7) -> Union[pd.Series, xr.DataArray]:
        """
        时间序列平滑
        
        Args:
            data: 时间序列数据
            method: 平滑方法 ('rolling_mean', 'savgol', 'exponential')
            window: 窗口大小
            
        Returns:
            平滑后的数据
        """
        logger.info(f"时间序列平滑: {method}, 窗口大小: {window}")
        
        if method == 'rolling_mean':
            if isinstance(data, pd.Series):
                return data.rolling(window=window, center=True).mean()
            elif isinstance(data, xr.DataArray):
                return data.rolling(time=window, center=True).mean()
                
        elif method == 'savgol':
            from scipy.signal import savgol_filter
            if isinstance(data, pd.Series):
                smoothed_values = savgol_filter(data.values, window, polyorder=3)
                return pd.Series(smoothed_values, index=data.index)
            elif isinstance(data, xr.DataArray):
                smoothed_values = savgol_filter(data.values, window, polyorder=3)
                return xr.DataArray(smoothed_values, dims=data.dims, coords=data.coords)
                
        elif method == 'exponential':
            if isinstance(data, pd.Series):
                return data.ewm(span=window).mean()
            else:
                raise ValueError("指数平滑目前仅支持pandas Series")
                
        else:
            raise ValueError(f"不支持的平滑方法: {method}")
    
    def resample_time_series(self, 
                           data: Union[pd.DataFrame, xr.Dataset],
                           target_frequency: str = 'D',
                           aggregation_method: str = 'mean') -> Union[pd.DataFrame, xr.Dataset]:
        """
        时间序列重采样
        
        Args:
            data: 时间序列数据
            target_frequency: 目标频率 ('D', 'H', 'M', 'W')
            aggregation_method: 聚合方法 ('mean', 'sum', 'max', 'min')
            
        Returns:
            重采样后的数据
        """
        logger.info(f"时间序列重采样: {target_frequency}, 聚合方法: {aggregation_method}")
        
        if isinstance(data, pd.DataFrame):
            resampler = data.resample(target_frequency)
        elif isinstance(data, xr.Dataset):
            resampler = data.resample(time=target_frequency)
        else:
            raise ValueError("不支持的数据类型")
        
        # 应用聚合方法
        if aggregation_method == 'mean':
            return resampler.mean()
        elif aggregation_method == 'sum':
            return resampler.sum()
        elif aggregation_method == 'max':
            return resampler.max()
        elif aggregation_method == 'min':
            return resampler.min()
        else:
            raise ValueError(f"不支持的聚合方法: {aggregation_method}")
    
    def spatial_interpolation(self, 
                            data: xr.Dataset,
                            target_resolution: float,
                            method: str = 'linear') -> xr.Dataset:
        """
        空间插值
        
        Args:
            data: 空间数据
            target_resolution: 目标分辨率
            method: 插值方法 ('linear', 'nearest', 'cubic')
            
        Returns:
            插值后的数据
        """
        logger.info(f"空间插值: {method}, 目标分辨率: {target_resolution}")
        
        # 创建新的坐标网格
        lon_min, lon_max = data.lon.min().values, data.lon.max().values
        lat_min, lat_max = data.lat.min().values, data.lat.max().values
        
        new_lon = np.arange(lon_min, lon_max, target_resolution)
        new_lat = np.arange(lat_min, lat_max, target_resolution)
        
        # 执行插值
        interpolated = data.interp(lon=new_lon, lat=new_lat, method=method)
        
        return interpolated

if __name__ == "__main__":
    # 测试代码
    preprocessor = DataPreprocessor()
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'temperature': np.random.normal(20, 5, 100) + 273.15,  # 开尔文温度
        'humidity': np.random.normal(60, 10, 100),
        'timestamp': dates
    })
    
    # 添加一些缺失值
    test_data.loc[10:15, 'temperature'] = np.nan
    test_data.loc[50:55, 'humidity'] = np.nan
    
    print("原始数据:")
    print(test_data.head())
    print(f"缺失值数量: {test_data.isnull().sum().sum()}")
    
    # 处理缺失值
    processed_data = preprocessor.handle_missing_values(test_data, method='interpolate')
    print(f"处理后缺失值数量: {processed_data.isnull().sum().sum()}")
    
    # 温度单位转换
    processed_data['temperature'] = preprocessor.convert_temperature_units(
        processed_data['temperature'], 
        from_unit='kelvin', 
        to_unit='celsius'
    )
    print(f"转换后温度范围: {processed_data['temperature'].min():.2f} - {processed_data['temperature'].max():.2f}")
    
    # 异常值检测
    outliers = preprocessor.detect_outliers(processed_data['temperature'])
    print(f"检测到{len(outliers)}个异常值")
    
    # 数据标准化
    numerical_cols = ['temperature', 'humidity']
    normalized_data = preprocessor.normalize_data(processed_data[numerical_cols])
    print("标准化后数据:")
    print(normalized_data.describe()) 