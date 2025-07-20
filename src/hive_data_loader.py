import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['create_time_s'] = pd.to_datetime(df['create_time_s'])
    df = df.set_index('create_time_s')
    return df

# 数据预处理
def preprocess_data(df):
    # 处理缺失值
    df['value'] = df['value'].fillna(method='ffill').fillna(method='bfill')

    # 检测并处理异常值（使用3σ原则）
    mean = df['value'].mean()
    std = df['value'].std()
    df['value'] = np.where(np.abs(df['value'] - mean) > 3 * std, mean, df['value'])

    return df

# 时间序列分解
def decompose_time_series(df, period=24):
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(df['value'], model='multiplicative', period=period)
    result.plot()
    plt.show()
    return result

# 异常检测
def detect_anomalies(df, window=24, threshold=3):
    df['rolling_mean'] = df['value'].rolling(window=window).mean()
    df['rolling_std'] = df['value'].rolling(window=window).std()

    df['z_score'] = (df['value'] - df['rolling_mean']) / df['rolling_std']
    df['anomaly'] = np.where(np.abs(df['z_score']) > threshold, 1, 0)

    # 绘制异常点
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['value'], label='Value')
    anomalies = df[df['anomaly'] == 1]
    plt.scatter(anomalies.index, anomalies['value'], color='red', label='Anomalies')
    plt.legend()
    plt.title('Detected Anomalies')
    plt.show()

    return df

# 预测建模
def build_forecast_model(df, steps=24):
    # 训练模型
    model = SARIMAX(df['value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    results = model.fit(disp=False)

    # 进行预测
    forecast = results.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int()

    # 绘制预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-100:], df['value'][-100:], label='Observed')
    plt.plot(pred_mean.index, pred_mean, label='Forecast', color='r')
    plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title('Time Series Forecast')
    plt.show()

    return results, pred_mean

# 主分析流程
def main():
    # 加载和预处理数据
    df = load_data('mud_monitor_cement_location.csv')
    df = preprocess_data(df)

    # 时间序列分解
    decomposition = decompose_time_series(df)

    # 异常检测
    df_anomalies = detect_anomalies(df)

    # 预测建模
    model, forecast = build_forecast_model(df)

    # 输出异常点
    anomalies = df_anomalies[df_anomalies['anomaly'] == 1]
    print("Detected anomalies:")
    print(anomalies[['value', 'anomaly']])

    # 输出预测结果
    print("\nForecast results:")
    print(forecast)

if __name__ == "__main__":
    main()
"""
Hive数据加载模块
负责从Hive数据库加载7张地质灾害监测数据表
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings
import sys
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# 添加配置路径
sys.path.append(str(Path(__file__).parent.parent / 'config'))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HiveDataLoader:
    """Hive数据加载器类，专门用于地质灾害监测数据"""
    
    def __init__(self, hive_connection=None, config_name=None, config_dict=None, use_env=False):
        """
        初始化Hive数据加载器
        
        Args:
            hive_connection: Hive连接对象 (可选，如果没有则使用模拟数据)
            config_name: 预定义配置名称 (local, production, kerberos, development)
            config_dict: 配置字典
            use_env: 是否使用环境变量
        """
        self.hive_conn = hive_connection
        self.table_info = self._get_table_info()
        
        # 如果没有直接传入连接对象，尝试通过配置创建连接
        if not self.hive_conn:
            self.hive_conn = self._create_hive_connection(config_name, config_dict, use_env)
    
    def _create_hive_connection(self, config_name=None, config_dict=None, use_env=False):
        """
        创建Hive连接
        
        Args:
            config_name: 预定义配置名称
            config_dict: 配置字典
            use_env: 是否使用环境变量
            
        Returns:
            Hive连接对象或None
        """
        try:
            # 尝试导入配置模块
            from hive_config import get_hive_connection
            
            connection = get_hive_connection(
                config_name=config_name,
                config_dict=config_dict,
                use_env=use_env
            )
            
            if connection:
                logger.info("成功创建Hive连接")
                return connection
            else:
                logger.warning("创建Hive连接失败，将使用模拟数据")
                return None
                
        except ImportError:
            logger.warning("无法导入hive_config模块，将使用模拟数据")
            return None
        except Exception as e:
            logger.error(f"创建Hive连接异常: {e}")
            return None
        
    def _get_table_info(self) -> Dict:
        """获取表信息配置"""
        return {
            "mud_monitor": {
                "table_name": "dwd_mud_monitor_detail_di",
                "description": "泥水位监测表",
                "key_columns": ["value", "str_val"],
                "feature_name": "mud_level"
            },
            "angle_monitor": {
                "table_name": "dwd_angle_monitor_detail_di", 
                "description": "倾角监测表",
                "key_columns": ["value_x", "value_y", "value_z", "value_angle", "value_trend"],
                "feature_name": "angle"
            },
            "acc_monitor": {
                "table_name": "dwd_acc_monitor_detail_di",
                "description": "加速度监测表", 
                "key_columns": ["acc_x", "acc_y", "acc_z"],
                "feature_name": "acceleration"
            },
            "gnss_monitor": {
                "table_name": "dwd_gnss_monitor_detail_di",
                "description": "GNSS位移监测表",
                "key_columns": ["value_x", "value_y", "value_z"], 
                "feature_name": "displacement"
            },
            "moisture_monitor": {
                "table_name": "dwd_moisture_monitor_detail_di",
                "description": "含水率监测表",
                "key_columns": ["moisture_content_value"],
                "feature_name": "moisture"
            },
            "rainfall_monitor": {
                "table_name": "dwd_rainfall_monitor_detail_di",
                "description": "雨量监测表",
                "key_columns": ["rainfall_value", "total_value"],
                "feature_name": "rainfall"
            },
            "crack_monitor": {
                "table_name": "dwd_crack_monitor_detail_di",
                "description": "裂缝监测表",
                "key_columns": ["crack_value"],
                "feature_name": "crack"
            }
        }
    
    def load_single_table(self, 
                         table_key: str,
                         start_time: Optional[str] = None,
                         end_time: Optional[str] = None,
                         monitor_points: Optional[List[str]] = None) -> pd.DataFrame:
        """
        加载单个监测表数据
        
        Args:
            table_key: 表标识符
            start_time: 开始时间 (yyyy-mm-dd hh:mm:ss)
            end_time: 结束时间 (yyyy-mm-dd hh:mm:ss)
            monitor_points: 监测点列表
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        if table_key not in self.table_info:
            raise ValueError(f"未知的表标识符: {table_key}")
        
        table_info = self.table_info[table_key]
        table_name = table_info["table_name"]
        
        logger.info(f"正在加载表: {table_name}")
        
        # 构建SQL查询
        sql = self._build_sql_query(table_name, start_time, end_time, monitor_points)
        
        # 如果有Hive连接，执行真实查询
        if self.hive_conn:
            try:
                df = pd.read_sql(sql, self.hive_conn)
                logger.info(f"成功加载 {len(df)} 条记录")
                return df
            except Exception as e:
                logger.error(f"加载数据失败: {e}")
                return self._generate_mock_data(table_key)
        else:
            # 否则返回模拟数据
            logger.warning("未提供Hive连接，使用模拟数据")
            return self._generate_mock_data(table_key)
    
    def _build_sql_query(self, 
                        table_name: str,
                        start_time: Optional[str] = None,
                        end_time: Optional[str] = None,
                        monitor_points: Optional[List[str]] = None) -> str:
        """
        构建SQL查询语句
        
        Args:
            table_name: 表名
            start_time: 开始时间
            end_time: 结束时间
            monitor_points: 监测点列表
            
        Returns:
            str: SQL查询语句
        """
        sql = f"SELECT * FROM {table_name}"
        
        conditions = []
        
        # 时间条件
        if start_time:
            conditions.append(f"create_time_s >= '{start_time}'")
        if end_time:
            conditions.append(f"create_time_s <= '{end_time}'")
        
        # 监测点条件
        if monitor_points:
            points_str = "','".join(monitor_points)
            conditions.append(f"monitor_point_code IN ('{points_str}')")
        
        # 添加WHERE子句
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        # 按时间排序
        sql += " ORDER BY create_time_s"
        
        return sql
    
    def _generate_mock_data(self, table_key: str) -> pd.DataFrame:
        """
        生成模拟数据用于测试
        
        Args:
            table_key: 表标识符
            
        Returns:
            pd.DataFrame: 模拟数据
        """
        np.random.seed(42)
        n_records = 1000
        
        # 生成时间序列
        start_time = datetime.now() - timedelta(days=30)
        time_series = pd.date_range(start_time, periods=n_records, freq='1H')
        
        # 基础数据
        base_data = {
            'id': [f'id_{i}' for i in range(n_records)],
            'sensor_code': [f'sensor_{i%10}' for i in range(n_records)],
            'device_id': [f'device_{i%5}' for i in range(n_records)],
            'sensor_type_code': [f'type_{i%3}' for i in range(n_records)],
            'monitor_point_code': [f'point_{i%20}' for i in range(n_records)],
            'monitor_point_addr': [f'地址_{i%20}' for i in range(n_records)],
            'create_time_s': time_series.strftime('%Y-%m-%d %H:%M:%S'),
            'create_time': [int(t.timestamp()) for t in time_series]
        }
        
        # 根据表类型添加特定数据
        if table_key == "mud_monitor":
            values = np.random.normal(10, 2, n_records)
            base_data.update({
                'value': values,
                'str_val': [f'{v:.2f}' for v in values]
            })
            
        elif table_key == "angle_monitor":
            base_data.update({
                'value_x': np.random.normal(0, 5, n_records),
                'value_y': np.random.normal(0, 5, n_records), 
                'value_z': np.random.normal(0, 5, n_records),
                'value_angle': np.random.uniform(0, 360, n_records),
                'value_trend': np.random.uniform(0, 360, n_records)
            })
            
        elif table_key == "acc_monitor":
            base_data.update({
                'acc_x': np.random.normal(0, 1, n_records),
                'acc_y': np.random.normal(0, 1, n_records),
                'acc_z': np.random.normal(9.8, 0.5, n_records)
            })
            
        elif table_key == "gnss_monitor":
            base_data.update({
                'value_x': np.cumsum(np.random.normal(0, 0.1, n_records)),
                'value_y': np.cumsum(np.random.normal(0, 0.1, n_records)),
                'value_z': np.cumsum(np.random.normal(0, 0.05, n_records))
            })
            
        elif table_key == "moisture_monitor":
            base_data.update({
                'moisture_content_value': np.random.uniform(10, 60, n_records),
                'sensor_level': [f'level_{i%5}' for i in range(n_records)]
            })
            
        elif table_key == "rainfall_monitor":
            rainfall_values = np.random.exponential(2, n_records)
            base_data.update({
                'rainfall_value': rainfall_values,
                'total_value': np.cumsum(rainfall_values)
            })
            
        elif table_key == "crack_monitor":
            base_data.update({
                'crack_value': np.cumsum(np.random.normal(0, 0.01, n_records))
            })
        
        # 添加一些异常值
        if len(base_data) > 0:
            anomaly_indices = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
            
            # 根据不同表类型添加异常值
            if table_key == "mud_monitor" and 'value' in base_data:
                values_array = np.array(base_data['value'])
                for idx in anomaly_indices:
                    values_array[idx] *= 3
                base_data['value'] = values_array.tolist()
                base_data['str_val'] = [f'{v:.2f}' for v in values_array]
                    
            elif table_key == "angle_monitor":
                for idx in anomaly_indices:
                    base_data['value_angle'][idx] = np.random.uniform(300, 360)
                    
            elif table_key == "acc_monitor":
                for idx in anomaly_indices:
                    base_data['acc_x'][idx] *= 5
                    base_data['acc_y'][idx] *= 5
                    
            elif table_key == "rainfall_monitor":
                for idx in anomaly_indices:
                    base_data['rainfall_value'][idx] *= 10
        
        df = pd.DataFrame(base_data)
        df['create_time_s'] = pd.to_datetime(df['create_time_s'])
        
        return df
    
    def load_all_tables(self,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None,
                       monitor_points: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        加载所有监测表数据
        
        Args:
            start_time: 开始时间 (yyyy-mm-dd hh:mm:ss)
            end_time: 结束时间 (yyyy-mm-dd hh:mm:ss)
            monitor_points: 监测点列表
            
        Returns:
            Dict[str, pd.DataFrame]: 所有表的数据字典
        """
        logger.info("开始加载所有监测表数据")
        
        all_data = {}
        
        for table_key in self.table_info.keys():
            try:
                df = self.load_single_table(table_key, start_time, end_time, monitor_points)
                all_data[table_key] = df
                logger.info(f"成功加载表 {table_key}: {len(df)} 条记录")
            except Exception as e:
                logger.error(f"加载表 {table_key} 失败: {e}")
                continue
        
        logger.info(f"完成加载，共 {len(all_data)} 个表")
        return all_data
    
    def get_feature_matrix(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        将多表数据转换为特征矩阵
        
        Args:
            all_data: 所有表的数据字典
            
        Returns:
            pd.DataFrame: 特征矩阵
        """
        logger.info("开始构建特征矩阵")
        
        feature_dfs = []
        
        for table_key, df in all_data.items():
            if df.empty:
                continue
                
            table_info = self.table_info[table_key]
            key_columns = table_info["key_columns"]
            feature_name = table_info["feature_name"]
            
            # 选择关键特征列
            available_columns = [col for col in key_columns if col in df.columns]
            
            if not available_columns:
                logger.warning(f"表 {table_key} 没有可用的特征列")
                continue
            
            # 创建特征数据
            feature_data = df[['monitor_point_code', 'create_time_s'] + available_columns].copy()
            
            # 重命名特征列
            rename_dict = {}
            for col in available_columns:
                rename_dict[col] = f"{feature_name}_{col}"
            
            feature_data = feature_data.rename(columns=rename_dict)
            feature_dfs.append(feature_data)
        
        # 合并所有特征
        if not feature_dfs:
            logger.error("没有可用的特征数据")
            return pd.DataFrame()
        
        # 以时间和监测点为键进行合并
        result_df = feature_dfs[0]
        
        for df in feature_dfs[1:]:
            result_df = pd.merge(
                result_df, df, 
                on=['monitor_point_code', 'create_time_s'], 
                how='outer'
            )
        
        # 按时间排序
        result_df = result_df.sort_values('create_time_s').reset_index(drop=True)
        
        logger.info(f"特征矩阵构建完成: {result_df.shape}")
        return result_df
    
    def get_monitor_point_summary(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        获取监测点汇总信息
        
        Args:
            all_data: 所有表的数据字典
            
        Returns:
            pd.DataFrame: 监测点汇总信息
        """
        summary_data = []
        
        for table_key, df in all_data.items():
            if df.empty:
                continue
                
            table_info = self.table_info[table_key]
            
            point_summary = df.groupby('monitor_point_code').agg({
                'create_time_s': ['count', 'min', 'max'],
                'sensor_code': 'nunique'
            }).reset_index()
            
            point_summary.columns = ['monitor_point_code', 'record_count', 'start_time', 'end_time', 'sensor_count']
            point_summary['table_type'] = table_info['description']
            
            summary_data.append(point_summary)
        
        if summary_data:
            return pd.concat(summary_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_table_list(self) -> List[str]:
        """获取所有表的列表"""
        return list(self.table_info.keys())
    
    def get_table_description(self, table_key: str) -> str:
        """获取表描述信息"""
        if table_key in self.table_info:
            return self.table_info[table_key]["description"]
        return "未知表"

if __name__ == "__main__":
    # 测试代码
    loader = HiveDataLoader()
    
    # 测试加载单个表
    print("=== 测试加载单个表 ===")
    mud_data = loader.load_single_table("mud_monitor")
    print(f"泥水位数据: {mud_data.shape}")
    print(mud_data.head())
    
    # 测试加载所有表
    print("\n=== 测试加载所有表 ===")
    all_data = loader.load_all_tables()
    for table_key, df in all_data.items():
        print(f"{table_key}: {df.shape}")
    
    # 测试构建特征矩阵
    print("\n=== 测试构建特征矩阵 ===")
    feature_matrix = loader.get_feature_matrix(all_data)
    print(f"特征矩阵: {feature_matrix.shape}")
    print(feature_matrix.columns.tolist())
    
    # 测试监测点汇总
    print("\n=== 测试监测点汇总 ===")
    summary = loader.get_monitor_point_summary(all_data)
    print(summary.head())
    
    print("\n测试完成") 