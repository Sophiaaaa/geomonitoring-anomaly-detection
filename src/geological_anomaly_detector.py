"""
地质灾害监测多维异常检测系统
专门针对泥水位、倾角、加速度、GNSS、含水率、雨量、裂缝等7类监测数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeologicalAnomalyDetector:
    """地质灾害监测多维异常检测器"""
    
    def __init__(self):
        """初始化异常检测器"""
        self.scalers = {}
        self.models = {}
        self.thresholds = {}
        self.detection_results = {}
        self.feature_importance = {}
        
        # 各监测类型的正常范围 (根据实际业务调整)
        self.normal_ranges = {
            'mud_level': {'min': 0, 'max': 50},
            'angle': {'min': -180, 'max': 180},
            'acceleration': {'min': -20, 'max': 20},
            'displacement': {'min': -1, 'max': 1},
            'moisture': {'min': 0, 'max': 100},
            'rainfall': {'min': 0, 'max': 200},
            'crack': {'min': 0, 'max': 10}
        }
        
        # 异常等级定义
        self.anomaly_levels = {
            'low': {'threshold': 0.3, 'color': 'yellow', 'description': '轻微异常'},
            'medium': {'threshold': 0.6, 'color': 'orange', 'description': '中等异常'},
            'high': {'threshold': 0.8, 'color': 'red', 'description': '严重异常'}
        }
    
    def preprocess_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        预处理特征矩阵
        
        Args:
            feature_matrix: 特征矩阵
            
        Returns:
            pd.DataFrame: 预处理后的特征矩阵
        """
        logger.info("开始预处理特征矩阵")
        
        df = feature_matrix.copy()
        
        # 处理时间列
        if 'create_time_s' in df.columns:
            df['create_time_s'] = pd.to_datetime(df['create_time_s'])
            df = df.sort_values('create_time_s')
        
        # 获取数值特征列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除非特征列
        exclude_cols = ['monitor_point_code', 'create_time_s']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not numeric_cols:
            logger.error("没有找到数值特征列")
            return df
        
        # 处理缺失值
        for col in numeric_cols:
            # 使用中位数填充缺失值
            df[col] = df[col].fillna(df[col].median())
        
        # 处理无穷值
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 标准化特征
        scaler = RobustScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.scalers['feature_scaler'] = scaler
        
        logger.info(f"特征预处理完成，数值特征数量: {len(numeric_cols)}")
        return df
    
    def detect_univariate_anomalies(self, 
                                   feature_matrix: pd.DataFrame,
                                   method: str = 'iqr') -> pd.DataFrame:
        """
        单变量异常检测
        
        Args:
            feature_matrix: 特征矩阵
            method: 检测方法 ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            pd.DataFrame: 包含异常标记的数据
        """
        logger.info(f"开始单变量异常检测: {method}")
        
        df = feature_matrix.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['monitor_point_code', 'create_time_s']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 为每个特征检测异常
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomalies = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                anomalies = z_scores > 3
                
            elif method == 'isolation_forest':
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomalies = iso_forest.fit_predict(df[[col]].dropna()) == -1
                
            df[f'{col}_anomaly'] = anomalies
        
        logger.info("单变量异常检测完成")
        return df
    
    def detect_multivariate_anomalies(self, 
                                     feature_matrix: pd.DataFrame,
                                     contamination: float = 0.1) -> Dict[str, np.ndarray]:
        """
        多变量异常检测
        
        Args:
            feature_matrix: 特征矩阵
            contamination: 异常值比例
            
        Returns:
            Dict[str, np.ndarray]: 不同方法的异常检测结果
        """
        logger.info("开始多变量异常检测")
        
        # 准备数据
        df = feature_matrix.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['monitor_point_code', 'create_time_s']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols and not col.endswith('_anomaly')]
        
        if len(numeric_cols) < 2:
            logger.warning("特征数量不足以进行多变量异常检测")
            return {}
        
        # 准备特征矩阵
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        
        results = {}
        
        # 1. 孤立森林
        logger.info("执行孤立森林异常检测")
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_anomalies = iso_forest.fit_predict(X)
        results['isolation_forest'] = iso_anomalies
        self.models['isolation_forest'] = iso_forest
        
        # 2. 局部异常因子
        logger.info("执行LOF异常检测")
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        lof_anomalies = lof.fit_predict(X)
        results['lof'] = lof_anomalies
        self.models['lof'] = lof
        
        # 3. 椭圆包络
        logger.info("执行椭圆包络异常检测")
        elliptic_env = EllipticEnvelope(contamination=contamination, random_state=42)
        elliptic_anomalies = elliptic_env.fit_predict(X)
        results['elliptic_envelope'] = elliptic_anomalies
        self.models['elliptic_envelope'] = elliptic_env
        
        # 4. DBSCAN聚类
        logger.info("执行DBSCAN异常检测")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(X)
        dbscan_anomalies = np.where(cluster_labels == -1, -1, 1)
        results['dbscan'] = dbscan_anomalies
        self.models['dbscan'] = dbscan
        
        # 5. 集成方法
        logger.info("执行集成异常检测")
        ensemble_votes = np.array([results['isolation_forest'], results['lof'], results['elliptic_envelope']])
        ensemble_anomalies = np.where(np.sum(ensemble_votes == -1, axis=0) >= 2, -1, 1)
        results['ensemble'] = ensemble_anomalies
        
        self.detection_results = results
        logger.info("多变量异常检测完成")
        return results
    
    def detect_time_series_anomalies(self, 
                                    feature_matrix: pd.DataFrame,
                                    window_size: int = 24) -> pd.DataFrame:
        """
        时间序列异常检测
        
        Args:
            feature_matrix: 特征矩阵
            window_size: 滑动窗口大小
            
        Returns:
            pd.DataFrame: 包含时间序列异常标记的数据
        """
        logger.info("开始时间序列异常检测")
        
        df = feature_matrix.copy()
        
        if 'create_time_s' not in df.columns:
            logger.error("缺少时间列，无法进行时间序列异常检测")
            return df
        
        # 按监测点分组处理
        for point_code in df['monitor_point_code'].unique():
            point_data = df[df['monitor_point_code'] == point_code].copy()
            point_data = point_data.sort_values('create_time_s')
            
            numeric_cols = point_data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['monitor_point_code', 'create_time_s']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols and not col.endswith('_anomaly')]
            
            for col in numeric_cols:
                # 计算滑动窗口统计量
                point_data[f'{col}_rolling_mean'] = point_data[col].rolling(window=window_size).mean()
                point_data[f'{col}_rolling_std'] = point_data[col].rolling(window=window_size).std()
                
                # 检测异常 (超过3倍标准差)
                upper_bound = point_data[f'{col}_rolling_mean'] + 3 * point_data[f'{col}_rolling_std']
                lower_bound = point_data[f'{col}_rolling_mean'] - 3 * point_data[f'{col}_rolling_std']
                
                ts_anomalies = (point_data[col] > upper_bound) | (point_data[col] < lower_bound)
                df.loc[df['monitor_point_code'] == point_code, f'{col}_ts_anomaly'] = ts_anomalies
        
        logger.info("时间序列异常检测完成")
        return df
    
    def calculate_anomaly_scores(self, 
                                feature_matrix: pd.DataFrame,
                                anomaly_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        计算异常分数
        
        Args:
            feature_matrix: 特征矩阵
            anomaly_results: 异常检测结果
            
        Returns:
            pd.DataFrame: 包含异常分数的数据
        """
        logger.info("开始计算异常分数")
        
        df = feature_matrix.copy()
        
        # 获取用于训练的特征列（与训练时一致）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['monitor_point_code', 'create_time_s']
        # 排除所有异常相关的列，确保与训练时特征集一致
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols 
                       and not col.endswith('_anomaly') 
                       and not col.endswith('_score')
                       and col != 'anomaly_score']
        
        # 计算每种方法的异常分数
        for method, results in anomaly_results.items():
            if method in self.models:
                model = self.models[method]
                
                # 获取异常分数
                if hasattr(model, 'decision_function'):
                    try:
                        X = df[numeric_cols].fillna(df[numeric_cols].median())
                        scores = model.decision_function(X)
                        
                        # 标准化分数到0-1范围
                        if scores.max() != scores.min():
                            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
                        else:
                            scores_normalized = np.zeros_like(scores)
                        df[f'{method}_score'] = scores_normalized
                    except Exception as e:
                        logger.warning(f"计算{method}异常分数失败: {e}")
                        # 使用二元异常结果作为分数
                        df[f'{method}_score'] = (results == -1).astype(float)
                        
                elif hasattr(model, 'negative_outlier_factor_'):
                    # LOF的情况
                    try:
                        scores = -model.negative_outlier_factor_
                        if scores.max() != scores.min():
                            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
                        else:
                            scores_normalized = np.zeros_like(scores)
                        df[f'{method}_score'] = scores_normalized
                    except Exception as e:
                        logger.warning(f"计算{method}异常分数失败: {e}")
                        # 使用二元异常结果作为分数
                        df[f'{method}_score'] = (results == -1).astype(float)
                else:
                    # 其他情况直接使用二元异常结果
                    df[f'{method}_score'] = (results == -1).astype(float)
        
        # 计算综合异常分数
        score_cols = [col for col in df.columns if col.endswith('_score')]
        if score_cols:
            df['anomaly_score'] = df[score_cols].mean(axis=1)
            
            # 分类异常等级
            df['anomaly_level'] = 'normal'
            df.loc[df['anomaly_score'] > self.anomaly_levels['low']['threshold'], 'anomaly_level'] = 'low'
            df.loc[df['anomaly_score'] > self.anomaly_levels['medium']['threshold'], 'anomaly_level'] = 'medium'
            df.loc[df['anomaly_score'] > self.anomaly_levels['high']['threshold'], 'anomaly_level'] = 'high'
        
        logger.info("异常分数计算完成")
        return df
    
    def analyze_feature_importance(self, feature_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        分析特征重要性
        
        Args:
            feature_matrix: 特征矩阵
            
        Returns:
            Dict[str, float]: 特征重要性分数
        """
        logger.info("开始分析特征重要性")
        
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['monitor_point_code', 'create_time_s']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols and not col.endswith('_anomaly') and not col.endswith('_score')]
        
        if len(numeric_cols) < 2:
            return {}
        
        # 使用方差分析特征重要性
        X = feature_matrix[numeric_cols].fillna(feature_matrix[numeric_cols].median())
        
        # 计算特征方差
        variances = X.var()
        
        # 计算特征与异常分数的相关性
        if 'anomaly_score' in feature_matrix.columns:
            correlations = X.corrwith(feature_matrix['anomaly_score']).abs()
        else:
            correlations = pd.Series(1.0, index=numeric_cols)
        
        # 综合重要性分数
        importance_scores = {}
        for col in numeric_cols:
            importance_scores[col] = (variances[col] * correlations[col])
        
        # 归一化
        max_importance = max(importance_scores.values())
        if max_importance > 0:
            importance_scores = {k: v/max_importance for k, v in importance_scores.items()}
        
        self.feature_importance = importance_scores
        logger.info("特征重要性分析完成")
        return importance_scores
    
    def generate_anomaly_report(self, 
                               feature_matrix: pd.DataFrame,
                               anomaly_results: Dict[str, np.ndarray]) -> Dict:
        """
        生成异常检测报告
        
        Args:
            feature_matrix: 特征矩阵
            anomaly_results: 异常检测结果
            
        Returns:
            Dict: 异常检测报告
        """
        logger.info("开始生成异常检测报告")
        
        report = {
            'summary': {},
            'by_method': {},
            'by_monitor_point': {},
            'by_feature': {},
            'recommendations': []
        }
        
        # 总体统计
        total_records = len(feature_matrix)
        report['summary']['total_records'] = total_records
        report['summary']['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 按方法统计
        for method, results in anomaly_results.items():
            anomaly_count = np.sum(results == -1)
            report['by_method'][method] = {
                'anomaly_count': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / total_records),
                'normal_count': int(total_records - anomaly_count)
            }
        
        # 按监测点统计
        if 'monitor_point_code' in feature_matrix.columns:
            for point_code in feature_matrix['monitor_point_code'].unique():
                point_mask = feature_matrix['monitor_point_code'] == point_code
                point_anomalies = 0
                
                for method, results in anomaly_results.items():
                    point_anomalies += np.sum(results[point_mask] == -1)
                
                report['by_monitor_point'][point_code] = {
                    'total_records': int(np.sum(point_mask)),
                    'anomaly_count': int(point_anomalies),
                    'anomaly_rate': float(point_anomalies / np.sum(point_mask)) if np.sum(point_mask) > 0 else 0
                }
        
        # 按特征统计
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['monitor_point_code', 'create_time_s']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols and not col.endswith('_anomaly') and not col.endswith('_score')]
        
        for col in numeric_cols:
            col_data = feature_matrix[col].dropna()
            if len(col_data) > 0:
                report['by_feature'][col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'outlier_count': int(len(col_data[np.abs(stats.zscore(col_data)) > 3]))
                }
        
        # 生成建议
        recommendations = []
        
        # 检查异常率
        ensemble_anomaly_rate = report['by_method'].get('ensemble', {}).get('anomaly_rate', 0)
        if ensemble_anomaly_rate > 0.2:
            recommendations.append("异常率较高，建议加强监测频率")
        
        # 检查特征重要性
        if self.feature_importance:
            top_feature = max(self.feature_importance.items(), key=lambda x: x[1])
            recommendations.append(f"重点关注特征: {top_feature[0]}")
        
        # 检查监测点
        high_anomaly_points = []
        for point_code, stats in report['by_monitor_point'].items():
            if stats['anomaly_rate'] > 0.3:
                high_anomaly_points.append(point_code)
        
        if high_anomaly_points:
            recommendations.append(f"高异常率监测点: {', '.join(high_anomaly_points[:3])}")
        
        report['recommendations'] = recommendations
        
        logger.info("异常检测报告生成完成")
        return report
    
    def plot_anomaly_results(self, 
                            feature_matrix: pd.DataFrame,
                            anomaly_results: Dict[str, np.ndarray],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制异常检测结果
        
        Args:
            feature_matrix: 特征矩阵
            anomaly_results: 异常检测结果
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('地质灾害监测异常检测结果', fontsize=16)
        
        # 准备数据
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['monitor_point_code', 'create_time_s']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols and not col.endswith('_anomaly') and not col.endswith('_score')]
        
        if len(numeric_cols) >= 2:
            X = feature_matrix[numeric_cols[:2]].fillna(feature_matrix[numeric_cols[:2]].median())
            
            # 1. 散点图 - 不同方法对比
            methods = ['isolation_forest', 'lof', 'elliptic_envelope', 'ensemble']
            for i, method in enumerate(methods):
                if method in anomaly_results:
                    ax = axes[i//2, i%2]
                    
                    normal_mask = anomaly_results[method] == 1
                    anomaly_mask = anomaly_results[method] == -1
                    
                    ax.scatter(X.iloc[normal_mask, 0], X.iloc[normal_mask, 1], 
                              c='blue', alpha=0.6, s=20, label='正常')
                    ax.scatter(X.iloc[anomaly_mask, 0], X.iloc[anomaly_mask, 1], 
                              c='red', alpha=0.8, s=30, label='异常')
                    
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel(numeric_cols[1])
                    ax.set_title(f'{method} - 异常数量: {np.sum(anomaly_mask)}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_time_series_anomalies(self, 
                                  feature_matrix: pd.DataFrame,
                                  monitor_point: str = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制时间序列异常检测结果
        
        Args:
            feature_matrix: 特征矩阵
            monitor_point: 监测点编码
            save_path: 保存路径
            
        Returns:
            plt.Figure: 图形对象
        """
        if 'create_time_s' not in feature_matrix.columns:
            logger.error("缺少时间列，无法绘制时间序列图")
            return None
        
        # 选择监测点
        if monitor_point:
            df = feature_matrix[feature_matrix['monitor_point_code'] == monitor_point].copy()
        else:
            # 选择第一个监测点
            monitor_point = feature_matrix['monitor_point_code'].iloc[0]
            df = feature_matrix[feature_matrix['monitor_point_code'] == monitor_point].copy()
        
        df = df.sort_values('create_time_s')
        
        # 获取数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['monitor_point_code', 'create_time_s']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols and not col.endswith('_anomaly') and not col.endswith('_score') and not col.endswith('_rolling_mean') and not col.endswith('_rolling_std')]
        
        # 限制特征数量
        numeric_cols = numeric_cols[:4]  # 最多显示4个特征
        
        if not numeric_cols:
            logger.error("没有可用的数值特征")
            return None
        
        # 创建子图
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(15, 4*len(numeric_cols)))
        if len(numeric_cols) == 1:
            axes = [axes]
        
        fig.suptitle(f'监测点 {monitor_point} 时间序列异常检测', fontsize=16)
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            
            # 绘制时间序列
            ax.plot(df['create_time_s'], df[col], 'b-', alpha=0.7, label='数值')
            
            # 标记异常点
            if f'{col}_ts_anomaly' in df.columns:
                anomaly_mask = df[f'{col}_ts_anomaly'] == True
                if anomaly_mask.any():
                    ax.scatter(df.loc[anomaly_mask, 'create_time_s'], 
                              df.loc[anomaly_mask, col], 
                              c='red', s=50, alpha=0.8, label='异常点')
            
            # 绘制滚动均值
            if f'{col}_rolling_mean' in df.columns:
                ax.plot(df['create_time_s'], df[f'{col}_rolling_mean'], 
                       'g--', alpha=0.7, label='滚动均值')
            
            ax.set_xlabel('时间')
            ax.set_ylabel(col)
            ax.set_title(f'{col} 时间序列')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

if __name__ == "__main__":
    # 测试代码
    from hive_data_loader import HiveDataLoader
    
    # 加载数据
    loader = HiveDataLoader()
    all_data = loader.load_all_tables()
    feature_matrix = loader.get_feature_matrix(all_data)
    
    # 初始化异常检测器
    detector = GeologicalAnomalyDetector()
    
    # 预处理特征
    processed_features = detector.preprocess_features(feature_matrix)
    
    # 多变量异常检测
    anomaly_results = detector.detect_multivariate_anomalies(processed_features)
    
    # 计算异常分数
    results_with_scores = detector.calculate_anomaly_scores(processed_features, anomaly_results)
    
    # 分析特征重要性
    importance = detector.analyze_feature_importance(results_with_scores)
    
    # 生成报告
    report = detector.generate_anomaly_report(processed_features, anomaly_results)
    
    # 输出结果
    print("=== 异常检测结果 ===")
    for method, result in report['by_method'].items():
        print(f"{method}: 异常数量={result['anomaly_count']}, 异常率={result['anomaly_rate']:.2%}")
    
    print("\n=== 特征重要性 ===")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{feature}: {score:.3f}")
    
    print("\n=== 建议 ===")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    print("\n异常检测测试完成") 