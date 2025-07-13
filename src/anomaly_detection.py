"""
异常检测模块
实现多种异常检测算法，包括孤立森林、LOF、DBSCAN等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """异常检测器类"""
    
    def __init__(self):
        """初始化异常检测器"""
        self.models = {}
        self.scalers = {}
        self.detection_results = {}
        
    def isolation_forest_detection(self, 
                                 data: Union[pd.DataFrame, np.ndarray],
                                 contamination: float = 0.1,
                                 n_estimators: int = 100,
                                 random_state: int = 42) -> np.ndarray:
        """
        孤立森林异常检测
        
        Args:
            data: 输入数据
            contamination: 异常值比例
            n_estimators: 树的数量
            random_state: 随机种子
            
        Returns:
            异常值标签 (1: 正常, -1: 异常)
        """
        logger.info(f"孤立森林异常检测，异常比例: {contamination}")
        
        # 初始化模型
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        # 训练和预测
        anomaly_labels = model.fit_predict(data)
        
        # 保存模型
        self.models['isolation_forest'] = model
        
        # 计算异常分数
        anomaly_scores = model.decision_function(data)
        
        # 保存结果
        self.detection_results['isolation_forest'] = {
            'labels': anomaly_labels,
            'scores': anomaly_scores,
            'anomaly_count': np.sum(anomaly_labels == -1),
            'normal_count': np.sum(anomaly_labels == 1)
        }
        
        logger.info(f"检测到{np.sum(anomaly_labels == -1)}个异常值")
        
        return anomaly_labels
    
    def local_outlier_factor_detection(self, 
                                     data: Union[pd.DataFrame, np.ndarray],
                                     n_neighbors: int = 20,
                                     contamination: float = 0.1) -> np.ndarray:
        """
        局部异常因子(LOF)检测
        
        Args:
            data: 输入数据
            n_neighbors: 邻居数量
            contamination: 异常值比例
            
        Returns:
            异常值标签 (1: 正常, -1: 异常)
        """
        logger.info(f"LOF异常检测，邻居数量: {n_neighbors}")
        
        # 初始化模型
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
        
        # 训练和预测
        anomaly_labels = model.fit_predict(data)
        
        # 保存模型
        self.models['lof'] = model
        
        # 计算异常分数
        anomaly_scores = model.negative_outlier_factor_
        
        # 保存结果
        self.detection_results['lof'] = {
            'labels': anomaly_labels,
            'scores': anomaly_scores,
            'anomaly_count': np.sum(anomaly_labels == -1),
            'normal_count': np.sum(anomaly_labels == 1)
        }
        
        logger.info(f"检测到{np.sum(anomaly_labels == -1)}个异常值")
        
        return anomaly_labels
    
    def one_class_svm_detection(self, 
                              data: Union[pd.DataFrame, np.ndarray],
                              nu: float = 0.1,
                              kernel: str = 'rbf',
                              gamma: str = 'scale') -> np.ndarray:
        """
        单类支持向量机异常检测
        
        Args:
            data: 输入数据
            nu: 异常值比例上界
            kernel: 核函数类型
            gamma: 核函数参数
            
        Returns:
            异常值标签 (1: 正常, -1: 异常)
        """
        logger.info(f"One-Class SVM异常检测，核函数: {kernel}")
        
        # 初始化模型
        model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
        
        # 训练和预测
        anomaly_labels = model.fit_predict(data)
        
        # 保存模型
        self.models['one_class_svm'] = model
        
        # 计算异常分数
        anomaly_scores = model.decision_function(data)
        
        # 保存结果
        self.detection_results['one_class_svm'] = {
            'labels': anomaly_labels,
            'scores': anomaly_scores,
            'anomaly_count': np.sum(anomaly_labels == -1),
            'normal_count': np.sum(anomaly_labels == 1)
        }
        
        logger.info(f"检测到{np.sum(anomaly_labels == -1)}个异常值")
        
        return anomaly_labels
    
    def dbscan_detection(self, 
                        data: Union[pd.DataFrame, np.ndarray],
                        eps: float = 0.5,
                        min_samples: int = 5) -> np.ndarray:
        """
        DBSCAN聚类异常检测
        
        Args:
            data: 输入数据
            eps: 邻域半径
            min_samples: 核心点最小样本数
            
        Returns:
            异常值标签 (1: 正常, -1: 异常)
        """
        logger.info(f"DBSCAN异常检测，eps: {eps}, min_samples: {min_samples}")
        
        # 初始化模型
        model = DBSCAN(eps=eps, min_samples=min_samples)
        
        # 训练和预测
        cluster_labels = model.fit_predict(data)
        
        # 噪声点(-1)视为异常值
        anomaly_labels = np.where(cluster_labels == -1, -1, 1)
        
        # 保存模型
        self.models['dbscan'] = model
        
        # 保存结果
        self.detection_results['dbscan'] = {
            'labels': anomaly_labels,
            'cluster_labels': cluster_labels,
            'anomaly_count': np.sum(anomaly_labels == -1),
            'normal_count': np.sum(anomaly_labels == 1),
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        }
        
        logger.info(f"检测到{np.sum(anomaly_labels == -1)}个异常值")
        logger.info(f"聚类数量: {self.detection_results['dbscan']['n_clusters']}")
        
        return anomaly_labels
    
    def statistical_detection(self, 
                            data: Union[pd.Series, np.ndarray],
                            method: str = 'iqr',
                            threshold: float = 1.5) -> np.ndarray:
        """
        统计方法异常检测
        
        Args:
            data: 输入数据
            method: 检测方法 ('iqr', 'zscore', 'modified_zscore')
            threshold: 阈值
            
        Returns:
            异常值标签 (1: 正常, -1: 异常)
        """
        logger.info(f"统计方法异常检测: {method}")
        
        if isinstance(data, pd.Series):
            data_values = data.values
        else:
            data_values = data.flatten()
        
        if method == 'iqr':
            Q1 = np.percentile(data_values, 25)
            Q3 = np.percentile(data_values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            anomalies = (data_values < lower_bound) | (data_values > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data_values - np.mean(data_values)) / np.std(data_values))
            anomalies = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = np.median(data_values)
            mad = np.median(np.abs(data_values - median))
            modified_z_scores = 0.6745 * (data_values - median) / mad
            anomalies = np.abs(modified_z_scores) > threshold
            
        else:
            raise ValueError(f"不支持的统计方法: {method}")
        
        # 转换为标签格式
        anomaly_labels = np.where(anomalies, -1, 1)
        
        # 保存结果
        self.detection_results[f'statistical_{method}'] = {
            'labels': anomaly_labels,
            'anomaly_count': np.sum(anomaly_labels == -1),
            'normal_count': np.sum(anomaly_labels == 1)
        }
        
        logger.info(f"检测到{np.sum(anomaly_labels == -1)}个异常值")
        
        return anomaly_labels
    
    def ensemble_detection(self, 
                          data: Union[pd.DataFrame, np.ndarray],
                          methods: List[str] = ['isolation_forest', 'lof', 'one_class_svm'],
                          voting: str = 'majority') -> np.ndarray:
        """
        集成异常检测
        
        Args:
            data: 输入数据
            methods: 使用的检测方法列表
            voting: 投票策略 ('majority', 'unanimous')
            
        Returns:
            异常值标签 (1: 正常, -1: 异常)
        """
        logger.info(f"集成异常检测，方法: {methods}")
        
        # 存储各方法的结果
        all_predictions = []
        
        # 执行各种检测方法
        for method in methods:
            if method == 'isolation_forest':
                labels = self.isolation_forest_detection(data)
            elif method == 'lof':
                labels = self.local_outlier_factor_detection(data)
            elif method == 'one_class_svm':
                labels = self.one_class_svm_detection(data)
            elif method == 'dbscan':
                labels = self.dbscan_detection(data)
            else:
                logger.warning(f"不支持的方法: {method}")
                continue
            
            all_predictions.append(labels)
        
        # 转换为数组
        all_predictions = np.array(all_predictions)
        
        # 投票决策
        if voting == 'majority':
            # 多数投票
            anomaly_votes = np.sum(all_predictions == -1, axis=0)
            threshold = len(methods) / 2
            ensemble_labels = np.where(anomaly_votes > threshold, -1, 1)
        elif voting == 'unanimous':
            # 全体一致
            ensemble_labels = np.where(np.all(all_predictions == -1, axis=0), -1, 1)
        else:
            raise ValueError(f"不支持的投票策略: {voting}")
        
        # 保存结果
        self.detection_results['ensemble'] = {
            'labels': ensemble_labels,
            'individual_results': all_predictions,
            'anomaly_count': np.sum(ensemble_labels == -1),
            'normal_count': np.sum(ensemble_labels == 1)
        }
        
        logger.info(f"集成检测到{np.sum(ensemble_labels == -1)}个异常值")
        
        return ensemble_labels
    
    def evaluate_detection(self, 
                          predictions: np.ndarray,
                          true_labels: np.ndarray) -> Dict:
        """
        评估异常检测结果
        
        Args:
            predictions: 预测标签
            true_labels: 真实标签
            
        Returns:
            评估指标
        """
        logger.info("评估异常检测结果")
        
        # 转换标签格式 (假设真实标签中1表示异常，0表示正常)
        y_true = np.where(true_labels == 1, -1, 1)
        y_pred = predictions
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
        
        # 计算各项指标
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        evaluation_results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        logger.info(f"准确率: {accuracy:.4f}")
        logger.info(f"精确率: {precision:.4f}")
        logger.info(f"召回率: {recall:.4f}")
        logger.info(f"F1分数: {f1_score:.4f}")
        
        return evaluation_results
    
    def plot_anomaly_detection_results(self, 
                                     data: Union[pd.DataFrame, np.ndarray],
                                     anomaly_labels: np.ndarray,
                                     method_name: str = "异常检测",
                                     feature_names: Optional[List[str]] = None) -> plt.Figure:
        """
        绘制异常检测结果
        
        Args:
            data: 原始数据
            anomaly_labels: 异常标签
            method_name: 方法名称
            feature_names: 特征名称
            
        Returns:
            matplotlib图形对象
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            if feature_names is None:
                feature_names = data.columns.tolist()
        else:
            data_array = data
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(data_array.shape[1])]
        
        # 如果数据维度大于2，使用PCA降维
        if data_array.shape[1] > 2:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data_array)
            feature_names = ['PC1', 'PC2']
        else:
            data_2d = data_array
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{method_name}结果', fontsize=16)
        
        # 散点图
        normal_mask = anomaly_labels == 1
        anomaly_mask = anomaly_labels == -1
        
        axes[0].scatter(data_2d[normal_mask, 0], data_2d[normal_mask, 1], 
                       c='blue', label='正常', alpha=0.6, s=20)
        axes[0].scatter(data_2d[anomaly_mask, 0], data_2d[anomaly_mask, 1], 
                       c='red', label='异常', alpha=0.8, s=30)
        axes[0].set_xlabel(feature_names[0])
        axes[0].set_ylabel(feature_names[1])
        axes[0].set_title('异常检测结果')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 异常分数分布（如果有的话）
        if hasattr(self, 'detection_results') and method_name.lower() in self.detection_results:
            result = self.detection_results[method_name.lower()]
            if 'scores' in result:
                scores = result['scores']
                axes[1].hist(scores[normal_mask], bins=30, alpha=0.7, label='正常', color='blue')
                axes[1].hist(scores[anomaly_mask], bins=30, alpha=0.7, label='异常', color='red')
                axes[1].set_xlabel('异常分数')
                axes[1].set_ylabel('频数')
                axes[1].set_title('异常分数分布')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, '无异常分数信息', ha='center', va='center', 
                            transform=axes[1].transAxes, fontsize=12)
        else:
            axes[1].text(0.5, 0.5, '无异常分数信息', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, 
                       data: Union[pd.DataFrame, np.ndarray],
                       methods: List[str] = ['isolation_forest', 'lof', 'one_class_svm']) -> plt.Figure:
        """
        比较不同异常检测方法的结果
        
        Args:
            data: 输入数据
            methods: 要比较的方法列表
            
        Returns:
            matplotlib图形对象
        """
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        fig.suptitle('异常检测方法比较', fontsize=16)
        
        if n_methods == 1:
            axes = [axes]
        
        # 数据降维（如果需要）
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
        else:
            data_2d = data
        
        for i, method in enumerate(methods):
            if method in self.detection_results:
                labels = self.detection_results[method]['labels']
                
                normal_mask = labels == 1
                anomaly_mask = labels == -1
                
                axes[i].scatter(data_2d[normal_mask, 0], data_2d[normal_mask, 1], 
                               c='blue', label='正常', alpha=0.6, s=20)
                axes[i].scatter(data_2d[anomaly_mask, 0], data_2d[anomaly_mask, 1], 
                               c='red', label='异常', alpha=0.8, s=30)
                axes[i].set_title(f'{method}\n异常数量: {np.sum(labels == -1)}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f'{method}\n未执行', ha='center', va='center', 
                            transform=axes[i].transAxes)
        
        plt.tight_layout()
        return fig
    
    def get_detection_summary(self) -> pd.DataFrame:
        """
        获取所有检测结果的摘要
        
        Returns:
            检测结果摘要DataFrame
        """
        summary_data = []
        
        for method, result in self.detection_results.items():
            summary_data.append({
                '方法': method,
                '异常数量': result['anomaly_count'],
                '正常数量': result['normal_count'],
                '异常比例': result['anomaly_count'] / (result['anomaly_count'] + result['normal_count'])
            })
        
        return pd.DataFrame(summary_data)

if __name__ == "__main__":
    # 测试代码
    detector = AnomalyDetector()
    
    # 生成测试数据
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (800, 2))
    anomaly_data = np.random.normal(3, 0.5, (200, 2))
    
    # 合并数据
    X = np.vstack([normal_data, anomaly_data])
    y_true = np.hstack([np.zeros(800), np.ones(200)])  # 0: 正常, 1: 异常
    
    print("测试数据:")
    print(f"数据形状: {X.shape}")
    print(f"异常值数量: {np.sum(y_true == 1)}")
    print(f"正常值数量: {np.sum(y_true == 0)}")
    
    # 测试各种检测方法
    print("\n=== 孤立森林检测 ===")
    if_labels = detector.isolation_forest_detection(X, contamination=0.2)
    
    print("\n=== LOF检测 ===")
    lof_labels = detector.local_outlier_factor_detection(X, contamination=0.2)
    
    print("\n=== One-Class SVM检测 ===")
    svm_labels = detector.one_class_svm_detection(X, nu=0.2)
    
    print("\n=== DBSCAN检测 ===")
    dbscan_labels = detector.dbscan_detection(X, eps=0.5, min_samples=5)
    
    print("\n=== 集成检测 ===")
    ensemble_labels = detector.ensemble_detection(X, methods=['isolation_forest', 'lof', 'one_class_svm'])
    
    # 评估结果
    print("\n=== 评估结果 ===")
    if_eval = detector.evaluate_detection(if_labels, y_true)
    print(f"孤立森林 F1分数: {if_eval['f1_score']:.4f}")
    
    lof_eval = detector.evaluate_detection(lof_labels, y_true)
    print(f"LOF F1分数: {lof_eval['f1_score']:.4f}")
    
    svm_eval = detector.evaluate_detection(svm_labels, y_true)
    print(f"One-Class SVM F1分数: {svm_eval['f1_score']:.4f}")
    
    ensemble_eval = detector.evaluate_detection(ensemble_labels, y_true)
    print(f"集成方法 F1分数: {ensemble_eval['f1_score']:.4f}")
    
    # 获取检测摘要
    summary = detector.get_detection_summary()
    print("\n=== 检测摘要 ===")
    print(summary)
    
    # 绘制结果
    fig = detector.plot_anomaly_detection_results(X, if_labels, "孤立森林")
    # plt.show()  # 如果需要显示图形，取消注释
    
    print("\n异常检测测试完成") 