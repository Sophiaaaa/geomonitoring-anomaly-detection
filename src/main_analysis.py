"""
地质灾害监测异常检测主分析脚本
整合数据加载、预处理、异常检测、报告生成等所有功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from hive_data_loader import HiveDataLoader
from geological_anomaly_detector import GeologicalAnomalyDetector
from preprocessing import DataPreprocessor
from decomposition import TimeSeriesDecomposer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anomaly_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeologicalAnomalyAnalyzer:
    """地质灾害监测异常检测分析系统"""
    
    def __init__(self, 
                 hive_connection=None,
                 output_dir: str = "../outputs"):
        """
        初始化分析系统
        
        Args:
            hive_connection: Hive连接对象
            output_dir: 输出目录
        """
        self.hive_conn = hive_connection
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        
        # 初始化组件
        self.data_loader = HiveDataLoader(hive_connection)
        self.preprocessor = DataPreprocessor()
        self.detector = GeologicalAnomalyDetector()
        self.decomposer = TimeSeriesDecomposer()
        
        # 分析结果
        self.analysis_results = {}
        
    def run_complete_analysis(self,
                             start_time: Optional[str] = None,
                             end_time: Optional[str] = None,
                             monitor_points: Optional[List[str]] = None,
                             contamination: float = 0.1,
                             window_size: int = 24) -> Dict:
        """
        运行完整的异常检测分析
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            monitor_points: 监测点列表
            contamination: 异常值比例
            window_size: 时间窗口大小
            
        Returns:
            Dict: 分析结果
        """
        logger.info("开始运行完整的异常检测分析")
        
        analysis_start_time = datetime.now()
        
        try:
            # 步骤1: 数据加载
            logger.info("步骤1: 数据加载")
            all_data = self.data_loader.load_all_tables(start_time, end_time, monitor_points)
            
            if not all_data:
                raise ValueError("没有加载到任何数据")
            
            # 步骤2: 构建特征矩阵
            logger.info("步骤2: 构建特征矩阵")
            feature_matrix = self.data_loader.get_feature_matrix(all_data)
            
            if feature_matrix.empty:
                raise ValueError("特征矩阵为空")
            
            logger.info(f"特征矩阵形状: {feature_matrix.shape}")
            
            # 步骤3: 数据预处理
            logger.info("步骤3: 数据预处理")
            processed_features = self.detector.preprocess_features(feature_matrix)
            
            # 步骤4: 单变量异常检测
            logger.info("步骤4: 单变量异常检测")
            univariate_results = self.detector.detect_univariate_anomalies(processed_features)
            
            # 步骤5: 多变量异常检测
            logger.info("步骤5: 多变量异常检测")
            multivariate_results = self.detector.detect_multivariate_anomalies(
                processed_features, contamination=contamination
            )
            
            # 步骤6: 时间序列异常检测
            logger.info("步骤6: 时间序列异常检测")
            ts_results = self.detector.detect_time_series_anomalies(
                processed_features, window_size=window_size
            )
            
            # 步骤7: 计算异常分数
            logger.info("步骤7: 计算异常分数")
            results_with_scores = self.detector.calculate_anomaly_scores(
                processed_features, multivariate_results
            )
            
            # 步骤8: 分析特征重要性
            logger.info("步骤8: 分析特征重要性")
            feature_importance = self.detector.analyze_feature_importance(results_with_scores)
            
            # 步骤9: 生成报告
            logger.info("步骤9: 生成报告")
            report = self.detector.generate_anomaly_report(processed_features, multivariate_results)
            
            # 步骤10: 时间序列分解分析
            logger.info("步骤10: 时间序列分解分析")
            decomposition_results = self._analyze_time_series_decomposition(processed_features)
            
            # 整合结果
            analysis_results = {
                'data_summary': {
                    'total_records': len(feature_matrix),
                    'monitor_points': len(feature_matrix['monitor_point_code'].unique()) if 'monitor_point_code' in feature_matrix.columns else 0,
                    'features_count': len([col for col in feature_matrix.columns if col not in ['monitor_point_code', 'create_time_s']]),
                    'time_range': {
                        'start': str(feature_matrix['create_time_s'].min()) if 'create_time_s' in feature_matrix.columns else None,
                        'end': str(feature_matrix['create_time_s'].max()) if 'create_time_s' in feature_matrix.columns else None
                    }
                },
                'anomaly_detection': {
                    'univariate_results': self._summarize_univariate_results(univariate_results),
                    'multivariate_results': report,
                    'time_series_results': self._summarize_time_series_results(ts_results),
                    'feature_importance': feature_importance
                },
                'decomposition_results': decomposition_results,
                'analysis_metadata': {
                    'start_time': analysis_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': (datetime.now() - analysis_start_time).total_seconds(),
                    'parameters': {
                        'contamination': contamination,
                        'window_size': window_size
                    }
                }
            }
            
            # 保存结果
            self.analysis_results = analysis_results
            
            # 生成可视化
            self._generate_visualizations(feature_matrix, processed_features, 
                                        multivariate_results, ts_results)
            
            # 保存报告
            self._save_results(analysis_results, processed_features, multivariate_results)
            
            logger.info("完整异常检测分析完成")
            return analysis_results
            
        except Exception as e:
            logger.error(f"分析过程中发生错误: {e}")
            raise
    
    def _analyze_time_series_decomposition(self, feature_matrix: pd.DataFrame) -> Dict:
        """
        分析时间序列分解
        
        Args:
            feature_matrix: 特征矩阵
            
        Returns:
            Dict: 分解分析结果
        """
        logger.info("开始时间序列分解分析")
        
        decomposition_results = {}
        
        if 'create_time_s' not in feature_matrix.columns:
            logger.warning("缺少时间列，跳过时间序列分解")
            return decomposition_results
        
        # 获取数值特征
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['monitor_point_code', 'create_time_s']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols and not col.endswith('_anomaly') and not col.endswith('_score')]
        
        # 选择前3个特征进行分解
        selected_features = numeric_cols[:3]
        
        for feature in selected_features:
            try:
                # 选择第一个监测点的数据
                first_point = feature_matrix['monitor_point_code'].iloc[0]
                point_data = feature_matrix[feature_matrix['monitor_point_code'] == first_point].copy()
                point_data = point_data.sort_values('create_time_s')
                
                # 创建时间序列
                ts_data = pd.Series(
                    point_data[feature].values,
                    index=pd.DatetimeIndex(point_data['create_time_s'])
                )
                
                # 去除空值
                ts_data = ts_data.dropna()
                
                if len(ts_data) < 14:  # 至少需要2周的数据
                    continue
                
                # STL分解
                decomposition = self.decomposer.stl_decomposition(ts_data, seasonal=7)
                
                # 基于残差的异常检测
                residual_anomalies = self.decomposer.detect_anomalies_from_residuals(
                    decomposition['resid']
                )
                
                # 统计摘要
                summary = self.decomposer.summary_statistics(decomposition)
                
                decomposition_results[feature] = {
                    'monitor_point': first_point,
                    'seasonal_strength': summary['seasonal_strength'],
                    'trend_strength': summary['trend_strength'],
                    'residual_anomalies_count': len(residual_anomalies),
                    'residual_anomalies_rate': len(residual_anomalies) / len(ts_data)
                }
                
                # 保存分解图
                fig = self.decomposer.plot_decomposition(decomposition, 
                                                       title=f'{feature} 时间序列分解')
                fig.savefig(self.output_dir / "figures" / f"decomposition_{feature}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                logger.warning(f"特征 {feature} 分解失败: {e}")
                continue
        
        return decomposition_results
    
    def _summarize_univariate_results(self, univariate_results: pd.DataFrame) -> Dict:
        """汇总单变量异常检测结果"""
        summary = {}
        
        anomaly_cols = [col for col in univariate_results.columns if col.endswith('_anomaly')]
        
        for col in anomaly_cols:
            feature_name = col.replace('_anomaly', '')
            anomaly_count = univariate_results[col].sum()
            
            summary[feature_name] = {
                'anomaly_count': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(univariate_results))
            }
        
        return summary
    
    def _summarize_time_series_results(self, ts_results: pd.DataFrame) -> Dict:
        """汇总时间序列异常检测结果"""
        summary = {}
        
        ts_anomaly_cols = [col for col in ts_results.columns if col.endswith('_ts_anomaly')]
        
        for col in ts_anomaly_cols:
            feature_name = col.replace('_ts_anomaly', '')
            anomaly_count = ts_results[col].sum()
            
            summary[feature_name] = {
                'anomaly_count': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(ts_results))
            }
        
        return summary
    
    def _generate_visualizations(self,
                               feature_matrix: pd.DataFrame,
                               processed_features: pd.DataFrame,
                               multivariate_results: Dict[str, np.ndarray],
                               ts_results: pd.DataFrame):
        """生成可视化图表"""
        logger.info("开始生成可视化图表")
        
        try:
            # 1. 异常检测结果对比图
            fig1 = self.detector.plot_anomaly_results(
                processed_features, multivariate_results,
                save_path=self.output_dir / "figures" / "anomaly_detection_comparison.png"
            )
            plt.close(fig1)
            
            # 2. 时间序列异常检测图
            if 'create_time_s' in processed_features.columns:
                fig2 = self.detector.plot_time_series_anomalies(
                    ts_results,
                    save_path=self.output_dir / "figures" / "time_series_anomalies.png"
                )
                if fig2:
                    plt.close(fig2)
            
            # 3. 特征重要性图
            self._plot_feature_importance()
            
            # 4. 监测点异常分布图
            self._plot_monitor_point_distribution(processed_features, multivariate_results)
            
            # 5. 异常等级分布图
            self._plot_anomaly_level_distribution(processed_features)
            
        except Exception as e:
            logger.error(f"生成可视化图表时发生错误: {e}")
    
    def _plot_feature_importance(self):
        """绘制特征重要性图"""
        if not self.detector.feature_importance:
            return
        
        importance_df = pd.DataFrame(
            list(self.detector.feature_importance.items()),
            columns=['特征', '重要性']
        ).sort_values('重要性', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['特征'], importance_df['重要性'])
        plt.xlabel('重要性分数')
        plt.title('特征重要性排序')
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "feature_importance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_monitor_point_distribution(self, 
                                       feature_matrix: pd.DataFrame,
                                       multivariate_results: Dict[str, np.ndarray]):
        """绘制监测点异常分布图"""
        if 'monitor_point_code' not in feature_matrix.columns:
            return
        
        # 计算每个监测点的异常数量
        point_anomaly_counts = {}
        
        for point_code in feature_matrix['monitor_point_code'].unique():
            point_mask = feature_matrix['monitor_point_code'] == point_code
            anomaly_count = 0
            
            for method, results in multivariate_results.items():
                if method == 'ensemble':
                    anomaly_count = np.sum(results[point_mask] == -1)
                    break
            
            point_anomaly_counts[point_code] = anomaly_count
        
        # 绘制条形图
        plt.figure(figsize=(12, 6))
        points = list(point_anomaly_counts.keys())
        counts = list(point_anomaly_counts.values())
        
        plt.bar(points, counts)
        plt.xlabel('监测点')
        plt.ylabel('异常数量')
        plt.title('各监测点异常数量分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "monitor_point_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_anomaly_level_distribution(self, feature_matrix: pd.DataFrame):
        """绘制异常等级分布图"""
        if 'anomaly_level' not in feature_matrix.columns:
            return
        
        level_counts = feature_matrix['anomaly_level'].value_counts()
        
        plt.figure(figsize=(8, 6))
        colors = ['green', 'yellow', 'orange', 'red']
        level_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors)
        plt.title('异常等级分布')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "anomaly_level_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, 
                     analysis_results: Dict,
                     processed_features: pd.DataFrame,
                     multivariate_results: Dict[str, np.ndarray]):
        """保存分析结果"""
        logger.info("开始保存分析结果")
        
        # 保存JSON报告
        json_report = {
            'analysis_summary': analysis_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / "reports" / "analysis_report.json", 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)
        
        # 保存详细的CSV结果
        results_df = processed_features.copy()
        
        # 添加异常检测结果
        for method, results in multivariate_results.items():
            results_df[f'{method}_result'] = results
        
        results_df.to_csv(self.output_dir / "reports" / "detailed_results.csv", 
                         index=False, encoding='utf-8')
        
        # 生成HTML报告
        self._generate_html_report(analysis_results)
        
        logger.info(f"分析结果已保存到: {self.output_dir}")
    
    def _generate_html_report(self, analysis_results: Dict):
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>地质灾害监测异常检测报告</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .anomaly-high {{ color: red; font-weight: bold; }}
                .anomaly-medium {{ color: orange; font-weight: bold; }}
                .anomaly-low {{ color: #f0ad4e; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>地质灾害监测异常检测报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>数据概览</h2>
                <div class="metric">总记录数: {analysis_results['data_summary']['total_records']}</div>
                <div class="metric">监测点数: {analysis_results['data_summary']['monitor_points']}</div>
                <div class="metric">特征数量: {analysis_results['data_summary']['features_count']}</div>
            </div>
            
            <div class="section">
                <h2>异常检测结果</h2>
                <table>
                    <tr><th>检测方法</th><th>异常数量</th><th>异常率</th></tr>
        """
        
        for method, result in analysis_results['anomaly_detection']['multivariate_results']['by_method'].items():
            html_content += f"""
                    <tr>
                        <td>{method}</td>
                        <td>{result['anomaly_count']}</td>
                        <td>{result['anomaly_rate']:.2%}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>建议</h2>
                <ul>
        """
        
        for rec in analysis_results['anomaly_detection']['multivariate_results']['recommendations']:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>可视化图表</h2>
                <p>详细的可视化图表请查看 figures 文件夹</p>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / "reports" / "analysis_report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def get_analysis_summary(self) -> Dict:
        """获取分析摘要"""
        if not self.analysis_results:
            return {"error": "尚未运行分析"}
        
        return {
            "数据量": self.analysis_results['data_summary']['total_records'],
            "监测点数": self.analysis_results['data_summary']['monitor_points'],
            "分析时长": f"{self.analysis_results['analysis_metadata']['duration']:.2f}秒",
            "主要异常": self.analysis_results['anomaly_detection']['multivariate_results']['by_method'].get('ensemble', {}).get('anomaly_count', 0),
            "建议数量": len(self.analysis_results['anomaly_detection']['multivariate_results']['recommendations'])
        }

def main():
    """主函数"""
    logger.info("开始地质灾害监测异常检测分析")
    
    # 创建分析器实例
    analyzer = GeologicalAnomalyAnalyzer()
    
    # 运行完整分析
    try:
        results = analyzer.run_complete_analysis(
            # start_time="2024-01-01 00:00:00",
            # end_time="2024-01-31 23:59:59",
            contamination=0.1,
            window_size=24
        )
        
        # 输出摘要
        summary = analyzer.get_analysis_summary()
        print("\n" + "="*50)
        print("分析摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("="*50)
        
        print(f"\n详细结果已保存到: {analyzer.output_dir}")
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise

if __name__ == "__main__":
    main() 