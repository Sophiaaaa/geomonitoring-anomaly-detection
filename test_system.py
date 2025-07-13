#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
地质灾害监测异常检测系统测试脚本
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入模块
from src.hive_data_loader import HiveDataLoader
from src.geological_anomaly_detector import GeologicalAnomalyDetector
from src.main_analysis import GeologicalAnomalyAnalyzer

def test_hive_data_loader():
    """测试Hive数据加载器"""
    print("=== 测试 HiveDataLoader ===")
    
    loader = HiveDataLoader()
    
    # 测试表列表
    tables = loader.get_table_list()
    print(f"可用表数量: {len(tables)}")
    assert len(tables) == 7, "应该有7个表"
    
    # 测试单个表加载
    mud_data = loader.load_single_table("mud_monitor")
    print(f"泥水位数据形状: {mud_data.shape}")
    assert not mud_data.empty, "泥水位数据不应为空"
    
    # 测试所有表加载
    all_data = loader.load_all_tables()
    print(f"加载的表数量: {len(all_data)}")
    assert len(all_data) == 7, "应该加载7个表"
    
    # 测试特征矩阵构建
    feature_matrix = loader.get_feature_matrix(all_data)
    print(f"特征矩阵形状: {feature_matrix.shape}")
    assert not feature_matrix.empty, "特征矩阵不应为空"
    
    print("✓ HiveDataLoader 测试通过")
    return loader, all_data, feature_matrix

def test_geological_anomaly_detector(feature_matrix):
    """测试地质异常检测器"""
    print("\n=== 测试 GeologicalAnomalyDetector ===")
    
    detector = GeologicalAnomalyDetector()
    
    # 测试预处理
    processed_features = detector.preprocess_features(feature_matrix)
    print(f"预处理后形状: {processed_features.shape}")
    assert not processed_features.empty, "预处理后数据不应为空"
    
    # 测试多变量异常检测
    multivariate_results = detector.detect_multivariate_anomalies(processed_features)
    print(f"多变量异常检测方法数量: {len(multivariate_results)}")
    assert len(multivariate_results) > 0, "应该有异常检测结果"
    
    # 测试异常分数计算
    results_with_scores = detector.calculate_anomaly_scores(processed_features, multivariate_results)
    print(f"异常分数计算完成，数据形状: {results_with_scores.shape}")
    
    # 测试特征重要性
    importance = detector.analyze_feature_importance(results_with_scores)
    print(f"特征重要性分析完成，特征数量: {len(importance)}")
    
    # 测试报告生成
    report = detector.generate_anomaly_report(processed_features, multivariate_results)
    print(f"报告生成完成，包含 {len(report)} 个部分")
    
    print("✓ GeologicalAnomalyDetector 测试通过")
    return detector, processed_features, multivariate_results

def test_geological_anomaly_analyzer():
    """测试地质异常分析器"""
    print("\n=== 测试 GeologicalAnomalyAnalyzer ===")
    
    # 创建临时输出目录
    temp_output = Path("test_outputs")
    temp_output.mkdir(exist_ok=True)
    
    analyzer = GeologicalAnomalyAnalyzer(output_dir=str(temp_output))
    
    # 运行完整分析
    try:
        results = analyzer.run_complete_analysis(
            contamination=0.1,
            window_size=24
        )
        print(f"完整分析完成，结果包含 {len(results)} 个部分")
        
        # 测试分析摘要
        summary = analyzer.get_analysis_summary()
        print(f"分析摘要: {summary}")
        
        # 验证输出文件
        assert (temp_output / "reports" / "analysis_report.json").exists(), "JSON报告应该存在"
        assert (temp_output / "reports" / "analysis_report.html").exists(), "HTML报告应该存在"
        assert (temp_output / "reports" / "detailed_results.csv").exists(), "详细结果CSV应该存在"
        
        print("✓ GeologicalAnomalyAnalyzer 测试通过")
        return analyzer, results
        
    except Exception as e:
        print(f"✗ GeologicalAnomalyAnalyzer 测试失败: {e}")
        raise

def run_performance_test():
    """运行性能测试"""
    print("\n=== 性能测试 ===")
    
    import time
    
    # 测试数据加载性能
    start_time = time.time()
    loader = HiveDataLoader()
    all_data = loader.load_all_tables()
    feature_matrix = loader.get_feature_matrix(all_data)
    load_time = time.time() - start_time
    print(f"数据加载时间: {load_time:.2f}秒")
    
    # 测试异常检测性能
    start_time = time.time()
    detector = GeologicalAnomalyDetector()
    processed_features = detector.preprocess_features(feature_matrix)
    multivariate_results = detector.detect_multivariate_anomalies(processed_features)
    detect_time = time.time() - start_time
    print(f"异常检测时间: {detect_time:.2f}秒")
    
    print(f"总处理时间: {load_time + detect_time:.2f}秒")
    
    # 内存使用检查
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"内存使用: {memory_mb:.1f}MB")
    
    print("✓ 性能测试完成")

def run_integration_test():
    """运行集成测试"""
    print("\n=== 集成测试 ===")
    
    try:
        # 测试完整流程
        loader, all_data, feature_matrix = test_hive_data_loader()
        detector, processed_features, multivariate_results = test_geological_anomaly_detector(feature_matrix)
        analyzer, results = test_geological_anomaly_analyzer()
        
        # 验证结果一致性
        assert len(all_data) == 7, "应该有7个数据表"
        assert not feature_matrix.empty, "特征矩阵不应为空"
        assert len(multivariate_results) > 0, "应该有异常检测结果"
        assert 'anomaly_detection' in results, "结果应包含异常检测部分"
        
        print("✓ 集成测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始地质灾害监测异常检测系统测试")
    print("=" * 60)
    
    try:
        # 运行集成测试
        if run_integration_test():
            print("\n" + "=" * 60)
            print("✓ 所有测试通过！系统运行正常")
            
            # 运行性能测试
            run_performance_test()
            
        else:
            print("\n" + "=" * 60)
            print("✗ 测试失败！系统存在问题")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("测试完成！")

if __name__ == "__main__":
    main() 