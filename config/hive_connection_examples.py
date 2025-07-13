#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hive连接配置使用示例
演示多种配置方式的使用方法
"""

import os
import sys
from pathlib import Path

# 添加源代码路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent))

from hive_config import get_hive_connection, HiveConnectionConfig, HiveConnectionManager
from hive_data_loader import HiveDataLoader

def example_1_predefined_config():
    """示例1: 使用预定义配置"""
    print("=" * 50)
    print("示例1: 使用预定义配置")
    print("=" * 50)
    
    # 使用本地配置
    try:
        loader = HiveDataLoader(config_name="local")
        tables = loader.get_table_list()
        print(f"✓ 使用本地配置成功，可用表数量: {len(tables)}")
    except Exception as e:
        print(f"✗ 使用本地配置失败: {e}")
    
    # 使用生产环境配置（需要修改配置）
    try:
        loader = HiveDataLoader(config_name="production")
        tables = loader.get_table_list()
        print(f"✓ 使用生产环境配置成功，可用表数量: {len(tables)}")
    except Exception as e:
        print(f"✗ 使用生产环境配置失败: {e}")

def example_2_config_dict():
    """示例2: 使用配置字典"""
    print("\n" + "=" * 50)
    print("示例2: 使用配置字典")
    print("=" * 50)
    
    # 基本配置
    config_dict = {
        "host": "localhost",
        "port": 10000,
        "database": "default",
        "username": None,
        "password": None,
        "auth_mechanism": "PLAIN"
    }
    
    try:
        loader = HiveDataLoader(config_dict=config_dict)
        tables = loader.get_table_list()
        print(f"✓ 使用配置字典成功，可用表数量: {len(tables)}")
    except Exception as e:
        print(f"✗ 使用配置字典失败: {e}")
    
    # 带认证的配置
    auth_config = {
        "host": "hive.example.com",
        "port": 10000,
        "database": "geological_monitoring",
        "username": "monitoring_user",
        "password": "your_password",
        "auth_mechanism": "PLAIN"
    }
    
    try:
        loader = HiveDataLoader(config_dict=auth_config)
        tables = loader.get_table_list()
        print(f"✓ 使用认证配置成功，可用表数量: {len(tables)}")
    except Exception as e:
        print(f"✗ 使用认证配置失败: {e}")

def example_3_environment_variables():
    """示例3: 使用环境变量"""
    print("\n" + "=" * 50)
    print("示例3: 使用环境变量")
    print("=" * 50)
    
    # 设置环境变量
    os.environ['HIVE_HOST'] = 'localhost'
    os.environ['HIVE_PORT'] = '10000'
    os.environ['HIVE_DATABASE'] = 'default'
    os.environ['HIVE_USERNAME'] = ''
    os.environ['HIVE_PASSWORD'] = ''
    os.environ['HIVE_AUTH_MECHANISM'] = 'PLAIN'
    
    try:
        loader = HiveDataLoader(use_env=True)
        tables = loader.get_table_list()
        print(f"✓ 使用环境变量成功，可用表数量: {len(tables)}")
    except Exception as e:
        print(f"✗ 使用环境变量失败: {e}")

def example_4_direct_connection():
    """示例4: 直接传入连接对象"""
    print("\n" + "=" * 50)
    print("示例4: 直接传入连接对象")
    print("=" * 50)
    
    # 直接创建连接
    connection = get_hive_connection(config_name="local")
    
    if connection:
        try:
            loader = HiveDataLoader(hive_connection=connection)
            tables = loader.get_table_list()
            print(f"✓ 直接传入连接对象成功，可用表数量: {len(tables)}")
        except Exception as e:
            print(f"✗ 直接传入连接对象失败: {e}")
        finally:
            # 关闭连接
            manager = HiveConnectionManager()
            manager.close_connection(connection)
    else:
        print("✗ 创建连接失败")

def example_5_advanced_usage():
    """示例5: 高级用法"""
    print("\n" + "=" * 50)
    print("示例5: 高级用法")
    print("=" * 50)
    
    # 创建连接管理器
    manager = HiveConnectionManager()
    
    # 测试多种连接方式
    configs = [
        ("local", HiveConnectionConfig(host="localhost", port=10000, database="default")),
        ("production", HiveConnectionConfig(
            host="hive-cluster.example.com",
            port=10000,
            database="geological_monitoring",
            username="monitoring_user",
            password="your_password"
        )),
        ("kerberos", HiveConnectionConfig(
            host="secure-hive.example.com",
            port=10000,
            database="geological_monitoring",
            username="monitoring_user",
            auth_mechanism="KERBEROS"
        ))
    ]
    
    for config_name, config in configs:
        try:
            connection = manager.create_connection_from_config(config)
            if connection:
                # 测试连接
                is_valid = manager.test_connection(connection)
                print(f"✓ {config_name} 配置连接成功，连接有效: {is_valid}")
                
                # 创建数据加载器
                loader = HiveDataLoader(hive_connection=connection)
                tables = loader.get_table_list()
                print(f"  可用表数量: {len(tables)}")
                
                # 关闭连接
                manager.close_connection(connection)
            else:
                print(f"✗ {config_name} 配置连接失败")
        except Exception as e:
            print(f"✗ {config_name} 配置异常: {e}")

def example_6_data_loading():
    """示例6: 数据加载完整流程"""
    print("\n" + "=" * 50)
    print("示例6: 数据加载完整流程")
    print("=" * 50)
    
    try:
        # 创建数据加载器
        loader = HiveDataLoader(config_name="local")
        
        # 获取表列表
        tables = loader.get_table_list()
        print(f"可用表: {tables}")
        
        # 加载单个表
        print("\n加载单个表数据...")
        mud_data = loader.load_single_table("mud_monitor")
        print(f"泥水位数据: {mud_data.shape}")
        print(f"列名: {list(mud_data.columns)}")
        
        # 加载所有表
        print("\n加载所有表数据...")
        all_data = loader.load_all_tables()
        for table_key, df in all_data.items():
            print(f"{table_key}: {df.shape}")
        
        # 构建特征矩阵
        print("\n构建特征矩阵...")
        feature_matrix = loader.get_feature_matrix(all_data)
        print(f"特征矩阵: {feature_matrix.shape}")
        print(f"特征列: {[col for col in feature_matrix.columns if col not in ['monitor_point_code', 'create_time_s']]}")
        
        # 监测点汇总
        print("\n监测点汇总...")
        summary = loader.get_monitor_point_summary(all_data)
        print(f"监测点汇总: {summary.shape}")
        
        print("✓ 数据加载流程完成")
        
    except Exception as e:
        print(f"✗ 数据加载流程失败: {e}")

def example_7_error_handling():
    """示例7: 错误处理和重试"""
    print("\n" + "=" * 50)
    print("示例7: 错误处理和重试")
    print("=" * 50)
    
    # 错误配置测试
    error_configs = [
        {"host": "nonexistent.host", "port": 10000, "database": "default"},
        {"host": "localhost", "port": 9999, "database": "default"},
        {"host": "localhost", "port": 10000, "database": "nonexistent_db"}
    ]
    
    for i, config in enumerate(error_configs):
        try:
            print(f"\n测试错误配置 {i+1}...")
            loader = HiveDataLoader(config_dict=config)
            tables = loader.get_table_list()
            print(f"意外成功: {len(tables)} 个表")
        except Exception as e:
            print(f"预期错误: {e}")
    
    # 测试无连接时的行为
    print("\n测试无连接时的行为...")
    try:
        loader = HiveDataLoader()  # 不提供任何配置
        tables = loader.get_table_list()
        print(f"无连接模式: {len(tables)} 个表")
        
        # 加载模拟数据
        mud_data = loader.load_single_table("mud_monitor")
        print(f"模拟数据: {mud_data.shape}")
        
    except Exception as e:
        print(f"无连接模式失败: {e}")

def main():
    """主函数，运行所有示例"""
    print("Hive连接配置使用示例")
    print("=" * 80)
    
    # 运行所有示例
    examples = [
        example_1_predefined_config,
        example_2_config_dict,
        example_3_environment_variables,
        example_4_direct_connection,
        example_5_advanced_usage,
        example_6_data_loading,
        example_7_error_handling
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"示例运行失败: {e}")
    
    print("\n" + "=" * 80)
    print("所有示例运行完成")
    
    # 输出使用建议
    print("\n使用建议:")
    print("1. 开发环境: 使用 config_name='local' 或 use_env=True")
    print("2. 生产环境: 使用 config_dict 或修改预定义配置")
    print("3. 安全环境: 使用 Kerberos 认证")
    print("4. 无连接时: 系统会自动使用模拟数据")
    print("5. 错误处理: 检查日志获取详细错误信息")

if __name__ == "__main__":
    main() 