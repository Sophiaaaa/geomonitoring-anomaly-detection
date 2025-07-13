# Hive连接配置指南

## 概述

本文档详细介绍了如何配置和使用Hive连接，以便从Hive数据库加载地质灾害监测数据。系统支持多种配置方式，包括预定义配置、配置字典、环境变量等。

## 目录

1. [快速开始](#快速开始)
2. [安装依赖](#安装依赖)
3. [配置方式](#配置方式)
4. [认证方式](#认证方式)
5. [使用示例](#使用示例)
6. [错误处理](#错误处理)
7. [最佳实践](#最佳实践)
8. [常见问题](#常见问题)

## 快速开始

### 最简单的使用方式

```python
from src.hive_data_loader import HiveDataLoader

# 使用默认配置（本地Hive）
loader = HiveDataLoader(config_name="local")
data = loader.load_single_table("mud_monitor")
```

### 使用环境变量

```bash
# 设置环境变量
export HIVE_HOST=your-hive-host
export HIVE_PORT=10000
export HIVE_DATABASE=your_database
export HIVE_USERNAME=your_username
export HIVE_PASSWORD=your_password
```

```python
# 使用环境变量
loader = HiveDataLoader(use_env=True)
data = loader.load_single_table("mud_monitor")
```

## 安装依赖

### 更新Conda环境

```bash
# 更新环境
conda env update -f environment.yml
```

### 手动安装Hive客户端

```bash
# 选择以下任一方式安装

# 方式1: PyHive (推荐)
pip install pyhive[hive] thrift thrift-sasl

# 方式2: Impyla
pip install impyla

# 方式3: SQLAlchemy
pip install sqlalchemy pyhive[hive]
```

## 配置方式

### 1. 预定义配置

系统提供了4种预定义配置：

```python
# 本地开发环境
loader = HiveDataLoader(config_name="local")

# 生产环境
loader = HiveDataLoader(config_name="production")

# 开发测试环境
loader = HiveDataLoader(config_name="development")

# Kerberos认证环境
loader = HiveDataLoader(config_name="kerberos")
```

### 2. 配置字典

```python
config_dict = {
    "host": "hive.example.com",
    "port": 10000,
    "database": "geological_monitoring",
    "username": "your_username",
    "password": "your_password",
    "auth_mechanism": "PLAIN"
}

loader = HiveDataLoader(config_dict=config_dict)
```

### 3. 环境变量

```python
# 首先设置环境变量（参考 config/hive_env_template.txt）
loader = HiveDataLoader(use_env=True)
```

### 4. 直接传入连接对象

```python
from config.hive_config import get_hive_connection

# 创建连接
connection = get_hive_connection(config_name="local")

# 传入连接对象
loader = HiveDataLoader(hive_connection=connection)
```

## 认证方式

### 1. 无认证 (PLAIN)

```python
config = {
    "host": "localhost",
    "port": 10000,
    "database": "default",
    "auth_mechanism": "PLAIN"
}
```

### 2. 用户名密码认证

```python
config = {
    "host": "hive.example.com",
    "port": 10000,
    "database": "geological_monitoring",
    "username": "monitoring_user",
    "password": "secure_password",
    "auth_mechanism": "PLAIN"
}
```

### 3. Kerberos认证

```python
config = {
    "host": "secure-hive.example.com",
    "port": 10000,
    "database": "geological_monitoring",
    "username": "monitoring_user",
    "auth_mechanism": "KERBEROS",
    "kerberos_service_name": "hive"
}
```

## 使用示例

### 完整的数据加载流程

```python
from src.hive_data_loader import HiveDataLoader
from src.geological_anomaly_detector import GeologicalAnomalyDetector

# 1. 创建数据加载器
loader = HiveDataLoader(config_name="local")

# 2. 查看可用表
tables = loader.get_table_list()
print(f"可用表: {tables}")

# 3. 加载单个表
mud_data = loader.load_single_table(
    "mud_monitor",
    start_time="2024-01-01 00:00:00",
    end_time="2024-01-31 23:59:59"
)

# 4. 加载所有表
all_data = loader.load_all_tables()

# 5. 构建特征矩阵
feature_matrix = loader.get_feature_matrix(all_data)

# 6. 进行异常检测
detector = GeologicalAnomalyDetector()
anomaly_results = detector.detect_multivariate_anomalies(feature_matrix)
```

### 生产环境配置

```python
# 生产环境配置
production_config = {
    "host": "hive-cluster.company.com",
    "port": 10000,
    "database": "geological_monitoring",
    "username": "monitoring_service",
    "password": "prod_password_123",
    "auth_mechanism": "PLAIN"
}

# 创建加载器
loader = HiveDataLoader(config_dict=production_config)

# 指定监测点和时间范围
data = loader.load_single_table(
    "mud_monitor",
    start_time="2024-01-01 00:00:00",
    end_time="2024-01-31 23:59:59",
    monitor_points=["point_001", "point_002", "point_003"]
)
```

### 批量处理多个时间段

```python
from datetime import datetime, timedelta

# 按月批量处理
loader = HiveDataLoader(config_name="production")

start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

results = []
current_date = start_date

while current_date < end_date:
    month_end = current_date + timedelta(days=30)
    
    # 加载月度数据
    monthly_data = loader.load_all_tables(
        start_time=current_date.strftime('%Y-%m-%d %H:%M:%S'),
        end_time=month_end.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # 处理数据
    feature_matrix = loader.get_feature_matrix(monthly_data)
    results.append(feature_matrix)
    
    current_date = month_end
```

## 错误处理

### 常见错误及解决方案

#### 1. 连接超时

```python
# 错误信息: Connection timeout
# 解决方案: 检查网络连接和防火墙设置

# 增加连接超时时间
config = {
    "host": "hive.example.com",
    "port": 10000,
    "database": "default",
    "configuration": {
        "hive.server2.thrift.client.connect.timeout": "60000",
        "hive.server2.thrift.client.socket.timeout": "60000"
    }
}
```

#### 2. 认证失败

```python
# 错误信息: Authentication failed
# 解决方案: 检查用户名、密码和认证机制

# 确保认证信息正确
config = {
    "host": "hive.example.com",
    "port": 10000,
    "database": "geological_monitoring",
    "username": "correct_username",
    "password": "correct_password",
    "auth_mechanism": "PLAIN"
}
```

#### 3. 数据库不存在

```python
# 错误信息: Database does not exist
# 解决方案: 确认数据库名称和权限

# 使用默认数据库
config = {
    "host": "hive.example.com",
    "port": 10000,
    "database": "default",  # 使用默认数据库
    "username": "your_username",
    "password": "your_password"
}
```

### 错误处理代码示例

```python
from src.hive_data_loader import HiveDataLoader
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

try:
    # 尝试连接
    loader = HiveDataLoader(config_name="production")
    data = loader.load_single_table("mud_monitor")
    print(f"成功加载数据: {data.shape}")
    
except Exception as e:
    logging.error(f"连接失败: {e}")
    
    # 回退到模拟数据
    print("使用模拟数据...")
    loader = HiveDataLoader()  # 无连接配置
    data = loader.load_single_table("mud_monitor")
    print(f"模拟数据: {data.shape}")
```

## 最佳实践

### 1. 开发环境配置

```python
# 开发环境推荐配置
config = {
    "host": "dev-hive.company.com",
    "port": 10000,
    "database": "dev_geological_monitoring",
    "username": "dev_user",
    "password": "dev_password",
    "auth_mechanism": "PLAIN"
}

loader = HiveDataLoader(config_dict=config)
```

### 2. 生产环境配置

```python
# 生产环境使用环境变量
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 验证关键环境变量
required_vars = ['HIVE_HOST', 'HIVE_DATABASE', 'HIVE_USERNAME', 'HIVE_PASSWORD']
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"必需的环境变量 {var} 未设置")

# 使用环境变量
loader = HiveDataLoader(use_env=True)
```

### 3. 连接池管理

```python
from config.hive_config import HiveConnectionManager

# 创建连接管理器
manager = HiveConnectionManager()

# 创建连接
connection = manager.create_connection_from_env()

# 测试连接
if manager.test_connection(connection):
    print("连接有效")
    
    # 使用连接
    loader = HiveDataLoader(hive_connection=connection)
    data = loader.load_single_table("mud_monitor")
    
    # 关闭连接
    manager.close_connection(connection)
else:
    print("连接无效")
```

### 4. 安全配置

```python
# 使用Kerberos认证
kerberos_config = {
    "host": "secure-hive.company.com",
    "port": 10000,
    "database": "geological_monitoring",
    "username": "monitoring_user",
    "auth_mechanism": "KERBEROS",
    "kerberos_service_name": "hive"
}

# 不要在代码中硬编码密码
# 使用环境变量或密钥管理系统
```

### 5. 性能优化

```python
# 优化查询性能
config = {
    "host": "hive.example.com",
    "port": 10000,
    "database": "geological_monitoring",
    "username": "monitoring_user",
    "password": "password",
    "configuration": {
        "hive.exec.dynamic.partition": "true",
        "hive.exec.dynamic.partition.mode": "nonstrict",
        "hive.exec.max.dynamic.partitions": "1000"
    }
}

# 指定合适的时间范围和监测点
loader = HiveDataLoader(config_dict=config)
data = loader.load_single_table(
    "mud_monitor",
    start_time="2024-01-01 00:00:00",
    end_time="2024-01-07 23:59:59",  # 一周的数据
    monitor_points=["point_001", "point_002"]  # 特定监测点
)
```

## 常见问题

### Q1: 如何知道Hive服务器地址和端口？

**A1:** 联系系统管理员或查看Hive配置文件。默认端口通常是10000。

### Q2: 连接失败时如何调试？

**A2:** 
1. 检查网络连接：`ping hive-server-host`
2. 检查端口是否开放：`telnet hive-server-host 10000`
3. 查看日志文件：检查应用日志和Hive日志
4. 验证认证信息：确认用户名和密码正确

### Q3: 如何处理大数据量？

**A3:** 
1. 使用时间范围过滤
2. 指定特定监测点
3. 分批加载数据
4. 使用数据缓存

### Q4: 如何在无网络环境下开发？

**A4:** 
```python
# 不提供任何连接配置，系统会自动使用模拟数据
loader = HiveDataLoader()
data = loader.load_single_table("mud_monitor")
```

### Q5: 如何自定义SQL查询？

**A5:** 
```python
# 系统会自动构建SQL查询
# 可以通过参数控制查询条件
data = loader.load_single_table(
    "mud_monitor",
    start_time="2024-01-01 00:00:00",
    end_time="2024-01-31 23:59:59",
    monitor_points=["point_001", "point_002"]
)
```

## 配置文件示例

### 环境变量配置 (.env)

```bash
# 复制 config/hive_env_template.txt 为 .env
cp config/hive_env_template.txt .env

# 编辑 .env 文件
HIVE_HOST=your-hive-host
HIVE_PORT=10000
HIVE_DATABASE=geological_monitoring
HIVE_USERNAME=your_username
HIVE_PASSWORD=your_password
HIVE_AUTH_MECHANISM=PLAIN
```

### Python配置文件

```python
# config/hive_custom.py
HIVE_CONFIGS = {
    "my_config": {
        "host": "my-hive-host",
        "port": 10000,
        "database": "my_database",
        "username": "my_username",
        "password": "my_password",
        "auth_mechanism": "PLAIN"
    }
}
```

## 支持与反馈

如果您在使用过程中遇到问题，请：

1. 检查日志文件获取详细错误信息
2. 参考本文档的错误处理部分
3. 联系技术支持团队
4. 提交问题到项目仓库

---

**注意:** 请不要在代码中硬编码敏感信息（如密码），使用环境变量或安全的密钥管理系统。 