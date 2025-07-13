"""
Hive连接配置模块
提供多种Hive连接配置方案
"""

import os
import logging
from typing import Dict, Optional, Union
from dataclasses import dataclass
import warnings

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class HiveConnectionConfig:
    """Hive连接配置数据类"""
    host: str
    port: int = 10000
    database: str = ""
    username: Optional[str] = None
    password: Optional[str] = None
    auth_mechanism: str = "PLAIN"  # PLAIN, NOSASL, KERBEROS
    kerberos_service_name: Optional[str] = None
    configuration: Optional[Dict] = None
    
    def __post_init__(self):
        """配置验证"""
        if not self.host:
            raise ValueError("Host不能为空")
        if self.port <= 0:
            raise ValueError("Port必须大于0")

class HiveConnectionManager:
    """Hive连接管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化连接管理器
        
        Args:
            config_file: 配置文件路径 (可选)
        """
        self.config_file = config_file
        self.connection = None
        self.connection_pools = {}
        
    def create_connection_from_config(self, config: HiveConnectionConfig):
        """
        根据配置创建Hive连接
        
        Args:
            config: Hive连接配置
            
        Returns:
            Hive连接对象
        """
        try:
            # 方案1: 使用pyhive (推荐)
            connection = self._create_pyhive_connection(config)
            if connection:
                return connection
                
        except ImportError:
            logger.warning("pyhive未安装，尝试其他连接方式")
            
        try:
            # 方案2: 使用impyla
            connection = self._create_impyla_connection(config)
            if connection:
                return connection
                
        except ImportError:
            logger.warning("impyla未安装，尝试其他连接方式")
            
        try:
            # 方案3: 使用PyHive with SQLAlchemy
            connection = self._create_sqlalchemy_connection(config)
            if connection:
                return connection
                
        except ImportError:
            logger.warning("sqlalchemy未安装")
            
        # 如果都失败，返回None
        logger.error("无法创建Hive连接，请检查依赖库安装")
        return None
    
    def _create_pyhive_connection(self, config: HiveConnectionConfig):
        """使用pyhive创建连接"""
        try:
            from pyhive import hive
            import thrift
            
            logger.info(f"使用pyhive连接到 {config.host}:{config.port}")
            
            # 基本连接参数
            conn_params = {
                'host': config.host,
                'port': config.port,
                'database': config.database,
                'auth': config.auth_mechanism,
            }
            
            # 添加用户名和密码
            if config.username:
                conn_params['username'] = config.username
            if config.password:
                conn_params['password'] = config.password
                
            # Kerberos认证
            if config.auth_mechanism == 'KERBEROS':
                conn_params['kerberos_service_name'] = config.kerberos_service_name or 'hive'
                
            # 其他配置
            if config.configuration:
                conn_params['configuration'] = config.configuration
                
            connection = hive.Connection(**conn_params)
            
            # 测试连接
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            logger.info("pyhive连接成功")
            return connection
            
        except Exception as e:
            logger.error(f"pyhive连接失败: {e}")
            return None
    
    def _create_impyla_connection(self, config: HiveConnectionConfig):
        """使用impyla创建连接"""
        try:
            from impyla import dbapi
            
            logger.info(f"使用impyla连接到 {config.host}:{config.port}")
            
            connection = dbapi.connect(
                host=config.host,
                port=config.port,
                database=config.database,
                user=config.username,
                password=config.password,
                auth_mechanism=config.auth_mechanism
            )
            
            # 测试连接
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            logger.info("impyla连接成功")
            return connection
            
        except Exception as e:
            logger.error(f"impyla连接失败: {e}")
            return None
    
    def _create_sqlalchemy_connection(self, config: HiveConnectionConfig):
        """使用SQLAlchemy创建连接"""
        try:
            from sqlalchemy import create_engine
            
            logger.info(f"使用SQLAlchemy连接到 {config.host}:{config.port}")
            
            # 构建连接URL
            if config.username and config.password:
                url = f"hive://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
            else:
                url = f"hive://{config.host}:{config.port}/{config.database}"
                
            engine = create_engine(url, echo=False)
            connection = engine.connect()
            
            # 测试连接
            result = connection.execute("SELECT 1")
            result.fetchone()
            
            logger.info("SQLAlchemy连接成功")
            return connection
            
        except Exception as e:
            logger.error(f"SQLAlchemy连接失败: {e}")
            return None
    
    def create_connection_from_env(self) -> Optional[object]:
        """
        从环境变量创建连接
        
        Returns:
            Hive连接对象或None
        """
        config = HiveConnectionConfig(
            host=os.getenv('HIVE_HOST', 'localhost'),
            port=int(os.getenv('HIVE_PORT', 10000)),
            database=os.getenv('HIVE_DATABASE', 'default'),
            username=os.getenv('HIVE_USERNAME'),
            password=os.getenv('HIVE_PASSWORD'),
            auth_mechanism=os.getenv('HIVE_AUTH_MECHANISM', 'PLAIN')
        )
        
        return self.create_connection_from_config(config)
    
    def create_connection_from_dict(self, config_dict: Dict) -> Optional[object]:
        """
        从字典创建连接
        
        Args:
            config_dict: 配置字典
            
        Returns:
            Hive连接对象或None
        """
        config = HiveConnectionConfig(**config_dict)
        return self.create_connection_from_config(config)
    
    def get_connection_pool(self, pool_name: str = "default"):
        """
        获取连接池
        
        Args:
            pool_name: 连接池名称
            
        Returns:
            连接池对象
        """
        if pool_name not in self.connection_pools:
            logger.warning(f"连接池 {pool_name} 不存在")
            return None
        return self.connection_pools[pool_name]
    
    def close_connection(self, connection):
        """关闭连接"""
        if connection:
            try:
                connection.close()
                logger.info("连接已关闭")
            except Exception as e:
                logger.error(f"关闭连接失败: {e}")
    
    def test_connection(self, connection) -> bool:
        """
        测试连接是否可用
        
        Args:
            connection: 连接对象
            
        Returns:
            bool: 连接是否可用
        """
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result is not None
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False

# 预定义配置
HIVE_CONFIGS = {
    # 本地开发环境
    "local": HiveConnectionConfig(
        host="localhost",
        port=10000,
        database="default",
        username='hdfs',
        password=None,
        auth_mechanism="PLAIN"
    ),
    
    # 生产环境示例
    "production": HiveConnectionConfig(
        host="hive-cluster.example.com",
        port=10000,
        database="geological_monitoring",
        username="hdfs",
        password=None,
        auth_mechanism="PLAIN"
    ),
    
    # Kerberos认证环境
    "kerberos": HiveConnectionConfig(
        host="secure-hive.example.com",
        port=10000,
        database="geological_monitoring",
        username="hdfs",
        auth_mechanism="KERBEROS",
        kerberos_service_name="hive"
    ),
    
    # 开发测试环境
    "development": HiveConnectionConfig(
        host="dev-hive.example.com",
        port=10000,
        database="dev_geological_monitoring",
        username="hdfs",
        password=None,
        auth_mechanism="PLAIN"
    )
}

def get_hive_connection(config_name: str = None, 
                       config_dict: Dict = None,
                       use_env: bool = False) -> Optional[object]:
    """
    便捷函数获取Hive连接
    
    Args:
        config_name: 预定义配置名称
        config_dict: 配置字典
        use_env: 是否使用环境变量
        
    Returns:
        Hive连接对象或None
    """
    manager = HiveConnectionManager()
    
    if use_env:
        return manager.create_connection_from_env()
    elif config_dict:
        return manager.create_connection_from_dict(config_dict)
    elif config_name and config_name in HIVE_CONFIGS:
        return manager.create_connection_from_config(HIVE_CONFIGS[config_name])
    else:
        logger.error("必须提供配置名称、配置字典或使用环境变量")
        return None

# 使用示例
if __name__ == "__main__":
    # 示例1: 使用预定义配置
    print("=== 测试预定义配置 ===")
    conn = get_hive_connection("local")
    if conn:
        print("连接成功")
        manager = HiveConnectionManager()
        manager.close_connection(conn)
    
    # 示例2: 使用配置字典
    print("\n=== 测试配置字典 ===")
    config_dict = {
        "host": "localhost",
        "port": 10000,
        "database": "wz_data_quality_dwd",
        "username": 'hdfs',
        "password": None
    }
    conn = get_hive_connection(config_dict=config_dict)
    if conn:
        print("连接成功")
        manager = HiveConnectionManager()
        manager.close_connection(conn)
    
    # 示例3: 使用环境变量
    print("\n=== 测试环境变量 ===")
    os.environ['HIVE_HOST'] = 'localhost'
    os.environ['HIVE_PORT'] = '10000'
    conn = get_hive_connection(use_env=True)
    if conn:
        print("连接成功")
        manager = HiveConnectionManager()
        manager.close_connection(conn)
    
    print("\n配置测试完成") 