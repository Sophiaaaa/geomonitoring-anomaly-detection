# 地理监测异常检测项目

## 项目概述
本项目旨在利用多源遥感数据进行地理监测异常检测，主要包括：
- MODIS地表温度数据的时间序列分析
- 超高分辨率地质灾害遥感图像处理
- 实地监测数据融合
- 基于STL分解的异常检测

## 数据来源
1. **MODIS地表温度数据** - 来自Google Earth Engine平台
2. **超高分辨率遥感图像** - 0.59m分辨率地质灾害影像
3. **实地监测数据** - 传感器读数等现场观测数据
4. **外部数据集** - 行政区划边界等辅助数据

## 项目结构
```
geomonitoring-anomaly-detection/
├── data/                       # 原始数据与预处理
├── notebooks/                  # 探索性分析与实验
├── src/                        # 核心代码
├── outputs/                    # 结果输出
├── config/                     # 配置参数
├── environment.yml             # Python环境依赖
└── README.md                   # 项目说明
```

## 安装与使用

### 环境配置
```bash
conda env create -f environment.yml
conda activate geomonitoring
```

### 数据预处理
```bash
python src/data_loader.py
python src/preprocessing.py
```

### 异常检测
```bash
python src/anomaly_detection.py
```

## 主要技术栈
- **数据处理**: pandas, numpy, xarray
- **遥感处理**: geemap, earthengine-api
- **时间序列分析**: statsmodels
- **异常检测**: scikit-learn, isolation forest
- **可视化**: matplotlib, seaborn, plotly

## 分析流程
1. 数据加载与预处理
2. 时间序列分解（趋势、季节、残差）
3. 异常检测模型训练
4. 结果可视化与报告生成

## 参考文献
[1] 基于MODIS地表温度数据的时间序列异常检测
[2] 超高分辨率遥感图像地质灾害监测技术

## 联系方式
如有问题，请联系项目维护者。 