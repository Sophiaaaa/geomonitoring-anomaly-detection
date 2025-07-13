#!/bin/bash
# 快速安装脚本 - 在远程环境中安装依赖

echo "开始安装依赖包..."

# 更新pip
python -m pip install --upgrade pip

# 安装基本的科学计算包
echo "安装基本科学计算包..."
pip install numpy pandas matplotlib seaborn plotly scipy scikit-learn statsmodels

# 安装xarray和相关包
echo "安装xarray和相关包..."
pip install xarray netcdf4

# 安装其他必需包
echo "安装其他必需包..."
pip install tqdm pyyaml psutil openpyxl jupyter ipywidgets

# 安装Hive连接包
echo "安装Hive连接包..."
pip install pyhive thrift thrift-sasl impyla python-dotenv

# 安装绘图包
echo "安装绘图包..."
pip install plotly-express kaleido

# 尝试安装地理空间包（可选）
echo "尝试安装地理空间包（可选）..."
pip install geopandas folium rasterio || echo "地理空间包安装失败，可稍后手动安装"

echo "依赖安装完成！"
echo "现在可以运行: python test_system.py" 