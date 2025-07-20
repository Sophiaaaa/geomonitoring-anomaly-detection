SELECT 
    id,
    sensor_code,
    device_id,
    sensor_type_code,
    value AS elevation,  -- 核心监测值：泥水面高程
    create_time_s AS monitor_time,  -- 监测时间
    storage_time_s AS storage_time,
    province_name,
    city_name,
    district_name,
    monitor_point_code,
    monitor_point_addr
FROM geology_monitor_data
WHERE sensor_type_code = '水泥位置传感器类型编码'  -- 替换实际传感器类型编码
  AND value IS NOT NULL  -- 排除空值
  AND create_time_s >= '2024-01-01'  -- 指定时间范围
  AND create_time_s <= '2025-06-30'
ORDER BY monitor_point_code, create_time_s;