# 基于PyTorch LSTM的空气质量预报预警系统

## 系统概述

本系统是一个完整的空气质量预报预警解决方案，基于PyTorch深度学习框架实现，避免了TensorFlow依赖问题。系统结合气象、传输路径等多种因素，为重点点位提供精准的空气质量预报预警服务，并在污染过程中分析成因，提出针对性管控建议。

## 核心功能

### 1. 多因子空气质量预报
- **污染物浓度预报**: PM2.5, PM10, O3, NO2, SO2, CO
- **多维特征融合**: 污染物浓度 + 气象要素 + 时间特征 + 空间特征
- **时序建模**: 基于LSTM的深度时序预测
- **注意力机制**: 增强模型对关键时间步的关注
- **不确定性估计**: Monte Carlo Dropout方法

### 2. 分级预警系统
- **6级污染等级**: 优、良、轻度污染、中度污染、重度污染、严重污染
- **实时预警**: 基于国标阈值的自动预警生成
- **预警消息**: 针对不同等级的定制化预警信息
- **管控措施**: 分级别的应急响应建议

### 3. 污染成因分析
- **气象因子分析**: 风速、湿度、混合层高度等影响评估
- **排放因子分析**: 一次污染物排放变化识别
- **传输因子分析**: 风向、静稳天气对污染传输的影响
- **二次生成分析**: 臭氧等二次污染物生成机制
- **贡献度量化**: 各因子对污染过程的定量贡献评估

### 4. 管控建议生成
- **立即措施**: 基于污染等级的应急响应
- **源头管控**: 针对主要排放源的管控建议
- **气象应对**: 基于不利气象条件的应对策略
- **长期措施**: 系统性的污染防控建议

## 技术架构

### 模型结构
```
输入层 (多维时序特征)
    ↓
LSTM层 (双层，带Dropout)
    ↓
多头注意力机制
    ↓
全连接层 (3层，逐步降维)
    ↓
输出层 (预报时长维度)
```

### 特征工程
1. **污染物特征**: 6种主要污染物浓度
2. **气象特征**: 温度、湿度、气压、风速、风向、降水、能见度
3. **时间特征**: 小时、星期、月份、季节
4. **衍生特征**: 
   - 大气稳定度指数
   - 混合层高度估算
   - 风场分量 (u, v)

### 数据处理
- **标准化**: StandardScaler用于特征，MinMaxScaler用于目标
- **序列构建**: 滑动窗口方式构建训练序列
- **数据增强**: 时序统计特征提取

## 安装和使用

### 环境要求
```bash
Python >= 3.8
PyTorch >= 1.12.0
NumPy >= 1.21.0
Pandas >= 1.5.0
Scikit-learn >= 1.1.0
Matplotlib >= 3.5.0
```

### 安装依赖
```bash
pip install -r requirements.txt
```

### 快速开始

#### 1. 基本使用
```python
from air_quality_forecast_system import AirQualityForecastSystem

# 创建预报系统
forecast_system = AirQualityForecastSystem(
    sequence_length=24,  # 使用24小时历史数据
    forecast_horizon=72  # 预报未来72小时
)

# 训练模型
history = forecast_system.train_model(historical_data, target_col='PM2.5')

# 生成预报
predictions, confidence_intervals = forecast_system.predict_air_quality(
    recent_data, 'PM2.5'
)

# 生成预警
warnings = forecast_system.generate_warnings(predictions, 'PM2.5')
```

#### 2. 完整系统运行
```python
# 运行完整预报预警系统
system_output = forecast_system.run_forecast_system(
    historical_data=historical_data,
    current_data=current_data,
    target_pollutant='PM2.5'
)

# 获取预报结果
predictions = system_output['predictions']
warnings = system_output['warnings']

# 如果检测到污染，获取成因分析
if system_output['pollution_detected']:
    cause_analysis = system_output['cause_analysis']
    recommendations = system_output['control_recommendations']
```

#### 3. 运行演示
```bash
# 快速演示
python demo_forecast_system.py

# 完整测试
python test_pytorch_forecast_system.py
```

## 数据格式要求

### 输入数据格式
数据应为pandas DataFrame，包含以下列：

**必需列**:
- `datetime`: 时间戳
- `PM2.5`: PM2.5浓度 (μg/m³)
- `temperature`: 温度 (°C)
- `humidity`: 相对湿度 (%)
- `wind_speed`: 风速 (m/s)

**可选列**:
- `PM10`, `O3`, `NO2`, `SO2`, `CO`: 其他污染物浓度
- `pressure`: 气压 (hPa)
- `wind_direction`: 风向 (度)
- `precipitation`: 降水量 (mm)
- `visibility`: 能见度 (km)

### 输出结果格式
```python
{
    'forecast_time': datetime,           # 预报时间
    'target_pollutant': str,            # 目标污染物
    'predictions': list,                # 预报浓度值
    'confidence_intervals': dict,       # 置信区间
    'warnings': list,                   # 预警信息
    'max_pollution_level': int,         # 最高污染等级
    'pollution_detected': bool,         # 是否检测到污染
    'cause_analysis': dict,             # 成因分析结果
    'control_recommendations': dict     # 管控建议
}
```

## 系统优势

1. **技术先进**: 基于PyTorch的现代深度学习架构
2. **功能完整**: 预报-预警-分析-建议的闭环系统
3. **实用性强**: 针对实际业务需求设计
4. **可扩展性**: 模块化设计，易于功能扩展
5. **兼容性好**: 避免TensorFlow依赖问题
6. **性能优化**: 支持GPU加速，训练效率高

## 应用场景

- **环保部门**: 空气质量预报预警业务
- **重点企业**: 污染排放管控决策支持
- **科研院所**: 空气质量研究和分析
- **智慧城市**: 环境监测和管理系统

## 技术支持

如有技术问题或改进建议，请联系开发团队。

## 版本信息

- 当前版本: v1.0.0
- 更新日期: 2024年
- 开发框架: PyTorch
- 支持Python: 3.8+
