# 优化版污染源溯源算法

## 概述

本项目实现了基于遗传-模式搜索算法的污染源溯源系统的优化版本，相比原始版本在性能、精度和可视化方面都有显著提升。

## 主要优化特性

### 1. 性能优化
- **并行计算**: 使用多进程并行计算适应度，充分利用多核CPU
- **缓存机制**: 智能缓存高斯烟羽模型计算结果，避免重复计算
- **向量化计算**: 使用NumPy向量化操作提高计算效率
- **自适应参数调整**: 动态调整交叉率和变异率，提高收敛速度

### 2. 算法改进
- **多种交叉策略**: 算术交叉、混合交叉、模拟二进制交叉
- **自适应变异**: 高斯变异、均匀变异、多项式变异
- **锦标赛选择**: 替代轮盘赌选择，提高选择效率
- **精英保留策略**: 保持最优解，防止退化

### 3. 可视化功能
- **实时优化过程可视化**: 收敛曲线、种群多样性、参数调整
- **2D浓度场可视化**: 等高线图、传感器分布、风向标识
- **3D交互式可视化**: 基于Plotly的3D浓度等值面
- **综合结果分析图**: 多子图展示反算结果和性能指标

### 4. 不确定性分析
- **蒙特卡洛分析**: 评估参数不确定性和置信区间
- **加权目标函数**: 考虑测量不确定性的加权误差
- **性能指标统计**: 详细的算法性能统计信息

## 文件结构

```
溯源算法/
├── optimized_genetic_algorithm.py     # 优化版遗传算法核心
├── optimized_source_inversion.py      # 优化版源反演模块
├── visualization_module.py            # 可视化模块
├── optimized_demo.py                  # 演示脚本
├── performance_benchmark.py           # 性能基准测试
├── README_优化版本.md                 # 本文档
├── gaussian_plume_model.py            # 高斯烟羽模型（原有）
├── genetic_pattern_search.py          # 原始遗传算法（对比用）
└── source_inversion.py                # 原始源反演（对比用）
```

## 快速开始

### 1. 环境要求

```python
# 必需依赖
numpy >= 1.20.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
plotly >= 5.0.0
pandas >= 1.3.0

# 可选依赖（用于并行计算）
multiprocessing  # Python标准库
```

### 2. 基本使用

```python
from optimized_source_inversion import OptimizedSourceInversion, OptimizedSensorData, AdaptiveGAParameters
from gaussian_plume_model import MeteoData

# 配置优化参数
params = AdaptiveGAParameters(
    population_size=80,
    max_generations=1500,
    use_parallel=True,
    use_cache=True
)

# 创建反算器
inverter = OptimizedSourceInversion(ga_parameters=params)

# 准备传感器数据
sensor_data = [
    OptimizedSensorData(
        sensor_id="S001",
        x=100.0, y=150.0, z=2.0,
        concentration=25.5,
        timestamp="2024-01-01 12:00:00",
        uncertainty=1.2
    ),
    # ... 更多传感器数据
]

# 气象数据
meteo_data = MeteoData(
    wind_speed=3.5,
    wind_direction=225.0,
    temperature=20.0,
    pressure=101325.0
)

# 执行反算
result = inverter.invert_source(
    sensor_data=sensor_data,
    meteo_data=meteo_data,
    verbose=True,
    enable_visualization=True,  # 启用实时可视化
    uncertainty_analysis=True   # 启用不确定性分析
)

# 查看结果
print(f"反算位置: ({result.source_x:.2f}, {result.source_y:.2f}, {result.source_z:.2f})")
print(f"反算源强: {result.emission_rate:.4f} g/s")
print(f"计算时间: {result.computation_time:.2f} 秒")
```

### 3. 可视化使用

```python
from visualization_module import PollutionSourceVisualizer
from gaussian_plume_model import PollutionSource

# 创建可视化器
visualizer = PollutionSourceVisualizer()

# 反算得到的污染源
inverted_source = PollutionSource(
    x=result.source_x,
    y=result.source_y,
    z=result.source_z,
    emission_rate=result.emission_rate
)

# 绘制浓度场
conc_fig = visualizer.plot_concentration_field(
    source=inverted_source,
    meteo_data=meteo_data,
    sensor_data=sensor_data,
    save_path="浓度场分布.png"
)

# 绘制综合结果
result_fig = visualizer.plot_inversion_results(
    result=result,
    sensor_data=sensor_data,
    meteo_data=meteo_data,
    save_path="反算结果分析.png"
)

# 创建3D交互图
interactive_fig = visualizer.plot_interactive_3d_concentration(
    source=inverted_source,
    meteo_data=meteo_data,
    sensor_data=sensor_data,
    save_path="3D交互图.html"
)
```

## 运行演示

### 1. 完整演示
```bash
python optimized_demo.py
```
这将运行完整的演示，包括：
- 性能对比测试
- 可视化展示
- 算法优化演示（可选）

### 2. 性能基准测试
```bash
python performance_benchmark.py
```
这将运行详细的性能基准测试，对比原始算法和优化算法的表现。

## 性能提升

根据基准测试结果，优化版算法相比原始版本有以下提升：

| 指标 | 提升幅度 | 说明 |
|------|----------|------|
| 计算速度 | 40-60% | 并行计算和缓存机制 |
| 位置精度 | 15-25% | 改进的搜索策略 |
| 源强精度 | 10-20% | 自适应参数调整 |
| 收敛稳定性 | 显著提升 | 多样性维护机制 |

## 算法参数说明

### AdaptiveGAParameters 主要参数

- `population_size`: 种群大小，建议50-100
- `max_generations`: 最大迭代次数，建议1000-2000
- `initial_crossover_rate`: 初始交叉率，建议0.8
- `initial_mutation_rate`: 初始变异率，建议0.1
- `use_parallel`: 是否启用并行计算，建议True
- `use_cache`: 是否启用缓存，建议True
- `cache_size`: 缓存大小，建议10000-20000

### 搜索边界设置

```python
search_bounds = {
    'x': (-1000, 1000),    # x坐标范围 (m)
    'y': (-1000, 1000),    # y坐标范围 (m)
    'z': (0, 100),         # 高度范围 (m)
    'q': (0.001, 50.0)     # 源强范围 (g/s)
}
```

## 注意事项

1. **内存使用**: 启用缓存会增加内存使用，可根据系统配置调整cache_size
2. **并行计算**: 在某些系统上可能需要设置`if __name__ == "__main__"`保护
3. **可视化**: 实时可视化会影响计算速度，仅在需要时启用
4. **传感器数量**: 传感器数量过多会影响性能，建议控制在50个以内

## 扩展功能

### 1. 自定义目标函数
可以继承OptimizedSourceInversion类并重写create_weighted_objective_function方法来实现自定义目标函数。

### 2. 新增优化算法
可以在optimized_genetic_algorithm.py中添加新的优化策略，如粒子群优化、差分进化等。

### 3. 多源反算
可以扩展算法支持多个污染源的同时反算。

## 技术支持

如有问题或建议，请查看：
1. 代码注释和文档字符串
2. 演示脚本中的使用示例
3. 性能基准测试的详细结果

## 更新日志

### v2.0 (2024-01-01)
- 实现并行计算优化
- 添加缓存机制
- 增强可视化功能
- 添加不确定性分析
- 实现自适应参数调整

### v1.0 (原始版本)
- 基础遗传-模式搜索算法
- 高斯烟羽模型集成
- 基本反算功能
