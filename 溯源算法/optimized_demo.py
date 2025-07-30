"""
优化版污染源溯源算法演示脚本
展示性能优化效果和可视化功能
"""

import numpy as np
import time
import os
from typing import List
import matplotlib.pyplot as plt

from gaussian_plume_model import GaussianPlumeModel, PollutionSource, MeteoData
from optimized_source_inversion import OptimizedSourceInversion, OptimizedSensorData, AdaptiveGAParameters
from visualization_module import PollutionSourceVisualizer
from source_inversion import SourceInversion  # 原始版本用于对比

# 设置中文字体和警告过滤
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


def create_test_scenario() -> tuple:
    """创建测试场景"""
    
    # 真实污染源
    true_source = PollutionSource(
        x=150.0,
        y=200.0,
        z=25.0,
        emission_rate=2.5
    )
    
    # 气象条件
    meteo_data = MeteoData(
        wind_speed=3.5,
        wind_direction=225.0,  # 西南风
        temperature=20.0,
        pressure=101325.0,
        humidity=60.0,
        solar_radiation=500.0,
        cloud_cover=0.3
    )
    
    # 传感器布置（网格状布置）
    sensor_positions = []
    
    # 主网格
    for x in range(-300, 400, 100):
        for y in range(-200, 300, 100):
            sensor_positions.append((x, y, 2.0))
    
    # 在污染源附近加密
    for x in range(50, 250, 50):
        for y in range(100, 300, 50):
            sensor_positions.append((x, y, 2.0))
    
    # 创建高斯烟羽模型用于生成观测数据
    gaussian_model = GaussianPlumeModel()
    
    # 生成传感器观测数据
    sensor_data = []
    for i, (x, y, z) in enumerate(sensor_positions):
        # 计算理论浓度
        theoretical_conc = gaussian_model.calculate_concentration(
            true_source, x, y, z, meteo_data
        )
        
        # 添加观测噪声（5-10%的相对误差）
        noise_level = 0.05 + 0.05 * np.random.random()
        observed_conc = theoretical_conc * (1 + np.random.normal(0, noise_level))
        observed_conc = max(0, observed_conc)  # 确保非负
        
        # 计算测量不确定性
        uncertainty = max(0.01, theoretical_conc * noise_level)
        
        sensor = OptimizedSensorData(
            sensor_id=f"S{i+1:03d}",
            x=x,
            y=y,
            z=z,
            concentration=observed_conc,
            timestamp="2024-01-01 12:00:00",
            uncertainty=uncertainty
        )
        sensor_data.append(sensor)
    
    # 只保留有显著浓度的传感器（减少计算量）
    significant_sensors = [s for s in sensor_data if s.concentration > 0.1]
    
    print(f"创建测试场景完成:")
    print(f"  真实污染源: ({true_source.x}, {true_source.y}, {true_source.z}) m, {true_source.emission_rate} g/s")
    print(f"  气象条件: 风速{meteo_data.wind_speed}m/s, 风向{meteo_data.wind_direction}°")
    print(f"  传感器数量: {len(significant_sensors)} (总布置{len(sensor_data)}个)")
    
    return true_source, meteo_data, significant_sensors


def performance_comparison():
    """性能对比测试"""
    print("\n" + "="*60)
    print("性能对比测试")
    print("="*60)
    
    # 创建测试场景
    true_source, meteo_data, sensor_data = create_test_scenario()
    
    # 转换为原始格式的传感器数据
    from source_inversion import SensorData
    original_sensor_data = [
        SensorData(
            sensor_id=s.sensor_id,
            x=s.x,
            y=s.y,
            z=s.z,
            concentration=s.concentration,
            timestamp=s.timestamp
        ) for s in sensor_data
    ]
    
    # 测试原始算法
    print("\n1. 测试原始遗传算法...")
    original_inverter = SourceInversion()
    
    start_time = time.time()
    original_result = original_inverter.invert_source(
        original_sensor_data, meteo_data, true_source, verbose=False
    )
    original_time = time.time() - start_time
    
    print(f"原始算法结果:")
    print(f"  计算时间: {original_time:.2f}秒")
    print(f"  位置误差: {original_result.position_error:.2f}m")
    print(f"  源强误差: {original_result.emission_error:.2f}%")
    print(f"  目标函数值: {original_result.objective_value:.2e}")
    
    # 测试优化算法
    print("\n2. 测试优化版遗传算法...")
    
    # 优化参数配置（暂时禁用并行计算以避免序列化问题）
    optimized_params = AdaptiveGAParameters(
        population_size=50,
        max_generations=500,
        initial_crossover_rate=0.8,
        initial_mutation_rate=0.1,
        elite_rate=0.15,
        use_parallel=False,  # 暂时禁用并行计算
        use_cache=True,
        cache_size=10000
    )
    
    optimized_inverter = OptimizedSourceInversion(ga_parameters=optimized_params)
    
    start_time = time.time()
    optimized_result = optimized_inverter.invert_source(
        sensor_data, meteo_data, true_source, verbose=False, uncertainty_analysis=True
    )
    optimized_time = time.time() - start_time
    
    print(f"优化算法结果:")
    print(f"  计算时间: {optimized_time:.2f}秒")
    print(f"  位置误差: {optimized_result.position_error:.2f}m")
    print(f"  源强误差: {optimized_result.emission_error:.2f}%")
    print(f"  目标函数值: {optimized_result.objective_value:.2e}")
    print(f"  缓存命中率: {optimized_result.performance_metrics['cache_hit_rate']:.1f}%")
    print(f"  评估速度: {optimized_result.performance_metrics['evaluations_per_second']:.1f} 次/秒")
    
    # 性能提升分析
    print(f"\n3. 性能提升分析:")
    time_improvement = (original_time - optimized_time) / original_time * 100
    accuracy_improvement_pos = (original_result.position_error - optimized_result.position_error) / original_result.position_error * 100
    accuracy_improvement_emission = (original_result.emission_error - optimized_result.emission_error) / original_result.emission_error * 100
    
    print(f"  时间提升: {time_improvement:.1f}% ({'加速' if time_improvement > 0 else '减慢'})")
    print(f"  位置精度提升: {accuracy_improvement_pos:.1f}%")
    print(f"  源强精度提升: {accuracy_improvement_emission:.1f}%")
    
    return optimized_result, sensor_data, meteo_data, true_source


def visualization_demo(result, sensor_data, meteo_data, true_source):
    """可视化演示"""
    print("\n" + "="*60)
    print("可视化演示")
    print("="*60)
    
    # 创建可视化器
    visualizer = PollutionSourceVisualizer()
    
    # 创建输出目录
    output_dir = "溯源算法/可视化结果"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. 生成浓度场图...")
    
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
        save_path=os.path.join(output_dir, "浓度场分布.png")
    )
    
    print("\n2. 生成反算结果综合图...")
    
    # 绘制反算结果
    result_fig = visualizer.plot_inversion_results(
        result=result,
        sensor_data=sensor_data,
        meteo_data=meteo_data,
        true_source=true_source,
        save_path=os.path.join(output_dir, "反算结果综合分析.png")
    )
    
    print("\n3. 生成响应式3D交互可视化...")

    # 创建响应式3D交互图
    interactive_fig = visualizer.plot_responsive_3d_concentration(
        source=inverted_source,
        meteo_data=meteo_data,
        sensor_data=sensor_data[:20],  # 限制传感器数量以提高性能
        save_path=os.path.join(output_dir, "3D响应式交互浓度分布.html")
    )
    
    print(f"\n所有可视化结果已保存至: {output_dir}")

    # 显示图形（非阻塞模式）
    try:
        plt.show(block=False)
        print("\n💡 提示: 图形窗口已打开，您可以关闭窗口继续程序执行")
    except KeyboardInterrupt:
        print("\n⚠️  用户中断了图形显示")
    except Exception as e:
        print(f"\n⚠️  图形显示出现问题: {e}")
        print("   可视化文件已保存，请直接查看文件")


def algorithm_optimization_demo():
    """算法优化演示"""
    print("\n" + "="*60)
    print("算法优化功能演示")
    print("="*60)
    
    # 创建测试场景
    true_source, meteo_data, sensor_data = create_test_scenario()
    
    print("\n1. 启用实时可视化的优化过程...")
    
    # 配置优化参数
    params = AdaptiveGAParameters(
        population_size=40,
        max_generations=300,
        initial_crossover_rate=0.8,
        initial_mutation_rate=0.1,
        use_parallel=False,  # 暂时禁用并行计算
        use_cache=True
    )
    
    # 创建优化器
    inverter = OptimizedSourceInversion(ga_parameters=params)
    
    # 执行带可视化的优化
    result = inverter.invert_source(
        sensor_data=sensor_data,
        meteo_data=meteo_data,
        true_source=true_source,
        verbose=True,
        enable_visualization=True,  # 启用实时可视化
        uncertainty_analysis=True
    )
    
    print(f"\n2. 不确定性分析结果:")
    if result.confidence_interval:
        for param, (lower, upper) in result.confidence_interval.items():
            print(f"  {param}: [{lower:.3f}, {upper:.3f}]")
    
    return result


def main():
    """主函数"""
    print("优化版污染源溯源算法演示")
    print("="*60)
    
    try:
        # 1. 性能对比测试
        result, sensor_data, meteo_data, true_source = performance_comparison()
        
        # 2. 可视化演示
        visualization_demo(result, sensor_data, meteo_data, true_source)
        
        # 3. 算法优化演示（可选，因为会弹出实时可视化窗口）
        print("\n是否进行算法优化演示（包含实时可视化）？[y/N]: ", end="")
        choice = input().strip().lower()
        
        if choice in ['y', 'yes']:
            algorithm_optimization_demo()
        
        print("\n演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
