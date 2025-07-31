"""
简单测试脚本
生成明确的污染数据来测试三色预警系统
"""

import numpy as np
from datetime import datetime, timedelta
import random

# 导入系统模块
from pollution_tracing_system import PollutionTracingSystem, SystemConfig
from genetic_pattern_search import GAParameters
from data_fusion import MonitoringData
from spatiotemporal_analysis import SpatialEvent


def create_test_monitoring_data():
    """创建测试监测数据，确保触发预警"""
    monitoring_data = []
    current_time = datetime.now()
    
    # 创建8个监测站
    stations = [
        {'id': 'GK001', 'location': (0, 0, 5), 'type': 'national'},
        {'id': 'GK002', 'location': (200, 100, 5), 'type': 'national'},
        {'id': 'XZ001', 'location': (100, 200, 5), 'type': 'township'},
        {'id': 'XZ002', 'location': (300, 300, 5), 'type': 'township'},
        {'id': 'WX001', 'location': (150, 150, 5), 'type': 'micro'},
        {'id': 'WX002', 'location': (250, 250, 5), 'type': 'micro'},
        {'id': 'WX003', 'location': (-100, 100, 5), 'type': 'micro'},
        {'id': 'WX004', 'location': (400, 200, 5), 'type': 'micro'}
    ]
    
    # 污染源位置
    pollution_source = (150, 200, 8)  # 在XZ001和WX001附近
    
    # 生成6个小时的数据
    for hour in range(6):
        timestamp = (current_time - timedelta(hours=5-hour)).isoformat()
        
        for station in stations:
            x, y, z = station['location']
            
            # 计算到污染源的距离
            distance = np.sqrt((x - pollution_source[0])**2 + (y - pollution_source[1])**2)
            
            # 基础浓度
            base_pm25 = 30
            base_pm10 = 50
            base_o3 = 60
            base_no2 = 25
            base_vocs = 20
            
            # 添加污染源影响
            if distance < 400:
                # 距离衰减函数
                influence = np.exp(-distance / 150)
                
                # 根据时间变化污染强度（模拟污染事件发展）
                time_factor = 1.0 + 0.5 * hour  # 污染逐渐加重
                
                # 添加污染物浓度
                base_pm25 += influence * time_factor * 120  # 确保超过黄色预警阈值75
                base_pm10 += influence * time_factor * 180  # 确保超过黄色预警阈值150
                base_no2 += influence * time_factor * 100   # 确保超过黄色预警阈值80
                base_vocs += influence * time_factor * 250  # 确保超过黄色预警阈值200
                base_o3 += influence * time_factor * 150    # 确保超过黄色预警阈值160
            
            # 添加随机噪声
            pm25 = max(0, base_pm25 + random.gauss(0, 5))
            pm10 = max(0, base_pm10 + random.gauss(0, 8))
            o3 = max(0, base_o3 + random.gauss(0, 10))
            no2 = max(0, base_no2 + random.gauss(0, 5))
            vocs = max(0, base_vocs + random.gauss(0, 15))
            
            # 气象数据
            temperature = 22.0 + random.gauss(0, 2)
            humidity = 65.0 + random.gauss(0, 5)
            wind_speed = 2.0 + random.gauss(0, 0.5)
            wind_direction = 180.0 + random.gauss(0, 30)  # 主要南风
            pressure = 1013.25 + random.gauss(0, 5)
            
            data = MonitoringData(
                timestamp=timestamp,
                station_id=station['id'],
                station_type=station['type'],
                location=station['location'],
                pm25=pm25,
                pm10=pm10,
                o3=o3,
                no2=no2,
                vocs=vocs,
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                pressure=pressure
            )
            
            monitoring_data.append(data)
    
    return monitoring_data


def create_test_spatial_events():
    """创建测试空间事件"""
    current_time = datetime.now()
    
    events = [
        SpatialEvent(
            event_id='industrial_001',
            event_type='industrial',
            location=(150, 200, 10),  # 与污染源位置一致
            start_time=(current_time - timedelta(hours=3)).isoformat(),
            end_time=None,
            intensity=8.5,
            description='化工企业异常排放',
            responsible_unit='XX化工厂'
        ),
        SpatialEvent(
            event_id='construction_001',
            event_type='construction',
            location=(180, 220, 0),  # 附近建筑工地
            start_time=(current_time - timedelta(hours=2)).isoformat(),
            end_time=None,
            intensity=6.0,
            description='建筑工地扬尘作业',
            responsible_unit='XX建筑公司'
        )
    ]
    
    return events


def run_simple_test():
    """运行简单测试"""
    print("=== 简单污染溯源测试 ===\n")
    
    # 1. 生成测试数据
    print("1. 生成测试数据...")
    monitoring_data = create_test_monitoring_data()
    spatial_events = create_test_spatial_events()
    
    print(f"   监测数据: {len(monitoring_data)} 条")
    print(f"   空间事件: {len(spatial_events)} 个")
    
    # 显示部分数据样本
    print("\n   数据样本:")
    for i, data in enumerate(monitoring_data[:3]):
        print(f"   站点{data.station_id}: PM2.5={data.pm25:.1f} PM10={data.pm10:.1f} "
              f"NO2={data.no2:.1f} VOCs={data.vocs:.1f}")
    
    # 2. 配置系统
    print("\n2. 配置系统...")
    config = SystemConfig(
        region_bounds=(-500, 500, -500, 500),
        grid_resolution=30,
        search_bounds={
            'x': (-300, 400),
            'y': (-100, 400),
            'z': (0, 20),
            'q': (0.5, 8.0)
        },
        ga_parameters=GAParameters(
            population_size=40,
            max_generations=500,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_rate=0.2
        )
    )
    
    # 3. 初始化系统
    tracing_system = PollutionTracingSystem(config)
    
    # 4. 准备数据源
    monitoring_data_sources = {'national': [], 'township': [], 'micro': []}
    
    for data in monitoring_data:
        if data.station_id.startswith('GK'):
            monitoring_data_sources['national'].append(data)
        elif data.station_id.startswith('XZ'):
            monitoring_data_sources['township'].append(data)
        elif data.station_id.startswith('WX'):
            monitoring_data_sources['micro'].append(data)
    
    print(f"   国控站数据: {len(monitoring_data_sources['national'])} 条")
    print(f"   乡镇站数据: {len(monitoring_data_sources['township'])} 条")
    print(f"   微型站数据: {len(monitoring_data_sources['micro'])} 条")
    
    # 5. 执行分析（禁用数据归一化以保持原始浓度值用于预警判断）
    print("\n3. 执行污染溯源分析...")

    # 临时修改系统配置以禁用归一化
    original_enable_normalization = tracing_system.config.enable_data_normalization
    tracing_system.config.enable_data_normalization = False

    result = tracing_system.process_real_time_data(
        monitoring_data_sources=monitoring_data_sources,
        spatial_events=spatial_events,
        verbose=True
    )

    # 恢复原始配置
    tracing_system.config.enable_data_normalization = original_enable_normalization
    
    # 6. 详细结果分析
    print(f"\n{'='*60}")
    print("详细结果分析")
    print(f"{'='*60}")
    
    print(f"总执行时间: {result.computation_time:.2f}秒")
    print(f"数据质量评分: {result.system_performance.get('data_quality_score', 0):.3f}")
    
    # 预警事件分析
    print(f"\n预警事件分析 ({len(result.warning_events)}个):")
    if result.warning_events:
        warning_stats = {}
        for warning in result.warning_events:
            level = warning.warning_level.value
            pollutant = warning.pollutant
            key = f"{pollutant}-{level}"
            warning_stats[key] = warning_stats.get(key, 0) + 1
            
            print(f"  - {warning.station_id}: {pollutant} {level}预警 "
                  f"浓度:{warning.concentration:.1f} 超标:{warning.threshold_exceeded:.2f}倍")
        
        print(f"\n预警统计:")
        for key, count in warning_stats.items():
            print(f"  {key}: {count}个")
    else:
        print("  未检测到预警事件")
    
    # 污染源反算结果
    print(f"\n污染源反算结果 ({len(result.inversion_results)}个):")
    if result.inversion_results:
        for i, source in enumerate(result.inversion_results):
            print(f"  源{i+1}: 位置({source.source_x:.1f}, {source.source_y:.1f}, {source.source_z:.1f})")
            print(f"       源强:{source.emission_rate:.3f}g/s 目标函数:{source.objective_value:.6f}")
            print(f"       计算时间:{source.computation_time:.2f}s")
            
            # 与真实污染源比较
            true_source = (150, 200, 8)
            distance_error = np.sqrt(
                (source.source_x - true_source[0])**2 + 
                (source.source_y - true_source[1])**2 + 
                (source.source_z - true_source[2])**2
            )
            print(f"       与真实源距离误差:{distance_error:.1f}m")
    else:
        print("  未识别到污染源")
    
    # 时空关联分析
    print(f"\n时空关联分析:")
    print(f"  生成热力图: {len(result.heatmap_data)}个")
    for pollutant, heatmap in result.heatmap_data.items():
        print(f"    {pollutant}: 最大浓度{heatmap.max_concentration:.1f} 热点{len(heatmap.hotspots)}个")
    
    print(f"  关联分析: {len(result.correlation_results)}个")
    high_confidence = sum(1 for r in result.correlation_results if r.confidence_level == 'high')
    medium_confidence = sum(1 for r in result.correlation_results if r.confidence_level == 'medium')
    print(f"    高置信度: {high_confidence}个")
    print(f"    中置信度: {medium_confidence}个")
    
    # 系统建议
    print(f"\n系统建议 ({len(result.recommendations)}条):")
    for i, rec in enumerate(result.recommendations):
        print(f"  {i+1}. {rec}")
    
    # 性能评估
    print(f"\n性能评估:")
    target_time = 2.44  # 论文目标时间
    target_position_error = 10  # 论文目标位置误差
    
    print(f"  响应时间: {result.computation_time:.2f}s (目标≤{target_time}s) "
          f"{'✓' if result.computation_time <= target_time else '✗'}")
    
    if result.inversion_results:
        avg_position_error = np.mean([
            np.sqrt((r.source_x - 150)**2 + (r.source_y - 200)**2 + (r.source_z - 8)**2)
            for r in result.inversion_results
        ])
        print(f"  平均位置误差: {avg_position_error:.1f}m (目标≤{target_position_error}m) "
              f"{'✓' if avg_position_error <= target_position_error else '✗'}")
    
    # 保存结果
    export_path = "溯源算法/simple_test_result.json"
    tracing_system.export_results(result, export_path)
    print(f"\n结果已保存到: {export_path}")
    
    return result


if __name__ == "__main__":
    result = run_simple_test()
    print(f"\n测试完成！")
