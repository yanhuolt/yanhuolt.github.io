"""
空气质量预报预警系统演示脚本
快速验证基于PyTorch LSTM的预报功能
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from air_quality_forecast_system import AirQualityForecastSystem
from visualization_utils import setup_chinese_font, plot_simple_forecast

def create_demo_data():
    """创建演示数据"""
    # 生成30天的小时数据
    hours = 30 * 24
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    
    # 简单的污染物浓度模拟
    hour_of_day = [(t.hour) for t in timestamps]
    
    # PM2.5浓度：基础值 + 日变化 + 随机噪声
    pm25 = 45 + 25 * np.sin(np.array(hour_of_day) * 2 * np.pi / 24) + 15 * np.random.randn(hours)
    pm25 = np.maximum(pm25, 5)  # 确保非负
    
    # 其他污染物
    pm10 = pm25 * 1.5 + 10 * np.random.randn(hours)
    pm10 = np.maximum(pm10, 10)
    
    o3 = 80 + 30 * np.sin(np.array(hour_of_day) * 2 * np.pi / 24 - np.pi/4) + 12 * np.random.randn(hours)
    o3 = np.maximum(o3, 10)
    
    no2 = pm25 * 0.6 + 8 * np.random.randn(hours)
    no2 = np.maximum(no2, 5)
    
    so2 = pm25 * 0.2 + 5 * np.random.randn(hours)
    so2 = np.maximum(so2, 2)
    
    co = pm25 * 0.03 + 0.5 * np.random.randn(hours)
    co = np.maximum(co, 0.1)
    
    # 气象数据
    temperature = 20 + 8 * np.sin(np.array(hour_of_day) * 2 * np.pi / 24 - np.pi/2) + 3 * np.random.randn(hours)
    humidity = 60 + 20 * np.sin(np.array(hour_of_day) * 2 * np.pi / 24) + 10 * np.random.randn(hours)
    humidity = np.clip(humidity, 30, 90)
    
    pressure = 1013 + 5 * np.random.randn(hours)
    wind_speed = 2 + np.abs(np.random.randn(hours))
    wind_direction = 180 + 60 * np.sin(np.arange(hours) * 2 * np.pi / (24*3)) + 30 * np.random.randn(hours)
    wind_direction = wind_direction % 360
    
    precipitation = np.random.exponential(0.05, hours)
    precipitation[precipitation > 1] = 0
    
    visibility = 15 - pm25 * 0.1 + 2 * np.random.randn(hours)
    visibility = np.maximum(visibility, 1)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'datetime': timestamps,
        'PM2.5': pm25,
        'PM10': pm10,
        'O3': o3,
        'NO2': no2,
        'SO2': so2,
        'CO': co,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'precipitation': precipitation,
        'visibility': visibility
    })
    
    return data

def demo_basic_forecast():
    """演示基本预报功能"""
    print("=== 空气质量预报演示 ===")
    
    # 1. 准备数据
    print("1. 准备演示数据...")
    data = create_demo_data()
    print(f"数据时间范围: {data['datetime'].min()} 至 {data['datetime'].max()}")
    print(f"数据记录数: {len(data)}")
    
    # 2. 创建预报系统
    print("\n2. 创建预报系统...")
    forecast_system = AirQualityForecastSystem(
        sequence_length=12,  # 使用12小时历史
        forecast_horizon=24  # 预报24小时
    )
    
    # 3. 训练模型（使用较少的epochs以加快演示）
    print("\n3. 训练LSTM模型...")
    try:
        history = forecast_system.train_model(
            data, 
            target_col='PM2.5',
            epochs=20,  # 减少训练轮数
            batch_size=16
        )
        print("模型训练完成！")
        
        # 4. 进行预报
        print("\n4. 生成预报...")
        recent_data = data.tail(24)  # 最近24小时
        predictions, confidence_intervals = forecast_system.predict_air_quality(
            recent_data, 'PM2.5'
        )
        
        print(f"预报结果:")
        print(f"  平均浓度: {np.mean(predictions):.1f} μg/m³")
        print(f"  最高浓度: {np.max(predictions):.1f} μg/m³")
        print(f"  最低浓度: {np.min(predictions):.1f} μg/m³")
        
        # 5. 生成预警
        print("\n5. 生成预警信息...")
        warnings = forecast_system.generate_warnings(predictions, 'PM2.5')
        
        # 统计各等级预警
        level_counts = {}
        for warning in warnings:
            level = warning['level_name']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print("预警等级分布:")
        for level, count in level_counts.items():
            print(f"  {level}: {count} 小时")
        
        # 显示污染时段
        pollution_warnings = [w for w in warnings if w['pollution_level'] >= 2]
        if pollution_warnings:
            print(f"\n检测到 {len(pollution_warnings)} 个污染时段:")
            for i, warning in enumerate(pollution_warnings[:3]):
                print(f"  第{warning['hour']}小时: {warning['level_name']}, "
                      f"浓度{warning['predicted_concentration']:.1f}μg/m³")
                if warning['control_measures']:
                    print(f"    建议措施: {warning['control_measures'][0]}")
        else:
            print("\n预报期内无污染预警")
        
        # 6. 简单的成因分析演示
        if pollution_warnings:
            print("\n6. 污染成因分析演示...")
            
            # 模拟成因分析
            pollution_start = datetime.now()
            pollution_end = pollution_start + timedelta(hours=24)
            
            try:
                cause_analysis = forecast_system.analyze_pollution_causes(
                    data, [pollution_start, pollution_end]
                )
                
                contributions = cause_analysis['comprehensive_assessment']
                print("各因子贡献度:")
                print(f"  气象因子: {contributions['meteorological']:.1f}%")
                print(f"  排放因子: {contributions['emission']:.1f}%")
                print(f"  传输因子: {contributions['transport']:.1f}%")
                print(f"  二次生成: {contributions['secondary']:.1f}%")
                
                # 生成管控建议
                max_level = max([w['pollution_level'] for w in warnings])
                recommendations = forecast_system.generate_control_recommendations(
                    cause_analysis, max_level
                )
                
                if recommendations['immediate_measures']:
                    print("\n建议采取的措施:")
                    for measure in recommendations['immediate_measures'][:2]:
                        print(f"  - {measure}")
                
            except Exception as e:
                print(f"成因分析出现错误: {e}")
        
        print("\n✅ 预报演示完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 演示过程出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_system_info():
    """显示系统信息"""
    print("=== 系统信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("\n=== 系统功能 ===")
    print("1. 基于PyTorch LSTM的时序预报")
    print("2. 多因子特征工程（污染物+气象+时间+空间）")
    print("3. 注意力机制增强的LSTM模型")
    print("4. 分级预警系统（6级污染等级）")
    print("5. 污染成因分析（气象+排放+传输+二次生成）")
    print("6. 针对性管控建议生成")
    print("7. 不确定性估计（Monte Carlo Dropout）")

def main():
    """主函数"""
    print("基于PyTorch LSTM的空气质量预报预警系统")
    print("=" * 50)
    
    # 显示系统信息
    demo_system_info()
    
    print("\n" + "=" * 50)
    
    # 运行演示
    success = demo_basic_forecast()
    
    if success:
        print("\n🎉 演示成功完成！")
        print("\n系统特点:")
        print("- 使用PyTorch实现，避免了TensorFlow依赖")
        print("- 支持GPU加速训练")
        print("- 集成了完整的预报预警流程")
        print("- 提供污染成因分析和管控建议")
        print("- 适用于重点点位的空气质量预报预警服务")
    else:
        print("\n⚠️ 演示失败，请检查环境配置")

if __name__ == "__main__":
    main()
