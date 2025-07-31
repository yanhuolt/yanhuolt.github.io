"""
基于PyTorch LSTM的空气质量预报预警系统测试脚本
演示完整的预报预警和污染成因分析功能
"""

import numpy as np
import pandas as pd
import torch
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from air_quality_forecast_system import AirQualityForecastSystem
from visualization_utils import setup_chinese_font, plot_forecast_results

def generate_synthetic_air_quality_data(days=60, hours_per_day=24):
    """
    生成合成的空气质量数据用于测试
    包含污染物浓度、气象要素等多维特征
    """
    total_hours = days * hours_per_day
    
    # 时间序列
    start_time = datetime.now() - timedelta(days=days)
    timestamps = [start_time + timedelta(hours=i) for i in range(total_hours)]
    
    # 基础时间特征
    hours = [t.hour for t in timestamps]
    days_of_week = [t.weekday() for t in timestamps]
    
    # 生成基础污染物浓度（带有日周期和随机波动）
    base_pm25 = 50 + 30 * np.sin(np.array(hours) * 2 * np.pi / 24) + 20 * np.random.randn(total_hours)
    
    # 添加周末效应（周末污染稍低）
    weekend_effect = [-10 if dow >= 5 else 0 for dow in days_of_week]
    base_pm25 += np.array(weekend_effect)
    
    # 添加一些污染事件
    pollution_events = np.random.choice(total_hours, size=int(total_hours * 0.03), replace=False)
    for event in pollution_events:
        duration = np.random.randint(8, 36)  # 污染持续8-36小时
        intensity = np.random.uniform(80, 150)  # 污染强度
        for i in range(duration):
            if event + i < total_hours:
                base_pm25[event + i] += intensity * np.exp(-i/12)  # 指数衰减
    
    # 确保浓度非负
    base_pm25 = np.maximum(base_pm25, 5)
    
    # 生成其他污染物（与PM2.5相关）
    pm10 = base_pm25 * 1.6 + 15 * np.random.randn(total_hours)
    pm10 = np.maximum(pm10, 10)
    
    o3 = 100 + 40 * np.sin(np.array(hours) * 2 * np.pi / 24 - np.pi/4) + 20 * np.random.randn(total_hours)
    o3 = np.maximum(o3, 10)
    
    no2 = base_pm25 * 0.7 + 12 * np.random.randn(total_hours)
    no2 = np.maximum(no2, 5)
    
    so2 = base_pm25 * 0.25 + 8 * np.random.randn(total_hours)
    so2 = np.maximum(so2, 2)
    
    co = base_pm25 * 0.04 + 0.8 * np.random.randn(total_hours)
    co = np.maximum(co, 0.1)
    
    # 生成气象数据
    temperature = 18 + 12 * np.sin(np.array(hours) * 2 * np.pi / 24 - np.pi/2) + 5 * np.random.randn(total_hours)
    humidity = 65 + 25 * np.sin(np.array(hours) * 2 * np.pi / 24) + 12 * np.random.randn(total_hours)
    humidity = np.clip(humidity, 20, 95)
    
    pressure = 1013 + 8 * np.random.randn(total_hours)
    wind_speed = 2.5 + np.abs(1.5 * np.random.randn(total_hours))
    wind_direction = 180 + 80 * np.sin(np.arange(total_hours) * 2 * np.pi / (24*5)) + 40 * np.random.randn(total_hours)
    wind_direction = wind_direction % 360
    
    precipitation = np.random.exponential(0.08, total_hours)
    precipitation[precipitation > 3] = 0  # 大部分时间无降水
    
    visibility = 18 - base_pm25 * 0.12 + 3 * np.random.randn(total_hours)
    visibility = np.maximum(visibility, 1)
    
    # 组装数据
    data = pd.DataFrame({
        'datetime': timestamps,
        'PM2.5': base_pm25,
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

def test_pytorch_lstm_system():
    """测试基于PyTorch LSTM的预报预警系统"""
    print("=== PyTorch LSTM空气质量预报预警系统测试 ===\n")

    # 创建必要的目录
    checkpoints_dir = "lstm算法/checkpoints"
    visualization_dir = "lstm算法/可视化结果"

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    print(f"模型保存目录: {checkpoints_dir}")
    print(f"可视化保存目录: {visualization_dir}")

    # 检查PyTorch和CUDA可用性
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")

    # 1. 生成测试数据
    print("\n1. 生成合成测试数据...")
    historical_data = generate_synthetic_air_quality_data(days=90)  # 90天历史数据
    current_data = historical_data.tail(48)  # 最近48小时数据用于预报

    print(f"历史数据: {len(historical_data)} 条记录")
    print(f"当前数据: {len(current_data)} 条记录")
    print(f"数据时间范围: {historical_data['datetime'].min()} 至 {historical_data['datetime'].max()}")

    # 2. 创建预报系统
    print("\n2. 初始化LSTM预报系统...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建时间戳用于文件命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(checkpoints_dir, f"best_lstm_model_{timestamp}.pth")

    forecast_system = AirQualityForecastSystem(
        sequence_length=24,  # 使用24小时历史数据
        forecast_horizon=48,  # 预报未来48小时
        device=device,
        model_save_path=model_save_path  # 传递模型保存路径
    )
    
    # 3. 运行完整系统
    print("\n3. 运行预报预警系统...")
    try:
        system_output = forecast_system.run_forecast_system(
            historical_data=historical_data,
            current_data=current_data,
            target_pollutant='PM2.5'
        )
        
        # 4. 显示结果
        print("\n4. 预报结果摘要:")
        print(f"预报时间: {system_output['forecast_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"目标污染物: {system_output['target_pollutant']}")
        print(f"预报时长: {len(system_output['predictions'])} 小时")
        
        predictions = system_output['predictions']
        print(f"平均预报浓度: {np.mean(predictions):.1f} μg/m³")
        print(f"最高预报浓度: {np.max(predictions):.1f} μg/m³")
        print(f"最低预报浓度: {np.min(predictions):.1f} μg/m³")
        print(f"最高污染等级: {forecast_system.warning_levels[system_output['max_pollution_level']]}")
        
        # 5. 预警信息
        print("\n5. 预警信息:")
        pollution_warnings = [w for w in system_output['warnings'] if w['pollution_level'] >= 2]
        if pollution_warnings:
            print(f"检测到 {len(pollution_warnings)} 个污染时段")
            for i, warning in enumerate(pollution_warnings[:5]):  # 显示前5个
                print(f"  时段{i+1}: 第{warning['hour']}小时, {warning['level_name']}, "
                      f"浓度{warning['predicted_concentration']:.1f}μg/m³")
        else:
            print("预报期内无污染预警")
        
        # 6. 成因分析和管控建议
        if system_output.get('pollution_detected', False):
            print("\n6. 污染成因分析:")
            contributions = system_output['cause_analysis']['comprehensive_assessment']
            print(f"  气象因子贡献: {contributions['meteorological']:.1f}%")
            print(f"  排放因子贡献: {contributions['emission']:.1f}%")
            print(f"  传输因子贡献: {contributions['transport']:.1f}%")
            print(f"  二次生成贡献: {contributions['secondary']:.1f}%")
            
            print("\n7. 管控建议:")
            recommendations = system_output['control_recommendations']
            
            if recommendations['immediate_measures']:
                print("  立即措施:")
                for measure in recommendations['immediate_measures']:
                    print(f"    - {measure}")
            
            if recommendations['source_control']:
                print("  源头管控:")
                for measure in recommendations['source_control'][:3]:
                    print(f"    - {measure}")
        
        # 7. 生成可视化图表
        print("\n8. 生成可视化图表...")
        try:
            # 创建可视化文件名
            viz_filename = f"LSTM预报结果_{timestamp}.png"
            viz_save_path = os.path.join(visualization_dir, viz_filename)

            # 使用专业的可视化工具
            plot_forecast_results(system_output, current_data, save_path=viz_save_path)
            print(f"✅ 可视化图表已保存至: {viz_save_path}")

        except Exception as e:
            print(f"可视化失败: {e}")
            # 如果专业可视化失败，使用简单的备用方案
            try:
                setup_chinese_font()
                plt.figure(figsize=(12, 8))

                recent_pm25 = current_data['PM2.5'].values[-24:]
                recent_hours = list(range(-24, 0))
                pred_hours = list(range(1, len(predictions) + 1))

                plt.plot(recent_hours, recent_pm25, 'b-', label='历史观测', linewidth=2, marker='o', markersize=4)
                plt.plot(pred_hours, predictions, 'r-', label='预测值', linewidth=2, marker='s', markersize=4)

                plt.axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='轻度污染(75)')
                plt.axhline(y=115, color='red', linestyle='--', alpha=0.7, label='中度污染(115)')
                plt.axhline(y=150, color='purple', linestyle='--', alpha=0.7, label='重度污染(150)')

                plt.xlabel('时间 (小时)')
                plt.ylabel('PM2.5 浓度 (μg/m³)')
                plt.title(f'PM2.5浓度预报 - {timestamp}')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # 添加预报起始线
                plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
                plt.text(0, plt.ylim()[1]*0.9, '预报起始', ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

                plt.tight_layout()

                # 保存备用可视化
                backup_viz_path = os.path.join(visualization_dir, f"简单预报图_{timestamp}.png")
                plt.savefig(backup_viz_path, dpi=300, bbox_inches='tight')
                print(f"✅ 备用可视化图表已保存至: {backup_viz_path}")
                plt.show()

            except Exception as e2:
                print(f"备用可视化也失败: {e2}")
        
        # 8. 生成结果总结
        print("\n9. 生成结果总结...")
        summary_path = os.path.join(visualization_dir, f"预报系统总结_{timestamp}.txt")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== PyTorch LSTM空气质量预报预警系统测试总结 ===\n\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"PyTorch版本: {torch.__version__}\n")
            f.write(f"CUDA可用: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA设备: {torch.cuda.get_device_name()}\n")
            f.write(f"计算设备: {device}\n\n")

            f.write("数据信息:\n")
            f.write(f"  历史数据: {len(historical_data)} 条记录\n")
            f.write(f"  当前数据: {len(current_data)} 条记录\n")
            f.write(f"  数据时间范围: {historical_data['datetime'].min()} 至 {historical_data['datetime'].max()}\n\n")

            f.write("预报结果:\n")
            f.write(f"  预报时间: {system_output['forecast_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  目标污染物: {system_output['target_pollutant']}\n")
            f.write(f"  预报时长: {len(system_output['predictions'])} 小时\n")
            f.write(f"  平均预报浓度: {np.mean(predictions):.1f} μg/m³\n")
            f.write(f"  最高预报浓度: {np.max(predictions):.1f} μg/m³\n")
            f.write(f"  最低预报浓度: {np.min(predictions):.1f} μg/m³\n")
            f.write(f"  最高污染等级: {forecast_system.warning_levels[system_output['max_pollution_level']]}\n\n")

            f.write("文件保存位置:\n")
            f.write(f"  最佳模型: {model_save_path}\n")
            f.write(f"  可视化结果: {visualization_dir}\n")
            f.write(f"  结果总结: {summary_path}\n\n")

            if system_output.get('pollution_detected', False):
                f.write("污染成因分析:\n")
                contributions = system_output['cause_analysis']['comprehensive_assessment']
                f.write(f"  气象因子贡献: {contributions['meteorological']:.1f}%\n")
                f.write(f"  排放因子贡献: {contributions['emission']:.1f}%\n")
                f.write(f"  传输因子贡献: {contributions['transport']:.1f}%\n")
                f.write(f"  二次生成贡献: {contributions['secondary']:.1f}%\n\n")

        print(f"✅ 结果总结已保存至: {summary_path}")
        print("\n✅ PyTorch LSTM预报预警系统测试完成！")
        print(f"\n📁 所有结果文件保存位置:")
        print(f"   模型文件: {model_save_path}")
        print(f"   可视化结果: {visualization_dir}")
        print(f"   结果总结: {summary_path}")

        return True
        
    except Exception as e:
        print(f"\n❌ 系统运行出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试基于PyTorch LSTM的空气质量预报预警系统...\n")
    
    # 测试完整系统
    success = test_pytorch_lstm_system()
    
    if success:
        print("\n🎉 所有测试成功完成！")
        print("系统已成功使用PyTorch LSTM实现空气质量预报预警功能")
        print("包含：多因子预报、污染预警、成因分析、管控建议等完整功能")
    else:
        print("\n⚠️ 测试失败，请检查系统配置和依赖")

if __name__ == "__main__":
    main()
