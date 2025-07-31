"""
活性炭预警系统测试脚本
演示预警功能和可视化效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integrated_warning_processor import IntegratedWarningProcessor
from activated_carbon_warning_system import WarningLevel

def create_test_data_with_breakthrough():
    """创建包含穿透过程的测试数据"""
    print("创建测试数据...")
    
    np.random.seed(42)  # 确保结果可重现
    
    data_points = []
    
    # 模拟8小时的运行数据，每8秒切换一次进出口
    total_time = 8 * 3600  # 8小时
    switch_interval = 8    # 8秒切换间隔
    
    for t in range(0, total_time, switch_interval):
        # 模拟活性炭逐渐饱和的过程
        time_hours = t / 3600
        
        # 进口浓度（相对稳定，有小幅波动）
        inlet_conc = np.random.normal(200, 20)
        inlet_conc = max(inlet_conc, 100)  # 最低100
        
        # 出口浓度（随时间增加，模拟穿透过程）
        if time_hours < 1:
            # 前1小时：很低的出口浓度（高效率）
            breakthrough_ratio = 0.02 + np.random.normal(0, 0.005)
        elif time_hours < 3:
            # 1-3小时：缓慢增加
            breakthrough_ratio = 0.02 + (time_hours - 1) * 0.015 + np.random.normal(0, 0.01)
        elif time_hours < 5:
            # 3-5小时：加速增加（穿透开始）
            breakthrough_ratio = 0.05 + (time_hours - 3) * 0.1 + np.random.normal(0, 0.02)
        elif time_hours < 7:
            # 5-7小时：快速增加（接近饱和）
            breakthrough_ratio = 0.25 + (time_hours - 5) * 0.3 + np.random.normal(0, 0.03)
        else:
            # 7-8小时：接近完全穿透
            breakthrough_ratio = 0.85 + (time_hours - 7) * 0.1 + np.random.normal(0, 0.02)
        
        # 确保穿透率在合理范围内
        breakthrough_ratio = max(0.01, min(0.98, breakthrough_ratio))
        outlet_conc = inlet_conc * breakthrough_ratio
        
        # 风速（模拟工作状态）
        wind_speed = np.random.normal(2.0, 0.3)
        wind_speed = max(wind_speed, 0.8)
        
        # 添加进口数据点
        data_points.append({
            'time': t,
            'inlet_outlet': 0,  # 进口
            'concentration': inlet_conc,
            'wind_speed': wind_speed
        })
        
        # 添加出口数据点
        data_points.append({
            'time': t + 4,  # 4秒后切换到出口
            'inlet_outlet': 1,  # 出口
            'concentration': outlet_conc,
            'wind_speed': wind_speed
        })
    
    df = pd.DataFrame(data_points)
    print(f"生成测试数据: {len(df)} 行")
    
    return df

def test_with_real_data():
    """使用真实数据测试"""
    print("\n=== 使用真实数据测试 ===")
    
    processor = IntegratedWarningProcessor()
    
    # 检查是否存在真实数据文件
    real_data_file = "可视化项目/7.24.csv"
    if os.path.exists(real_data_file):
        try:
            results = processor.process_complete_workflow(
                file_path=real_data_file,
                wind_speed_threshold=0.5
            )
            
            if "error" not in results:
                # 保存可视化结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fig_filename = f"可视化项目/预警系统_真实数据_{timestamp}.png"
                results['visualization'].savefig(fig_filename, dpi=300, bbox_inches='tight')
                print(f"真实数据预警分析图已保存: {fig_filename}")
                
                # 显示报告
                print("\n" + results['report'])
                
                # 显示图形
                plt.show()
                
                return results
            else:
                print(f"真实数据处理失败: {results['error']}")
                
        except Exception as e:
            print(f"真实数据测试出错: {e}")
    else:
        print(f"真实数据文件不存在: {real_data_file}")
    
    return None

def test_with_simulated_data():
    """使用模拟数据测试"""
    print("\n=== 使用模拟数据测试 ===")
    
    # 创建测试数据
    test_data = create_test_data_with_breakthrough()
    
    # 保存测试数据
    test_data_file = "可视化项目/test_breakthrough_data.csv"
    test_data.to_csv(test_data_file, index=False, encoding='utf-8-sig')
    print(f"测试数据已保存: {test_data_file}")
    
    # 处理数据
    processor = IntegratedWarningProcessor()
    
    try:
        results = processor.process_complete_workflow(
            file_path=test_data_file,
            wind_speed_threshold=0.5
        )
        
        if "error" not in results:
            # 保存可视化结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_filename = f"可视化项目/预警系统_模拟数据_{timestamp}.png"
            results['visualization'].savefig(fig_filename, dpi=300, bbox_inches='tight')
            print(f"模拟数据预警分析图已保存: {fig_filename}")
            
            # 显示报告
            print("\n" + results['report'])
            
            # 显示图形
            plt.show()
            
            # 分析预警事件
            warning_events = results['warning_results'].get('warning_events', [])
            if warning_events:
                print(f"\n检测到 {len(warning_events)} 个预警事件:")
                for i, event in enumerate(warning_events, 1):
                    print(f"  {i}. {event.warning_level.value} - 穿透率{event.breakthrough_ratio:.1f}% (时间{event.timestamp}s)")
            
            return results
        else:
            print(f"模拟数据处理失败: {results['error']}")
            
    except Exception as e:
        print(f"模拟数据测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def demonstrate_warning_levels():
    """演示不同预警等级的判断"""
    print("\n=== 预警等级演示 ===")
    
    from activated_carbon_warning_system import ActivatedCarbonWarningSystem
    
    warning_system = ActivatedCarbonWarningSystem()
    
    # 测试不同穿透率对应的预警等级
    test_ratios = [0.02, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.98]
    
    print("穿透率 -> 预警等级:")
    for ratio in test_ratios:
        level = warning_system.determine_warning_level(ratio)
        print(f"  {ratio*100:5.1f}% -> {level.value}")
    
    print("\n预警阈值设置:")
    print(f"  穿透起始点: {warning_system.breakthrough_start_threshold*100:.1f}%")
    print(f"  预警点: {warning_system.warning_threshold*100:.1f}%")
    print(f"  饱和点: {warning_system.saturation_threshold*100:.1f}%")

def main():
    """主函数"""
    print("活性炭更换预警系统测试")
    print("=" * 50)
    
    # 1. 演示预警等级判断
    demonstrate_warning_levels()
    
    # 2. 使用模拟数据测试
    simulated_results = test_with_simulated_data()
    
    # 3. 使用真实数据测试（如果存在）
    real_results = test_with_real_data()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    
    # 总结
    if simulated_results:
        sim_events = len(simulated_results['warning_results'].get('warning_events', []))
        sim_status = simulated_results['warning_results'].get('current_status', WarningLevel.GREEN)
        print(f"模拟数据: {sim_events} 个预警事件, 当前状态: {sim_status.value}")
    
    if real_results:
        real_events = len(real_results['warning_results'].get('warning_events', []))
        real_status = real_results['warning_results'].get('current_status', WarningLevel.GREEN)
        print(f"真实数据: {real_events} 个预警事件, 当前状态: {real_status.value}")

if __name__ == "__main__":
    main()
