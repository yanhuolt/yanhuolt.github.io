#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试预警点设置 - 验证基于拐点的预警点计算
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime

def create_realistic_breakthrough_data():
    """创建更真实的穿透数据 - 模拟实际活性炭吸附过程"""
    print("创建真实的穿透过程数据...")

    # 创建48小时的数据，每小时一个数据点
    start_time = datetime(2024, 1, 1, 8, 0, 0)

    data = []
    for i in range(48):  # 48小时
        time_point = start_time + pd.Timedelta(hours=i)

        # 模拟进口浓度（稳定在100左右，有小幅波动）
        inlet_conc = 100 + np.random.normal(0, 3)

        # 模拟更真实的穿透过程：
        # 1. 初期很长时间保持低穿透率（活性炭有效期）
        # 2. 然后开始缓慢增长
        # 3. 最后快速增长到饱和

        t = i  # 时间（小时）

        if t < 20:  # 前20小时，穿透率很低且增长缓慢
            breakthrough_ratio = 0.005 + 0.002 * t + np.random.normal(0, 0.002)
        elif t < 35:  # 20-35小时，开始缓慢增长
            breakthrough_ratio = 0.045 + 0.01 * (t - 20) + np.random.normal(0, 0.005)
        else:  # 35小时后，快速增长
            # 使用指数增长模拟快速穿透
            breakthrough_ratio = 0.195 + 0.6 * (1 - np.exp(-0.3 * (t - 35))) + np.random.normal(0, 0.01)

        # 限制在合理范围内
        breakthrough_ratio = max(0.001, min(0.85, breakthrough_ratio))

        outlet_conc = inlet_conc * breakthrough_ratio
        
        # 进口数据
        data.append({
            '创建时间': time_point,
            '进口0出口1': 0,
            '进口voc': inlet_conc,
            '出口voc': 0,
            '风管内风速值': 2.5,
            '风量': 1000
        })
        
        # 出口数据
        data.append({
            '创建时间': time_point,
            '进口0出口1': 1,
            '进口voc': 0,
            '出口voc': outlet_conc,
            '风管内风速值': 2.5,
            '风量': 1000
        })
    
    df = pd.DataFrame(data)
    df.to_csv('test_realistic_data.csv', index=False, encoding='utf-8-sig')
    print(f"真实穿透数据已保存，共{len(df)}条记录")
    print(f"数据特点:")
    print(f"  初期阶段(0-20h): 低穿透率，缓慢增长")
    print(f"  过渡阶段(20-35h): 开始明显增长")
    print(f"  快速阶段(35h+): 快速增长到饱和")
    return 'test_realistic_data.csv'

def test_dynamic_fitting():
    """测试动态权重和动态增长率的拟合效果"""
    print("\n开始动态拟合测试...")

    try:
        # 创建真实测试数据
        test_file = create_realistic_breakthrough_data()

        # 导入算法模块
        from 完整数据处理与可视化算法 import AdsorptionCurveProcessor

        print("算法模块导入成功")

        # 创建处理器
        processor = AdsorptionCurveProcessor(test_file)
        print("处理器创建成功")

        # 运行处理
        print("\n=== 开始数据处理和模型拟合 ===")
        processor.process_and_visualize()
        print("=== 处理完成 ===")
        
        # 详细分析拟合结果
        if hasattr(processor, 'warning_model') and processor.warning_model.fitted:
            A, k, t0 = processor.warning_model.params
            print(f"\n=== 动态拟合结果分析 ===")
            print(f"模型参数:")
            print(f"  A (最大穿透率): {A:.3f}")
            print(f"  k (增长率): {k:.6f}")
            print(f"  t0 (拐点时间): {t0:.1f}秒 ({t0/3600:.2f}小时)")

            print(f"\n关键时间点:")
            if processor.warning_model.breakthrough_start_time:
                start_h = processor.warning_model.breakthrough_start_time / 3600
                print(f"  穿透起始时间: {start_h:.2f}小时")

            if processor.warning_model.predicted_saturation_time:
                sat_h = processor.warning_model.predicted_saturation_time / 3600
                sat_ratio = processor.warning_model.predict_breakthrough(
                    np.array([processor.warning_model.predicted_saturation_time]))[0]
                print(f"  实际饱和时间: {sat_h:.2f}小时")
                print(f"  实际饱和穿透率: {sat_ratio:.1%}")

            if processor.warning_model.warning_time:
                warning_h = processor.warning_model.warning_time / 3600
                warning_ratio = processor.warning_model.predict_breakthrough(
                    np.array([processor.warning_model.warning_time]))[0]
                print(f"  预警时间: {warning_h:.2f}小时")
                print(f"  预警点穿透率: {warning_ratio:.1%}")

                # 验证预警时间计算逻辑
                if processor.warning_model.breakthrough_start_time and processor.warning_model.predicted_saturation_time:
                    start_h = processor.warning_model.breakthrough_start_time / 3600
                    expected_warning_h = start_h + (sat_h - start_h) * 0.8
                    print(f"  预期预警时间: {expected_warning_h:.2f}小时")
                    print(f"  计算误差: {abs(warning_h - expected_warning_h):.3f}小时")

            # 评估拟合质量
            print(f"\n=== 拟合质量评估 ===")
            # 读取数据进行质量评估
            test_data = pd.read_csv(test_file, encoding='utf-8-sig')
            inlet_data = test_data[test_data['进口0出口1'] == 0]
            outlet_data = test_data[test_data['进口0出口1'] == 1]

            if len(inlet_data) > 0 and len(outlet_data) > 0:
                # 计算实际穿透率
                actual_breakthrough = []
                time_points = []
                for _, inlet_row in inlet_data.iterrows():
                    outlet_row = outlet_data[outlet_data['创建时间'] == inlet_row['创建时间']]
                    if not outlet_row.empty:
                        inlet_conc = inlet_row['进口voc']
                        outlet_conc = outlet_row.iloc[0]['出口voc']
                        if inlet_conc > 0:
                            breakthrough = outlet_conc / inlet_conc
                            actual_breakthrough.append(breakthrough)
                            time_diff = (inlet_row['创建时间'] - inlet_data.iloc[0]['创建时间']).total_seconds()
                            time_points.append(time_diff)

                if len(actual_breakthrough) > 5:
                    actual_breakthrough = np.array(actual_breakthrough)
                    time_points = np.array(time_points)

                    # 预测对应时间点的穿透率
                    predicted_breakthrough = processor.warning_model.predict_breakthrough(time_points)

                    # 计算拟合指标
                    mse = np.mean((actual_breakthrough - predicted_breakthrough)**2)
                    mae = np.mean(np.abs(actual_breakthrough - predicted_breakthrough))
                    r_squared = 1 - np.sum((actual_breakthrough - predicted_breakthrough)**2) / np.sum((actual_breakthrough - np.mean(actual_breakthrough))**2)

                    print(f"  均方误差 (MSE): {mse:.6f}")
                    print(f"  平均绝对误差 (MAE): {mae:.6f}")
                    print(f"  决定系数 (R²): {r_squared:.3f}")

                    if r_squared > 0.8:
                        print("  ✓ 拟合质量: 优秀")
                    elif r_squared > 0.6:
                        print("  ○ 拟合质量: 良好")
                    else:
                        print("  ✗ 拟合质量: 需要改进")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dynamic_fitting()
    if success:
        print("\n" + "="*50)
        print("✓ 动态权重和动态增长率拟合测试完成")
        print("请查看生成的图表文件验证拟合效果:")
        print("  - 检查预测曲线是否更好地拟合实际数据")
        print("  - 验证预警点位置是否合理")
        print("  - 观察拟合质量指标")
        print("="*50)
    else:
        print("\n✗ 动态拟合测试失败")
