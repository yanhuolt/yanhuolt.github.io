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

def create_test_data_with_clear_inflection():
    """创建具有明显拐点的测试数据"""
    print("创建具有明显拐点的测试数据...")
    
    # 创建72小时的数据，每小时一个数据点
    start_time = datetime(2024, 1, 1, 8, 0, 0)
    
    data = []
    for i in range(72):  # 72小时
        time_point = start_time + pd.Timedelta(hours=i)
        
        # 模拟进口浓度（稳定在100左右）
        inlet_conc = 100 + np.random.normal(0, 2)
        
        # 使用Logistic函数模拟真实的穿透过程
        # 参数设置：A=0.85, k=0.15, t0=36小时
        t = i  # 时间（小时）
        A = 0.85
        k = 0.15
        t0 = 36  # 拐点在36小时
        
        # Logistic函数：breakthrough_ratio = A / (1 + exp(-k*(t-t0)))
        breakthrough_ratio = A / (1 + np.exp(-k * (t - t0)))
        
        # 添加一些噪声
        breakthrough_ratio += np.random.normal(0, 0.01)
        breakthrough_ratio = max(0.001, min(0.95, breakthrough_ratio))
        
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
    df.to_csv('test_inflection_data.csv', index=False, encoding='utf-8-sig')
    print(f"测试数据已保存，共{len(df)}条记录")
    print(f"理论拐点时间: 36小时")
    print(f"理论穿透起始点: 约在{t0 - 3/k:.1f}小时 (5%穿透率)")

    # 计算理论饱和时间（95%最大值）
    theoretical_sat_time = t0 - np.log(A / (A * 0.95) - 1) / k
    theoretical_warning_time = (t0 - 3/k) + (theoretical_sat_time - (t0 - 3/k)) * 0.8
    print(f"理论饱和时间: 约在{theoretical_sat_time:.1f}小时 (95%最大穿透率)")
    print(f"理论预警点: 约在{theoretical_warning_time:.1f}小时")
    return 'test_inflection_data.csv'

def test_warning_point_calculation():
    """测试预警点计算"""
    print("\n开始预警点计算测试...")
    
    try:
        # 创建测试数据
        test_file = create_test_data_with_clear_inflection()
        
        # 导入算法模块
        from 完整数据处理与可视化算法 import AdsorptionCurveProcessor
        
        print("算法模块导入成功")
        
        # 创建处理器
        processor = AdsorptionCurveProcessor(test_file)
        print("处理器创建成功")
        
        # 运行处理
        processor.process_and_visualize()
        print("处理完成")
        
        # 检查预警模型参数
        if hasattr(processor, 'warning_model') and processor.warning_model.fitted:
            A, k, t0 = processor.warning_model.params
            print(f"\n模型参数:")
            print(f"  A (最大穿透率): {A:.3f}")
            print(f"  k (增长率): {k:.3f}")
            print(f"  t0 (拐点时间): {t0:.1f}秒 ({t0/3600:.2f}小时)")
            
            print(f"\n关键时间点:")
            if processor.warning_model.breakthrough_start_time:
                start_h = processor.warning_model.breakthrough_start_time / 3600
                print(f"  穿透起始时间: {start_h:.2f}小时")
            
            inflection_h = t0 / 3600
            print(f"  拐点时间: {inflection_h:.2f}小时")

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

                # 验证预警时间计算
                if processor.warning_model.breakthrough_start_time:
                    start_h = processor.warning_model.breakthrough_start_time / 3600
                    expected_warning_h = start_h + (sat_h - start_h) * 0.8
                    print(f"  预期预警时间: {expected_warning_h:.2f}小时")
                    print(f"  计算误差: {abs(warning_h - expected_warning_h):.3f}小时")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_warning_point_calculation()
    if success:
        print("\n✓ 预警点计算测试通过")
        print("请查看生成的图表文件验证预警点位置")
    else:
        print("\n✗ 预警点计算测试失败")
