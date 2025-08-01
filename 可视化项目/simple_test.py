#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试脚本 - 验证修改后的预警点设置
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime

def create_simple_test_data():
    """创建简单的测试数据 - 模拟真实的穿透过程"""
    print("创建简单测试数据...")

    # 创建48小时的数据，每小时一个数据点，模拟完整的穿透过程
    start_time = datetime(2024, 1, 1, 8, 0, 0)

    data = []
    for i in range(48):  # 48小时，确保有完整的穿透过程
        time_point = start_time + pd.Timedelta(hours=i)

        # 模拟进口浓度（稳定在100左右）
        inlet_conc = 100 + np.random.normal(0, 3)

        # 模拟真实的穿透过程：
        # 0-12小时：低穿透（<5%）
        # 12-24小时：开始穿透（5%-30%）
        # 24-36小时：快速穿透（30%-80%）
        # 36-48小时：接近饱和并稳定（80%-85%，小幅波动）
        time_factor = i / 48

        if i < 12:  # 初期低穿透
            breakthrough_ratio = 0.01 + 0.03 * (i / 12) + np.random.normal(0, 0.005)
        elif i < 24:  # 开始穿透
            breakthrough_ratio = 0.04 + 0.26 * ((i - 12) / 12) + np.random.normal(0, 0.01)
        elif i < 36:  # 快速穿透
            breakthrough_ratio = 0.3 + 0.5 * ((i - 24) / 12) + np.random.normal(0, 0.02)
        else:  # 接近饱和并稳定
            # 在80%-85%之间稳定波动
            base_ratio = 0.82
            breakthrough_ratio = base_ratio + np.random.normal(0, 0.015)

        # 确保穿透率在合理范围内
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
    df.to_csv('simple_test_data.csv', index=False, encoding='utf-8-sig')
    print(f"测试数据已保存，共{len(df)}条记录")
    print(f"数据时间范围: {df['创建时间'].min()} 到 {df['创建时间'].max()}")
    return 'simple_test_data.csv'

def test_basic_functionality():
    """测试基本功能"""
    print("开始基本功能测试...")
    
    try:
        # 创建测试数据
        test_file = create_simple_test_data()
        
        # 导入算法模块
        from 完整数据处理与可视化算法 import AdsorptionCurveProcessor
        
        print("算法模块导入成功")
        
        # 创建处理器
        processor = AdsorptionCurveProcessor(test_file)
        print("处理器创建成功")
        
        # 运行处理
        processor.process_and_visualize()
        print("处理完成")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("✓ 基本功能测试通过")
    else:
        print("✗ 基本功能测试失败")
