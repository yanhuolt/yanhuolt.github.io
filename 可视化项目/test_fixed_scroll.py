#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的鼠标滚轮拉伸功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from 副本.完整数据处理与可视化算法 import AdsorptionCurveProcessor

def test_fixed_scroll_stretch():
    """测试修复后的鼠标滚轮拉伸功能"""
    print("=== 测试修复后的鼠标滚轮拉伸功能 ===")
    
    # 创建处理器实例
    processor = AdsorptionCurveProcessor("可视化项目/7.24.csv")
    
    # 加载数据
    if not processor.load_data():
        print("数据加载失败")
        return
    
    print(f"原始数据加载成功: {len(processor.raw_data)} 行")
    
    # 基础数据清洗
    basic_cleaned = processor.basic_data_cleaning(processor.raw_data)
    if len(basic_cleaned) == 0:
        print("基础清洗后无数据")
        return
    
    print(f"基础清洗完成: {len(basic_cleaned)} 行")
    
    # K-S检验清洗
    processor.cleaned_data_ks = processor.ks_test_cleaning(basic_cleaned)
    print(f"K-S检验清洗完成: {len(processor.cleaned_data_ks)} 行")
    
    # 计算效率数据
    if len(processor.cleaned_data_ks) > 0:
        processor.efficiency_data_ks = processor.calculate_efficiency_data(
            processor.cleaned_data_ks, "K-S检验"
        )
        
        print(f"效率数据计算完成: {len(processor.efficiency_data_ks)} 个时间段")
        
        # 训练预警模型
        time_data = processor.efficiency_data_ks['time'].values
        breakthrough_data = processor.efficiency_data_ks['breakthrough_ratio'].values
        
        print("开始训练预警模型...")
        model_fitted = processor.warning_model.fit_model(time_data, breakthrough_data)
        
        if model_fitted:
            print("✅ 预警模型训练成功")
        else:
            print("⚠️ 预警模型训练失败，但仍可测试滚轮功能")
        
        # 创建带有滚轮拉伸功能的可视化
        print("\n" + "="*50)
        print("🎯 创建预警系统可视化图表...")
        print("\n📋 使用说明:")
        print("1. 在主图区域使用鼠标滚轮可调整横坐标拉伸")
        print("2. 向上滚动：拉伸数据点间距（数据点和曲线会相应移动）")
        print("3. 向下滚动：压缩数据点间距（数据点和曲线会相应移动）")
        print("4. 拉伸系数范围：0.1 - 10.0")
        print("5. 控制台会显示详细的调试信息")
        print("6. 修复内容：")
        print("   - 增加了拉伸变化幅度（从10%改为20%）")
        print("   - 添加了详细的调试输出")
        print("   - 修复了缩进问题")
        print("   - 确保图形强制重绘")
        print("\n🔍 请观察:")
        print("   - 滚轮操作时控制台的调试信息")
        print("   - 数据点和曲线是否随拉伸系数变化而移动")
        print("   - 横坐标标签是否显示当前拉伸系数")
        
        fig = processor.create_warning_visualization(processor.efficiency_data_ks)
        
        import matplotlib.pyplot as plt
        plt.show()
        
        print("\n✅ 测试完成！")
        print("如果数据点和曲线仍然没有移动，请检查控制台的调试信息")
    else:
        print("❌ 没有有效的效率数据")

if __name__ == "__main__":
    test_fixed_scroll_stretch()
