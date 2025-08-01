#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("开始最小测试...")

try:
    import pandas as pd
    print("✓ pandas导入成功")
    
    import numpy as np
    print("✓ numpy导入成功")
    
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    print("✓ matplotlib导入成功")
    
    from datetime import datetime
    print("✓ datetime导入成功")
    
    # 测试基本功能
    data = {'x': [1, 2, 3], 'y': [1, 4, 9]}
    df = pd.DataFrame(data)
    print(f"✓ DataFrame创建成功: {len(df)} 行")
    
    # 测试绘图
    plt.figure(figsize=(8, 6))
    plt.plot(df['x'], df['y'], 'b-', label='test')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Test Plot')
    plt.legend()
    plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 测试图表保存成功")
    
    print("所有基础组件测试通过！")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
