#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改7.24.csv数据，使其满足算法测试需求
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def modify_data_for_testing():
    """修改数据以满足测试需求"""
    
    # 读取原始数据
    data = pd.read_csv('7.24.csv')
    data['创建时间'] = pd.to_datetime(data['创建时间'])
    
    print(f"原始数据行数: {len(data)}")
    
    # 1. 修改风速数据，创建风速段切分
    print("\n1. 修改风速数据，创建风速段...")
    
    # 将数据按时间排序
    data = data.sort_values('创建时间').reset_index(drop=True)
    
    # 创建风速段：交替设置高风速和低风速段
    segment_size = 200  # 每个段大约200行数据
    
    for i in range(len(data)):
        segment_idx = i // segment_size
        
        if segment_idx % 3 == 0:  # 每3个段中第1个段：高风速段
            data.loc[i, '风管内风速值'] = np.random.uniform(0.8, 1.5)
        elif segment_idx % 3 == 1:  # 每3个段中第2个段：低风速段（用于分隔）
            data.loc[i, '风管内风速值'] = np.random.uniform(0.1, 0.4)  # <0.5
        else:  # 每3个段中第3个段：高风速段
            data.loc[i, '风管内风速值'] = np.random.uniform(0.6, 1.2)
    
    print(f"修改后风速>=0.5的行数: {len(data[data['风管内风速值'] >= 0.5])}")
    print(f"修改后风速<0.5的行数: {len(data[data['风管内风速值'] < 0.5])}")
    
    # 2. 添加一些同时记录型数据（进口0出口1=2）
    print("\n2. 添加同时记录型数据...")
    
    # 选择一部分数据改为同时记录型
    simultaneous_indices = []
    for i in range(0, len(data), 50):  # 每50行选一些
        if i + 10 < len(data):
            simultaneous_indices.extend(range(i, min(i + 10, len(data))))
    
    # 修改选中的行为同时记录型
    for idx in simultaneous_indices:
        data.loc[idx, '进口0出口1'] = 2
        # 确保进口和出口都有合理的值
        if data.loc[idx, '进口voc'] == 0:
            data.loc[idx, '进口voc'] = np.random.uniform(50, 100)
        if data.loc[idx, '出口voc'] == 0:
            # 出口浓度应该小于进口浓度
            inlet_conc = data.loc[idx, '进口voc']
            data.loc[idx, '出口voc'] = np.random.uniform(0, inlet_conc * 0.8)
    
    print(f"添加了 {len(simultaneous_indices)} 行同时记录型数据")
    
    # 3. 创建一些无法匹配的时间段
    print("\n3. 创建无法匹配的时间段...")
    
    # 找到一些连续的进口或出口段，移除它们的匹配段
    unmatched_segments = []
    
    # 在数据中间插入一些孤立的进口段（后面没有对应出口段）
    for i in range(500, 1500, 300):  # 选择几个位置
        if i + 20 < len(data):
            # 创建孤立的进口段
            for j in range(i, min(i + 20, len(data))):
                data.loc[j, '进口0出口1'] = 0
                data.loc[j, '进口voc'] = np.random.uniform(60, 90)
                data.loc[j, '出口voc'] = 0
            unmatched_segments.append(f"进口段 {i}-{i+19}")
    
    # 创建一些孤立的出口段（前面没有对应进口段）
    for i in range(2000, 2500, 200):
        if i + 15 < len(data):
            # 创建孤立的出口段
            for j in range(i, min(i + 15, len(data))):
                data.loc[j, '进口0出口1'] = 1
                data.loc[j, '出口voc'] = np.random.uniform(5, 25)
                data.loc[j, '进口voc'] = 0
            unmatched_segments.append(f"出口段 {i}-{i+14}")
    
    print(f"创建了 {len(unmatched_segments)} 个无法匹配的时间段:")
    for segment in unmatched_segments:
        print(f"  - {segment}")
    
    # 4. 添加一些异常穿透率数据点
    print("\n4. 添加异常穿透率数据点...")
    
    # 在同时记录型数据中添加一些异常值
    simultaneous_data_indices = data[data['进口0出口1'] == 2].index.tolist()
    
    if len(simultaneous_data_indices) > 0:
        # 添加一些出口浓度>进口浓度的异常数据
        abnormal_indices = np.random.choice(simultaneous_data_indices, 
                                          min(20, len(simultaneous_data_indices)//4), 
                                          replace=False)
        
        for idx in abnormal_indices:
            # 让出口浓度大于进口浓度（异常情况）
            inlet_conc = data.loc[idx, '进口voc']
            data.loc[idx, '出口voc'] = inlet_conc * np.random.uniform(1.2, 2.0)
        
        print(f"添加了 {len(abnormal_indices)} 个异常穿透率数据点")
    
    # 5. 验证修改结果
    print("\n=== 修改后数据验证 ===")
    print(f"总行数: {len(data)}")
    print(f"进口0出口1值分布:")
    print(data['进口0出口1'].value_counts())
    print(f"风速>=0.5的行数: {len(data[data['风管内风速值'] >= 0.5])}")
    print(f"风速<0.5的行数: {len(data[data['风管内风速值'] < 0.5])}")
    print(f"同时记录型数据(进口0出口1=2): {len(data[data['进口0出口1'] == 2])}")
    
    # 6. 保存修改后的数据
    backup_filename = '7.24_original_backup.csv'
    modified_filename = '7.24.csv'
    
    # 备份原始数据
    original_data = pd.read_csv('7.24.csv')
    original_data.to_csv(backup_filename, index=False, encoding='utf-8-sig')
    print(f"\n原始数据已备份到: {backup_filename}")
    
    # 保存修改后的数据
    data.to_csv(modified_filename, index=False, encoding='utf-8-sig')
    print(f"修改后数据已保存到: {modified_filename}")
    
    return data

def create_test_scenarios():
    """创建具体的测试场景"""
    print("\n=== 创建的测试场景 ===")
    print("1. 风速切分场景:")
    print("   - 高风速段(>=0.5): 用于有效数据处理")
    print("   - 低风速段(<0.5): 用于时间段分隔，这些数据会被剔除")
    
    print("\n2. 数据类型场景:")
    print("   - 切换型数据(进口0出口1=0或1): 主要数据类型")
    print("   - 同时记录型数据(进口0出口1=2): 少量数据用于测试不同清洗规则")
    
    print("\n3. 时间段匹配场景:")
    print("   - 可匹配时间段: 进口段后跟出口段，或出口段后跟进口段")
    print("   - 无法匹配时间段: 孤立的进口段或出口段，用于测试弃用逻辑")
    
    print("\n4. 异常数据场景:")
    print("   - 异常穿透率: 出口浓度>进口浓度的数据点")
    print("   - 零值数据: 进口或出口浓度为0的记录")
    
    print("\n5. 算法测试点:")
    print("   - 风速时间段切分功能")
    print("   - 两套不同的数据清洗规则")
    print("   - 时间段匹配和拼接逻辑")
    print("   - 异常值检测和清理")
    print("   - 无法匹配时间段的弃用处理")

if __name__ == "__main__":
    print("开始修改7.24.csv数据以满足算法测试需求...")
    
    # 修改数据
    modified_data = modify_data_for_testing()
    
    # 创建测试场景说明
    create_test_scenarios()
    
    print("\n✅ 数据修改完成！现在可以运行算法进行全面测试。")
