#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析7.24.csv数据结构，为测试需求做准备
"""

import pandas as pd
import numpy as np

def analyze_data():
    """分析数据结构"""
    data = pd.read_csv('7.24.csv')
    
    print('=== 数据基本信息 ===')
    print(f'数据总行数: {len(data)}')
    print(f'列数: {len(data.columns)}')
    
    print('\n=== 进口0出口1列的值分布 ===')
    print(data['进口0出口1'].value_counts())
    
    print('\n=== 风速统计 ===')
    print(f'最小风速: {data["风管内风速值"].min():.2f}')
    print(f'最大风速: {data["风管内风速值"].max():.2f}')
    print(f'风速>=0.5的行数: {len(data[data["风管内风速值"] >= 0.5])}')
    print(f'风速<0.5的行数: {len(data[data["风管内风速值"] < 0.5])}')
    
    print('\n=== 进口voc和出口voc为0的情况 ===')
    print(f'进口voc=0的行数: {len(data[data["进口voc"] == 0])}')
    print(f'出口voc=0的行数: {len(data[data["出口voc"] == 0])}')
    
    print('\n=== 进口0出口1=2的情况 ===')
    simultaneous_data = data[data['进口0出口1'] == 2]
    print(f'进口0出口1=2的行数: {len(simultaneous_data)}')
    if len(simultaneous_data) > 0:
        print(f'其中进口voc=0或出口voc=0的行数: {len(simultaneous_data[(simultaneous_data["进口voc"] == 0) | (simultaneous_data["出口voc"] == 0)])}')
    
    print('\n=== 切换型数据情况 ===')
    switching_data = data[data['进口0出口1'].isin([0, 1])]
    print(f'切换型数据总行数: {len(switching_data)}')
    print(f'进口0出口1=0的行数: {len(data[data["进口0出口1"] == 0])}')
    print(f'进口0出口1=1的行数: {len(data[data["进口0出口1"] == 1])}')
    
    print('\n=== 时间范围 ===')
    data['创建时间'] = pd.to_datetime(data['创建时间'])
    print(f'开始时间: {data["创建时间"].min()}')
    print(f'结束时间: {data["创建时间"].max()}')
    print(f'时间跨度: {data["创建时间"].max() - data["创建时间"].min()}')

if __name__ == "__main__":
    analyze_data()
