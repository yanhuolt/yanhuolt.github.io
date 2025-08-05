#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理6.1.xlsx数据：
1. 将--替换为0
2. 按分钟合并数据（同一分钟内的记录求和）
3. 删除系统时间相关列
"""

import pandas as pd
import numpy as np
from datetime import datetime

def process_6_1_data():
    """处理6.1.xlsx数据"""
    print("=== 处理6.1.xlsx数据 ===")
    
    # 读取Excel数据
    try:
        df = pd.read_excel("建德/6.1.xlsx", engine='openpyxl')
        print(f"读取成功: {len(df)} 行, {len(df.columns)} 列")
    except Exception as e:
        print(f"读取失败: {e}")
        return
    
    print(f"原始列名: {list(df.columns[:10])}...")  # 显示前10列
    
    # 1. 将所有"--"替换为0
    print("\n步骤1: 替换空值...")
    df_processed = df.replace('--', 0)
    print("空值替换完成")
    
    # 2. 转换时间列
    time_col = df.columns[0]  # 第一列是时间
    print(f"时间列: {time_col}")
    
    try:
        df_processed[time_col] = pd.to_datetime(df_processed[time_col])
        print("时间转换成功")
    except Exception as e:
        print(f"时间转换失败: {e}")
        return
    
    # 3. 创建分钟级别的时间分组
    print("\n步骤2: 按分钟分组...")
    df_processed['minute_group'] = df_processed[time_col].dt.floor('min')
    
    # 显示分组情况
    group_counts = df_processed.groupby('minute_group').size()
    print(f"分组统计: {len(group_counts)} 个分钟组")
    print(f"每组记录数范围: {group_counts.min()} - {group_counts.max()}")
    
    # 4. 识别需要删除的系统时间列
    time_related_cols = ['系统时间年', '系统时间月', '系统时间日', '系统时间小时', '系统时间分钟', '系统时间秒']
    existing_time_cols = [col for col in time_related_cols if col in df_processed.columns]
    print(f"找到系统时间列: {existing_time_cols}")
    
    # 5. 识别数值列（除了时间列和系统时间列）
    exclude_cols = [time_col, 'minute_group'] + existing_time_cols
    numeric_cols = [col for col in df_processed.columns if col not in exclude_cols]
    print(f"数值列数量: {len(numeric_cols)}")
    
    # 6. 转换数值列为数值类型
    print("\n步骤3: 转换数值类型...")
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
    
    # 7. 按分钟分组并求和
    print("\n步骤4: 按分钟分组求和...")
    
    # 创建聚合字典
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = 'sum'  # 对数值列求和
    
    # 分组聚合
    result_df = df_processed.groupby('minute_group').agg(agg_dict).reset_index()
    
    # 重命名时间列
    result_df = result_df.rename(columns={'minute_group': time_col})
    
    print(f"合并后数据: {len(result_df)} 行, {len(result_df.columns)} 列")
    
    # 8. 格式化时间为标准格式
    result_df[time_col] = result_df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 9. 保存结果为Excel文件
    output_file = "建德/6.1_processed.xlsx"
    result_df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\n✅ 处理完成，结果已保存: {output_file}")
    
    # 10. 显示处理结果统计
    print(f"\n📊 处理结果统计:")
    print(f"- 原始数据: {len(df)} 行")
    print(f"- 处理后数据: {len(result_df)} 行")
    print(f"- 数据压缩比: {len(result_df)/len(df)*100:.1f}%")
    print(f"- 保留列数: {len(result_df.columns)}")
    print(f"- 删除的系统时间列: {existing_time_cols}")
    
    # 11. 显示前几行结果
    print(f"\n📋 前5行处理结果:")
    print(result_df.head().to_string(index=False))
    
    # 12. 显示时间范围
    if len(result_df) > 0:
        print(f"\n⏰ 时间范围:")
        print(f"- 开始时间: {result_df[time_col].iloc[0]}")
        print(f"- 结束时间: {result_df[time_col].iloc[-1]}")
    
    # 13. 检查数据质量
    print(f"\n🔍 数据质量检查:")
    null_counts = result_df.isnull().sum()
    if null_counts.sum() == 0:
        print("- ✅ 无空值")
    else:
        print("- ❌ 存在空值:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col}: {count} 个空值")
    
    return result_df

if __name__ == "__main__":
    process_6_1_data()
