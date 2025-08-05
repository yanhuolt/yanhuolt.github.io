#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理建德数据文件夹下的所有xlsx文件：
1. 将--替换为0
2. 按分钟合并数据（同一分钟内的记录求和）
3. 删除系统时间相关列
4. 输出文件名为原文件名_process.xlsx
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path

def process_single_file(file_path):
    """处理单个xlsx文件"""
    print(f"\n处理文件: {file_path}")
    
    try:
        # 读取Excel数据
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"  读取成功: {len(df)} 行, {len(df.columns)} 列")
        
        if len(df) == 0:
            print("  ⚠️ 文件为空，跳过处理")
            return False
            
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return False
    
    try:
        # 1. 将所有"--"替换为0
        df_processed = df.replace('--', 0)
        
        # 2. 转换时间列（假设第一列是时间）
        time_col = df.columns[0]
        
        # 尝试转换时间
        try:
            df_processed[time_col] = pd.to_datetime(df_processed[time_col])
        except Exception as e:
            print(f"  ⚠️ 时间转换失败: {e}，跳过时间处理")
            # 如果时间转换失败，只做简单的空值替换和列删除
            time_related_cols = ['系统时间年', '系统时间月', '系统时间日', '系统时间小时', '系统时间分钟', '系统时间秒']
            existing_time_cols = [col for col in time_related_cols if col in df_processed.columns]
            if existing_time_cols:
                df_processed = df_processed.drop(columns=existing_time_cols)
                print(f"  删除系统时间列: {existing_time_cols}")
            
            # 转换数值列
            for col in df_processed.columns:
                if col != time_col:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            
            result_df = df_processed
        else:
            # 3. 创建分钟级别的时间分组
            df_processed['minute_group'] = df_processed[time_col].dt.floor('min')
            
            # 4. 识别需要删除的系统时间列
            time_related_cols = ['系统时间年', '系统时间月', '系统时间日', '系统时间小时', '系统时间分钟', '系统时间秒']
            existing_time_cols = [col for col in time_related_cols if col in df_processed.columns]
            
            # 5. 识别数值列（除了时间列和系统时间列）
            exclude_cols = [time_col, 'minute_group'] + existing_time_cols
            numeric_cols = [col for col in df_processed.columns if col not in exclude_cols]
            
            # 6. 转换数值列为数值类型
            for col in numeric_cols:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            
            # 7. 按分钟分组并求和
            if len(numeric_cols) > 0:
                agg_dict = {col: 'sum' for col in numeric_cols}
                result_df = df_processed.groupby('minute_group').agg(agg_dict).reset_index()
                result_df = result_df.rename(columns={'minute_group': time_col})
                
                # 格式化时间
                result_df[time_col] = result_df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                print("  ⚠️ 未找到数值列，保持原始数据")
                result_df = df_processed.drop(columns=['minute_group'] + existing_time_cols)
        
        # 8. 生成输出文件路径
        file_path_obj = Path(file_path)
        output_file = file_path_obj.parent / f"{file_path_obj.stem}_process.xlsx"
        
        # 9. 保存结果
        result_df.to_excel(output_file, index=False, engine='openpyxl')
        
        print(f"  ✅ 处理完成: {len(df)} 行 → {len(result_df)} 行")
        print(f"  📁 输出文件: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False

def batch_process_data():
    """批量处理建德数据文件夹下的所有xlsx文件"""
    print("=== 批量处理建德数据文件 ===")
    
    # 数据文件夹路径
    data_folder = Path("建德/建德数据")
    
    if not data_folder.exists():
        print(f"❌ 数据文件夹不存在: {data_folder}")
        return
    
    # 查找所有xlsx文件
    xlsx_files = list(data_folder.rglob("*.xlsx"))
    
    if not xlsx_files:
        print("❌ 未找到任何xlsx文件")
        return
    
    print(f"📁 找到 {len(xlsx_files)} 个xlsx文件")
    
    # 统计变量
    success_count = 0
    fail_count = 0
    
    # 逐个处理文件
    for file_path in xlsx_files:
        # 跳过已经处理过的文件（文件名包含_process）
        if "_process" in file_path.stem:
            print(f"⏭️ 跳过已处理文件: {file_path}")
            continue
            
        if process_single_file(file_path):
            success_count += 1
        else:
            fail_count += 1
    
    # 输出处理结果统计
    print(f"\n📊 批量处理完成:")
    print(f"  ✅ 成功处理: {success_count} 个文件")
    print(f"  ❌ 处理失败: {fail_count} 个文件")
    print(f"  📁 总计文件: {len(xlsx_files)} 个")
    
    if success_count > 0:
        print(f"\n🎉 处理成功！所有输出文件都保存在原文件夹中，文件名添加了'_process'后缀")

if __name__ == "__main__":
    batch_process_data()
