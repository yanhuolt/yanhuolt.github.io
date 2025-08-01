#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试光大衢州垃圾焚烧预警系统 (两炉配置)
基于waste_incineration_warning_system_guangda_quzhou.py
"""

import os
import sys
from waste_incineration_warning_system_guangda_quzhou import WasteIncinerationWarningSystemGuangdaQuzhou

def test_guangda_quzhou_system():
    """使用光大衢州数据测试预警系统"""

    # 数据文件路径
    data_file = "衢州/数据上/数据上/光大（衢州）/5.25.xlsx"
    output_dir = "预警输出_光大衢州"

    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在 - {data_file}")
        # 尝试使用xlsx格式作为备选
        data_file_backup = "数据上/数据上/光大（衢州）/5.23.xlsx"
        if os.path.exists(data_file_backup):
            print(f"使用备选数据文件: {data_file_backup}")
            data_file = data_file_backup
        else:
            print(f"备选数据文件也不存在: {data_file_backup}")
            return

    print("=" * 70)
    print("垃圾焚烧预警系统测试 - 光大衢州 (两炉配置)")
    print("=" * 70)

    # 创建预警系统实例
    warning_system = WasteIncinerationWarningSystemGuangdaQuzhou()
    
    # 处理数据
    print(f"开始处理数据文件: {data_file}")
    warning_df = warning_system.process_data(data_file, output_dir)
    
    if not warning_df.empty:
        print(f"\n预警处理完成! 输出目录: {output_dir}")
        
        # 显示预警摘要
        print("\n预警事件摘要:")
        print("-" * 50)
        
        # 按炉号统计
        print("按炉号统计:")
        furnace_stats = warning_df['炉号'].value_counts().sort_index()
        for furnace, count in furnace_stats.items():
            print(f"  {furnace}号炉: {count} 条预警")
        
        # 按预警类型统计
        print("\n按预警事件统计:")
        event_stats = warning_df['预警/报警事件'].value_counts()
        for event, count in event_stats.head(10).items():  # 显示前10个
            print(f"  {event}: {count} 次")
        
        # 显示前几条预警记录
        print(f"\n前5条预警记录:")
        print("-" * 50)
        for idx, row in warning_df.head(5).iterrows():
            print(f"{row['时间']} | {row['炉号']}号炉 | {row['预警/报警事件']}")
        
        # 显示各类预警的数量分布
        print(f"\n预警类型分布:")
        print("-" * 50)
        temp_warnings = warning_df[warning_df['预警/报警事件'].str.contains('温度')]
        pressure_warnings = warning_df[warning_df['预警/报警事件'].str.contains('压力')]
        o2_warnings = warning_df[warning_df['预警/报警事件'].str.contains('氧含量')]
        pollutant_warnings = warning_df[warning_df['预警/报警事件'].str.contains('浓度')]
        
        print(f"  温度相关预警: {len(temp_warnings)} 条")
        print(f"  压力相关预警: {len(pressure_warnings)} 条")
        print(f"  氧含量预警: {len(o2_warnings)} 条")
        print(f"  污染物浓度预警: {len(pollutant_warnings)} 条")
        
        # 各炉预警详细分布
        print(f"\n各炉预警详细分布:")
        print("-" * 50)
        for furnace_id in range(1, 3):  # 光大衢州有2个炉子
            furnace_data = warning_df[warning_df['炉号'] == str(furnace_id)]
            if not furnace_data.empty:
                print(f"\n{furnace_id}号炉 ({len(furnace_data)} 条预警):")
                furnace_event_stats = furnace_data['预警/报警事件'].value_counts()
                for event, count in furnace_event_stats.head(5).items():
                    print(f"  {event}: {count} 次")
            else:
                print(f"\n{furnace_id}号炉: 无预警事件")
        
        # 验证输出格式是否符合模板要求
        print(f"\n输出格式验证:")
        print("-" * 50)
        required_columns = ['时间', '炉号', '预警/报警类型', '预警/报警事件']
        missing_columns = [col for col in required_columns if col not in warning_df.columns]
        
        if not missing_columns:
            print("✓ 输出格式符合模板要求")
            print(f"  包含所有必需列: {required_columns}")
        else:
            print(f"✗ 缺少必需列: {missing_columns}")
        
        # 显示预警规则覆盖情况
        print(f"\n预警规则覆盖情况:")
        print("-" * 50)
        expected_warnings = [
            '瞬时低炉温焚烧',
            '炉膛温度偏高',
            '炉膛温度过高',
            '布袋除尘器压力损失偏高',
            '布袋除尘器压力损失偏低',
            '焚烧炉出口氧含量偏高',
            '焚烧炉出口氧含量偏低',
            '烟气中颗粒物（PM）浓度较高',
            '烟气中氮氧化物（NOx）浓度较高',
            '烟气中二氧化硫（SO₂）浓度较高',
            '烟气中氯化氢（HCl）浓度较高',
            '烟气中一氧化碳（CO）浓度较高'
        ]
        
        actual_warnings = set(warning_df['预警/报警事件'].unique())
        for expected in expected_warnings:
            if expected in actual_warnings:
                print(f"  ✓ {expected}")
            else:
                print(f"  - {expected} (未触发)")
        
    else:
        print("\n数据处理完成，未发现预警事件。")
        print("这可能是因为:")
        print("1. 数据中的所有参数都在正常范围内")
        print("2. 数据格式或字段映射需要调整")
        print("3. 预警阈值设置过于严格")

if __name__ == "__main__":
    test_guangda_quzhou_system()
