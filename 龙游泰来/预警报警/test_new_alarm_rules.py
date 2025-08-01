#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新增的报警规则
"""

import os
import sys
from shishi_data_yujing_gz import WasteIncinerationWarningSystemLongyou

def test_new_alarm_rules():
    """测试新增的报警规则"""
    
    # 测试数据文件路径
    test_file = "衢州/数据上/数据上/龙游泰来/5.23.xlsx"
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    print("=== 测试龙游泰来新增报警规则 ===")
    print(f"测试文件: {test_file}")
    
    # 创建预警系统实例
    warning_system = WasteIncinerationWarningSystemLongyou()
    
    # 处理数据
    result_df = warning_system.process_data(test_file, "./test_output")
    
    if not result_df.empty:
        print(f"\n=== 测试结果 ===")
        print(f"共检测到 {len(result_df)} 条预警报警记录")
        
        # 按类型统计
        type_stats = result_df['预警/报警类型'].value_counts()
        print(f"\n预警报警类型分布:")
        for alarm_type, count in type_stats.items():
            print(f"  {alarm_type}: {count} 条")
        
        # 按事件类型统计
        event_stats = result_df['预警/报警事件'].value_counts()
        print(f"\n预警报警事件分布:")
        for event, count in event_stats.items():
            print(f"  {event}: {count} 条")
        
        # 显示前几条记录
        print(f"\n前5条记录:")
        print("-" * 80)
        for idx, row in result_df.head(5).iterrows():
            print(f"{row['时间']} | {row['炉号']}号炉 | {row['预警/报警类型']} | {row['预警/报警事件']} | {row.get('预警/报警区分', 'N/A')}")
        
        # 检查新增的报警规则
        print(f"\n=== 新增报警规则检查 ===")
        
        # 1. 低炉温焚烧报警
        low_temp_alarms = result_df[result_df['预警/报警事件'] == '低炉温焚烧']
        print(f"低炉温焚烧报警: {len(low_temp_alarms)} 条")
        
        # 2-6. 污染物日均值超标报警
        pollutant_alarms = result_df[result_df['预警/报警事件'].str.contains('排放超标', na=False)]
        print(f"污染物排放超标报警: {len(pollutant_alarms)} 条")
        
        if len(pollutant_alarms) > 0:
            pollutant_types = pollutant_alarms['预警/报警事件'].value_counts()
            for pollutant, count in pollutant_types.items():
                print(f"  {pollutant}: {count} 条")
        
        # 检查是否包含预警/报警区分列
        if '预警/报警区分' in result_df.columns:
            distinction_stats = result_df['预警/报警区分'].value_counts()
            print(f"\n预警/报警区分统计:")
            for distinction, count in distinction_stats.items():
                print(f"  {distinction}: {count} 条")
        else:
            print("\n警告: 缺少'预警/报警区分'列")
        
    else:
        print("\n未检测到预警报警事件")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_new_alarm_rules()
