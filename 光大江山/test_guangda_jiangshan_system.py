#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试光大江山预警报警系统
"""

import os
import sys
from waste_incineration_warning_system_guangda_jiangshan import WasteIncinerationWarningSystemGuangdaJiangshan

def test_guangda_jiangshan_system():
    """测试光大江山预警报警系统"""
    
    # 测试数据文件路径
    test_file = "衢州/数据下/数据下/光大（江山）/7.3.csv"
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    print("=== 测试光大江山预警报警系统 ===")
    print(f"测试文件: {test_file}")
    
    # 创建预警系统实例
    warning_system = WasteIncinerationWarningSystemGuangdaJiangshan()
    
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
        
        # 按区分统计
        if '预警/报警区分' in result_df.columns:
            distinction_stats = result_df['预警/报警区分'].value_counts()
            print(f"\n预警报警区分统计:")
            for distinction, count in distinction_stats.items():
                print(f"  {distinction}: {count} 条")
        
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
        print(f"\n=== 报警规则检查 ===")
        
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
        
        # 检查预警规则
        print(f"\n=== 预警规则检查 ===")
        
        # 瞬时低炉温焚烧预警
        low_temp_warnings = result_df[result_df['预警/报警事件'] == '瞬时低炉温焚烧']
        print(f"瞬时低炉温焚烧预警: {len(low_temp_warnings)} 条")
        
        # 炉膛温度预警
        temp_warnings = result_df[result_df['预警/报警事件'].str.contains('炉膛温度', na=False)]
        print(f"炉膛温度预警: {len(temp_warnings)} 条")
        
        # 压力预警
        pressure_warnings = result_df[result_df['预警/报警事件'].str.contains('压力损失', na=False)]
        print(f"布袋除尘器压力预警: {len(pressure_warnings)} 条")
        
        # 氧含量预警
        o2_warnings = result_df[result_df['预警/报警事件'].str.contains('氧含量', na=False)]
        print(f"氧含量预警: {len(o2_warnings)} 条")
        
        # 活性炭预警
        carbon_warnings = result_df[result_df['预警/报警事件'].str.contains('活性炭', na=False)]
        print(f"活性炭投加量预警: {len(carbon_warnings)} 条")
        
        # 污染物浓度预警
        pollutant_warnings = result_df[result_df['预警/报警事件'].str.contains('浓度较高', na=False)]
        print(f"污染物浓度预警: {len(pollutant_warnings)} 条")
        
    else:
        print("\n未检测到预警报警事件")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_guangda_jiangshan_system()
