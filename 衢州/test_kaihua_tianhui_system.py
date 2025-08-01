#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试开化天汇预警报警系统
"""

import os
import sys
import pandas as pd
from datetime import datetime

def test_kaihua_tianhui_system():
    """测试开化天汇预警报警系统"""
    print("=== 测试开化天汇预警报警系统 ===")
    
    try:
        # 导入开化天汇预警系统
        from waste_incineration_warning_system_kaihua_tianhui import WasteIncinerationWarningSystemKaihuaTianhui
        print("✓ 开化天汇预警系统导入成功")
        
        # 创建实例
        warning_system = WasteIncinerationWarningSystemKaihuaTianhui()
        print("✓ 系统实例创建成功")
        
        # 检查测试文件是否存在
        test_file = "衢州/数据上/数据上/开化/5.23.csv"
        if os.path.exists(test_file):
            print(f"✓ 测试文件存在: {test_file}")
            
            # 尝试加载数据
            df = warning_system.load_data(test_file)
            if not df.empty:
                print(f"✓ 数据加载成功，行数: {len(df)}")
                print(f"✓ 数据列数: {len(df.columns)}")
                
                # 显示关键字段
                key_fields = [
                    '炉膛上部温度', '炉膛中部温度', '布袋除尘进出口压差', 
                    '烟气O2', '烟气粉尘', '烟气SO2', '烟气Nox', '烟气CO', '烟气HCL'
                ]
                
                print("\n关键数据字段检查:")
                for field in key_fields:
                    if field in df.columns:
                        print(f"  ✓ {field}")
                    else:
                        print(f"  ✗ {field} (缺失)")
                
                # 测试炉膛温度计算
                df_with_temp = warning_system.calculate_furnace_temperature(df.copy())
                if '炉膛温度' in df_with_temp.columns:
                    temp_stats = df_with_temp['炉膛温度'].describe()
                    print(f"\n✓ 炉膛温度计算成功")
                    print(f"  - 平均温度: {temp_stats['mean']:.2f}℃")
                    print(f"  - 最高温度: {temp_stats['max']:.2f}℃")
                    print(f"  - 最低温度: {temp_stats['min']:.2f}℃")
                else:
                    print("✗ 炉膛温度计算失败")
                
                # 测试预警检查功能
                print("\n测试预警检查功能:")
                
                # 测试低炉温预警
                low_temp_warnings = warning_system.check_low_furnace_temp_warning(df_with_temp)
                print(f"  - 低炉温预警: {len(low_temp_warnings)} 条")
                
                # 测试高炉温预警
                high_temp_warnings = warning_system.check_high_furnace_temp_warning(df_with_temp)
                print(f"  - 高炉温预警: {len(high_temp_warnings)} 条")
                
                # 测试压力预警
                pressure_warnings = warning_system.check_bag_pressure_warning(df_with_temp)
                print(f"  - 压力预警: {len(pressure_warnings)} 条")
                
                # 测试氧含量预警
                o2_warnings = warning_system.check_o2_warning(df_with_temp)
                print(f"  - 氧含量预警: {len(o2_warnings)} 条")
                
                # 测试污染物预警
                pollutant_warnings = warning_system.check_pollutant_warning(df_with_temp)
                print(f"  - 污染物预警: {len(pollutant_warnings)} 条")
                
                # 测试报警检查功能
                print("\n测试报警检查功能:")
                
                # 测试低炉温报警
                low_temp_alarms = warning_system.check_low_furnace_temp_alarm(df_with_temp)
                print(f"  - 低炉温报警: {len(low_temp_alarms)} 条")
                
                # 测试污染物日均值报警
                pollutant_alarms = warning_system.check_pollutant_daily_alarm(df_with_temp)
                print(f"  - 污染物日均值报警: {len(pollutant_alarms)} 条")
                
                total_events = (len(low_temp_warnings) + len(high_temp_warnings) + 
                              len(pressure_warnings) + len(o2_warnings) + 
                              len(pollutant_warnings) + len(low_temp_alarms) + 
                              len(pollutant_alarms))
                
                print(f"\n✓ 功能测试完成，共检测到 {total_events} 条预警报警事件")

                # 如果有预警报警事件，输出到Excel模板
                if total_events > 0:
                    print("\n正在生成预警报警报告...")

                    # 收集所有预警报警事件
                    all_events = []
                    all_events.extend(low_temp_warnings)
                    all_events.extend(high_temp_warnings)
                    all_events.extend(pressure_warnings)
                    all_events.extend(o2_warnings)
                    all_events.extend(pollutant_warnings)
                    all_events.extend(low_temp_alarms)
                    all_events.extend(pollutant_alarms)

                    # 转换为DataFrame
                    if all_events:
                        events_df = pd.DataFrame(all_events)
                        events_df = events_df.sort_values('时间')

                        # 保存到Excel
                        output_file = f"开化天汇_测试预警报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

                        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                            # 预警报警记录
                            events_df.to_excel(writer, sheet_name='预警报警记录', index=False)

                            # 统计信息
                            stats_data = {
                                '统计项目': [
                                    '测试文件',
                                    '数据行数',
                                    '总预警报警数量',
                                    '预警数量',
                                    '报警数量',
                                    '低炉温预警',
                                    '高炉温预警',
                                    '压力预警',
                                    '氧含量预警',
                                    '污染物预警',
                                    '低炉温报警',
                                    '污染物日均值报警',
                                    '测试时间'
                                ],
                                '数值': [
                                    test_file,
                                    len(df),
                                    total_events,
                                    len([e for e in all_events if e.get('预警/报警区分') == '预警']),
                                    len([e for e in all_events if e.get('预警/报警区分') == '报警']),
                                    len(low_temp_warnings),
                                    len(high_temp_warnings),
                                    len(pressure_warnings),
                                    len(o2_warnings),
                                    len(pollutant_warnings),
                                    len(low_temp_alarms),
                                    len(pollutant_alarms),
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                ]
                            }
                            stats_df = pd.DataFrame(stats_data)
                            stats_df.to_excel(writer, sheet_name='测试统计', index=False)

                            # 事件类型统计
                            if '预警/报警事件' in events_df.columns:
                                event_stats = events_df['预警/报警事件'].value_counts().reset_index()
                                event_stats.columns = ['预警报警事件', '数量']
                                event_stats.to_excel(writer, sheet_name='事件类型统计', index=False)

                        print(f"✓ 预警报告已保存: {output_file}")
                        print(f"  - 总计 {total_events} 条预警报警事件")
                        print(f"  - 预警: {len([e for e in all_events if e.get('预警/报警区分') == '预警'])} 条")
                        print(f"  - 报警: {len([e for e in all_events if e.get('预警/报警区分') == '报警'])} 条")

                        # 显示前几条事件示例
                        print(f"\n前5条预警报警事件:")
                        for i, event in enumerate(all_events[:5], 1):
                            print(f"  {i}. [{event['预警/报警区分']}] {event['预警/报警事件']}")
                            print(f"     时间: {event['时间']}")
                            print(f"     数值: {event['数值']}, 阈值: {event['阈值']}")
                            print(f"     描述: {event['描述']}")
                            print()

                else:
                    print("\n未检测到预警报警事件")

                print("\n可以运行完整处理:")
                print("python waste_incineration_warning_system_kaihua_tianhui.py 衢州/数据上/数据上/开化/5.23.csv ./output")

                print("\n=== 测试成功 ===")
                
            else:
                print("✗ 数据加载失败")
        else:
            print(f"✗ 测试文件不存在: {test_file}")
            print("请检查文件路径是否正确")
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_field_mapping():
    """测试字段映射"""
    print("\n=== 测试字段映射 ===")
    
    try:
        from waste_incineration_warning_system_kaihua_tianhui import KAIHUA_TIANHUI_FIELD_MAPPING, KAIHUA_TIANHUI_THRESHOLDS
        
        print("字段映射:")
        for key, value in KAIHUA_TIANHUI_FIELD_MAPPING.items():
            print(f"  {key}: {value}")
        
        print("\n预警阈值:")
        for key, value in KAIHUA_TIANHUI_THRESHOLDS['warning'].items():
            print(f"  {key}: {value}")
        
        print("\n报警阈值:")
        for key, value in KAIHUA_TIANHUI_THRESHOLDS['alarm'].items():
            print(f"  {key}: {value}")
            
        print("✓ 字段映射测试成功")
        
    except Exception as e:
        print(f"✗ 字段映射测试失败: {e}")

if __name__ == "__main__":
    test_kaihua_tianhui_system()
    test_field_mapping()
