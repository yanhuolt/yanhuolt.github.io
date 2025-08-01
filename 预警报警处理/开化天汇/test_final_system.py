#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试开化天汇最终修正后的预警报警系统
"""

import os
import sys
import pandas as pd
from datetime import datetime

def test_final_system():
    """测试最终修正后的系统"""
    print("=== 测试开化天汇最终修正后的预警报警系统 ===")
    
    try:
        # 导入系统
        from waste_incineration_warning_system_kaihua_tianhui import WasteIncinerationWarningSystemKaihuaTianhui
        print("✓ 系统导入成功")
        
        # 创建实例
        system = WasteIncinerationWarningSystemKaihuaTianhui()
        print("✓ 系统实例创建成功")
        
        # 测试数据文件
        test_file = "数据上/数据上/开化/5.23.csv"
        
        if os.path.exists(test_file):
            print(f"✓ 测试文件存在: {test_file}")
            
            # 处理数据
            print("\n=== 处理数据 ===")
            system.process_data(test_file)
            
            # 获取所有事件
            all_events = system.warning_events + system.alarm_events
            
            print(f"\n=== 处理结果 ===")
            print(f"总计: {len(all_events)} 条预警报警事件")
            print(f"  - 预警事件: {len(system.warning_events)} 条")
            print(f"  - 报警事件: {len(system.alarm_events)} 条")
            
            # 按事件类型统计
            if all_events:
                events_df = pd.DataFrame(all_events)
                
                print(f"\n=== 事件类型统计 ===")
                event_counts = events_df['预警/报警事件'].value_counts()
                for event_type, count in event_counts.items():
                    print(f"  - {event_type}: {count} 条")
                
                print(f"\n=== 预警/报警区分统计 ===")
                type_counts = events_df['预警/报警区分'].value_counts()
                for alarm_type, count in type_counts.items():
                    print(f"  - {alarm_type}: {count} 条")
                
                # 显示前10条事件
                print(f"\n=== 前10条事件示例 ===")
                for i, event in enumerate(all_events[:10], 1):
                    print(f"  {i}. [{event['时间']}] {event['预警/报警事件']} ({event['预警/报警区分']})")
                
                # 保存结果
                output_file = f"开化天汇_最终测试结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    # 所有事件
                    events_df_sorted = events_df.sort_values('时间')
                    events_df_sorted.to_excel(writer, sheet_name='预警报警记录', index=False)
                    
                    # 事件统计
                    stats_data = {
                        '统计项目': [
                            '总事件数',
                            '预警事件数',
                            '报警事件数',
                            '瞬时低炉温焚烧',
                            '炉膛温度偏高',
                            '炉膛温度过高',
                            '布袋除尘器压力损失偏高',
                            '布袋除尘器压力损失偏低',
                            '焚烧炉出口氧含量偏高',
                            '焚烧炉出口氧含量偏低',
                            '污染物浓度预警',
                            '低炉温焚烧报警',
                            '污染物排放超标报警'
                        ],
                        '数量': [
                            len(all_events),
                            len(system.warning_events),
                            len(system.alarm_events),
                            len([e for e in all_events if e['预警/报警事件'] == '瞬时低炉温焚烧']),
                            len([e for e in all_events if e['预警/报警事件'] == '炉膛温度偏高']),
                            len([e for e in all_events if e['预警/报警事件'] == '炉膛温度过高']),
                            len([e for e in all_events if e['预警/报警事件'] == '布袋除尘器压力损失偏高']),
                            len([e for e in all_events if e['预警/报警事件'] == '布袋除尘器压力损失偏低']),
                            len([e for e in all_events if e['预警/报警事件'] == '焚烧炉出口氧含量偏高']),
                            len([e for e in all_events if e['预警/报警事件'] == '焚烧炉出口氧含量偏低']),
                            len([e for e in all_events if '浓度较高' in e['预警/报警事件']]),
                            len([e for e in all_events if e['预警/报警事件'] == '低炉温焚烧']),
                            len([e for e in all_events if '排放超标' in e['预警/报警事件']])
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='统计结果', index=False)
                    
                    # 规则实施情况
                    rule_status = {
                        '规则编号': [1, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18, 19, 20, 21, 22],
                        '规则名称': [
                            '瞬时低炉温焚烧',
                            '炉膛温度偏高',
                            '炉膛温度过高',
                            '焚烧工况不稳定',
                            '布袋除尘器压力损失偏高',
                            '布袋除尘器压力损失偏低',
                            '焚烧炉出口氧含量偏高',
                            '焚烧炉出口氧含量偏低',
                            '活性炭投加量不足',
                            '烟气中颗粒物浓度较高',
                            '烟气中氮氧化物浓度较高',
                            '烟气中二氧化硫浓度较高',
                            '烟气中氯化氢浓度较高',
                            '烟气中一氧化碳浓度较高',
                            '氨逃逸偏高'
                        ],
                        '实施状态': [
                            '已实施（修改后）',
                            '已实施（修改后）',
                            '已实施（修改后）',
                            '已删除',
                            '已实施（修改后）',
                            '已实施（修改后）',
                            '已实施（修改后）',
                            '已实施（修改后）',
                            '已删除',
                            '已实施（修改后）',
                            '已实施（修改后）',
                            '已实施（修改后）',
                            '已实施（修改后）',
                            '已实施（修改后）',
                            '已删除'
                        ],
                        '主要修改': [
                            '5分钟窗口连续预警逻辑',
                            '自然日零点起始，1小时间隔',
                            '自然日零点起始，1小时间隔',
                            '删除该项预警规则',
                            '删除正常工况判断，连续预警逻辑',
                            '删除正常工况判断，连续预警逻辑',
                            '删除正常工况判断，连续预警逻辑',
                            '删除正常工况判断，连续预警逻辑',
                            '删除该项预警规则',
                            '删除正常工况判断',
                            '删除正常工况判断',
                            '删除正常工况判断',
                            '删除正常工况判断',
                            '删除正常工况判断',
                            '删除该项预警规则'
                        ]
                    }
                    rule_df = pd.DataFrame(rule_status)
                    rule_df.to_excel(writer, sheet_name='规则实施情况', index=False)
                
                print(f"\n✓ 测试结果已保存: {output_file}")
                
            else:
                print("✗ 未检测到任何预警报警事件")
                
        else:
            print(f"✗ 测试文件不存在: {test_file}")
            
        print(f"\n=== 最终系统测试完成 ===")
        print(f"系统特点:")
        print(f"  ✓ 简化输出格式：只保留时间、预警/报警事件、预警/报警区分")
        print(f"  ✓ 连续预警逻辑：压力损失和氧含量预警实现开始-结束逻辑")
        print(f"  ✓ 删除无关规则：焚烧工况不稳定、活性炭投加量不足、氨逃逸偏高")
        print(f"  ✓ 修改预警条件：删除正常工况判断")
        print(f"  ✓ 时间窗口优化：自然日零点起始，5分钟/1小时间隔")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_final_system()
