#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开化天汇批量处理器
基于龙游泰来批量处理代码修改，适用于开化天汇的预警报警系统
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import glob
from pathlib import Path
import traceback

# 导入开化天汇预警系统
from waste_incineration_warning_system_kaihua_tianhui import WasteIncinerationWarningSystemKaihuaTianhui

class BatchProcessorKaihuaTianhui:
    """开化天汇批量处理器"""
    
    def __init__(self):
        self.warning_system = WasteIncinerationWarningSystemKaihuaTianhui()
        self.all_events = []  # 存储所有预警报警事件
        self.processed_files = []
        self.failed_files = []
        
    def parse_date_from_filename(self, filename: str) -> datetime:
        """从文件名解析日期"""
        try:
            # 提取文件名中的日期部分 (如 "5.23.csv" -> "5.23")
            basename = os.path.basename(filename)
            # 处理csv和xlsx文件
            if basename.endswith('.csv'):
                date_str = basename.replace('.csv', '')
            elif basename.endswith('.xlsx'):
                date_str = basename.split('.')[0] + '.' + basename.split('.')[1]
            else:
                date_str = basename.split('.')[0] + '.' + basename.split('.')[1]
            
            # 解析月日
            month, day = date_str.split('.')
            month = int(month)
            day = int(day)
            
            # 根据月份判断年份
            if month >= 5:
                year = 2024
            else:
                year = 2025
                
            return datetime(year, month, day)
        except Exception as e:
            print(f"无法解析文件名日期: {filename}, 错误: {e}")
            return None
    
    def get_files_in_date_range(self, folder_path: str, start_date: datetime, end_date: datetime) -> list:
        """获取指定日期范围内的文件"""
        files_in_range = []
        
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return files_in_range
        
        # 查找所有csv和xlsx文件
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        all_files = csv_files + xlsx_files
        
        for file_path in all_files:
            file_date = self.parse_date_from_filename(file_path)
            if file_date and start_date <= file_date <= end_date:
                files_in_range.append(file_path)
        
        # 按日期排序
        files_in_range.sort(key=lambda x: self.parse_date_from_filename(x))
        return files_in_range
    
    def process_single_file(self, file_path: str) -> bool:
        """处理单个文件"""
        try:
            print(f"\n正在处理文件: {os.path.basename(file_path)}")
            
            # 重置预警系统状态
            self.warning_system.warning_events = []
            self.warning_system.alarm_events = []
            
            # 处理文件
            self.warning_system.process_data(file_path)
            
            # 收集所有事件
            all_file_events = self.warning_system.warning_events + self.warning_system.alarm_events
            
            # 如果有事件数据，添加文件信息
            if all_file_events:
                file_date = self.parse_date_from_filename(file_path)
                for event in all_file_events:
                    event['数据文件'] = os.path.basename(file_path)
                    event['文件日期'] = file_date.strftime('%Y-%m-%d') if file_date else 'Unknown'
                
                self.all_events.extend(all_file_events)
            
            self.processed_files.append(file_path)
            
            warning_count = len(self.warning_system.warning_events)
            alarm_count = len(self.warning_system.alarm_events)
            print(f"  - 处理完成，发现 {warning_count} 条预警，{alarm_count} 条报警")
            return True
            
        except Exception as e:
            print(f"  - 处理失败: {str(e)}")
            print(f"  - 错误详情: {traceback.format_exc()}")
            self.failed_files.append((file_path, str(e)))
            return False
    
    def process_date_range(self, folder_path: str, start_date: datetime, end_date: datetime, folder_name: str):
        """处理指定日期范围的文件"""
        print(f"\n{'='*60}")
        print(f"开始处理 {folder_name}")
        print(f"文件夹: {folder_path}")
        print(f"日期范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        files = self.get_files_in_date_range(folder_path, start_date, end_date)
        
        if not files:
            print(f"在指定日期范围内未找到文件")
            return
        
        print(f"找到 {len(files)} 个文件需要处理")
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] ", end="")
            self.process_single_file(file_path)
    
    def save_results(self, output_file: str = "开化天汇_批量预警报警结果.xlsx"):
        """保存处理结果"""
        try:
            if not self.all_events:
                print("\n没有预警报警数据需要保存")
                return
            
            # 转换为DataFrame
            df_events = pd.DataFrame(self.all_events)
            
            # 按时间排序
            if '时间' in df_events.columns:
                df_events = df_events.sort_values('时间')
            
            # 统计预警和报警数量
            warning_count = len([e for e in self.all_events if e['预警/报警区分'] == '预警'])
            alarm_count = len([e for e in self.all_events if e['预警/报警区分'] == '报警'])
            
            # 保存到Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 预警报警结果
                df_events.to_excel(writer, sheet_name='预警报警结果', index=False)
                
                # 处理统计
                stats_data = {
                    '统计项目': [
                        '总处理文件数',
                        '成功处理文件数', 
                        '失败处理文件数',
                        '总事件数量',
                        '预警事件数量',
                        '报警事件数量',
                        '处理时间'
                    ],
                    '数值': [
                        len(self.processed_files) + len(self.failed_files),
                        len(self.processed_files),
                        len(self.failed_files),
                        len(self.all_events),
                        warning_count,
                        alarm_count,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='处理统计', index=False)
                
                # 事件类型统计
                if '预警/报警事件' in df_events.columns:
                    event_counts = df_events['预警/报警事件'].value_counts()
                    event_stats_data = {
                        '事件类型': event_counts.index.tolist(),
                        '数量': event_counts.values.tolist()
                    }
                    df_event_stats = pd.DataFrame(event_stats_data)
                    df_event_stats.to_excel(writer, sheet_name='事件类型统计', index=False)
                
                # 失败文件列表
                if self.failed_files:
                    failed_data = {
                        '失败文件': [os.path.basename(f[0]) for f in self.failed_files],
                        '错误信息': [f[1] for f in self.failed_files]
                    }
                    df_failed = pd.DataFrame(failed_data)
                    df_failed.to_excel(writer, sheet_name='失败文件', index=False)
            
            print(f"\n结果已保存到: {output_file}")
            print(f"总共处理 {len(self.processed_files)} 个文件")
            print(f"发现 {len(self.all_events)} 条事件 (预警: {warning_count}, 报警: {alarm_count})")
            if self.failed_files:
                print(f"失败文件 {len(self.failed_files)} 个")
                
        except Exception as e:
            print(f"保存结果时出错: {e}")
            print(f"错误详情: {traceback.format_exc()}")

def main():
    """主函数"""
    processor = BatchProcessorKaihuaTianhui()

    # 定义开化天汇的数据文件夹路径和日期范围
    # 5.23 到 6.16 (数据上文件夹)
    start_date_1 = datetime(2024, 5, 23)
    end_date_1 = datetime(2024, 6, 16)
    folder_path_1 = "衢州/数据上/数据上/开化"

    # 6.17 到 7.20 (数据下文件夹)
    start_date_2 = datetime(2024, 6, 17)
    end_date_2 = datetime(2024, 7, 20)
    folder_path_2 = "衢州/数据下/数据下/开化"

    try:
        # 处理第一个日期范围 (5.23-6.16)
        processor.process_date_range(
            folder_path_1,
            start_date_1,
            end_date_1,
            "开化天汇数据上文件夹 (5.23-6.16)"
        )

        # 处理第二个日期范围 (6.17-7.20)
        processor.process_date_range(
            folder_path_2,
            start_date_2,
            end_date_2,
            "开化天汇数据下文件夹 (6.17-7.20)"
        )

        # 保存结果
        output_filename = f"开化天汇_批量预警报警结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        processor.save_results(output_filename)

        print(f"\n{'='*60}")
        print("开化天汇批量处理完成!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"批量处理过程中出现错误: {e}")
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
