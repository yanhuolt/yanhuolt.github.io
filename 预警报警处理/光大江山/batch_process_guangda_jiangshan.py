#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光大江山批量处理器
批量处理光大江山的预警报警数据文件
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import glob
from pathlib import Path
import traceback

# 导入光大江山预警系统
from waste_incineration_warning_system_guangda_jiangshan import WasteIncinerationWarningSystemGuangdaJiangshan

class BatchProcessorGuangdaJiangshan:
    """光大江山批量处理器"""
    
    def __init__(self):
        self.warning_system = WasteIncinerationWarningSystemGuangdaJiangshan()
        self.all_warnings = []
        self.processed_files = []
        self.failed_files = []
        
    def parse_date_from_filename(self, filename: str) -> datetime:
        """从文件名解析日期"""
        try:
            # 提取文件名中的日期部分 (如 "7.3.csv" -> "7.3")
            basename = os.path.basename(filename)
            name_parts = basename.split('.')
            
            # 处理不同的文件名格式
            if len(name_parts) >= 2:
                date_str = name_parts[0] + '.' + name_parts[1]
                
                # 解析月日
                month, day = date_str.split('.')
                month = int(month)
                day = int(day)
                
                # 根据月份判断年份 (光大江山数据主要在2025年)
                if month >= 7:  # 7月及以后的数据是2025年
                    year = 2025
                else:
                    year = 2025
                    
                return datetime(year, month, day)
            else:
                print(f"无法解析文件名格式: {filename}")
                return None
                
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
        files_in_range.sort(key=lambda x: self.parse_date_from_filename(x) or datetime.min)
        return files_in_range
    
    def process_single_file(self, file_path: str) -> bool:
        """处理单个文件"""
        try:
            print(f"\n正在处理文件: {os.path.basename(file_path)}")
            
            # 重置预警系统状态
            self.warning_system.warning_events = []
            
            # 处理文件
            warning_df = self.warning_system.process_data(file_path)

            # 如果有预警数据，添加文件信息
            if not warning_df.empty:
                file_date = self.parse_date_from_filename(file_path)
                warning_df['数据文件'] = os.path.basename(file_path)
                warning_df['文件日期'] = file_date.strftime('%Y-%m-%d') if file_date else 'Unknown'

                # 转换为字典列表并添加到总预警列表
                warnings_list = warning_df.to_dict('records')
                self.all_warnings.extend(warnings_list)
            else:
                warnings_list = []
                
            self.processed_files.append(file_path)
            
            print(f"  - 处理完成，发现 {len(warnings_list)} 条预警报警")
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
    
    def process_all_files_in_folder(self, folder_path: str, folder_name: str):
        """处理文件夹中的所有文件"""
        print(f"\n{'='*60}")
        print(f"开始处理 {folder_name}")
        print(f"文件夹: {folder_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return
        
        # 查找所有csv和xlsx文件
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        all_files = csv_files + xlsx_files
        
        if not all_files:
            print(f"文件夹中未找到数据文件")
            return
        
        # 按文件名排序
        all_files.sort()
        
        print(f"找到 {len(all_files)} 个文件需要处理")
        
        for i, file_path in enumerate(all_files, 1):
            print(f"\n[{i}/{len(all_files)}] ", end="")
            self.process_single_file(file_path)

    def save_results(self, output_file: str = "光大江山_批量预警结果.xlsx"):
        """保存处理结果"""
        try:
            if not self.all_warnings:
                print("\n没有预警报警数据需要保存")
                return

            # 转换为DataFrame
            df_warnings = pd.DataFrame(self.all_warnings)

            # 按时间排序
            if '时间' in df_warnings.columns:
                df_warnings = df_warnings.sort_values('时间')

            # 保存到Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 预警报警结果
                df_warnings.to_excel(writer, sheet_name='预警报警结果', index=False)

                # 处理统计
                stats_data = {
                    '统计项目': [
                        '总处理文件数',
                        '成功处理文件数',
                        '失败处理文件数',
                        '总预警报警数量',
                        '预警数量',
                        '报警数量',
                        '处理时间'
                    ],
                    '数值': [
                        len(self.processed_files) + len(self.failed_files),
                        len(self.processed_files),
                        len(self.failed_files),
                        len(self.all_warnings),
                        len([w for w in self.all_warnings if w.get('预警/报警区分') == '预警']),
                        len([w for w in self.all_warnings if w.get('预警/报警区分') == '报警']),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='处理统计', index=False)

                # 预警报警类型统计
                if '预警/报警事件' in df_warnings.columns:
                    event_stats = df_warnings['预警/报警事件'].value_counts().reset_index()
                    event_stats.columns = ['预警报警事件', '数量']
                    event_stats.to_excel(writer, sheet_name='事件类型统计', index=False)

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
            print(f"发现 {len(self.all_warnings)} 条预警报警")
            if self.all_warnings:
                warning_count = len([w for w in self.all_warnings if w.get('预警/报警区分') == '预警'])
                alarm_count = len([w for w in self.all_warnings if w.get('预警/报警区分') == '报警'])
                print(f"  - 预警: {warning_count} 条")
                print(f"  - 报警: {alarm_count} 条")
            if self.failed_files:
                print(f"失败文件 {len(self.failed_files)} 个")

        except Exception as e:
            print(f"保存结果时出错: {e}")
            print(f"错误详情: {traceback.format_exc()}")


def main():
    """主函数"""
    processor = BatchProcessorGuangdaJiangshan()

    # 光大江山数据文件夹路径
    folder_path = "衢州/数据下/数据下/光大（江山）"

    try:
        print("=== 光大江山批量预警报警处理 ===")
        print(f"处理文件夹: {folder_path}")

        # 处理文件夹中的所有文件
        processor.process_all_files_in_folder(folder_path, "光大江山数据文件夹")

        # 保存结果
        output_filename = f"光大江山_批量预警报警结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        processor.save_results(output_filename)

        print(f"\n{'='*60}")
        print("光大江山批量处理完成!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"批量处理过程中出现错误: {e}")
        print(f"错误详情: {traceback.format_exc()}")


def main_with_date_range():
    """带日期范围的主函数 - 如果需要按日期范围处理"""
    processor = BatchProcessorGuangdaJiangshan()

    # 定义日期范围 (根据实际需要调整)
    start_date = datetime(2025, 7, 1)   # 开始日期
    end_date = datetime(2025, 7, 31)    # 结束日期
    folder_path = "数据下/数据下/光大（江山）"

    try:
        print("=== 光大江山批量预警报警处理 (按日期范围) ===")

        # 处理指定日期范围的文件
        processor.process_date_range(
            folder_path,
            start_date,
            end_date,
            "光大江山数据文件夹"
        )

        # 保存结果
        output_filename = f"光大江山_批量预警报警结果_{start_date.strftime('%m%d')}到{end_date.strftime('%m%d')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        processor.save_results(output_filename)

        print(f"\n{'='*60}")
        print("光大江山批量处理完成!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"批量处理过程中出现错误: {e}")
        print(f"错误详情: {traceback.format_exc()}")


if __name__ == "__main__":
    # 默认处理所有文件
    main()

    # 如果需要按日期范围处理，可以调用：
    # main_with_date_range()
