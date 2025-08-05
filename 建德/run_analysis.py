#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建德垃圾焚烧预警报警分析 - 直接运行脚本
"""

import os
import sys
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path

# 导入算法
try:
    from shishi_data_yujing_gz import WasteIncinerationWarningSystemJiande
    print("✅ 成功导入建德预警报警算法")
except ImportError as e:
    print(f"❌ 导入算法失败: {e}")
    sys.exit(1)

def find_process_files(data_dir):
    """查找所有_process.xlsx文件"""
    pattern = os.path.join(data_dir, "**", "*_process.xlsx")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)

def analyze_single_file(file_path, warning_system, output_dir):
    """分析单个文件"""
    print(f"\n📊 分析文件: {os.path.basename(file_path)}")

    try:
        start_time = datetime.now()
        result_df = warning_system.process_data(file_path, output_dir)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if not result_df.empty:
            print(f"   ✅ 检测到 {len(result_df)} 条事件，耗时: {duration:.2f}秒")
            return len(result_df), result_df
        else:
            print(f"   ✅ 无事件，耗时: {duration:.2f}秒")
            return 0, result_df

    except Exception as e:
        print(f"   ❌ 分析失败: {e}")
        return 0, None

def generate_integrated_report(all_results, output_dir, total_files, successful_files,
                             failed_files, total_events, total_duration):
    """生成集成报告"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. 生成集成的预警报警事件Excel文件
    if all_results:
        print("   📊 合并所有预警报警事件...")
        all_events = []

        for file_path, result_df in all_results:
            # 添加数据来源列
            result_df_copy = result_df.copy()
            result_df_copy['数据来源'] = os.path.basename(file_path)

            # 提取日期信息
            file_name = os.path.basename(file_path)
            if '_process.xlsx' in file_name:
                date_part = file_name.replace('_process.xlsx', '')
                result_df_copy['数据日期'] = date_part

            all_events.append(result_df_copy)

        # 合并所有事件
        integrated_df = pd.concat(all_events, ignore_index=True)

        # 按时间排序
        integrated_df = integrated_df.sort_values('时间')

        # 保存集成Excel文件
        excel_file = os.path.join(output_dir, f"建德预警报警集成报告_{timestamp}.xlsx")

        # 创建多个工作表
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 主报告工作表
            integrated_df.to_excel(writer, sheet_name='预警报警事件', index=False)

            # 统计汇总工作表
            create_summary_sheet(writer, integrated_df, all_results, total_files,
                                successful_files, failed_files, total_events, total_duration)

            # 按类型分组工作表
            create_type_analysis_sheet(writer, integrated_df)

            # 按日期分组工作表
            create_date_analysis_sheet(writer, integrated_df)

        print(f"   ✅ 集成Excel报告: {excel_file}")

        # 保存集成CSV文件
        csv_file = os.path.join(output_dir, f"建德预警报警集成报告_{timestamp}.csv")
        integrated_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"   ✅ 集成CSV报告: {csv_file}")

    # 2. 生成汇总统计文件
    stats_file = os.path.join(output_dir, f"建德批量分析统计报告_{timestamp}.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("建德垃圾焚烧批量分析统计报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析耗时: {total_duration:.2f} 秒\n\n")

        f.write("处理概况:\n")
        f.write(f"  总文件数: {total_files}\n")
        f.write(f"  成功处理: {successful_files}\n")
        f.write(f"  处理失败: {failed_files}\n")
        f.write(f"  总事件数: {total_events}\n\n")

        if all_results:
            f.write("文件详情:\n")
            for file_path, result_df in all_results:
                file_name = os.path.basename(file_path)
                f.write(f"  {file_name}: {len(result_df)} 条事件\n")

                # 事件类型统计
                type_stats = result_df['预警/报警类型'].value_counts()
                for event_type, count in type_stats.items():
                    f.write(f"    - {event_type}: {count} 条\n")
                f.write("\n")

    print(f"   ✅ 统计报告: {stats_file}")

def create_summary_sheet(writer, integrated_df, all_results, total_files,
                        successful_files, failed_files, total_events, total_duration):
    """创建汇总统计工作表"""
    summary_data = []

    # 基本统计
    summary_data.append(['统计项目', '数值'])
    summary_data.append(['处理文件总数', total_files])
    summary_data.append(['成功处理文件', successful_files])
    summary_data.append(['处理失败文件', failed_files])
    summary_data.append(['总事件数', total_events])
    summary_data.append(['分析耗时(秒)', f"{total_duration:.2f}"])
    summary_data.append(['', ''])

    # 事件类型统计
    summary_data.append(['事件类型统计', ''])
    if not integrated_df.empty:
        type_stats = integrated_df['预警/报警类型'].value_counts()
        for event_type, count in type_stats.items():
            summary_data.append([event_type, count])

    summary_data.append(['', ''])

    # 事件详细统计
    summary_data.append(['事件详细统计', ''])
    if not integrated_df.empty:
        event_stats = integrated_df['预警/报警事件'].value_counts()
        for event, count in event_stats.items():
            summary_data.append([event, count])

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='统计汇总', index=False, header=False)

def create_type_analysis_sheet(writer, integrated_df):
    """创建按类型分析工作表"""
    if integrated_df.empty:
        return

    type_analysis = []
    type_analysis.append(['预警/报警类型', '事件名称', '数量', '占比(%)'])

    type_stats = integrated_df['预警/报警类型'].value_counts()
    total_count = len(integrated_df)

    for event_type in type_stats.index:
        type_df = integrated_df[integrated_df['预警/报警类型'] == event_type]
        event_stats = type_df['预警/报警事件'].value_counts()

        for event_name, count in event_stats.items():
            percentage = (count / total_count) * 100
            type_analysis.append([event_type, event_name, count, f"{percentage:.2f}"])

    type_df = pd.DataFrame(type_analysis[1:], columns=type_analysis[0])
    type_df.to_excel(writer, sheet_name='类型分析', index=False)

def create_date_analysis_sheet(writer, integrated_df):
    """创建按日期分析工作表"""
    if integrated_df.empty or '数据日期' not in integrated_df.columns:
        return

    date_analysis = []
    date_analysis.append(['数据日期', '预警数量', '报警数量', '总数量'])

    date_stats = integrated_df.groupby('数据日期').agg({
        '预警/报警类型': lambda x: (x == '预警').sum(),
        '预警/报警事件': 'count'
    }).reset_index()

    date_stats.columns = ['数据日期', '预警数量', '总数量']
    date_stats['报警数量'] = integrated_df.groupby('数据日期')['预警/报警类型'].apply(lambda x: (x == '报警').sum()).values

    # 重新排列列顺序
    date_stats = date_stats[['数据日期', '预警数量', '报警数量', '总数量']]

    date_stats.to_excel(writer, sheet_name='日期分析', index=False)

def batch_analysis():
    """批量分析模式"""
    print("=" * 80)
    print("🏭 建德垃圾焚烧预警报警分析系统 - 批量分析模式")
    print("=" * 80)

    # 配置路径
    data_dir = "建德/建德数据"
    output_dir = "./建德/预警输出"

    print(f"� 数据目录: {data_dir}")
    print(f"� 输出目录: {output_dir}")

    # 查找所有_process.xlsx文件
    print(f"\n🔍 搜索_process.xlsx文件...")
    process_files = find_process_files(data_dir)

    if not process_files:
        print("❌ 未找到任何_process.xlsx文件")
        return

    print(f"✅ 找到 {len(process_files)} 个文件")

    # 按月份分组显示
    files_by_month = {}
    for file_path in process_files:
        # 提取月份信息
        path_parts = file_path.split(os.sep)
        month_info = "未知月份"
        for part in path_parts:
            if "月" in part:
                month_info = part
                break

        if month_info not in files_by_month:
            files_by_month[month_info] = []
        files_by_month[month_info].append(file_path)

    print(f"\n📋 文件分布:")
    for month, files in files_by_month.items():
        print(f"   {month}: {len(files)} 个文件")

    # 创建预警系统实例
    print(f"\n🔧 创建预警系统实例...")
    warning_system = WasteIncinerationWarningSystemJiande()

    # 批量处理
    print(f"\n🚀 开始批量分析...")
    total_start_time = datetime.now()

    total_files = len(process_files)
    total_events = 0
    successful_files = 0
    failed_files = 0
    all_results = []

    for i, file_path in enumerate(process_files, 1):
        print(f"\n[{i}/{total_files}] 处理: {file_path}")

        event_count, result_df = analyze_single_file(file_path, warning_system, output_dir)

        if result_df is not None:
            successful_files += 1
            total_events += event_count
            if not result_df.empty:
                all_results.append((file_path, result_df))
        else:
            failed_files += 1

    total_end_time = datetime.now()
    total_duration = (total_end_time - total_start_time).total_seconds()

    # 生成集成报告
    print(f"\n📋 生成集成报告...")
    generate_integrated_report(all_results, output_dir, total_files, successful_files,
                             failed_files, total_events, total_duration)

    # 显示汇总结果
    print(f"\n" + "=" * 80)
    print("📊 批量分析汇总结果")
    print("=" * 80)
    print(f"📁 处理文件总数: {total_files}")
    print(f"✅ 成功处理: {successful_files}")
    print(f"❌ 处理失败: {failed_files}")
    print(f"🎯 总事件数: {total_events}")
    print(f"⏱️ 总耗时: {total_duration:.2f} 秒")
    print(f"📂 报告保存目录: {output_dir}")

    if all_results:
        print(f"\n📋 有事件的文件详情:")
        for file_path, result_df in all_results:
            file_name = os.path.basename(file_path)
            type_stats = result_df['预警/报警类型'].value_counts()
            stats_str = ", ".join([f"{t}:{c}条" for t, c in type_stats.items()])
            print(f"   {file_name}: {len(result_df)}条事件 ({stats_str})")

    print(f"\n🎉 批量分析完成！")

def single_analysis():
    """单文件分析模式"""
    print("=" * 80)
    print("🏭 建德垃圾焚烧预警报警分析系统 - 单文件分析模式")
    print("=" * 80)

    # 配置文件路径
    input_file = "建德/建德数据/2025年6月/6.1_process.xlsx"
    output_dir = "./建德/预警输出"

    print(f"📁 输入文件: {input_file}")
    print(f"📂 输出目录: {output_dir}")

    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return

    # 创建预警系统实例
    print(f"\n🔧 创建预警系统实例...")
    warning_system = WasteIncinerationWarningSystemJiande()

    # 开始分析
    print(f"\n🚀 开始分析数据...")
    start_time = datetime.now()

    try:
        # 处理数据
        result_df = warning_system.process_data(input_file, output_dir)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n⏱️ 分析完成，耗时: {duration:.2f} 秒")

        if not result_df.empty:
            print(f"\n🎯 分析结果:")
            print(f"   总计检测到 {len(result_df)} 条预警报警事件")

            # 显示事件类型统计
            type_stats = result_df['预警/报警类型'].value_counts()
            print(f"\n📊 事件类型统计:")
            for event_type, count in type_stats.items():
                print(f"   {event_type}: {count} 条")

            # 显示事件详细统计
            event_stats = result_df['预警/报警事件'].value_counts()
            print(f"\n📋 事件详细统计:")
            for event, count in event_stats.items():
                print(f"   {event}: {count} 条")

            # 显示前几条事件
            print(f"\n🔍 前5条事件:")
            for i, (_, row) in enumerate(result_df.head().iterrows()):
                print(f"   {i+1}. {row['时间']} - {row['预警/报警事件']} ({row['预警/报警类型']})")

            print(f"\n📁 报告已保存到: {output_dir}")
        else:
            print(f"\n✅ 分析完成，未发现预警报警事件")
            print("   数据正常，无需关注")

    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n" + "=" * 80)
    print("🎉 分析完成！")
    print("=" * 80)

def main():
    """主函数"""
    print("🚀 建德垃圾焚烧预警报警分析系统")
    print("请选择分析模式:")
    print("1. 单文件分析")
    print("2. 批量分析")

    # 自动选择批量分析模式
    choice = "2"  # 默认批量分析

    if choice == "1":
        single_analysis()
    elif choice == "2":
        batch_analysis()
    else:
        print("❌ 无效选择，默认使用批量分析模式")
        batch_analysis()

if __name__ == "__main__":
    main()
