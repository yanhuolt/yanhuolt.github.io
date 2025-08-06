import os
import pandas as pd
from shishi_data_yujing_gz import WasteIncinerationWarningSystemJiande  # 导入类

def test_waste_incineration_system(file_path: str, output_dir: str):
    """测试垃圾焚烧预警系统"""
    # 创建系统实例
    warning_system = WasteIncinerationWarningSystemJiande()

    # 处理数据
    print(f"开始处理数据文件: {file_path}")
    try:
        warning_df = warning_system.process_data(file_path, output_dir)

        if not warning_df.empty:
            print(f"\n✅ 预警处理完成! 输出目录: {output_dir}")
            print(f"📊 总计检测到 {len(warning_df)} 条预警报警事件")

            # 显示事件类型统计
            type_stats = warning_df['预警/报警类型'].value_counts()
            print("\n📈 事件类型统计:")
            for event_type, count in type_stats.items():
                print(f"  {event_type}: {count} 条")

            # 显示前几条事件
            print(f"\n📋 前5条事件:")
            for i, (_, row) in enumerate(warning_df.head().iterrows()):
                print(f"  {i + 1}. {row['时间']} - {row['预警/报警事件']} ({row['预警/报警类型']})")
        else:
            print("\n✅ 数据处理完成，未发现预警报警事件。")

    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 请替换为您的文件路径和输出目录
    input_file_path = "建德/建德数据/2025年4月/4.18_process.xlsx"  # 替换为您的xlsx文件路径
    output_directory = "./预警输出"  # 输出目录

    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"❌ 错误: 输入文件不存在 - {input_file_path}")
    else:
        # 运行测试
        test_waste_incineration_system(input_file_path, output_directory)
