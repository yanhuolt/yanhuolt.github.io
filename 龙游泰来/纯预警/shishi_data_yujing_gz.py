import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import os
import warnings
warnings.filterwarnings('ignore')

# 龙游泰来数据字段映射 (基于5.23.csv的字段结构 - 三炉配置)
LONGYOU_FIELD_MAPPING = {
    # 炉膛温度相关字段 (三个炉子)
    "furnace_1_top_temp": "1#炉膛上部温度",
    "furnace_1_mid_temp": "1#炉膛中部温度",
    "furnace_2_top_temp": "2#炉膛上部温度",
    "furnace_2_mid_temp": "2#炉膛中部温度",
    "furnace_3_top_temp": "3#炉膛上部温度",
    "furnace_3_mid_temp": "3#炉膛中部温度",

    # 布袋除尘器压差 (三个炉子)
    "furnace_1_bag_pressure": "1#炉布袋除尘器压差",
    "furnace_2_bag_pressure": "2#炉布袋除尘器压差",
    "furnace_3_bag_pressure": "3#炉布袋除尘器压差",

    # 氧含量 (三个炉子)
    "furnace_1_o2": "1#炉烟气O2",
    "furnace_2_o2": "2#炉烟气O2",
    "furnace_3_o2": "3#炉烟气O2",

    # 污染物浓度 (三个炉子)
    # 1号炉
    "furnace_1_dust": "1#炉烟气KLW（折算）",
    "furnace_1_so2": "1#炉烟气SO2（折算）",
    "furnace_1_nox": "1#炉烟气NOX（折算）",
    "furnace_1_co": "1#炉烟气CO（折算）",
    "furnace_1_hcl": "1#炉烟气HCL（折算）",

    # 2号炉
    "furnace_2_dust": "2#炉烟气KLW（折算）",
    "furnace_2_so2": "2#炉烟气SO2（折算）",
    "furnace_2_nox": "2#炉烟气NOX（折算）",
    "furnace_2_co": "2#炉烟气CO（折算）",
    "furnace_2_hcl": "2#炉烟气HCL（折算）",

    # 3号炉
    "furnace_3_dust": "3#炉烟气粉尘（折算）",
    "furnace_3_so2": "3#炉烟气SO2（折算）",
    "furnace_3_nox": "3#炉烟气NOX（折算）",
    "furnace_3_co": "3#炉烟气CO（折算）",
    "furnace_3_hcl": "3#炉烟气HCL（折算）",

    # 活性炭投加量 (假设字段名，需要根据实际数据调整)
    "furnace_1_carbon_dosage": "1#炉活性炭投加量",
    "furnace_2_carbon_dosage": "2#炉活性炭投加量",
    "furnace_3_carbon_dosage": "3#炉活性炭投加量",
}

# 龙游泰来预警阈值配置
LONGYOU_WARNING_THRESHOLDS = {
    # 瞬时低炉温焚烧 (5分钟平均值)
    "low_furnace_temp": 850,  # 低于850℃

    # 炉膛温度预警
    "high_furnace_temp": 1200,  # 高于1200℃ (偏高)
    "very_high_furnace_temp": 1300,  # 高于1300℃ (过高)

    # 布袋除尘器压力损失
    "bag_pressure_high": 2000,  # 高于2000Pa (偏高)
    "bag_pressure_low": 500,    # 低于500Pa (偏低)

    # 焚烧炉出口氧含量
    "o2_high": 10,  # 高于10% (偏高)
    "o2_low": 6,    # 低于6% (偏低)

    # 活性炭投加量
    "carbon_dosage_low": 3.0,  # 低于3.0kg/h (不足)

    # 污染物浓度阈值 (小时均值)
    "dust_limit": 30,    # 颗粒物 ≤30mg/m³
    "nox_limit": 300,    # 氮氧化物 ≤300mg/m³
    "so2_limit": 100,    # 二氧化硫 ≤100mg/m³
    "hcl_limit": 60,     # 氯化氢 ≤60mg/m³
    "co_limit": 100,     # 一氧化碳 ≤100mg/m³
}

class WasteIncinerationWarningSystemLongyou:
    """垃圾焚烧预警系统 - 龙游泰来 (三炉配置)"""

    def __init__(self):
        self.warning_events = []
        self.warning_status = {}
        self.furnace_count = 3  # 龙游泰来有3个炉子

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据文件 (支持csv和xlsx)"""
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                raise ValueError("不支持的文件格式，请使用csv或xlsx文件")

            # 转换时间列
            if '数据时间' in df.columns:
                df['数据时间'] = pd.to_datetime(df['数据时间'])

            # 清理和转换数值列
            df = self.clean_numeric_data(df)

            print(f"成功加载数据文件: {file_path}")
            print(f"数据行数: {len(df)}, 列数: {len(df.columns)}")
            return df

        except Exception as e:
            print(f"加载数据文件失败: {e}")
            return pd.DataFrame()

    def clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数值数据，处理异常值和字符串"""
        df_clean = df.copy()

        # 获取所有需要处理的数值列
        numeric_columns = []
        for column_name in LONGYOU_FIELD_MAPPING.values():
            if column_name in df_clean.columns:
                numeric_columns.append(column_name)

        # 清理数值列
        for col in numeric_columns:
            if col in df_clean.columns:
                # 转换为字符串，然后处理异常格式
                df_clean[col] = df_clean[col].astype(str)

                # 处理 '--' 和其他非数值字符
                df_clean[col] = df_clean[col].replace('--', '0')
                df_clean[col] = df_clean[col].replace('nan', '0')

                # 处理连续数字的情况（如 '465.96645.97657.15'）
                # 取第一个有效数字
                df_clean[col] = df_clean[col].str.extract(r'(-?\d+\.?\d*)', expand=False)

                # 转换为数值，无法转换的设为NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

                # 填充NaN值为0
                df_clean[col] = df_clean[col].fillna(0)

        return df_clean

    def calculate_furnace_temperature(self, df: pd.DataFrame, furnace_id: int) -> pd.DataFrame:
        """计算指定炉子的炉膛温度 (上部和中部温度的算术平均值)"""
        result_df = df.copy()

        top_temp_col = LONGYOU_FIELD_MAPPING[f"furnace_{furnace_id}_top_temp"]
        mid_temp_col = LONGYOU_FIELD_MAPPING[f"furnace_{furnace_id}_mid_temp"]

        if top_temp_col in df.columns and mid_temp_col in df.columns:
            # 计算炉膛总体温度 (上部和中部的算术平均值)
            result_df[f'{furnace_id}号炉膛温度'] = (result_df[top_temp_col] + result_df[mid_temp_col]) / 2

        return result_df

    def calculate_time_windows(self, df: pd.DataFrame, window_type: str = '5min') -> pd.DataFrame:
        """计算时间窗口数据 (5分钟、1小时)"""
        if '数据时间' not in df.columns:
            return df

        df_copy = df.copy()
        df_copy.set_index('数据时间', inplace=True)

        # 只选择数值列进行重采样
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_numeric = df_copy[numeric_cols]

        if window_type == '5min':
            # 5分钟窗口
            resampled = df_numeric.resample('5T').mean()
        elif window_type == '1hour':
            # 1小时窗口
            resampled = df_numeric.resample('1H').mean()
        else:
            return df

        resampled.reset_index(inplace=True)
        return resampled

    def check_low_furnace_temp_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查瞬时低炉温焚烧预警 (三个炉子)"""
        warnings = []

        # 计算5分钟平均温度
        df_5min = self.calculate_time_windows(df, '5min')

        # 检查每个炉子
        for furnace_id in range(1, self.furnace_count + 1):
            df_5min = self.calculate_furnace_temperature(df_5min, furnace_id)
            temp_col = f'{furnace_id}号炉膛温度'

            if temp_col not in df_5min.columns:
                continue

            # 检查低于850℃的情况
            low_temp_mask = df_5min[temp_col] < LONGYOU_WARNING_THRESHOLDS['low_furnace_temp']

            if low_temp_mask.any():
                for _, row in df_5min[low_temp_mask].iterrows():
                    warnings.append({
                        '时间': row['数据时间'],
                        '炉号': str(furnace_id),
                        '预警/报警类型': '预警',
                        '预警/报警事件': '瞬时低炉温焚烧'
                    })

        return warnings

    def check_high_furnace_temp_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查炉膛温度偏高/过高预警 (三个炉子)"""
        warnings = []

        # 计算1小时平均温度
        df_1hour = self.calculate_time_windows(df, '1hour')

        # 检查每个炉子
        for furnace_id in range(1, self.furnace_count + 1):
            df_1hour = self.calculate_furnace_temperature(df_1hour, furnace_id)
            temp_col = f'{furnace_id}号炉膛温度'

            if temp_col not in df_1hour.columns:
                continue

            # 检查温度过高 (>1300℃)
            very_high_mask = df_1hour[temp_col] > LONGYOU_WARNING_THRESHOLDS['very_high_furnace_temp']
            for _, row in df_1hour[very_high_mask].iterrows():
                warnings.append({
                    '时间': row['数据时间'],
                    '炉号': str(furnace_id),
                    '预警/报警类型': '预警',
                    '预警/报警事件': '炉膛温度过高'
                })

            # 检查温度偏高 (>1200℃ 且 ≤1300℃)
            high_mask = (df_1hour[temp_col] > LONGYOU_WARNING_THRESHOLDS['high_furnace_temp']) & \
                       (df_1hour[temp_col] <= LONGYOU_WARNING_THRESHOLDS['very_high_furnace_temp'])
            for _, row in df_1hour[high_mask].iterrows():
                warnings.append({
                    '时间': row['数据时间'],
                    '炉号': str(furnace_id),
                    '预警/报警类型': '预警',
                    '预警/报警事件': '炉膛温度偏高'
                })

        return warnings

    def check_bag_pressure_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查布袋除尘器压力损失预警 (三个炉子) - 连续状态跟踪"""
        warnings = []

        # 按时间排序确保正确的状态跟踪
        df_sorted = df.sort_values('数据时间')

        # 检查每个炉子
        for furnace_id in range(1, self.furnace_count + 1):
            pressure_field = LONGYOU_FIELD_MAPPING[f'furnace_{furnace_id}_bag_pressure']

            if pressure_field not in df_sorted.columns:
                continue

            # 状态跟踪变量
            high_pressure_start = None
            low_pressure_start = None

            for _, row in df_sorted.iterrows():
                current_time = row['数据时间']
                pressure_value = row[pressure_field]

                # 检查压力偏高状态 (>2000Pa)
                if pressure_value > LONGYOU_WARNING_THRESHOLDS['bag_pressure_high']:
                    if high_pressure_start is None:
                        # 开始新的高压预警
                        high_pressure_start = current_time
                elif high_pressure_start is not None:
                    # 结束高压预警
                    warnings.append({
                        '时间': high_pressure_start,
                        '炉号': str(furnace_id),
                        '预警/报警类型': '预警',
                        '预警/报警事件': '布袋除尘器压力损失偏高'
                    })
                    high_pressure_start = None

                # 检查压力偏低状态 (<500Pa)
                if pressure_value < LONGYOU_WARNING_THRESHOLDS['bag_pressure_low']:
                    if low_pressure_start is None:
                        # 开始新的低压预警
                        low_pressure_start = current_time
                elif low_pressure_start is not None:
                    # 结束低压预警
                    warnings.append({
                        '时间': low_pressure_start,
                        '炉号': str(furnace_id),
                        '预警/报警类型': '预警',
                        '预警/报警事件': '布袋除尘器压力损失偏低'
                    })
                    low_pressure_start = None

            # 处理到数据结束时仍在进行的预警
            if high_pressure_start is not None:
                warnings.append({
                    '时间': high_pressure_start,
                    '炉号': str(furnace_id),
                    '预警/报警类型': '预警',
                    '预警/报警事件': '布袋除尘器压力损失偏高'
                })

            if low_pressure_start is not None:
                warnings.append({
                    '时间': low_pressure_start,
                    '炉号': str(furnace_id),
                    '预警/报警类型': '预警',
                    '预警/报警事件': '布袋除尘器压力损失偏低'
                })

        return warnings

    def check_o2_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查焚烧炉出口氧含量预警 (三个炉子) - 连续状态跟踪"""
        warnings = []

        # 按时间排序确保正确的状态跟踪
        df_sorted = df.sort_values('数据时间')

        # 检查每个炉子
        for furnace_id in range(1, self.furnace_count + 1):
            o2_field = LONGYOU_FIELD_MAPPING[f'furnace_{furnace_id}_o2']

            if o2_field not in df_sorted.columns:
                continue

            # 状态跟踪变量
            high_o2_start = None
            low_o2_start = None

            for _, row in df_sorted.iterrows():
                current_time = row['数据时间']
                o2_value = row[o2_field]

                # 检查氧含量偏高状态 (>10%)
                if o2_value > LONGYOU_WARNING_THRESHOLDS['o2_high']:
                    if high_o2_start is None:
                        # 开始新的高氧含量预警
                        high_o2_start = current_time
                elif high_o2_start is not None:
                    # 结束高氧含量预警
                    warnings.append({
                        '时间': high_o2_start,
                        '炉号': str(furnace_id),
                        '预警/报警类型': '预警',
                        '预警/报警事件': '焚烧炉出口氧含量偏高'
                    })
                    high_o2_start = None

                # 检查氧含量偏低状态 (<6%)
                if o2_value < LONGYOU_WARNING_THRESHOLDS['o2_low']:
                    if low_o2_start is None:
                        # 开始新的低氧含量预警
                        low_o2_start = current_time
                elif low_o2_start is not None:
                    # 结束低氧含量预警
                    warnings.append({
                        '时间': low_o2_start,
                        '炉号': str(furnace_id),
                        '预警/报警类型': '预警',
                        '预警/报警事件': '焚烧炉出口氧含量偏低'
                    })
                    low_o2_start = None

            # 处理到数据结束时仍在进行的预警
            if high_o2_start is not None:
                warnings.append({
                    '时间': high_o2_start,
                    '炉号': str(furnace_id),
                    '预警/报警类型': '预警',
                    '预警/报警事件': '焚烧炉出口氧含量偏高'
                })

            if low_o2_start is not None:
                warnings.append({
                    '时间': low_o2_start,
                    '炉号': str(furnace_id),
                    '预警/报警类型': '预警',
                    '预警/报警事件': '焚烧炉出口氧含量偏低'
                })

        return warnings

    def check_pollutant_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查污染物浓度预警 (小时均值, 三个炉子)"""
        warnings = []

        # 计算1小时平均值
        df_1hour = self.calculate_time_windows(df, '1hour')

        # 检查每个炉子的各种污染物
        pollutants = {
            'dust': ('烟气中颗粒物（PM）浓度较高', 'dust_limit'),
            'nox': ('烟气中氮氧化物（NOx）浓度较高', 'nox_limit'),
            'so2': ('烟气中二氧化硫（SO₂）浓度较高', 'so2_limit'),
            'hcl': ('烟气中氯化氢（HCl）浓度较高', 'hcl_limit'),
            'co': ('烟气中一氧化碳（CO）浓度较高', 'co_limit')
        }

        for furnace_id in range(1, self.furnace_count + 1):
            for pollutant, (event_name, threshold_key) in pollutants.items():
                field = LONGYOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_{pollutant}')

                if field and field in df_1hour.columns:
                    threshold = LONGYOU_WARNING_THRESHOLDS[threshold_key]
                    high_mask = df_1hour[field] > threshold

                    for _, row in df_1hour[high_mask].iterrows():
                        warnings.append({
                            '时间': row['数据时间'],
                            '炉号': str(furnace_id),
                            '预警/报警类型': '预警',
                            '预警/报警事件': event_name
                        })

        return warnings

    def check_carbon_dosage_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查活性炭投加量不足预警 (三个炉子) - 连续状态跟踪"""
        warnings = []

        # 按时间排序确保正确的状态跟踪
        df_sorted = df.sort_values('数据时间')

        # 检查每个炉子
        for furnace_id in range(1, self.furnace_count + 1):
            carbon_field = LONGYOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_carbon_dosage')

            if not carbon_field or carbon_field not in df_sorted.columns:
                continue

            # 状态跟踪变量
            low_carbon_start = None

            for _, row in df_sorted.iterrows():
                current_time = row['数据时间']
                carbon_value = row[carbon_field]

                # 检查活性炭投加量不足状态 (<3.0kg/h)
                if carbon_value < LONGYOU_WARNING_THRESHOLDS['carbon_dosage_low']:
                    if low_carbon_start is None:
                        # 开始新的活性炭不足预警
                        low_carbon_start = current_time
                elif low_carbon_start is not None:
                    # 结束活性炭不足预警
                    warnings.append({
                        '时间': low_carbon_start,
                        '炉号': str(furnace_id),
                        '预警/报警类型': '预警',
                        '预警/报警事件': '活性炭投加量不足'
                    })
                    low_carbon_start = None

            # 处理到数据结束时仍在进行的预警
            if low_carbon_start is not None:
                warnings.append({
                    '时间': low_carbon_start,
                    '炉号': str(furnace_id),
                    '预警/报警类型': '预警',
                    '预警/报警事件': '活性炭投加量不足'
                })

        return warnings

    def process_data(self, file_path: str, output_dir: str = None) -> pd.DataFrame:
        """处理数据并生成预警报告"""
        # 加载数据
        df = self.load_data(file_path)
        if df.empty:
            return pd.DataFrame()

        # 清空之前的预警事件
        self.warning_events = []

        print(f"\n检查龙游泰来焚烧炉预警 (3个炉子)...")

        # 瞬时低炉温焚烧预警
        low_temp_warnings = self.check_low_furnace_temp_warning(df)
        self.warning_events.extend(low_temp_warnings)
        print(f"低炉温预警: {len(low_temp_warnings)} 条")

        # 炉膛温度偏高/过高预警
        high_temp_warnings = self.check_high_furnace_temp_warning(df)
        self.warning_events.extend(high_temp_warnings)
        print(f"高炉温预警: {len(high_temp_warnings)} 条")

        # 布袋除尘器压力损失预警
        pressure_warnings = self.check_bag_pressure_warning(df)
        self.warning_events.extend(pressure_warnings)
        print(f"压力预警: {len(pressure_warnings)} 条")

        # 焚烧炉出口氧含量预警
        o2_warnings = self.check_o2_warning(df)
        self.warning_events.extend(o2_warnings)
        print(f"氧含量预警: {len(o2_warnings)} 条")

        # 活性炭投加量预警
        carbon_warnings = self.check_carbon_dosage_warning(df)
        self.warning_events.extend(carbon_warnings)
        print(f"活性炭预警: {len(carbon_warnings)} 条")

        # 污染物浓度预警
        pollutant_warnings = self.check_pollutant_warning(df)
        self.warning_events.extend(pollutant_warnings)
        print(f"污染物预警: {len(pollutant_warnings)} 条")

        # 转换为DataFrame
        if self.warning_events:
            warning_df = pd.DataFrame(self.warning_events)
            # 按时间排序
            warning_df = warning_df.sort_values('时间')

            print(f"\n共检测到 {len(warning_df)} 条预警事件")

            # 按炉号统计
            furnace_stats = warning_df['炉号'].value_counts().sort_index()
            print("各炉预警分布:")
            for furnace, count in furnace_stats.items():
                print(f"  {furnace}号炉: {count} 条预警")

            # 保存预警报告
            if output_dir:
                self.save_warning_report(warning_df, output_dir, file_path)

            return warning_df
        else:
            print("\n未检测到预警事件")
            return pd.DataFrame()

    def save_warning_report(self, warning_df: pd.DataFrame, output_dir: str, input_file: str):
        """保存预警报告"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存Excel格式
        excel_file = os.path.join(output_dir, f"{base_name}_龙游预警报告_{timestamp}.xlsx")
        warning_df.to_excel(excel_file, index=False)
        print(f"预警报告已保存: {excel_file}")

        # 保存CSV格式 (与输出模板格式一致)
        csv_file = os.path.join(output_dir, f"{base_name}_龙游预警报告_{timestamp}.csv")
        # 只保留模板需要的列
        template_df = warning_df[['时间', '炉号', '预警/报警类型', '预警/报警事件']].copy()
        template_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"预警报告(模板格式)已保存: {csv_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='垃圾焚烧预警系统 - 龙游泰来 (三炉配置)')
    parser.add_argument('input_file', help='输入数据文件路径 (csv或xlsx)')
    parser.add_argument('-o', '--output', default='./预警输出', help='输出目录 (默认: ./预警输出)')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在 - {args.input_file}")
        return

    # 创建预警系统实例
    warning_system = WasteIncinerationWarningSystemLongyou()

    # 处理数据
    print(f"开始处理数据文件: {args.input_file}")
    warning_df = warning_system.process_data(args.input_file, args.output)

    if not warning_df.empty:
        print(f"\n预警处理完成! 输出目录: {args.output}")
    else:
        print("\n数据处理完成，未发现预警事件。")


if __name__ == "__main__":
    main()