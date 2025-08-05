import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import os
import warnings
warnings.filterwarnings('ignore')

# 建德数据字段映射 (基于实际CSV文件的列名)
JIANDE_FIELD_MAPPING = {
    # 炉膛温度相关字段 (9个测点) - 对应规则中的I+J+K+L+M+N+O+P+Q)/9
    "furnace_temp_points": [
        "上部烟气温度左", "上部烟气温度中", "上部烟气温度右",  # 上部断面
        "中部烟气温度左", "中部烟气温度中", "中部烟气温度右",  # 中部断面
        "下部烟气温度左", "下部烟气温度中", "下部烟气温度右"   # 下部断面
    ],

    # 根据实际数据列名映射
    "furnace_temp_1": "上部烟气温度左",
    "furnace_temp_2": "上部烟气温度中",
    "furnace_temp_3": "上部烟气温度右",
    "furnace_temp_4": "中部烟气温度左",
    "furnace_temp_5": "中部烟气温度中",
    "furnace_temp_6": "中部烟气温度右",
    "furnace_temp_7": "下部烟气温度左",
    "furnace_temp_8": "下部烟气温度中",
    "furnace_temp_9": "下部烟气温度右",

    # 布袋除尘器压差 - 对应规则中的AO列
    "bag_pressure": "除尘器差压",

    # 氧含量 - 对应规则中的AT列
    "o2": "烟气氧量",

    # 污染物浓度字段 - 对应规则中的各列，需要进行折算
    "dust": "烟气烟尘",      # AU列 - 颗粒物（实测值，需折算）
    "so2": "SO2浓度",       # AV列 - 二氧化硫（实测值，需折算）
    "nox": "NOX浓度",       # AW列 - 氮氧化物（实测值，需折算）
    "co": "CO浓度",         # AX列 - 一氧化碳（实测值，需折算）
    "hcl": "HCL浓度",       # AY列 - 氯化氢（实测值，需折算）
}

# 建德预警报警阈值配置 (根据新规则)
JIANDE_WARNING_THRESHOLDS = {
    # 预警阈值
    "low_furnace_temp": 850,      # 瞬时低炉温焚烧 <850℃
    "high_furnace_temp": 1200,    # 炉膛温度偏高 >1200℃
    "very_high_furnace_temp": 1300, # 炉膛温度过高 >1300℃
    "bag_pressure_high": 2000,    # 布袋除尘器压力损失偏高 >2000Pa
    "bag_pressure_low": 500,      # 布袋除尘器压力损失偏低 <500Pa
    "o2_high": 10,                # 焚烧炉出口氧含量偏高 >10%
    "o2_low": 6,                  # 焚烧炉出口氧含量偏低 <6%

    # 污染物浓度预警阈值（小时均值，折算后）
    "dust_warning_limit": 30,     # 颗粒物（PM）预警 >30mg/m³
    "nox_warning_limit": 300,     # 氮氧化物（NOx）预警 >300mg/m³
    "so2_warning_limit": 100,     # 二氧化硫（SO₂）预警 >100mg/m³
    "hcl_warning_limit": 60,      # 氯化氢（HCl）预警 >60mg/m³
    "co_warning_limit": 100,      # 一氧化碳（CO）预警 >100mg/m³
}

# 建德报警阈值配置
JIANDE_ALARM_THRESHOLDS = {
    # 温度报警阈值
    "low_furnace_temp": 850,      # 低炉温焚烧 <850℃

    # 污染物浓度报警阈值（日均值，折算后）
    "dust_alarm_limit": 20,       # 颗粒物（PM）报警 >20mg/m³
    "nox_alarm_limit": 250,       # 氮氧化物（NOx）报警 >250mg/m³
    "so2_alarm_limit": 80,        # 二氧化硫（SO₂）报警 >80mg/m³
    "hcl_alarm_limit": 50,        # 氯化氢（HCl）报警 >50mg/m³
    "co_alarm_limit": 80,         # 一氧化碳（CO）报警 >80mg/m³
}

class WasteIncinerationWarningSystemJiande:
    """垃圾焚烧预警系统 - 建德 (单炉配置)"""

    def __init__(self):
        self.warning_events = []
        self.warning_status = {}
        self.furnace_count = 1  # 建德有1个炉子

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
        for column_name in JIANDE_FIELD_MAPPING.values():
            if isinstance(column_name, str) and column_name in df_clean.columns:
                numeric_columns.append(column_name)
            elif isinstance(column_name, list):
                # 处理温度测点列表
                for col in column_name:
                    if col in df_clean.columns:
                        numeric_columns.append(col)

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

    def calculate_furnace_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算炉膛温度 (建德单炉，9个温度测点)
        根据预警规则：对焚烧炉炉膛的中部和上部两个断面，各自取所有热电偶测量温度的中位数
        计算这两个中位数的算术平均值，作为该断面的代表温度
        """
        result_df = df.copy()

        # 获取温度测点列名
        temp_points = JIANDE_FIELD_MAPPING.get('furnace_temp_points', [])
        available_temp_cols = [col for col in temp_points if col in df.columns]

        if len(available_temp_cols) >= 6:
            # 按照规则：上部3个，中部3个，下部3个
            upper_cols = available_temp_cols[:3]   # 上部断面
            middle_cols = available_temp_cols[3:6] # 中部断面

            # 计算上部和中部断面的中位数
            upper_median = df[upper_cols].median(axis=1)
            middle_median = df[middle_cols].median(axis=1)

            # 计算两个中位数的算术平均值，作为该断面的代表温度
            result_df['furnace_temp'] = (upper_median + middle_median) / 2

            print(f"✅ 使用上部断面({len(upper_cols)}个测点)和中部断面({len(middle_cols)}个测点)计算炉膛温度")
        elif len(available_temp_cols) > 0:
            # 如果测点不足，使用所有可用测点的平均值
            result_df['furnace_temp'] = df[available_temp_cols].mean(axis=1)
            print(f"⚠️ 温度测点不足，使用 {len(available_temp_cols)} 个测点的平均值")
        else:
            print("❌ 未找到温度数据列")
            result_df['furnace_temp'] = 0

        return result_df

    def calculate_time_windows(self, df: pd.DataFrame, window_type: str = '5min') -> pd.DataFrame:
        """计算时间窗口数据 (5分钟、1小时、24小时)"""
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
        elif window_type == '1day' or window_type == '24hour':
            # 24小时窗口（日均值）
            resampled = df_numeric.resample('24H').mean()
        else:
            return df

        resampled.reset_index(inplace=True)
        return resampled

    def check_low_furnace_temp_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查瞬时低炉温焚烧预警 (建德单炉) - 根据新规则"""
        warnings = []

        # 计算5分钟平均温度
        df_5min = self.calculate_time_windows(df, '5min')

        # 计算炉膛温度（建德单炉）
        df_with_temp = self.calculate_furnace_temperature(df_5min)
        temp_col = 'furnace_temp'

        if temp_col not in df_with_temp.columns:
            return warnings

        # 检查低于850℃的情况 - 触发预警
        low_temp_mask = df_with_temp[temp_col] < JIANDE_WARNING_THRESHOLDS['low_furnace_temp']

        if low_temp_mask.any():
            for _, row in df_with_temp[low_temp_mask].iterrows():
                warnings.append({
                    '时间': row['数据时间'],
                    '炉号': '1',
                    '预警/报警类型': '预警',
                    '预警/报警事件': '瞬时低炉温焚烧',
                    '预警/报警区分': '预警'
                })

        return warnings

    def check_low_furnace_temp_alarm(self, df: pd.DataFrame) -> List[Dict]:
        """检查低炉温焚烧报警 (建德单炉) - 根据新规则"""
        alarms = []

        # 计算5分钟平均温度
        df_5min = self.calculate_time_windows(df, '5min')

        # 计算炉膛温度（建德单炉）
        df_with_temp = self.calculate_furnace_temperature(df_5min)
        temp_col = 'furnace_temp'

        if temp_col not in df_with_temp.columns:
            return alarms

        # 检查低于850℃的情况 - 触发报警
        low_temp_mask = df_with_temp[temp_col] < JIANDE_ALARM_THRESHOLDS['low_furnace_temp']

        if low_temp_mask.any():
            for _, row in df_with_temp[low_temp_mask].iterrows():
                alarms.append({
                    '时间': row['数据时间'],
                    '炉号': '1',
                    '预警/报警类型': '报警',
                    '预警/报警事件': '低炉温焚烧',
                    '预警/报警区分': '报警'
                })

        return alarms

    def check_pollutant_daily_alarm(self, df: pd.DataFrame) -> List[Dict]:
        """检查污染物日均值排放超标报警 (建德单炉) - 根据新规则"""
        alarms = []

        # 按日期分组计算日均值
        df_daily = self.calculate_time_windows(df, '1day')

        # 检查各种污染物日均值（需要进行折算）
        pollutants = {
            'dust': ('烟气中颗粒物（PM）排放超标', 'dust_alarm_limit'),
            'nox': ('烟气中氮氧化物（NOx）排放超标', 'nox_alarm_limit'),
            'so2': ('烟气中二氧化硫（SO₂）排放超标', 'so2_alarm_limit'),
            'hcl': ('烟气中氯化氢（HCl）排放超标', 'hcl_alarm_limit'),
            'co': ('烟气中一氧化碳（CO）排放超标', 'co_alarm_limit')
        }

        # 获取氧含量字段用于折算
        o2_field = JIANDE_FIELD_MAPPING.get('o2')

        if not o2_field or o2_field not in df_daily.columns:
            print("警告：未找到氧含量字段，无法进行污染物浓度折算")
            return alarms

        for pollutant, (event_name, threshold_key) in pollutants.items():
            field = JIANDE_FIELD_MAPPING.get(pollutant)

            if field and field in df_daily.columns:
                threshold = JIANDE_ALARM_THRESHOLDS[threshold_key]

                # 计算折算后的浓度
                measured_conc = df_daily[field].dropna()
                measured_o2 = df_daily[o2_field].dropna()

                if len(measured_conc) > 0 and len(measured_o2) > 0:
                    # 确保数据长度一致
                    min_len = min(len(measured_conc), len(measured_o2))
                    if min_len > 0:
                        # 折算公式：ρ（标准）=ρ（实测）*10/(21-ρ（实测O2））
                        corrected_conc = measured_conc.iloc[:min_len] * 10 / (21 - measured_o2.iloc[:min_len])

                        # 检查是否超过阈值
                        exceed_mask = corrected_conc > threshold

                        for i, is_exceed in enumerate(exceed_mask):
                            if is_exceed:
                                alarms.append({
                                    '时间': df_daily.iloc[i]['数据时间'],
                                    '炉号': '1',
                                    '预警/报警类型': '报警',
                                    '预警/报警事件': event_name,
                                    '预警/报警区分': '报警'
                                })

        return alarms

    def check_high_furnace_temp_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查炉膛温度偏高/过高预警 (建德单炉) - 根据新规则"""
        warnings = []

        # 计算1小时平均温度
        df_1hour = self.calculate_time_windows(df, '1hour')

        # 计算炉膛温度（建德单炉）
        df_with_temp = self.calculate_furnace_temperature(df_1hour)
        temp_col = 'furnace_temp'

        if temp_col not in df_with_temp.columns:
            return warnings

        # 检查温度过高 (>1300℃)
        very_high_mask = df_with_temp[temp_col] > JIANDE_WARNING_THRESHOLDS['very_high_furnace_temp']
        for _, row in df_with_temp[very_high_mask].iterrows():
            warnings.append({
                '时间': row['数据时间'],
                '炉号': '1',
                '预警/报警类型': '预警',
                '预警/报警事件': '炉膛温度过高',
                '预警/报警区分': '预警'
            })

        # 检查温度偏高 (>1200℃ 且 ≤1300℃)
        high_mask = (df_with_temp[temp_col] > JIANDE_WARNING_THRESHOLDS['high_furnace_temp']) & \
                   (df_with_temp[temp_col] <= JIANDE_WARNING_THRESHOLDS['very_high_furnace_temp'])
        for _, row in df_with_temp[high_mask].iterrows():
            warnings.append({
                '时间': row['数据时间'],
                '炉号': '1',
                '预警/报警类型': '预警',
                '预警/报警事件': '炉膛温度偏高',
                '预警/报警区分': '预警'
            })

        return warnings

    def check_bag_pressure_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查布袋除尘器压力损失预警 (建德单炉) - 连续状态跟踪"""
        warnings = []

        # 按时间排序确保正确的状态跟踪
        df_sorted = df.sort_values('数据时间')

        # 获取压力字段
        pressure_field = JIANDE_FIELD_MAPPING.get('bag_pressure')

        if not pressure_field or pressure_field not in df_sorted.columns:
            return warnings

        # 状态跟踪变量
        high_pressure_start = None
        low_pressure_start = None

        for _, row in df_sorted.iterrows():
            current_time = row['数据时间']
            pressure_value = row[pressure_field]

            if pd.isna(pressure_value):
                continue

            # 检查压力偏高状态 (>2000Pa)
            if pressure_value > JIANDE_WARNING_THRESHOLDS['bag_pressure_high']:
                if high_pressure_start is None:
                    # 开始新的高压预警
                    high_pressure_start = current_time
            elif high_pressure_start is not None:
                # 结束高压预警
                warnings.append({
                    '时间': high_pressure_start,
                    '炉号': '1',
                    '预警/报警类型': '预警',
                    '预警/报警事件': '布袋除尘器压力损失偏高',
                    '预警/报警区分': '预警'
                })
                high_pressure_start = None

            # 检查压力偏低状态 (<500Pa)
            if pressure_value < JIANDE_WARNING_THRESHOLDS['bag_pressure_low']:
                if low_pressure_start is None:
                    # 开始新的低压预警
                    low_pressure_start = current_time
            elif low_pressure_start is not None:
                # 结束低压预警
                warnings.append({
                    '时间': low_pressure_start,
                    '炉号': '1',
                    '预警/报警类型': '预警',
                    '预警/报警事件': '布袋除尘器压力损失偏低',
                    '预警/报警区分': '预警'
                })
                low_pressure_start = None

        # 处理到数据结束时仍在进行的预警
        if high_pressure_start is not None:
            warnings.append({
                '时间': high_pressure_start,
                '炉号': '1',
                '预警/报警类型': '预警',
                '预警/报警事件': '布袋除尘器压力损失偏高',
                '预警/报警区分': '预警'
            })

        if low_pressure_start is not None:
            warnings.append({
                '时间': low_pressure_start,
                '炉号': '1',
                '预警/报警类型': '预警',
                '预警/报警事件': '布袋除尘器压力损失偏低',
                '预警/报警区分': '预警'
            })

        return warnings

    def check_o2_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查焚烧炉出口氧含量预警 (建德单炉) - 连续状态跟踪"""
        warnings = []

        # 按时间排序确保正确的状态跟踪
        df_sorted = df.sort_values('数据时间')

        # 获取氧含量字段
        o2_field = JIANDE_FIELD_MAPPING.get('o2')

        if not o2_field or o2_field not in df_sorted.columns:
            return warnings

        # 状态跟踪变量
        high_o2_start = None
        low_o2_start = None

        for _, row in df_sorted.iterrows():
            current_time = row['数据时间']
            o2_value = row[o2_field]

            if pd.isna(o2_value):
                continue

            # 检查氧含量偏高状态 (>10%)
            if o2_value > JIANDE_WARNING_THRESHOLDS['o2_high']:
                if high_o2_start is None:
                    # 开始新的高氧含量预警
                    high_o2_start = current_time
            elif high_o2_start is not None:
                # 结束高氧含量预警
                warnings.append({
                    '时间': high_o2_start,
                    '炉号': '1',
                    '预警/报警类型': '预警',
                    '预警/报警事件': '焚烧炉出口氧含量偏高',
                    '预警/报警区分': '预警'
                })
                high_o2_start = None

            # 检查氧含量偏低状态 (<6%)
            if o2_value < JIANDE_WARNING_THRESHOLDS['o2_low']:
                if low_o2_start is None:
                    # 开始新的低氧含量预警
                    low_o2_start = current_time
            elif low_o2_start is not None:
                # 结束低氧含量预警
                warnings.append({
                    '时间': low_o2_start,
                    '炉号': '1',
                    '预警/报警类型': '预警',
                    '预警/报警事件': '焚烧炉出口氧含量偏低',
                    '预警/报警区分': '预警'
                })
                low_o2_start = None

        # 处理到数据结束时仍在进行的预警
        if high_o2_start is not None:
            warnings.append({
                '时间': high_o2_start,
                '炉号': '1',
                '预警/报警类型': '预警',
                '预警/报警事件': '焚烧炉出口氧含量偏高',
                '预警/报警区分': '预警'
            })

        if low_o2_start is not None:
            warnings.append({
                '时间': low_o2_start,
                '炉号': '1',
                '预警/报警类型': '预警',
                '预警/报警事件': '焚烧炉出口氧含量偏低',
                '预警/报警区分': '预警'
            })

        return warnings

    def calculate_corrected_concentration(self, measured_conc, measured_o2):
        """计算标准状态下的浓度（折算）"""
        # ρ（标准）=ρ（实测）*10/(21-ρ（实测O2））
        corrected = measured_conc * 10 / (21 - measured_o2)
        return corrected

    def check_pollutant_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查污染物浓度预警 (建德单炉) - 小时均值，需要折算"""
        warnings = []

        # 计算1小时平均浓度
        df_1hour = self.calculate_time_windows(df, '1hour')

        # 获取氧含量字段用于折算
        o2_field = JIANDE_FIELD_MAPPING.get('o2')

        if not o2_field or o2_field not in df_1hour.columns:
            print("警告：未找到氧含量字段，无法进行污染物浓度折算")
            return warnings

        # 检查各种污染物（需要折算的）
        pollutants = {
            'dust': ('烟气中颗粒物（PM）浓度较高', 'dust_warning_limit'),
            'nox': ('烟气中氮氧化物（NOx）浓度较高', 'nox_warning_limit'),
            'so2': ('烟气中二氧化硫（SO₂）浓度较高', 'so2_warning_limit'),
            'hcl': ('烟气中氯化氢（HCl）浓度较高', 'hcl_warning_limit'),
            'co': ('烟气中一氧化碳（CO）浓度较高', 'co_warning_limit')
        }

        for pollutant, (event_name, threshold_key) in pollutants.items():
            field = JIANDE_FIELD_MAPPING.get(pollutant)

            if field and field in df_1hour.columns:
                threshold = JIANDE_WARNING_THRESHOLDS[threshold_key]

                # 计算折算后的浓度
                measured_conc = df_1hour[field].dropna()
                measured_o2 = df_1hour[o2_field].dropna()

                if len(measured_conc) > 0 and len(measured_o2) > 0:
                    # 确保数据长度一致
                    min_len = min(len(measured_conc), len(measured_o2))
                    if min_len > 0:
                        corrected_conc = self.calculate_corrected_concentration(
                            measured_conc.iloc[:min_len], measured_o2.iloc[:min_len]
                        )

                        # 检查是否超过阈值
                        high_mask = corrected_conc > threshold

                        for i, is_high in enumerate(high_mask):
                            if is_high:
                                warnings.append({
                                    '时间': df_1hour.iloc[i]['数据时间'],
                                    '炉号': '1',
                                    '预警/报警类型': '预警',
                                    '预警/报警事件': event_name,
                                    '预警/报警区分': '预警'
                                })

        return warnings

    def check_pollutant_alarm(self, df: pd.DataFrame) -> List[Dict]:
        """检查污染物浓度报警 (建德单炉) - 日均值，需要折算"""
        alarms = []

        # 计算24小时平均浓度（日均值）
        df_daily = self.calculate_time_windows(df, '1day')

        # 获取氧含量字段用于折算
        o2_field = JIANDE_FIELD_MAPPING.get('o2')

        if not o2_field or o2_field not in df_daily.columns:
            print("警告：未找到氧含量字段，无法进行污染物浓度折算")
            return alarms

        # 检查各种污染物（需要折算的）
        pollutants = {
            'dust': ('烟气中颗粒物（PM）排放超标', 'dust_alarm_limit'),
            'nox': ('烟气中氮氧化物（NOx）排放超标', 'nox_alarm_limit'),
            'so2': ('烟气中二氧化硫（SO₂）排放超标', 'so2_alarm_limit'),
            'hcl': ('烟气中氯化氢（HCl）排放超标', 'hcl_alarm_limit'),
            'co': ('烟气中一氧化碳（CO）排放超标', 'co_alarm_limit')
        }

        for pollutant, (event_name, threshold_key) in pollutants.items():
            field = JIANDE_FIELD_MAPPING.get(pollutant)

            if field and field in df_daily.columns:
                threshold = JIANDE_ALARM_THRESHOLDS[threshold_key]

                # 计算折算后的浓度
                measured_conc = df_daily[field].dropna()
                measured_o2 = df_daily[o2_field].dropna()

                if len(measured_conc) > 0 and len(measured_o2) > 0:
                    # 确保数据长度一致
                    min_len = min(len(measured_conc), len(measured_o2))
                    if min_len > 0:
                        corrected_conc = self.calculate_corrected_concentration(
                            measured_conc.iloc[:min_len], measured_o2.iloc[:min_len]
                        )

                        # 检查是否超过阈值
                        high_mask = corrected_conc > threshold

                        for i, is_high in enumerate(high_mask):
                            if is_high:
                                alarms.append({
                                    '时间': df_daily.iloc[i]['数据时间'],
                                    '炉号': '1',
                                    '预警/报警类型': '报警',
                                    '预警/报警事件': event_name,
                                    '预警/报警区分': '报警'
                                })

        return alarms

    # 注意：建德预警规则中没有活性炭投加量预警，已删除该方法

    def process_data(self, file_path: str, output_dir: str = None) -> pd.DataFrame:
        """处理数据并生成预警报告"""
        # 加载数据
        df = self.load_data(file_path)
        if df.empty:
            return pd.DataFrame()

        # 清空之前的预警事件
        self.warning_events = []

        print(f"\n检查建德焚烧炉预警报警 (1个炉子)...")

        # === 预警规则 ===
        # 1. 瞬时低炉温焚烧预警
        low_temp_warnings = self.check_low_furnace_temp_warning(df)
        self.warning_events.extend(low_temp_warnings)
        print(f"瞬时低炉温预警: {len(low_temp_warnings)} 条")

        # 2. 炉膛温度偏高/过高预警
        high_temp_warnings = self.check_high_furnace_temp_warning(df)
        self.warning_events.extend(high_temp_warnings)
        print(f"高炉温预警: {len(high_temp_warnings)} 条")

        # 3. 布袋除尘器压力损失预警
        pressure_warnings = self.check_bag_pressure_warning(df)
        self.warning_events.extend(pressure_warnings)
        print(f"压力预警: {len(pressure_warnings)} 条")

        # 4. 焚烧炉出口氧含量预警
        o2_warnings = self.check_o2_warning(df)
        self.warning_events.extend(o2_warnings)
        print(f"氧含量预警: {len(o2_warnings)} 条")

        # 5. 污染物浓度预警（小时均值）
        pollutant_warnings = self.check_pollutant_warning(df)
        self.warning_events.extend(pollutant_warnings)
        print(f"污染物预警: {len(pollutant_warnings)} 条")

        # === 报警规则 ===
        # 1. 低炉温焚烧报警
        low_temp_alarms = self.check_low_furnace_temp_alarm(df)
        self.warning_events.extend(low_temp_alarms)
        print(f"低炉温报警: {len(low_temp_alarms)} 条")

        # 2. 污染物排放超标报警（日均值）
        pollutant_alarms = self.check_pollutant_alarm(df)
        self.warning_events.extend(pollutant_alarms)
        print(f"污染物报警: {len(pollutant_alarms)} 条")

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
        """保存预警报警报告"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存Excel格式
        excel_file = os.path.join(output_dir, f"{base_name}_建德预警报警报告_{timestamp}.xlsx")
        warning_df.to_excel(excel_file, index=False)
        print(f"📊 预警报警报告已保存: {excel_file}")

        # 保存CSV格式 (与输出模板格式一致)
        csv_file = os.path.join(output_dir, f"{base_name}_建德预警报警报告_{timestamp}.csv")

        # 确保所有记录都有预警/报警区分列
        if '预警/报警区分' not in warning_df.columns:
            # 如果没有该列，根据预警/报警类型自动填充
            warning_df['预警/报警区分'] = warning_df['预警/报警类型']

        # 保留模板需要的列
        required_columns = ['时间', '炉号', '预警/报警类型', '预警/报警事件', '预警/报警区分']
        template_df = warning_df[required_columns].copy()
        template_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"📋 CSV报告已保存: {csv_file}")

        # 生成统计报告
        stats_file = os.path.join(output_dir, f"{base_name}_建德预警报警统计_{timestamp}.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"建德垃圾焚烧预警报警统计报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {input_file}\n")
            f.write(f"总事件数量: {len(warning_df)}\n\n")

            # 按预警/报警类型统计
            type_stats = warning_df['预警/报警类型'].value_counts()
            f.write("事件类型统计:\n")
            for event_type, count in type_stats.items():
                f.write(f"  {event_type}: {count} 条\n")

            # 按事件详细统计
            event_stats = warning_df['预警/报警事件'].value_counts()
            f.write("\n事件详细统计:\n")
            for event, count in event_stats.items():
                f.write(f"  {event}: {count} 条\n")

            f.write(f"\n1号炉总事件数: {len(warning_df)} 条\n")

        print(f"📈 统计报告已保存: {stats_file}")


def main():
    """主函数 - 支持命令行和直接运行"""
    import sys

    # 配置区域 - 可以直接修改这里的路径
    DEFAULT_INPUT_FILE = "6.1_process.csv"  # 默认输入文件
    DEFAULT_OUTPUT_DIR = "./预警输出"  # 默认输出目录

    if len(sys.argv) >= 2:
        # 命令行模式
        input_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    else:
        # 直接运行模式
        print("🚀 直接运行模式")
        print("💡 提示: 可以修改代码中的DEFAULT_INPUT_FILE变量来指定要分析的文件")
        print("💡 提示: 使用 python shishi_data_yujing_gz.py <文件路径> 分析指定文件")

        input_file = DEFAULT_INPUT_FILE
        output_dir = DEFAULT_OUTPUT_DIR

        print(f"📁 使用默认输入文件: {input_file}")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在 - {input_file}")
        print("请修改代码中的DEFAULT_INPUT_FILE变量或使用命令行参数")
        return

    # 创建预警系统实例
    print("\n🔧 创建建德预警系统实例...")
    warning_system = WasteIncinerationWarningSystemJiande()

    # 处理数据
    print(f"📊 开始处理数据文件: {input_file}")
    try:
        warning_df = warning_system.process_data(input_file, output_dir)

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
                print(f"  {i+1}. {row['时间']} - {row['预警/报警事件']} ({row['预警/报警类型']})")
        else:
            print("\n✅ 数据处理完成，未发现预警报警事件。")

    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()