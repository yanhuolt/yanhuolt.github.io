#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
垃圾焚烧预警报警系统 - 光大江山 (单炉配置)
根据光大江山的预警报警规则实现
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import os
import warnings
warnings.filterwarnings('ignore')

# 光大江山数据字段映射 (基于7.3.csv的字段结构 - 单炉配置)
GUANGDA_JIANGSHAN_FIELD_MAPPING = {
    # 炉膛温度相关字段 (单炉)
    # 根据预警规则：上部温度F，中部温度G
    # 根据报警规则：光大江山只有1号炉，对应U1、V1、W1、X1、Y1、Z1点位
    # 实际CSV字段名
    "furnace_top_temp": "炉膛上部温度",      # F - 上部断面温度
    "furnace_mid_temp": "炉膛中部温度",      # G - 中部断面温度

    # 布袋除尘器压差
    "bag_pressure": "布袋进出口压差",        # AK - 布袋除尘器压力损失

    # 氧含量
    "o2": "烟气O2",                         # AR - 焚烧炉出口氧含量

    # 活性炭投加量
    "carbon_dosage": "活性炭喷射量",         # 活性炭投加量

    # 污染物浓度 - 预警规则 (小时均值)
    # 根据预警规则中的数据采集点位
    "dust": "烟气粉尘",                     # AS - 颗粒物预警
    "nox": "烟气Nox",                       # AU - 氮氧化物预警
    "so2": "烟气SO2",                       # AT - 二氧化硫预警
    "hcl": "烟气HCL",                       # AW - 氯化氢预警
    "co": "烟气CO",                         # AV - 一氧化碳预警

    # 污染物浓度 - 报警规则 (日均值)
    # 根据报警规则中的数据采集点位：光大江山只有1号炉
    "dust_alarm": "烟气粉尘",               # ES - 颗粒物报警 (1号炉)
    "nox_alarm": "烟气Nox",                 # EU - 氮氧化物报警 (1号炉)
    "so2_alarm": "烟气SO2",                 # ET - 二氧化硫报警 (1号炉)
    "hcl_alarm": "烟气HCL",                 # EW - 氯化氢报警 (1号炉)
    "co_alarm": "烟气CO",                   # EV - 一氧化碳报警 (1号炉)
}

# 光大江山预警报警阈值配置
GUANGDA_JIANGSHAN_THRESHOLDS = {
    # === 预警阈值 (根据之前提供的预警规则) ===
    # 1. 瞬时低炉温焚烧预警 (5分钟平均值)
    "low_furnace_temp_warning": 850,  # 低于850℃触发预警

    # 5. 炉膛温度偏高预警 (1小时平均值)
    "high_furnace_temp_warning": 1200,  # 高于1200℃触发预警

    # 6. 炉膛温度过高预警 (1小时平均值)
    "very_high_furnace_temp_warning": 1300,  # 高于1300℃触发预警

    # 8. 布袋除尘器压力损失偏高预警
    "bag_pressure_high_warning": 2000,  # 高于2000Pa触发预警

    # 9. 布袋除尘器压力损失偏低预警
    "bag_pressure_low_warning": 500,    # 低于500Pa触发预警

    # 12. 焚烧炉出口氧含量偏高预警
    "o2_high_warning": 10,  # 高于10%触发预警

    # 13. 焚烧炉出口氧含量偏低预警
    "o2_low_warning": 6,    # 低于6%触发预警

    # 14. 活性炭投加量不足预警
    "carbon_dosage_low_warning": 3.0,  # 低于3.0kg/h触发预警

    # 17-21. 污染物浓度预警阈值 (小时均值)
    "dust_warning": 30,     # 颗粒物 ≤30mg/m³
    "nox_warning": 300,     # 氮氧化物 ≤300mg/m³
    "so2_warning": 100,     # 二氧化硫 ≤100mg/m³
    "hcl_warning": 60,      # 氯化氢 ≤60mg/m³
    "co_warning": 100,      # 一氧化碳 ≤100mg/m³

    # === 报警阈值 (根据报警规则CSV文件) ===
    # 1. 低炉温焚烧报警 (5分钟平均值) - 与预警阈值相同
    "low_furnace_temp_alarm": 850,  # 低于850℃触发报警

    # 2-6. 污染物日均值排放超标报警阈值 (根据报警规则CSV)
    "dust_daily_alarm": 20,   # 颗粒物（PM）日均值 ≤20mg/m³
    "nox_daily_alarm": 250,   # 氮氧化物（NOx）日均值 ≤250mg/m³
    "so2_daily_alarm": 80,    # 二氧化硫（SO₂）日均值 ≤80mg/m³
    "hcl_daily_alarm": 50,    # 氯化氢（HCl）日均值 ≤50mg/m³
    "co_daily_alarm": 80,     # 一氧化碳（CO）日均值 ≤80mg/m³
}

class WasteIncinerationWarningSystemGuangdaJiangshan:
    """垃圾焚烧预警报警系统 - 光大江山 (单炉配置)"""

    def __init__(self):
        self.warning_events = []
        self.warning_states = {}  # 用于跟踪连续预警状态
        self.furnace_count = 1  # 光大江山只有1个炉子

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据文件 (支持csv和xlsx)"""
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                raise ValueError("不支持的文件格式，请使用csv或xlsx文件")

            if df.empty:
                print("警告: 数据文件为空")
                return pd.DataFrame()

            # 转换时间列
            if '数据时间' in df.columns:
                df['数据时间'] = pd.to_datetime(df['数据时间'])
            else:
                print("警告: 未找到'数据时间'列")
                return pd.DataFrame()

            # 数据清洗：处理非数值数据
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != '数据时间':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"成功加载数据文件: {file_path}")
            print(f"数据行数: {len(df)}, 列数: {len(df.columns)}")
            return df

        except Exception as e:
            print(f"加载数据文件失败: {e}")
            return pd.DataFrame()

    def calculate_time_windows(self, df: pd.DataFrame, window: str) -> pd.DataFrame:
        """计算时间窗口的平均值"""
        if df.empty or '数据时间' not in df.columns:
            return df

        try:
            # 设置时间索引
            df_temp = df.set_index('数据时间')
            
            # 根据窗口类型重采样
            if window == '5min':
                resampled = df_temp.resample('5min').mean()
            elif window == '1hour' or window == '1H':
                resampled = df_temp.resample('1H').mean()
            elif window == '1day' or window == '1D':
                resampled = df_temp.resample('1D').mean()
            else:
                print(f"不支持的时间窗口: {window}")
                return df

        except Exception as e:
            print(f"时间窗口计算失败: {e}")
            return df

        resampled.reset_index(inplace=True)
        return resampled

    def calculate_furnace_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算炉膛温度 - 根据预警报警规则

        规则要求：
        1. 对焚烧炉炉膛的中部和上部两个断面，各自取所有热电偶测量温度的中位数
        2. 计算这两个中位数的算术平均值，作为该断面的代表温度

        注意：光大江山实际只有一个上部温度和一个中部温度字段，
        所以直接使用这两个值的算术平均值作为炉膛温度
        """
        top_temp_field = GUANGDA_JIANGSHAN_FIELD_MAPPING.get('furnace_top_temp')
        mid_temp_field = GUANGDA_JIANGSHAN_FIELD_MAPPING.get('furnace_mid_temp')

        if top_temp_field in df.columns and mid_temp_field in df.columns:
            # 由于光大江山数据中每个断面只有一个温度值，直接计算算术平均值
            # 如果有多个热电偶，应该先计算各断面的中位数，再计算平均值
            df['炉膛温度'] = (df[top_temp_field] + df[mid_temp_field]) / 2

            # 处理异常值和缺失值
            df['炉膛温度'] = df['炉膛温度'].replace([np.inf, -np.inf], np.nan)

        else:
            print(f"警告: 缺少温度字段 {top_temp_field} 或 {mid_temp_field}")
            # 如果缺少字段，创建一个空的炉膛温度列
            df['炉膛温度'] = np.nan

        return df

    # === 预警检查函数 ===

    def check_low_furnace_temp_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查瞬时低炉温焚烧预警 (5分钟平均值)"""
        warnings = []

        # 计算5分钟平均温度
        df_5min = self.calculate_time_windows(df, '5min')
        df_5min = self.calculate_furnace_temperature(df_5min)

        if '炉膛温度' not in df_5min.columns:
            return warnings

        # 检查低于850℃的情况
        low_temp_mask = df_5min['炉膛温度'] < GUANGDA_JIANGSHAN_THRESHOLDS['low_furnace_temp_warning']

        for _, row in df_5min[low_temp_mask].iterrows():
            warnings.append({
                '时间': row['数据时间'],
                '炉号': '1',
                '预警/报警类型': '预警',
                '预警/报警事件': '瞬时低炉温焚烧',
                '预警/报警区分': '预警'
            })

        return warnings

    def check_high_furnace_temp_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查炉膛温度偏高/过高预警 (1小时平均值)"""
        warnings = []

        # 计算1小时平均温度
        df_1hour = self.calculate_time_windows(df, '1hour')
        df_1hour = self.calculate_furnace_temperature(df_1hour)

        if '炉膛温度' not in df_1hour.columns:
            return warnings

        # 检查温度过高 (>1300℃)
        very_high_mask = df_1hour['炉膛温度'] > GUANGDA_JIANGSHAN_THRESHOLDS['very_high_furnace_temp_warning']
        for _, row in df_1hour[very_high_mask].iterrows():
            warnings.append({
                '时间': row['数据时间'],
                '炉号': '1',
                '预警/报警类型': '预警',
                '预警/报警事件': '炉膛温度过高',
                '预警/报警区分': '预警'
            })

        # 检查温度偏高 (>1200℃ 且 ≤1300℃)
        high_mask = (df_1hour['炉膛温度'] > GUANGDA_JIANGSHAN_THRESHOLDS['high_furnace_temp_warning']) & \
                   (df_1hour['炉膛温度'] <= GUANGDA_JIANGSHAN_THRESHOLDS['very_high_furnace_temp_warning'])
        for _, row in df_1hour[high_mask].iterrows():
            warnings.append({
                '时间': row['数据时间'],
                '炉号': '1',
                '预警/报警类型': '预警',
                '预警/报警事件': '炉膛温度偏高',
                '预警/报警区分': '预警'
            })

        return warnings

    def check_bag_pressure_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查布袋除尘器压力损失预警 - 连续状态跟踪"""
        warnings = []

        # 按时间排序确保正确的状态跟踪
        df_sorted = df.sort_values('数据时间')

        pressure_field = GUANGDA_JIANGSHAN_FIELD_MAPPING.get('bag_pressure')
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

            # 检查压力偏高 (>2000Pa)
            if pressure_value > GUANGDA_JIANGSHAN_THRESHOLDS['bag_pressure_high_warning']:
                if high_pressure_start is None:
                    high_pressure_start = current_time
            else:
                if high_pressure_start is not None:
                    warnings.append({
                        '时间': high_pressure_start,
                        '炉号': '1',
                        '预警/报警类型': '预警',
                        '预警/报警事件': '布袋除尘器压力损失偏高',
                        '预警/报警区分': '预警'
                    })
                    high_pressure_start = None

            # 检查压力偏低 (<500Pa)
            if pressure_value < GUANGDA_JIANGSHAN_THRESHOLDS['bag_pressure_low_warning']:
                if low_pressure_start is None:
                    low_pressure_start = current_time
            else:
                if low_pressure_start is not None:
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
        """检查焚烧炉出口氧含量预警 - 连续状态跟踪"""
        warnings = []

        # 按时间排序确保正确的状态跟踪
        df_sorted = df.sort_values('数据时间')

        o2_field = GUANGDA_JIANGSHAN_FIELD_MAPPING.get('o2')
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

            # 检查氧含量偏高 (>10%)
            if o2_value > GUANGDA_JIANGSHAN_THRESHOLDS['o2_high_warning']:
                if high_o2_start is None:
                    high_o2_start = current_time
            else:
                if high_o2_start is not None:
                    warnings.append({
                        '时间': high_o2_start,
                        '炉号': '1',
                        '预警/报警类型': '预警',
                        '预警/报警事件': '焚烧炉出口氧含量偏高',
                        '预警/报警区分': '预警'
                    })
                    high_o2_start = None

            # 检查氧含量偏低 (<6%)
            if o2_value < GUANGDA_JIANGSHAN_THRESHOLDS['o2_low_warning']:
                if low_o2_start is None:
                    low_o2_start = current_time
            else:
                if low_o2_start is not None:
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

    def check_carbon_dosage_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查活性炭投加量不足预警 - 连续状态跟踪"""
        warnings = []

        # 按时间排序确保正确的状态跟踪
        df_sorted = df.sort_values('数据时间')

        carbon_field = GUANGDA_JIANGSHAN_FIELD_MAPPING.get('carbon_dosage')
        if not carbon_field or carbon_field not in df_sorted.columns:
            return warnings

        # 状态跟踪变量
        low_carbon_start = None

        for _, row in df_sorted.iterrows():
            current_time = row['数据时间']
            carbon_value = row[carbon_field]

            if pd.isna(carbon_value):
                continue

            # 检查活性炭投加量不足 (<3.0kg/h)
            if carbon_value < GUANGDA_JIANGSHAN_THRESHOLDS['carbon_dosage_low_warning']:
                if low_carbon_start is None:
                    low_carbon_start = current_time
            else:
                if low_carbon_start is not None:
                    warnings.append({
                        '时间': low_carbon_start,
                        '炉号': '1',
                        '预警/报警类型': '预警',
                        '预警/报警事件': '活性炭投加量不足',
                        '预警/报警区分': '预警'
                    })
                    low_carbon_start = None

        # 处理到数据结束时仍在进行的预警
        if low_carbon_start is not None:
            warnings.append({
                '时间': low_carbon_start,
                '炉号': '1',
                '预警/报警类型': '预警',
                '预警/报警事件': '活性炭投加量不足',
                '预警/报警区分': '预警'
            })

        return warnings

    def check_pollutant_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查污染物浓度预警 (小时均值)"""
        warnings = []

        # 计算1小时平均值
        df_1hour = self.calculate_time_windows(df, '1hour')

        # 检查各种污染物
        pollutants = {
            'dust': ('烟气中颗粒物（PM）浓度较高', 'dust_warning'),
            'nox': ('烟气中氮氧化物（NOx）浓度较高', 'nox_warning'),
            'so2': ('烟气中二氧化硫（SO₂）浓度较高', 'so2_warning'),
            'hcl': ('烟气中氯化氢（HCl）浓度较高', 'hcl_warning'),
            'co': ('烟气中一氧化碳（CO）浓度较高', 'co_warning')
        }

        for pollutant, (event_name, threshold_key) in pollutants.items():
            field = GUANGDA_JIANGSHAN_FIELD_MAPPING.get(pollutant)

            if field and field in df_1hour.columns:
                threshold = GUANGDA_JIANGSHAN_THRESHOLDS[threshold_key]
                high_mask = df_1hour[field] > threshold

                for _, row in df_1hour[high_mask].iterrows():
                    warnings.append({
                        '时间': row['数据时间'],
                        '炉号': '1',
                        '预警/报警类型': '预警',
                        '预警/报警事件': event_name,
                        '预警/报警区分': '预警'
                    })

        return warnings

    # === 报警检查函数 ===

    def check_low_furnace_temp_alarm(self, df: pd.DataFrame) -> List[Dict]:
        """检查低炉温焚烧报警 (5分钟平均值)"""
        alarms = []

        # 计算5分钟平均温度
        df_5min = self.calculate_time_windows(df, '5min')
        df_5min = self.calculate_furnace_temperature(df_5min)

        if '炉膛温度' not in df_5min.columns:
            return alarms

        # 检查低于850℃的情况 - 触发报警
        low_temp_mask = df_5min['炉膛温度'] < GUANGDA_JIANGSHAN_THRESHOLDS['low_furnace_temp_alarm']

        for _, row in df_5min[low_temp_mask].iterrows():
            alarms.append({
                '时间': row['数据时间'],
                '炉号': '1',
                '预警/报警类型': '报警',
                '预警/报警事件': '低炉温焚烧',
                '预警/报警区分': '报警'
            })

        return alarms

    def check_pollutant_daily_alarm(self, df: pd.DataFrame) -> List[Dict]:
        """检查污染物日均值排放超标报警 - 根据报警规则CSV文件"""
        alarms = []

        # 按日期分组计算日均值
        df_daily = self.calculate_time_windows(df, '1day')

        # 根据报警规则CSV文件，检查各种污染物日均值超标情况
        # 分析报警规则：虽然有些规则中写的是"触发预警"，但这些都是报警类型的规则
        # 所有污染物日均值排放超标都应该归类为"报警"
        pollutants = {
            # 序号2: 烟气中颗粒物（PM）排放超标 - 日均值≤20mg/m³
            'dust_alarm': ('烟气中颗粒物（PM）排放超标', 'dust_daily_alarm', '报警'),
            # 序号3: 烟气中氮氧化物（NOx）排放超标 - 日均值≤250mg/m³
            'nox_alarm': ('烟气中氮氧化物（NOx）排放超标', 'nox_daily_alarm', '报警'),
            # 序号4: 烟气中二氧化硫（SO₂）排放超标 - 日均值≤80mg/m³
            'so2_alarm': ('烟气中二氧化硫（SO₂）排放超标', 'so2_daily_alarm', '报警'),
            # 序号5: 烟气中氯化氢（HCl）排放超标 - 日均值≤50mg/m³
            'hcl_alarm': ('烟气中氯化氢（HCl）排放超标', 'hcl_daily_alarm', '报警'),
            # 序号6: 烟气中一氧化碳（CO）排放超标 - 日均值≤80mg/m³
            'co_alarm': ('烟气中一氧化碳（CO）排放超标', 'co_daily_alarm', '报警')
        }

        for pollutant, (event_name, threshold_key, alarm_type) in pollutants.items():
            field = GUANGDA_JIANGSHAN_FIELD_MAPPING.get(pollutant)

            if field and field in df_daily.columns:
                threshold = GUANGDA_JIANGSHAN_THRESHOLDS[threshold_key]
                exceed_mask = df_daily[field] > threshold

                for _, row in df_daily[exceed_mask].iterrows():
                    alarms.append({
                        '时间': row['数据时间'],
                        '炉号': '1',
                        '预警/报警类型': alarm_type,
                        '预警/报警事件': event_name,
                        '预警/报警区分': alarm_type
                    })

        return alarms

    def process_data(self, file_path: str, output_dir: str = None) -> pd.DataFrame:
        """处理数据并生成预警报警报告"""
        # 加载数据
        df = self.load_data(file_path)
        if df.empty:
            return pd.DataFrame()

        # 清空之前的预警事件
        self.warning_events = []

        print(f"\n检查光大江山焚烧炉预警报警 (1个炉子)...")

        # === 预警检查 ===
        # 1. 瞬时低炉温焚烧预警
        low_temp_warnings = self.check_low_furnace_temp_warning(df)
        self.warning_events.extend(low_temp_warnings)
        print(f"低炉温预警: {len(low_temp_warnings)} 条")

        # 5-6. 炉膛温度偏高/过高预警
        high_temp_warnings = self.check_high_furnace_temp_warning(df)
        self.warning_events.extend(high_temp_warnings)
        print(f"高炉温预警: {len(high_temp_warnings)} 条")

        # 8-9. 布袋除尘器压力损失预警
        pressure_warnings = self.check_bag_pressure_warning(df)
        self.warning_events.extend(pressure_warnings)
        print(f"压力预警: {len(pressure_warnings)} 条")

        # 12-13. 焚烧炉出口氧含量预警
        o2_warnings = self.check_o2_warning(df)
        self.warning_events.extend(o2_warnings)
        print(f"氧含量预警: {len(o2_warnings)} 条")

        # 14. 活性炭投加量预警
        carbon_warnings = self.check_carbon_dosage_warning(df)
        self.warning_events.extend(carbon_warnings)
        print(f"活性炭预警: {len(carbon_warnings)} 条")

        # 17-21. 污染物浓度预警 (小时均值)
        pollutant_warnings = self.check_pollutant_warning(df)
        self.warning_events.extend(pollutant_warnings)
        print(f"污染物浓度预警: {len(pollutant_warnings)} 条")

        # === 报警检查 ===
        # 1. 低炉温焚烧报警
        low_temp_alarms = self.check_low_furnace_temp_alarm(df)
        self.warning_events.extend(low_temp_alarms)
        print(f"低炉温报警: {len(low_temp_alarms)} 条")

        # 2-6. 污染物日均值排放超标报警
        pollutant_daily_alarms = self.check_pollutant_daily_alarm(df)
        self.warning_events.extend(pollutant_daily_alarms)
        print(f"污染物日均值超标报警: {len(pollutant_daily_alarms)} 条")

        # 转换为DataFrame
        if self.warning_events:
            warning_df = pd.DataFrame(self.warning_events)
            # 按时间排序
            warning_df = warning_df.sort_values('时间')

            print(f"\n共检测到 {len(warning_df)} 条预警报警事件")

            # 保存预警报告
            if output_dir:
                self.save_warning_report(warning_df, output_dir, file_path)

            return warning_df
        else:
            print("\n未检测到预警报警事件")
            return pd.DataFrame()

    def save_warning_report(self, warning_df: pd.DataFrame, output_dir: str, input_file: str):
        """保存预警报告"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存Excel格式
        excel_file = os.path.join(output_dir, f"{base_name}_光大江山预警报告_{timestamp}.xlsx")
        warning_df.to_excel(excel_file, index=False)
        print(f"预警报告已保存: {excel_file}")

        # 保存CSV格式 (与输出模板格式一致，包含预警/报警区分列)
        csv_file = os.path.join(output_dir, f"{base_name}_光大江山预警报告_{timestamp}.csv")

        # 确保所有记录都有预警/报警区分列
        if '预警/报警区分' not in warning_df.columns:
            # 如果没有该列，根据预警/报警类型自动填充
            warning_df['预警/报警区分'] = warning_df['预警/报警类型']

        # 保留模板需要的列，包含预警/报警区分
        required_columns = ['时间', '炉号', '预警/报警类型', '预警/报警事件', '预警/报警区分']
        template_df = warning_df[required_columns].copy()
        template_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"预警报告(模板格式)已保存: {csv_file}")


def main():
    """主函数 - 命令行接口"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python waste_incineration_warning_system_guangda_jiangshan.py <数据文件路径> [输出目录]")
        print("示例: python waste_incineration_warning_system_guangda_jiangshan.py 数据下/数据下/光大（江山）/7.3.csv ./output")
        return

    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"

    # 创建预警系统实例
    warning_system = WasteIncinerationWarningSystemGuangdaJiangshan()

    # 处理数据
    result_df = warning_system.process_data(file_path, output_dir)

    if not result_df.empty:
        print(f"\n处理完成！共生成 {len(result_df)} 条预警报警记录")
    else:
        print("\n处理完成！未发现预警报警事件")


if __name__ == "__main__":
    main()
