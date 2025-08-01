#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开化天汇垃圾焚烧预警报警系统
基于预警规则和报警规则实现完整的预警报警功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import os
import warnings
warnings.filterwarnings('ignore')

# 开化天汇数据字段映射 (基于5.23.csv的字段结构 - 单炉配置)
KAIHUA_TIANHUI_FIELD_MAPPING = {
    # 炉膛温度相关字段 (单炉)
    # 根据预警规则：上部温度L，中部温度M
    # 根据报警规则：开化天汇只有1号炉，对应U1、V1、W1、X1、Y1、Z1点位
    "furnace_top_temp": "炉膛上部温度",      # L - 上部断面温度
    "furnace_mid_temp": "炉膛中部温度",      # M - 中部断面温度
    
    # 布袋除尘器压差
    "bag_pressure": "布袋除尘进出口压差",    # AO - 布袋除尘器压力损失
    
    # 氧含量
    "o2": "烟气O2",                         # AV - 焚烧炉出口氧含量
    
    # 污染物浓度 - 预警规则 (小时均值)
    # 根据预警规则中的数据采集点位
    "dust": "烟气粉尘",                     # AW - 颗粒物预警
    "nox": "烟气Nox",                       # AY - 氮氧化物预警  
    "so2": "烟气SO2",                       # AX - 二氧化硫预警
    "hcl": "烟气HCL",                       # BA - 氯化氢预警
    "co": "烟气CO",                         # AZ - 一氧化碳预警
    
    # 污染物浓度 - 报警规则 (日均值)
    # 根据报警规则中的数据采集点位：开化天汇只有1号炉
    "dust_alarm": "烟气粉尘",               # ES - 颗粒物报警 (1号炉)
    "nox_alarm": "烟气Nox",                 # EU - 氮氧化物报警 (1号炉)
    "so2_alarm": "烟气SO2",                 # ET - 二氧化硫报警 (1号炉)
    "hcl_alarm": "烟气HCL",                 # EW - 氯化氢报警 (1号炉)
    "co_alarm": "烟气CO",                   # EV - 一氧化碳报警 (1号炉)
}

# 开化天汇预警报警阈值配置
KAIHUA_TIANHUI_THRESHOLDS = {
    # 预警阈值
    "warning": {
        # 炉膛温度预警阈值
        "low_furnace_temp": 850,        # 低炉温预警：< 850℃ (5分钟平均)
        "high_furnace_temp": 1200,      # 高炉温预警：> 1200℃ (1小时平均)
        "very_high_furnace_temp": 1300, # 过高炉温预警：> 1300℃ (1小时平均)
        
        # 布袋除尘器压差预警阈值
        "bag_pressure_high": 2000,      # 压差偏高：> 2000Pa (实时)
        "bag_pressure_low": 500,        # 压差偏低：< 500Pa (实时)
        
        # 氧含量预警阈值
        "o2_high": 10,                  # 氧含量偏高：> 10% (实时)
        "o2_low": 6,                    # 氧含量偏低：< 6% (实时)
        
        # 污染物浓度预警阈值 (小时均值)
        "dust_hourly": 30,              # 颗粒物：> 30mg/m³
        "nox_hourly": 300,              # 氮氧化物：> 300mg/m³
        "so2_hourly": 100,              # 二氧化硫：> 100mg/m³
        "hcl_hourly": 60,               # 氯化氢：> 60mg/m³
        "co_hourly": 100,               # 一氧化碳：> 100mg/m³
    },
    
    # 报警阈值
    "alarm": {
        # 炉膛温度报警阈值
        "low_furnace_temp": 850,        # 低炉温报警：< 850℃ (5分钟平均)
        
        # 污染物浓度报警阈值 (日均值)
        "dust_daily": 20,               # 颗粒物：> 20mg/m³
        "nox_daily": 250,               # 氮氧化物：> 250mg/m³
        "so2_daily": 80,                # 二氧化硫：> 80mg/m³
        "hcl_daily": 50,                # 氯化氢：> 50mg/m³
        "co_daily": 80,                 # 一氧化碳：> 80mg/m³
    }
}


class WasteIncinerationWarningSystemKaihuaTianhui:
    """开化天汇垃圾焚烧预警报警系统"""
    
    def __init__(self):
        self.warning_events = []
        self.field_mapping = KAIHUA_TIANHUI_FIELD_MAPPING
        self.thresholds = KAIHUA_TIANHUI_THRESHOLDS
        
    def clean_numeric_data(self, series: pd.Series) -> pd.Series:
        """清理数值数据"""
        # 转换为字符串，处理可能的字符串连接问题
        series_str = series.astype(str)
        
        # 替换常见的非数值字符
        series_str = series_str.str.replace(',', '')
        series_str = series_str.str.replace('，', '')
        series_str = series_str.str.replace(' ', '')
        
        # 转换为数值，无法转换的设为NaN
        series_numeric = pd.to_numeric(series_str, errors='coerce')
        
        # 替换无穷大值为NaN
        series_numeric = series_numeric.replace([np.inf, -np.inf], np.nan)
        
        return series_numeric

    def remove_pollutant_outliers_boxplot(self, series: pd.Series, pollutant_name: str) -> pd.Series:
        """
        使用箱型图方法去除污染物浓度中的异常值（极大值和极小值）

        箱型图异常值检测原理：
        - Q1: 第一四分位数（25%分位数）
        - Q3: 第三四分位数（75%分位数）
        - IQR: 四分位距 = Q3 - Q1
        - 下边界: Q1 - 1.5 * IQR
        - 上边界: Q3 + 1.5 * IQR
        - 超出边界的值被认为是异常值

        参数:
            series: 污染物浓度数据序列
            pollutant_name: 污染物名称（用于日志输出）

        返回:
            去除异常值后的数据序列（异常值用NaN替换）
        """
        if series.empty or series.isna().all():
            return series

        # 去除NaN值和零值进行计算（污染物浓度为0通常不是有效数据）
        valid_data = series[(series.notna()) & (series > 0)]

        if len(valid_data) < 4:  # 数据点太少，无法进行箱型图分析
            print(f"{pollutant_name}有效数据点不足，跳过异常值检测")
            return series

        # 计算四分位数
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1

        # 计算异常值边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 对于污染物浓度，下边界不应小于0
        lower_bound = max(0, lower_bound)

        # 识别异常值
        outliers_mask = (series < lower_bound) | (series > upper_bound)
        outliers_count = outliers_mask.sum()

        if outliers_count > 0:
            print(f"{pollutant_name}浓度异常值检测:")
            print(f"  有效数据范围: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
            print(f"  正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  检测到异常值: {outliers_count} 个")

            # 将异常值替换为NaN
            cleaned_series = series.copy()
            cleaned_series[outliers_mask] = np.nan

            # 统计信息
            remaining_valid = cleaned_series[(cleaned_series.notna()) & (cleaned_series > 0)]
            if len(remaining_valid) > 0:
                print(f"  清理后数据范围: [{remaining_valid.min():.2f}, {remaining_valid.max():.2f}]")
                print(f"  保留有效数据: {len(remaining_valid)} 个")

            return cleaned_series
        else:
            print(f"{pollutant_name}浓度未检测到异常值")
            return series

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据文件"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            # 转换时间列
            if '数据时间' in df.columns:
                df['数据时间'] = pd.to_datetime(df['数据时间'])
                df = df.sort_values('数据时间')
            
            # 清理数值列
            numeric_columns = [col for col in df.columns if col != '数据时间']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = self.clean_numeric_data(df[col])

            # 特殊处理：使用箱型图去除污染物浓度异常值
            pollutants_to_clean = {
                'dust': 'PM（颗粒物）',
                'so2': 'SO2（二氧化硫）',
                'nox': 'NOx（氮氧化物）',
                'co': 'CO（一氧化碳）',
                'hcl': 'HCL（氯化氢）'
            }

            print(f"\n开始污染物浓度异常值检测...")
            for pollutant_key, pollutant_display_name in pollutants_to_clean.items():
                field_name = self.field_mapping.get(pollutant_key)
                if field_name and field_name in df.columns:
                    print(f"\n对{pollutant_display_name}浓度字段 '{field_name}' 进行异常值检测...")
                    df[field_name] = self.remove_pollutant_outliers_boxplot(df[field_name], pollutant_display_name)

            print(f"成功加载数据: {len(df)} 行, {len(df.columns)} 列")
            return df
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_furnace_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算炉膛温度 - 根据预警报警规则
        
        规则要求：
        1. 对焚烧炉炉膛的中部和上部两个断面，各自取所有热电偶测量温度的中位数
        2. 计算这两个中位数的算术平均值，作为该断面的代表温度
        
        注意：开化天汇实际只有一个上部温度和一个中部温度字段，
        所以直接使用这两个值的算术平均值作为炉膛温度
        """
        top_temp_field = self.field_mapping.get('furnace_top_temp')
        mid_temp_field = self.field_mapping.get('furnace_mid_temp')

        if top_temp_field in df.columns and mid_temp_field in df.columns:
            # 由于开化天汇数据中每个断面只有一个温度值，直接计算算术平均值
            # 如果有多个热电偶，应该先计算各断面的中位数，再计算平均值
            df['炉膛温度'] = (df[top_temp_field] + df[mid_temp_field]) / 2
            
            # 处理异常值和缺失值
            df['炉膛温度'] = df['炉膛温度'].replace([np.inf, -np.inf], np.nan)
            
        else:
            print(f"警告: 缺少温度字段 {top_temp_field} 或 {mid_temp_field}")
            # 如果缺少字段，创建一个空的炉膛温度列
            df['炉膛温度'] = np.nan

        return df

    def check_low_furnace_temp_warning(self, df: pd.DataFrame) -> List[Dict]:
        """
        检查瞬时低炉温焚烧预警 - 修改后的规则

        规则要求：
        1. 时间划分：以自然日零点为起始点，按5分钟间隔划分时间
        2. 温度计算：对焚烧炉炉膛的中部和上部两个断面，各自取所有热电偶测量温度的中位数，
           计算这两个中位数的算术平均值，作为该断面的代表温度
        3. 以5分钟为一个时间窗口，计算温度累计平均值
        4. 当温度累计平均值低于850℃开始预警（记录一条预警记录），直至温度高于850℃或5分钟窗口预警结束
        """
        warnings = []

        if '炉膛温度' not in df.columns:
            return warnings

        # 设置时间索引
        df_temp = df.set_index('数据时间')

        # 按5分钟间隔计算平均值（以自然日零点为起始点）
        temp_5min = df_temp['炉膛温度'].resample('5min', origin='start_day').mean()

        threshold = self.thresholds['warning']['low_furnace_temp']

        # 跟踪预警状态，避免连续预警重复记录
        in_warning = False
        warning_start_time = None

        for timestamp, temp_avg in temp_5min.items():
            if pd.notna(temp_avg):
                if temp_avg < threshold:
                    # 温度低于阈值
                    if not in_warning:
                        # 开始新的预警
                        in_warning = True
                        warning_start_time = timestamp
                        warnings.append({
                            '时间': timestamp,
                            '预警/报警事件': '瞬时低炉温焚烧',
                            '预警/报警区分': '预警'
                        })
                else:
                    # 温度高于阈值
                    if in_warning:
                        # 结束预警
                        in_warning = False

        return warnings

    def check_high_furnace_temp_warning(self, df: pd.DataFrame) -> List[Dict]:
        """
        检查炉膛温度偏高/过高预警 - 修改后的规则

        规则要求：
        - 炉膛温度偏高：以一个自然日零点为起始计算点，以1小时为时间间隔进行划分
          计算焚烧炉炉膛内温度（上炉膛和中炉膛热电偶测量温度的平均值）计算1小时均值，
          高于1200℃，发出预警，并进行下一个小时的判断
        - 炉膛温度过高：1小时均值高于1300℃，发出预警，并进行下一个小时的判断
        """
        warnings = []

        if '炉膛温度' not in df.columns:
            return warnings

        # 设置时间索引
        df_temp = df.set_index('数据时间')

        # 以自然日零点为起始计算点，按1小时间隔划分
        temp_1hour = df_temp['炉膛温度'].resample('1H', origin='start_day').mean()

        high_threshold = self.thresholds['warning']['high_furnace_temp']
        very_high_threshold = self.thresholds['warning']['very_high_furnace_temp']

        for timestamp, temp_avg in temp_1hour.items():
            if pd.notna(temp_avg):
                if temp_avg > very_high_threshold:
                    warnings.append({
                        '时间': timestamp,
                        '预警/报警事件': '炉膛温度过高',
                        '预警/报警区分': '预警'
                    })
                elif temp_avg > high_threshold:
                    warnings.append({
                        '时间': timestamp,
                        '预警/报警事件': '炉膛温度偏高',
                        '预警/报警区分': '预警'
                    })

        return warnings

    def check_bag_pressure_warning(self, df: pd.DataFrame) -> List[Dict]:
        """
        检查布袋除尘器压力损失预警 - 连续预警逻辑

        规则要求：
        - 压力损失偏高：实时判断布袋除尘压差高于2000Pa发出预警信息，并开始计时，
          直至压差低于2000Pa，预警结束，并开始下一个预警的判断
        - 压力损失偏低：实时判断布袋除尘压差低于500Pa发出预警信息，并开始计时，
          直至压差高于500Pa，预警结束，并开始下一个预警的判断
        """
        warnings = []

        pressure_field = self.field_mapping.get('bag_pressure')
        if pressure_field not in df.columns:
            return warnings

        high_threshold = self.thresholds['warning']['bag_pressure_high']
        low_threshold = self.thresholds['warning']['bag_pressure_low']

        # 跟踪预警状态
        high_pressure_warning = False
        low_pressure_warning = False

        for _, row in df.iterrows():
            pressure = row[pressure_field]
            timestamp = row['数据时间']

            if pd.notna(pressure):
                # 检查压力偏高预警
                if pressure > high_threshold:
                    if not high_pressure_warning:
                        # 开始新的高压预警
                        high_pressure_warning = True
                        warnings.append({
                            '时间': timestamp,
                            '预警/报警事件': '布袋除尘器压力损失偏高',
                            '预警/报警区分': '预警'
                        })
                else:
                    if high_pressure_warning:
                        # 结束高压预警
                        high_pressure_warning = False

                # 检查压力偏低预警
                if pressure < low_threshold:
                    if not low_pressure_warning:
                        # 开始新的低压预警
                        low_pressure_warning = True
                        warnings.append({
                            '时间': timestamp,
                            '预警/报警事件': '布袋除尘器压力损失偏低',
                            '预警/报警区分': '预警'
                        })
                else:
                    if low_pressure_warning:
                        # 结束低压预警
                        low_pressure_warning = False

        return warnings

    def check_o2_warning(self, df: pd.DataFrame) -> List[Dict]:
        """
        检查焚烧炉出口氧含量预警 - 连续预警逻辑

        规则要求：
        - 氧含量偏高：实时判断焚烧炉出口氧含量，高于10%发出预警并开始计时，
          直至氧含量低于10%，预警结束，并开始下一个预警的判断
        - 氧含量偏低：实时判断焚烧炉出口氧含量，低于6%发出预警并开始计时，
          直至氧含量高于6%，预警结束，并开始下一个预警的判断
        """
        warnings = []

        o2_field = self.field_mapping.get('o2')
        if o2_field not in df.columns:
            return warnings

        high_threshold = self.thresholds['warning']['o2_high']
        low_threshold = self.thresholds['warning']['o2_low']

        # 跟踪预警状态
        high_o2_warning = False
        low_o2_warning = False

        for _, row in df.iterrows():
            o2_value = row[o2_field]
            timestamp = row['数据时间']

            if pd.notna(o2_value):
                # 检查氧含量偏高预警
                if o2_value > high_threshold:
                    if not high_o2_warning:
                        # 开始新的高氧含量预警
                        high_o2_warning = True
                        warnings.append({
                            '时间': timestamp,
                            '预警/报警事件': '焚烧炉出口氧含量偏高',
                            '预警/报警区分': '预警'
                        })
                else:
                    if high_o2_warning:
                        # 结束高氧含量预警
                        high_o2_warning = False

                # 检查氧含量偏低预警
                if o2_value < low_threshold:
                    if not low_o2_warning:
                        # 开始新的低氧含量预警
                        low_o2_warning = True
                        warnings.append({
                            '时间': timestamp,
                            '预警/报警事件': '焚烧炉出口氧含量偏低',
                            '预警/报警区分': '预警'
                        })
                else:
                    if low_o2_warning:
                        # 结束低氧含量预警
                        low_o2_warning = False

        return warnings

    def check_pollutant_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查污染物浓度预警 - 小时均值超标"""
        warnings = []

        # 设置时间索引
        df_temp = df.set_index('数据时间')

        # 污染物配置
        pollutants = {
            'dust': ('烟气中颗粒物（PM）浓度较高', 'dust_hourly'),
            'nox': ('烟气中氮氧化物（NOx）浓度较高', 'nox_hourly'),
            'so2': ('烟气中二氧化硫（SO₂）浓度较高', 'so2_hourly'),
            'hcl': ('烟气中氯化氢（HCl）浓度较高', 'hcl_hourly'),
            'co': ('烟气中一氧化碳（CO）浓度较高', 'co_hourly')
        }

        for pollutant_key, (event_name, threshold_key) in pollutants.items():
            field_name = self.field_mapping.get(pollutant_key)
            if field_name not in df.columns:
                continue

            # 计算1小时平均值
            hourly_avg = df_temp[field_name].resample('1H').mean()
            threshold = self.thresholds['warning'][threshold_key]

            for timestamp, avg_value in hourly_avg.items():
                if pd.notna(avg_value) and avg_value > threshold:
                    warnings.append({
                        '时间': timestamp,
                        '预警/报警事件': event_name,
                        '预警/报警区分': '预警'
                    })

        return warnings

    def check_low_furnace_temp_alarm(self, df: pd.DataFrame) -> List[Dict]:
        """检查低炉温焚烧报警 - 5分钟平均值 < 850℃"""
        alarms = []

        if '炉膛温度' not in df.columns:
            return alarms

        # 设置时间索引
        df_temp = df.set_index('数据时间')

        # 计算5分钟平均值
        temp_5min = df_temp['炉膛温度'].resample('5min').mean()

        threshold = self.thresholds['alarm']['low_furnace_temp']

        for timestamp, temp_avg in temp_5min.items():
            if pd.notna(temp_avg) and temp_avg < threshold:
                alarms.append({
                    '时间': timestamp,
                    '预警/报警事件': '低炉温焚烧',
                    '预警/报警区分': '报警'
                })

        return alarms

    def check_pollutant_daily_alarm(self, df: pd.DataFrame) -> List[Dict]:
        """检查污染物日均值排放超标报警"""
        alarms = []

        # 设置时间索引
        df_temp = df.set_index('数据时间')

        # 污染物配置 - 所有污染物日均值排放超标都应该归类为"报警"
        pollutants = {
            # 序号2: 烟气中颗粒物（PM）排放超标 - 日均值≤20mg/m³
            'dust_alarm': ('烟气中颗粒物（PM）排放超标', 'dust_daily', '报警'),
            # 序号3: 烟气中氮氧化物（NOx）排放超标 - 日均值≤250mg/m³
            'nox_alarm': ('烟气中氮氧化物（NOx）排放超标', 'nox_daily', '报警'),
            # 序号4: 烟气中二氧化硫（SO₂）排放超标 - 日均值≤80mg/m³
            'so2_alarm': ('烟气中二氧化硫（SO₂）排放超标', 'so2_daily', '报警'),
            # 序号5: 烟气中氯化氢（HCl）排放超标 - 日均值≤50mg/m³
            'hcl_alarm': ('烟气中氯化氢（HCl）排放超标', 'hcl_daily', '报警'),
            # 序号6: 烟气中一氧化碳（CO）排放超标 - 日均值≤80mg/m³
            'co_alarm': ('烟气中一氧化碳（CO）排放超标', 'co_daily', '报警')
        }

        for pollutant_key, (event_name, threshold_key, alarm_type) in pollutants.items():
            # 将报警键名映射到预警键名
            mapping_key = pollutant_key.replace('_alarm', '')
            field_name = self.field_mapping.get(mapping_key)
            if field_name not in df.columns:
                continue

            # 计算日均值
            daily_avg = df_temp[field_name].resample('1D').mean()
            threshold = self.thresholds['alarm'][threshold_key]

            for timestamp, avg_value in daily_avg.items():
                if pd.notna(avg_value) and avg_value > threshold:
                    alarms.append({
                        '时间': timestamp,
                        '预警/报警事件': event_name,
                        '预警/报警区分': alarm_type
                    })

        return alarms

    def save_warning_report(self, warning_df: pd.DataFrame, output_dir: str, input_file: str):
        """保存预警报告"""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_dir, f"开化天汇_预警报警_{base_name}_{timestamp}.xlsx")

            # 保存到Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                warning_df.to_excel(writer, sheet_name='预警报警记录', index=False)

            print(f"预警报告已保存: {output_file}")

        except Exception as e:
            print(f"保存预警报告失败: {e}")

    def process_data(self, file_path: str, output_dir: str = None) -> pd.DataFrame:
        """处理数据并生成预警报警报告"""
        # 加载数据
        df = self.load_data(file_path)
        if df.empty:
            return pd.DataFrame()

        # 清空之前的预警事件
        self.warning_events = []

        print(f"\n检查开化天汇焚烧炉预警报警 (1个炉子)...")

        # 计算炉膛温度
        df = self.calculate_furnace_temperature(df)

        # === 预警检查 ===
        # 1. 瞬时低炉温焚烧预警 (修改后规则：5分钟窗口，累计平均值)
        low_temp_warnings = self.check_low_furnace_temp_warning(df)
        self.warning_events.extend(low_temp_warnings)
        print(f"低炉温预警: {len(low_temp_warnings)} 条")

        # 5-6. 炉膛温度偏高/过高预警 (修改后规则：自然日零点起始，1小时间隔)
        high_temp_warnings = self.check_high_furnace_temp_warning(df)
        self.warning_events.extend(high_temp_warnings)
        print(f"高炉温预警: {len(high_temp_warnings)} 条")

        # 7. 焚烧工况不稳定预警 - 已删除该项预警规则

        # 8-9. 布袋除尘器压力损失预警 (修改后规则：删除正常工况判断)
        pressure_warnings = self.check_bag_pressure_warning(df)
        self.warning_events.extend(pressure_warnings)
        print(f"压力预警: {len(pressure_warnings)} 条")

        # 12-13. 焚烧炉出口氧含量预警 (修改后规则：删除正常工况判断)
        o2_warnings = self.check_o2_warning(df)
        self.warning_events.extend(o2_warnings)
        print(f"氧含量预警: {len(o2_warnings)} 条")

        # 14. 活性炭投加量不足预警 - 已删除该项预警规则

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


def main():
    """主函数"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python waste_incineration_warning_system_kaihua_tianhui.py <数据文件路径> [输出目录]")
        print("示例: python waste_incineration_warning_system_kaihua_tianhui.py 数据上/数据上/开化/5.23.csv ./output")
        return

    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"

    # 创建预警系统实例
    warning_system = WasteIncinerationWarningSystemKaihuaTianhui()

    # 处理数据
    result_df = warning_system.process_data(file_path, output_dir)

    if not result_df.empty:
        print(f"\n处理完成！共生成 {len(result_df)} 条预警报警记录")
    else:
        print("\n处理完成！未发现预警报警事件")


if __name__ == "__main__":
    main()