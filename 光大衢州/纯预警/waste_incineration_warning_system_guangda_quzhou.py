#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光大衢州垃圾焚烧预警系统
根据《垃圾焚烧预警报警规则（光大衢州）.csv》实现预警算法
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 光大衢州预警报警阈值配置
GUANGDA_QUZHOU_WARNING_THRESHOLDS = {
    # 温度报警阈值
    'low_furnace_temp': 850,      # 低炉温焚烧 <850℃ (报警)

    # 污染物日均值报警阈值
    'dust_daily_limit': 20,       # 颗粒物（PM）日均值 ≤20mg/m³ (报警)
    'nox_daily_limit': 250,       # 氮氧化物（NOx）日均值 ≤250mg/m³ (报警)
    'so2_daily_limit': 80,        # 二氧化硫（SO₂）日均值 ≤80mg/m³ (报警)
    'hcl_daily_limit': 50,        # 氯化氢（HCl）日均值 ≤50mg/m³ (报警)
    'co_daily_limit': 80,         # 一氧化碳（CO）日均值 ≤80mg/m³ (报警)
}

# 光大衢州字段映射（2个炉子）
GUANGDA_QUZHOU_FIELD_MAPPING = {
    # 炉膛温度相关字段 (根据新规则中的数据采集点位)
    # 1号炉温度字段 (U1、V1、W1为上部，X1、Y1、Z1为中部)
    "furnace_1_top_temp_1": "U1",
    "furnace_1_top_temp_2": "V1",
    "furnace_1_top_temp_3": "W1",
    "furnace_1_mid_temp_1": "X1",
    "furnace_1_mid_temp_2": "Y1",
    "furnace_1_mid_temp_3": "Z1",

    # 2号炉温度字段 (AA1、AB1、AC1为上部，AD1、AE1、AF1为中部)
    "furnace_2_top_temp_1": "AA1",
    "furnace_2_top_temp_2": "AB1",
    "furnace_2_top_temp_3": "AC1",
    "furnace_2_mid_temp_1": "AD1",
    "furnace_2_mid_temp_2": "AE1",
    "furnace_2_mid_temp_3": "AF1",

    # 污染物浓度字段 (根据新规则中的数据采集点位)
    "furnace_1_dust": "ES",      # 1号炉颗粒物
    "furnace_1_nox": "EU",       # 1号炉氮氧化物
    "furnace_1_so2": "ET",       # 1号炉二氧化硫
    "furnace_1_hcl": "EW",       # 1号炉氯化氢
    "furnace_1_co": "EV",        # 1号炉一氧化碳

    "furnace_2_dust": "FD",      # 2号炉颗粒物
    "furnace_2_nox": "FF",       # 2号炉氮氧化物
    "furnace_2_so2": "FE",       # 2号炉二氧化硫
    "furnace_2_hcl": "FH",       # 2号炉氯化氢
    "furnace_2_co": "FG",        # 2号炉一氧化碳
}

class WasteIncinerationWarningSystemGuangdaQuzhou:
    """光大衢州垃圾焚烧预警系统"""
    
    def __init__(self):
        self.furnace_count = 2  # 光大衢州有2个炉子
        self.warning_events = []
        self.warning_states = {}  # 用于跟踪连续预警状态
        
    def clean_numeric_data(self, value):
        """清理数值数据"""
        if pd.isna(value):
            return np.nan
        
        if isinstance(value, str):
            # 移除可能的字符串连接符和空格
            value = value.strip().replace(' ', '')
            if value == '' or value == 'nan':
                return np.nan
            try:
                return float(value)
            except ValueError:
                return np.nan
        
        return float(value) if not pd.isna(value) else np.nan
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据文件"""
        try:
            print(f"成功加载数据文件: {file_path}")
            
            # 根据文件扩展名选择读取方式
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            # 转换时间列
            if '数据时间' in df.columns:
                df['数据时间'] = pd.to_datetime(df['数据时间'])
            
            # 清理数值列
            numeric_columns = [col for col in df.columns if col != '数据时间']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self.clean_numeric_data)
            
            print(f"数据行数: {len(df)}, 列数: {len(df.columns)}")
            return df
            
        except Exception as e:
            print(f"加载数据文件失败: {e}")
            return pd.DataFrame()
    
    def calculate_furnace_temperature(self, df: pd.DataFrame, furnace_id: int) -> pd.Series:
        """
        计算炉膛温度
        根据预警规则：对焚烧炉炉膛的中部和上部两个断面，各自取所有热电偶测量温度的中位数
        计算这两个中位数的算术平均值，作为该断面的代表温度
        """
        # 获取上部温度字段
        top_temp_fields = [
            GUANGDA_QUZHOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_top_temp_1'),
            GUANGDA_QUZHOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_top_temp_2'),
            GUANGDA_QUZHOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_top_temp_3')
        ]
        
        # 获取中部温度字段
        mid_temp_fields = [
            GUANGDA_QUZHOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_mid_temp_1'),
            GUANGDA_QUZHOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_mid_temp_2'),
            GUANGDA_QUZHOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_mid_temp_3')
        ]
        
        # 过滤存在的字段
        top_temp_fields = [f for f in top_temp_fields if f and f in df.columns]
        mid_temp_fields = [f for f in mid_temp_fields if f and f in df.columns]
        
        if not top_temp_fields or not mid_temp_fields:
            return pd.Series([np.nan] * len(df))
        
        # 计算上部温度中位数
        top_temps = df[top_temp_fields]
        top_median = top_temps.median(axis=1)
        
        # 计算中部温度中位数
        mid_temps = df[mid_temp_fields]
        mid_median = mid_temps.median(axis=1)
        
        # 计算两个中位数的算术平均值
        furnace_temp = (top_median + mid_median) / 2
        
        return furnace_temp
    
    def check_low_furnace_temp_alarm(self, df: pd.DataFrame) -> List[Dict]:
        """检查低炉温焚烧报警"""
        alarms = []

        # 按5分钟间隔重采样
        df_5min = df.set_index('数据时间').resample('5T').mean().reset_index()

        for furnace_id in range(1, self.furnace_count + 1):
            # 计算炉膛温度
            furnace_temp = self.calculate_furnace_temperature(df_5min, furnace_id)
            df_5min[f'furnace_{furnace_id}_temp'] = furnace_temp

            # 检查低温报警 (<850℃)
            low_temp_mask = furnace_temp < GUANGDA_QUZHOU_WARNING_THRESHOLDS['low_furnace_temp']

            if low_temp_mask.any():
                for _, row in df_5min[low_temp_mask].iterrows():
                    alarms.append({
                        '时间': row['数据时间'],
                        '炉号': str(furnace_id),
                        '预警/报警类型': '报警',
                        '预警/报警事件': '低炉温焚烧',
                        '预警/报警区分': '报警'
                    })

        return alarms

    def check_pollutant_daily_alarm(self, df: pd.DataFrame) -> List[Dict]:
        """检查污染物日均值排放超标报警"""
        alarms = []

        # 按日期分组计算日均值
        df_daily = df.set_index('数据时间').resample('1D').mean().reset_index()

        # 检查每个炉子的各种污染物日均值
        pollutants = {
            'dust': ('烟气中颗粒物（PM）排放超标', 'dust_daily_limit'),
            'nox': ('烟气中氮氧化物（NOx）排放超标', 'nox_daily_limit'),
            'so2': ('烟气中二氧化硫（SO₂）排放超标', 'so2_daily_limit'),
            'hcl': ('烟气中氯化氢（HCl）排放超标', 'hcl_daily_limit'),
            'co': ('烟气中一氧化碳（CO）排放超标', 'co_daily_limit')
        }

        for furnace_id in range(1, self.furnace_count + 1):
            for pollutant, (event_name, threshold_key) in pollutants.items():
                field = GUANGDA_QUZHOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_{pollutant}')

                if field and field in df_daily.columns:
                    threshold = GUANGDA_QUZHOU_WARNING_THRESHOLDS[threshold_key]
                    exceed_mask = df_daily[field] > threshold

                    for _, row in df_daily[exceed_mask].iterrows():
                        alarms.append({
                            '时间': row['数据时间'],
                            '炉号': str(furnace_id),
                            '预警/报警类型': '报警',
                            '预警/报警事件': event_name,
                            '预警/报警区分': '报警'
                        })

        return alarms

    def process_data(self, file_path: str, output_dir: str = None) -> pd.DataFrame:
        """处理数据并生成预警报告"""
        # 加载数据
        df = self.load_data(file_path)
        if df.empty:
            return pd.DataFrame()

        # 清空之前的预警事件和状态
        self.warning_events = []
        self.warning_states = {}

        print(f"\n检查光大衢州焚烧炉预警报警 (2个炉子)...")

        # 低炉温焚烧报警
        low_temp_alarms = self.check_low_furnace_temp_alarm(df)
        self.warning_events.extend(low_temp_alarms)
        print(f"低炉温报警: {len(low_temp_alarms)} 条")

        # 污染物日均值排放超标报警
        pollutant_alarms = self.check_pollutant_daily_alarm(df)
        self.warning_events.extend(pollutant_alarms)
        print(f"污染物排放超标报警: {len(pollutant_alarms)} 条")

        # 转换为DataFrame
        if self.warning_events:
            warning_df = pd.DataFrame(self.warning_events)
            # 按时间排序
            warning_df = warning_df.sort_values('时间')

            print(f"\n共检测到 {len(warning_df)} 条预警报警事件")

            # 按炉号统计
            furnace_stats = warning_df['炉号'].value_counts().sort_index()
            print("各炉预警报警分布:")
            for furnace, count in furnace_stats.items():
                print(f"  {furnace}号炉: {count} 条")

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
        excel_file = os.path.join(output_dir, f"{base_name}_光大衢州预警报告_{timestamp}.xlsx")
        warning_df.to_excel(excel_file, index=False)
        print(f"预警报告已保存: {excel_file}")

        # 保存CSV格式 (与输出模板格式一致，包含预警/报警区分列)
        csv_file = os.path.join(output_dir, f"{base_name}_光大衢州预警报告_{timestamp}.csv")
        # 确保包含所有必需的列
        required_columns = ['时间', '炉号', '预警/报警类型', '预警/报警事件', '预警/报警区分']
        template_df = warning_df[required_columns].copy()
        template_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"CSV报告已保存: {csv_file}")

def main():
    """主函数 - 命令行接口"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python waste_incineration_warning_system_guangda_quzhou.py <数据文件路径> [输出目录]")
        print("示例: python waste_incineration_warning_system_guangda_quzhou.py 数据上/数据上/光大（衢州）/5.23.xlsx ./output")
        return

    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"

    # 创建预警系统实例
    warning_system = WasteIncinerationWarningSystemGuangdaQuzhou()

    # 处理数据
    result_df = warning_system.process_data(file_path, output_dir)

    if not result_df.empty:
        print(f"\n处理完成！共生成 {len(result_df)} 条预警报警记录")
    else:
        print("\n处理完成！未发现预警报警事件")

if __name__ == "__main__":
    main()
