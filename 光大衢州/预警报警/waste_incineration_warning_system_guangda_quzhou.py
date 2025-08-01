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

# 光大衢州预警阈值配置
GUANGDA_QUZHOU_WARNING_THRESHOLDS = {
    # 温度预警阈值
    'low_furnace_temp': 850,      # 瞬时低炉温焚烧 <850℃
    'high_furnace_temp': 1200,    # 炉膛温度偏高 >1200℃
    'very_high_furnace_temp': 1300, # 炉膛温度过高 >1300℃
    
    # 压力预警阈值
    'bag_pressure_high': 2000,    # 布袋除尘器压力损失偏高 >2000Pa
    'bag_pressure_low': 500,      # 布袋除尘器压力损失偏低 <500Pa
    
    # 氧含量预警阈值
    'o2_high': 10,                # 焚烧炉出口氧含量偏高 >10%
    'o2_low': 6,                  # 焚烧炉出口氧含量偏低 <6%
    
    # 活性炭投加量预警阈值（已删除该项预警规则）
    # 'carbon_dosage_low': 3.0,     # 活性炭投加量不足 <3.0kg/h
    
    # 污染物浓度预警阈值
    'dust_limit': 30,             # 颗粒物（PM）≤30mg/m³
    'nox_limit': 300,             # 氮氧化物（NOx）≤300mg/m³
    'so2_limit': 100,             # 二氧化硫（SO₂）≤100mg/m³
    'hcl_limit': 60,              # 氯化氢（HCl）≤60mg/m³
    'co_limit': 100,              # 一氧化碳（CO）≤100mg/m³
    # 'nh3_limit': 8,               # 氨逃逸（NH3）≤8ppm（已删除该项预警规则）
}

# 光大衢州字段映射（2个炉子）
GUANGDA_QUZHOU_FIELD_MAPPING = {
    # 炉膛温度相关字段 (两个炉子)
    # 1号炉温度字段
    "furnace_1_top_temp_1": "#1炉第一烟道上部温度T10（顶间左侧）",
    "furnace_1_top_temp_2": "#1炉第一烟道上部温度T11（顶间中侧）", 
    "furnace_1_top_temp_3": "#1炉第一烟道上部温度T12（顶间右侧）",
    "furnace_1_mid_temp_1": "#1炉第一烟道中部温度T20（中间左侧)",
    "furnace_1_mid_temp_2": "#1炉第一烟道中部温度T21（中间中侧）",
    "furnace_1_mid_temp_3": "#1炉第一烟道中部温度T22（中间右间）",
    
    # 2号炉温度字段
    "furnace_2_top_temp_1": "#2炉第一烟道上部温度T10（顶间左侧）",
    "furnace_2_top_temp_2": "#2炉第一烟道上部温度T11（顶间中侧）",
    "furnace_2_top_temp_3": "#2炉第一烟道上部温度T12（顶间右侧）",
    "furnace_2_mid_temp_1": "#2炉第一烟道中部温度T20（中间左侧)",
    "furnace_2_mid_temp_2": "#2炉第一烟道中部温度T21（中间中侧）",
    "furnace_2_mid_temp_3": "#2炉第一烟道中部温度T22（中间右间）",
    
    # 布袋除尘器压力字段
    "furnace_1_bag_pressure": "1#炉除尘器总差压  PT2001",
    "furnace_2_bag_pressure": "2#炉除尘器总差压  PT2001",
    
    # 焚烧炉出口氧含量字段
    "furnace_1_o2": "1#炉O2",
    "furnace_2_o2": "2#炉O2",
    
    # 污染物浓度字段
    "furnace_1_dust": "1#炉粉尘",
    "furnace_1_nox": "1#炉Nox",
    "furnace_1_so2": "1#炉SO2",
    "furnace_1_hcl": "1#炉HCL",
    "furnace_1_co": "1#炉CO",
    
    "furnace_2_dust": "2#炉粉尘",
    "furnace_2_nox": "2#炉Nox",
    "furnace_2_so2": "2#炉SO2",
    "furnace_2_hcl": "2#炉HCL",
    "furnace_2_co": "2#炉CO",
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
    
    def check_low_furnace_temp_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查瞬时低炉温焚烧预警"""
        warnings = []
        
        # 按5分钟间隔重采样
        df_5min = df.set_index('数据时间').resample('5T').mean().reset_index()
        
        for furnace_id in range(1, self.furnace_count + 1):
            # 计算炉膛温度
            furnace_temp = self.calculate_furnace_temperature(df_5min, furnace_id)
            df_5min[f'furnace_{furnace_id}_temp'] = furnace_temp
            
            # 检查低温预警
            low_temp_mask = furnace_temp < GUANGDA_QUZHOU_WARNING_THRESHOLDS['low_furnace_temp']
            
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
        """检查炉膛温度偏高/过高预警"""
        warnings = []

        # 按1小时间隔重采样
        df_1hour = df.set_index('数据时间').resample('1H').mean().reset_index()

        for furnace_id in range(1, self.furnace_count + 1):
            # 计算炉膛温度
            furnace_temp = self.calculate_furnace_temperature(df_1hour, furnace_id)
            df_1hour[f'furnace_{furnace_id}_temp'] = furnace_temp

            # 检查温度过高 (>1300℃)
            very_high_mask = furnace_temp > GUANGDA_QUZHOU_WARNING_THRESHOLDS['very_high_furnace_temp']
            for _, row in df_1hour[very_high_mask].iterrows():
                warnings.append({
                    '时间': row['数据时间'],
                    '炉号': str(furnace_id),
                    '预警/报警类型': '预警',
                    '预警/报警事件': '炉膛温度过高'
                })

            # 检查温度偏高 (>1200℃ 且 ≤1300℃)
            high_mask = (furnace_temp > GUANGDA_QUZHOU_WARNING_THRESHOLDS['high_furnace_temp']) & \
                       (furnace_temp <= GUANGDA_QUZHOU_WARNING_THRESHOLDS['very_high_furnace_temp'])
            for _, row in df_1hour[high_mask].iterrows():
                warnings.append({
                    '时间': row['数据时间'],
                    '炉号': str(furnace_id),
                    '预警/报警类型': '预警',
                    '预警/报警事件': '炉膛温度偏高'
                })

        return warnings

    def check_bag_pressure_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查布袋除尘器压力损失预警（连续状态跟踪）"""
        warnings = []

        for furnace_id in range(1, self.furnace_count + 1):
            pressure_field = GUANGDA_QUZHOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_bag_pressure')

            if not pressure_field or pressure_field not in df.columns:
                continue

            state_key_high = f'pressure_high_{furnace_id}'
            state_key_low = f'pressure_low_{furnace_id}'

            # 初始化状态
            if state_key_high not in self.warning_states:
                self.warning_states[state_key_high] = {'active': False, 'start_time': None}
            if state_key_low not in self.warning_states:
                self.warning_states[state_key_low] = {'active': False, 'start_time': None}

            for _, row in df.iterrows():
                pressure_value = row[pressure_field]
                current_time = row['数据时间']

                if pd.isna(pressure_value):
                    continue

                # 检查压力偏高 (>2000Pa)
                if pressure_value > GUANGDA_QUZHOU_WARNING_THRESHOLDS['bag_pressure_high']:
                    if not self.warning_states[state_key_high]['active']:
                        # 开始新的预警
                        self.warning_states[state_key_high]['active'] = True
                        self.warning_states[state_key_high]['start_time'] = current_time
                        warnings.append({
                            '时间': current_time,
                            '炉号': str(furnace_id),
                            '预警/报警类型': '预警',
                            '预警/报警事件': '布袋除尘器压力损失偏高'
                        })
                else:
                    # 压力恢复正常，结束预警
                    if self.warning_states[state_key_high]['active']:
                        self.warning_states[state_key_high]['active'] = False
                        self.warning_states[state_key_high]['start_time'] = None

                # 检查压力偏低 (<500Pa)
                if pressure_value < GUANGDA_QUZHOU_WARNING_THRESHOLDS['bag_pressure_low']:
                    if not self.warning_states[state_key_low]['active']:
                        # 开始新的预警
                        self.warning_states[state_key_low]['active'] = True
                        self.warning_states[state_key_low]['start_time'] = current_time
                        warnings.append({
                            '时间': current_time,
                            '炉号': str(furnace_id),
                            '预警/报警类型': '预警',
                            '预警/报警事件': '布袋除尘器压力损失偏低'
                        })
                else:
                    # 压力恢复正常，结束预警
                    if self.warning_states[state_key_low]['active']:
                        self.warning_states[state_key_low]['active'] = False
                        self.warning_states[state_key_low]['start_time'] = None

        return warnings

    def check_o2_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查焚烧炉出口氧含量预警（连续状态跟踪）"""
        warnings = []

        for furnace_id in range(1, self.furnace_count + 1):
            o2_field = GUANGDA_QUZHOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_o2')

            if not o2_field or o2_field not in df.columns:
                continue

            state_key_high = f'o2_high_{furnace_id}'
            state_key_low = f'o2_low_{furnace_id}'

            # 初始化状态
            if state_key_high not in self.warning_states:
                self.warning_states[state_key_high] = {'active': False, 'start_time': None}
            if state_key_low not in self.warning_states:
                self.warning_states[state_key_low] = {'active': False, 'start_time': None}

            for _, row in df.iterrows():
                o2_value = row[o2_field]
                current_time = row['数据时间']

                if pd.isna(o2_value):
                    continue

                # 检查氧含量偏高 (>10%)
                if o2_value > GUANGDA_QUZHOU_WARNING_THRESHOLDS['o2_high']:
                    if not self.warning_states[state_key_high]['active']:
                        # 开始新的预警
                        self.warning_states[state_key_high]['active'] = True
                        self.warning_states[state_key_high]['start_time'] = current_time
                        warnings.append({
                            '时间': current_time,
                            '炉号': str(furnace_id),
                            '预警/报警类型': '预警',
                            '预警/报警事件': '焚烧炉出口氧含量偏高'
                        })
                else:
                    # 氧含量恢复正常，结束预警
                    if self.warning_states[state_key_high]['active']:
                        self.warning_states[state_key_high]['active'] = False
                        self.warning_states[state_key_high]['start_time'] = None

                # 检查氧含量偏低 (<6%)
                if o2_value < GUANGDA_QUZHOU_WARNING_THRESHOLDS['o2_low']:
                    if not self.warning_states[state_key_low]['active']:
                        # 开始新的预警
                        self.warning_states[state_key_low]['active'] = True
                        self.warning_states[state_key_low]['start_time'] = current_time
                        warnings.append({
                            '时间': current_time,
                            '炉号': str(furnace_id),
                            '预警/报警类型': '预警',
                            '预警/报警事件': '焚烧炉出口氧含量偏低'
                        })
                else:
                    # 氧含量恢复正常，结束预警
                    if self.warning_states[state_key_low]['active']:
                        self.warning_states[state_key_low]['active'] = False
                        self.warning_states[state_key_low]['start_time'] = None

        return warnings

    def check_pollutant_warning(self, df: pd.DataFrame) -> List[Dict]:
        """检查污染物浓度预警"""
        warnings = []

        # 按1小时间隔重采样
        df_1hour = df.set_index('数据时间').resample('1H').mean().reset_index()

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
                field = GUANGDA_QUZHOU_FIELD_MAPPING.get(f'furnace_{furnace_id}_{pollutant}')

                if field and field in df_1hour.columns:
                    threshold = GUANGDA_QUZHOU_WARNING_THRESHOLDS[threshold_key]
                    high_mask = df_1hour[field] > threshold

                    for _, row in df_1hour[high_mask].iterrows():
                        warnings.append({
                            '时间': row['数据时间'],
                            '炉号': str(furnace_id),
                            '预警/报警类型': '预警',
                            '预警/报警事件': event_name
                        })

        return warnings

    def process_data(self, file_path: str, output_dir: str = None) -> pd.DataFrame:
        """处理数据并生成预警报告"""
        # 加载数据
        df = self.load_data(file_path)
        if df.empty:
            return pd.DataFrame()

        # 清空之前的预警事件和状态
        self.warning_events = []
        self.warning_states = {}

        print(f"\n检查光大衢州焚烧炉预警 (2个炉子)...")

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
        excel_file = os.path.join(output_dir, f"{base_name}_光大衢州预警报告_{timestamp}.xlsx")
        warning_df.to_excel(excel_file, index=False)
        print(f"预警报告已保存: {excel_file}")

        # 保存CSV格式 (与输出模板格式一致)
        csv_file = os.path.join(output_dir, f"{base_name}_光大衢州预警报告_{timestamp}.csv")
        # 只保留模板需要的列
        template_df = warning_df[['时间', '炉号', '预警/报警类型', '预警/报警事件']].copy()
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
        print(f"\n处理完成！共生成 {len(result_df)} 条预警记录")
    else:
        print("\n处理完成！未发现预警事件")

if __name__ == "__main__":
    main()
