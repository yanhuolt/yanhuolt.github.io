#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版抽取式吸附曲线数据处理算法
实现完整的数据清洗、异常值剔除和序号化时间轴可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class FinalAdsorptionCurveProcessor:
    """最终版抽取式吸附曲线数据处理器"""
    
    def __init__(self, switch_interval: int = 8):
        """
        初始化处理器
        
        Args:
            switch_interval: 进出口切换间隔时间(秒)
        """
        self.switch_interval = switch_interval
        self.raw_data = None
        self.cleaned_data = None
        self.efficiency_data = None
        self.segment_data = None
        
    def process_complete_workflow(self, data: pd.DataFrame,
                                wind_speed_threshold: float = 0.5,
                                time_intervals: int = 10,
                                outlier_method: str = 'ks_test',
                                custom_time_ranges: List[Tuple] = None) -> Dict:
        """
        完整的数据处理工作流程

        Args:
            data: 原始数据
            wind_speed_threshold: 风速阈值
            time_intervals: 时间间隔数量（当custom_time_ranges为None时使用）
            outlier_method: 异常值检测方法
            custom_time_ranges: 自定义时间范围列表，格式为[("7:15", "7:30"), ("7:50", "8:10"), ...]

        Returns:
            包含所有处理结果的字典
        """
        print("=== 开始完整数据处理工作流程 ===")
        
        # 1. 数据验证
        self._validate_data(data)
        self.raw_data = data.copy()
        
        # 2. 数据清洗
        print("\n1. 数据清洗...")
        self.cleaned_data = self._clean_data(data, wind_speed_threshold, outlier_method)
        
        # 3. 效率计算
        print("\n2. 效率计算...")
        self.efficiency_data = self._calculate_efficiency(self.cleaned_data)
        
        # 4. 时间段分析
        print("\n3. 时间段分析...")
        self.segment_data = self._create_time_segments(self.efficiency_data, time_intervals, custom_time_ranges)
        
        # 5. 创建可视化
        print("\n4. 创建可视化...")
        fig = self._create_final_visualization(self.segment_data)
        
        # 6. 生成报告
        print("\n5. 生成报告...")
        report = self._generate_comprehensive_report()
        
        return {
            'raw_data': self.raw_data,
            'cleaned_data': self.cleaned_data,
            'efficiency_data': self.efficiency_data,
            'segment_data': self.segment_data,
            'visualization': fig,
            'report': report
        }
    
    def _validate_data(self, data: pd.DataFrame):
        """验证数据格式"""
        required_columns = ['time', 'inlet_outlet', 'concentration', 'wind_speed']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"数据缺少必要的列: {missing_columns}")
        
        if len(data) == 0:
            raise ValueError("数据为空")
        
        print(f"数据验证通过: {len(data)} 条记录")
    
    def _clean_data(self, data: pd.DataFrame, wind_speed_threshold: float, 
                   outlier_method: str) -> pd.DataFrame:
        """数据清洗"""
        # 1. 工作状态筛选
        working_data = data[data['wind_speed'] > wind_speed_threshold].copy()
        print(f"   风速过滤: {len(data)} -> {len(working_data)} 条记录")
        
        # 2. 零值剔除
        working_data = working_data[
            (working_data['concentration'] > 0) & 
            (working_data['wind_speed'] > 0)
        ].copy()
        print(f"   零值过滤: -> {len(working_data)} 条记录")
        
        # 3. 剔除进口浓度小于出口浓度的异常值
        working_data = self._remove_invalid_concentration_pairs(working_data)

        # 4. 异常值检测
        if outlier_method == 'ks_test':
            cleaned_data = self._ks_test_outliers(working_data)
        elif outlier_method == 'boxplot':
            cleaned_data = self._boxplot_outliers(working_data)
        else:
            raise ValueError("outlier_method must be 'ks_test' or 'boxplot'")

        return cleaned_data

    def _remove_invalid_concentration_pairs(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        剔除进口浓度小于出口浓度的异常值

        Args:
            data: 清洗后的数据

        Returns:
            剔除异常浓度对后的数据
        """
        print("   剔除进口浓度小于出口浓度的异常值...")

        if len(data) == 0:
            return data

        # 按时间排序
        data_sorted = data.sort_values('time').copy()

        # 分离进口和出口数据
        inlet_data = data_sorted[data_sorted['inlet_outlet'] == 0].copy()
        outlet_data = data_sorted[data_sorted['inlet_outlet'] == 1].copy()

        if len(inlet_data) == 0 or len(outlet_data) == 0:
            print("     警告: 缺少进口或出口数据")
            return data_sorted

        # 为每个出口数据点找到最近的进口数据点
        valid_indices = []
        invalid_pairs = 0

        for _, outlet_row in outlet_data.iterrows():
            outlet_time = outlet_row['time']
            outlet_conc = outlet_row['concentration']

            # 找到时间最接近的进口数据点（在切换间隔内）
            time_diff = np.abs(inlet_data['time'] - outlet_time)

            if len(time_diff) > 0:
                closest_inlet_idx = time_diff.idxmin()

                if time_diff.loc[closest_inlet_idx] <= self.switch_interval:
                    inlet_conc = inlet_data.loc[closest_inlet_idx, 'concentration']

                    # 检查是否进口浓度 >= 出口浓度
                    if inlet_conc >= outlet_conc:
                        # 有效的浓度对，保留两个数据点
                        valid_indices.extend([closest_inlet_idx, outlet_row.name])
                    else:
                        # 无效的浓度对，记录但不保留
                        invalid_pairs += 1
                        print(f"     移除异常对: 进口{inlet_conc:.1f} < 出口{outlet_conc:.1f} (时间{outlet_time:.1f}s)")
                else:
                    # 没有匹配的进口数据，保留出口数据
                    valid_indices.append(outlet_row.name)
            else:
                # 没有进口数据，保留出口数据
                valid_indices.append(outlet_row.name)

        # 添加没有匹配出口数据的进口数据
        matched_inlet_indices = set()
        for _, outlet_row in outlet_data.iterrows():
            outlet_time = outlet_row['time']
            time_diff = np.abs(inlet_data['time'] - outlet_time)

            if len(time_diff) > 0:
                closest_inlet_idx = time_diff.idxmin()

                if time_diff.loc[closest_inlet_idx] <= self.switch_interval:
                    matched_inlet_indices.add(closest_inlet_idx)

        unmatched_inlet_indices = set(inlet_data.index) - matched_inlet_indices
        valid_indices.extend(list(unmatched_inlet_indices))

        # 获取有效数据
        valid_data = data_sorted.loc[list(set(valid_indices))].copy()

        print(f"     移除异常浓度对: {invalid_pairs} 对")
        print(f"     保留有效数据: {len(valid_data)} 条")

        return valid_data

    def _ks_test_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """K-S检验异常值检测"""
        concentration = data['concentration'].values
        
        # K-S正态性检验
        _, p_value = stats.kstest(concentration, 'norm', 
                                 args=(np.mean(concentration), np.std(concentration)))
        
        print(f"   K-S检验 p值: {p_value:.4f}")
        
        if p_value > 0.05:
            # 3σ准则
            mean_val = np.mean(concentration)
            std_val = np.std(concentration)
            mask = np.abs(concentration - mean_val) <= 3 * std_val
            print(f"   正态分布，3σ准则: {len(data)} -> {np.sum(mask)} 条记录")
        else:
            # Z-score方法
            z_scores = np.abs(stats.zscore(concentration))
            mask = z_scores <= 3
            print(f"   非正态分布，Z-score: {len(data)} -> {np.sum(mask)} 条记录")
        
        return data[mask].copy()
    
    def _boxplot_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """箱型图异常值检测"""
        concentration = data['concentration'].values
        
        Q1 = np.percentile(concentration, 25)
        Q3 = np.percentile(concentration, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (concentration >= lower_bound) & (concentration <= upper_bound)
        print(f"   箱型图方法: {len(data)} -> {np.sum(mask)} 条记录")
        
        return data[mask].copy()
    
    def _calculate_efficiency(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算吸附效率"""
        inlet_data = data[data['inlet_outlet'] == 0].sort_values('time')
        outlet_data = data[data['inlet_outlet'] == 1].sort_values('time')
        
        efficiency_records = []
        
        for _, outlet_row in outlet_data.iterrows():
            outlet_time = outlet_row['time']
            c1 = outlet_row['concentration']
            
            # 查找对应的进口浓度
            c0 = self._find_corresponding_inlet(inlet_data, outlet_time, c1)
            
            if c0 is not None and c0 > 0:
                efficiency = (1 - c1 / c0) * 100
                efficiency_records.append({
                    'time': outlet_time,
                    'c0': c0,
                    'c1': c1,
                    'efficiency': efficiency
                })
        
        result = pd.DataFrame(efficiency_records)
        print(f"   计算得到 {len(result)} 个效率数据点")
        return result
    
    def _find_corresponding_inlet(self, inlet_data: pd.DataFrame, 
                                outlet_time: float, c1: float) -> Optional[float]:
        """查找对应的进口浓度"""
        time_window = self.switch_interval
        time_mask = (inlet_data['time'] >= outlet_time - time_window) & \
                   (inlet_data['time'] <= outlet_time + time_window)
        
        candidate_inlet = inlet_data[time_mask]
        
        if len(candidate_inlet) == 0:
            return None
        
        # 优先选择大于c1的c0
        valid_inlet = candidate_inlet[candidate_inlet['concentration'] > c1]
        
        if len(valid_inlet) > 0:
            time_diff = np.abs(valid_inlet['time'] - outlet_time)
            best_idx = time_diff.idxmin()
            return valid_inlet.loc[best_idx, 'concentration']
        else:
            # 使用上一个小于c1的c0
            previous_inlet = candidate_inlet[candidate_inlet['concentration'] < c1]
            if len(previous_inlet) > 0:
                time_diff = np.abs(previous_inlet['time'] - outlet_time)
                best_idx = time_diff.idxmin()
                return previous_inlet.loc[best_idx, 'concentration']
        
        return None
    
    def _create_time_segments(self, efficiency_data: pd.DataFrame,
                            time_intervals: int = None,
                            custom_time_ranges: List[Tuple] = None) -> List[Dict]:
        """
        创建时间段数据 - 支持非连续时间段

        Args:
            efficiency_data: 效率数据
            time_intervals: 自动分割的时间间隔数量（当custom_time_ranges为None时使用）
            custom_time_ranges: 自定义时间范围列表，格式为[(start1, end1), (start2, end2), ...]
                               时间可以是秒数或时间字符串（如"7:15", "7:30"）
        """
        if len(efficiency_data) == 0:
            return []

        segment_data = []

        if custom_time_ranges is not None:
            # 使用自定义时间范围
            print(f"   使用自定义时间范围: {len(custom_time_ranges)} 个时间段")

            for i, (time_start, time_end) in enumerate(custom_time_ranges):
                # 转换时间格式
                start_seconds = self._convert_time_to_seconds(time_start)
                end_seconds = self._convert_time_to_seconds(time_end)

                # 查找该时间段内的数据
                mask = (efficiency_data['time'] >= start_seconds) & \
                       (efficiency_data['time'] <= end_seconds)

                if np.sum(mask) > 0:
                    avg_efficiency = efficiency_data[mask]['efficiency'].mean()
                    segment_data.append({
                        'segment': i + 1,
                        'time_start': start_seconds,
                        'time_end': end_seconds,
                        'time_start_str': str(time_start),
                        'time_end_str': str(time_end),
                        'efficiency': avg_efficiency,
                        'data_count': np.sum(mask)
                    })
                    print(f"     时段{i+1}: {time_start}-{time_end}, 数据点:{np.sum(mask)}个, 效率:{avg_efficiency:.1f}%")
                else:
                    print(f"     时段{i+1}: {time_start}-{time_end}, 无数据")

        else:
            # 使用自动分割（原有逻辑）
            if time_intervals is None:
                time_intervals = 10

            time_min = efficiency_data['time'].min()
            time_max = efficiency_data['time'].max()
            time_bins = np.linspace(time_min, time_max, time_intervals + 1)

            for i in range(len(time_bins) - 1):
                mask = (efficiency_data['time'] >= time_bins[i]) & \
                       (efficiency_data['time'] < time_bins[i + 1])

                if np.sum(mask) > 0:
                    avg_efficiency = efficiency_data[mask]['efficiency'].mean()
                    segment_data.append({
                        'segment': i + 1,
                        'time_start': time_bins[i],
                        'time_end': time_bins[i + 1],
                        'time_start_str': self._seconds_to_time_str(time_bins[i]),
                        'time_end_str': self._seconds_to_time_str(time_bins[i + 1]),
                        'efficiency': avg_efficiency,
                        'data_count': np.sum(mask)
                    })

        print(f"   创建了 {len(segment_data)} 个有效时间段")
        return segment_data

    def _convert_time_to_seconds(self, time_input) -> float:
        """
        将时间转换为秒数

        Args:
            time_input: 时间输入，可以是秒数(float/int)或时间字符串("7:15", "07:15:30")

        Returns:
            秒数
        """
        if isinstance(time_input, (int, float)):
            return float(time_input)

        if isinstance(time_input, str):
            # 解析时间字符串
            time_parts = time_input.split(':')

            if len(time_parts) == 2:  # "7:15" 格式
                hours, minutes = map(int, time_parts)
                return hours * 3600 + minutes * 60
            elif len(time_parts) == 3:  # "07:15:30" 格式
                hours, minutes, seconds = map(int, time_parts)
                return hours * 3600 + minutes * 60 + seconds
            else:
                raise ValueError(f"无法解析时间格式: {time_input}")

        raise ValueError(f"不支持的时间格式: {type(time_input)}")

    def _seconds_to_time_str(self, seconds: float) -> str:
        """
        将秒数转换为时间字符串

        Args:
            seconds: 秒数

        Returns:
            时间字符串 "HH:MM"
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours:02d}:{minutes:02d}"
    
    def _create_final_visualization(self, segment_data: List[Dict]) -> plt.Figure:
        """创建最终的可视化图像"""
        if not segment_data:
            raise ValueError("没有时间段数据可用于可视化")
        
        # 创建图像
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # 提取数据
        segments = [d['segment'] for d in segment_data]
        efficiencies = [d['efficiency'] for d in segment_data]
        
        # 绘制数据点
        ax.scatter(segments, efficiencies, color='red', s=150, zorder=5, 
                  label='实际数据点', edgecolors='darkred', linewidth=2)
        
        # 拟合曲线
        if len(segments) >= 3:
            try:
                # 多项式拟合
                degree = min(3, len(segments) - 1)
                coeffs = np.polyfit(segments, efficiencies, degree)
                poly_func = np.poly1d(coeffs)
                
                # 生成平滑曲线
                x_smooth = np.linspace(min(segments), max(segments), 200)
                y_smooth = poly_func(x_smooth)
                
                ax.plot(x_smooth, y_smooth, 'b-', linewidth=3,
                       label=f'拟合曲线', alpha=0.8)

            except Exception as e:
                print(f"拟合失败: {e}")
        
        # 在数据点上方添加效率标注
        for i, data in enumerate(segment_data):
            segment = data['segment']
            efficiency = data['efficiency']

            # 在数据点正上方显示效率
            ax.annotate(f'{efficiency:.1f}%',
                       xy=(segment, efficiency),
                       xytext=(0, 15),  # 向上偏移15个点
                       textcoords='offset points',
                       fontsize=12,
                       ha='center',
                       va='bottom',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='yellow',
                               alpha=0.8,
                               edgecolor='orange',
                               linewidth=1),
                       arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0',
                                     color='orange',
                                     linewidth=1.5))
        
        # 设置坐标轴
        ax.set_xlabel('时间段序号', fontsize=16, fontweight='bold')
        ax.set_ylabel('处理效率 (%)', fontsize=16, fontweight='bold')
        ax.set_title('抽取式吸附曲线 - 分段效率分析', fontsize=18, fontweight='bold', pad=20)
        
        # 设置x轴刻度和标签
        ax.set_xticks(segments)

        # 创建x轴标签：序号 + 时间段信息
        x_labels = []
        for data in segment_data:
            segment = data['segment']
            if 'time_start_str' in data and 'time_end_str' in data:
                time_range = f"{data['time_start_str']}-{data['time_end_str']}"
            else:
                time_range = f"{data['time_start']:.0f}-{data['time_end']:.0f}s"

            # 创建两行标签：第一行是序号，第二行是时间段
            x_labels.append(f"{segment}\n{time_range}")

        ax.set_xticklabels(x_labels, fontsize=12, ha='center')
        
        # 设置范围
        y_min = min(efficiencies) - 25
        y_max = max(efficiencies) + 25
        ax.set_ylim(y_min, y_max)
        
        # 美化
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1)
        ax.legend(fontsize=14, loc='upper right')
        
        # 移除拟合信息框 - 只保留可视化曲线
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        return fig
    
    def _generate_comprehensive_report(self) -> str:
        """生成综合报告"""
        report = "=== 抽取式吸附曲线数据处理综合报告 ===\n\n"
        
        if self.raw_data is not None:
            report += f"1. 原始数据统计:\n"
            report += f"   - 总记录数: {len(self.raw_data)} 条\n"
            report += f"   - 时间跨度: {self.raw_data['time'].min():.1f} - {self.raw_data['time'].max():.1f} 秒\n"
            report += f"   - 进口数据: {len(self.raw_data[self.raw_data['inlet_outlet']==0])} 条\n"
            report += f"   - 出口数据: {len(self.raw_data[self.raw_data['inlet_outlet']==1])} 条\n\n"
        
        if self.cleaned_data is not None:
            report += f"2. 清洗后数据统计:\n"
            report += f"   - 有效记录数: {len(self.cleaned_data)} 条\n"
            report += f"   - 数据保留率: {len(self.cleaned_data)/len(self.raw_data)*100:.1f}%\n\n"
        
        if self.efficiency_data is not None:
            report += f"3. 效率数据统计:\n"
            report += f"   - 有效效率数据点: {len(self.efficiency_data)} 个\n"
            report += f"   - 平均处理效率: {self.efficiency_data['efficiency'].mean():.2f}%\n"
            report += f"   - 效率标准差: {self.efficiency_data['efficiency'].std():.2f}%\n"
            report += f"   - 最高效率: {self.efficiency_data['efficiency'].max():.2f}%\n"
            report += f"   - 最低效率: {self.efficiency_data['efficiency'].min():.2f}%\n\n"
        
        if self.segment_data:
            report += f"4. 时间段分析:\n"
            report += f"   - 时间段数量: {len(self.segment_data)} 个\n"
            for segment in self.segment_data:
                report += f"   - 时段{segment['segment']}: {segment['time_start']:.1f}-{segment['time_end']:.1f}s, "
                report += f"效率:{segment['efficiency']:.1f}%, 数据点:{segment['data_count']}个\n"
        
        return report


def create_demo_data_with_time_gaps():
    """创建带有时间间隔的演示数据"""
    np.random.seed(400)

    # 定义非连续的时间段（模拟您的需求）
    time_ranges = [
        ("7:15", "7:30"),   # 7:15-7:30
        ("7:50", "8:10"),   # 7:50-8:10
        ("8:18", "8:30"),   # 8:18-8:30
        ("9:05", "9:25"),   # 9:05-9:25
        ("10:12", "10:28"), # 10:12-10:28
    ]

    data_records = []

    for time_start_str, time_end_str in time_ranges:
        # 解析时间
        start_parts = time_start_str.split(':')
        end_parts = time_end_str.split(':')

        start_abs = int(start_parts[0])*3600 + int(start_parts[1])*60
        end_abs = int(end_parts[0])*3600 + int(end_parts[1])*60

        # 在该时间段内生成数据
        duration = end_abs - start_abs
        n_points = duration // 8  # 每8秒一个数据点

        for i in range(n_points):
            cycle_time = start_abs + i * 8 + np.random.uniform(-2, 2)

            # 进口数据
            inlet_conc = np.random.normal(180, 25)
            inlet_conc = max(inlet_conc, 50)

            # 出口数据（随时间段变化效率）
            time_factor = (start_abs - 7*3600 - 15*60) / 3600  # 小时数
            base_efficiency = 0.85 - 0.05 * time_factor  # 效率随时间下降
            efficiency = base_efficiency + np.random.normal(0, 0.03)
            efficiency = max(0.6, min(0.95, efficiency))

            outlet_conc = inlet_conc * (1 - efficiency)
            outlet_conc = max(outlet_conc, 5)

            # 风速
            wind_speed = np.random.normal(2.0, 0.3)
            wind_speed = max(wind_speed, 0.5)

            # 添加数据
            data_records.extend([
                {
                    'time': cycle_time,
                    'inlet_outlet': 0,
                    'concentration': inlet_conc,
                    'wind_speed': wind_speed
                },
                {
                    'time': cycle_time + 4,
                    'inlet_outlet': 1,
                    'concentration': outlet_conc,
                    'wind_speed': wind_speed
                }
            ])

    df = pd.DataFrame(data_records)

    # 添加少量异常值
    n_outliers = int(len(df) * 0.02)
    outlier_indices = np.random.choice(len(df), size=n_outliers, replace=False)

    for idx in outlier_indices:
        if np.random.random() < 0.7:
            df.loc[idx, 'concentration'] *= np.random.uniform(3, 5)
        else:
            df.loc[idx, 'concentration'] = np.random.uniform(0, 2)

    print(f"创建演示数据: {len(df)} 条记录")
    print(f"时间范围: {df['time'].min():.0f} - {df['time'].max():.0f} 秒")

    return df, time_ranges

def main():
    """主函数示例 - 演示非连续时间段处理"""
    print("=== 最终版抽取式吸附曲线数据处理算法 ===")
    print("演示非连续时间段处理功能\n")

    # 创建带有时间间隔的演示数据
    demo_data, time_ranges = create_demo_data_with_time_gaps()

    # 创建处理器
    processor = FinalAdsorptionCurveProcessor(switch_interval=8)

    print("=== 方法1: 使用自定义时间范围 ===")
    try:
        # 使用自定义时间范围
        results1 = processor.process_complete_workflow(
            data=demo_data,
            wind_speed_threshold=0.5,
            outlier_method='ks_test',
            custom_time_ranges=time_ranges  # 使用自定义时间范围
        )

        # 保存结果
        results1['visualization'].savefig('可视化项目/非连续时间段处理结果.png',
                                        dpi=300, bbox_inches='tight', facecolor='white')
        print("\n图像已保存: 非连续时间段处理结果.png")

        # 显示图像
        plt.show()

        # 打印报告
        print("\n" + results1['report'])

    except Exception as e:
        print(f"自定义时间范围处理出错: {e}")

    print("\n" + "="*60)
    print("=== 方法2: 使用自动分割（对比） ===")
    try:
        # 使用自动分割作为对比
        results2 = processor.process_complete_workflow(
            data=demo_data,
            wind_speed_threshold=0.5,
            time_intervals=5,  # 自动分割为5段
            outlier_method='ks_test'
        )

        # 保存对比结果
        results2['visualization'].savefig('可视化项目/自动分割对比结果.png',
                                        dpi=300, bbox_inches='tight', facecolor='white')
        print("\n对比图像已保存: 自动分割对比结果.png")

        # 显示图像
        plt.show()

    except Exception as e:
        print(f"自动分割处理出错: {e}")

def demo_custom_time_ranges():
    """演示如何使用不同格式的自定义时间范围"""
    print("\n=== 自定义时间范围格式演示 ===")

    # 创建简单的测试数据
    test_data = pd.DataFrame({
        'time': [26100, 26200, 26300, 28800, 29000, 29200, 30000, 30100, 30200],  # 对应7:15, 8:00, 8:20等
        'inlet_outlet': [0, 1, 0, 1, 0, 1, 0, 1, 0],
        'concentration': [150, 30, 160, 35, 155, 32, 145, 28, 150],
        'wind_speed': [2.0, 2.0, 1.8, 1.8, 2.1, 2.1, 1.9, 1.9, 2.0]
    })

    processor = FinalAdsorptionCurveProcessor()

    # 示例1: 使用时间字符串
    print("1. 使用时间字符串格式:")
    time_ranges_str = [("7:15", "7:20"), ("8:00", "8:05"), ("8:20", "8:25")]

    # 示例2: 使用秒数
    print("2. 使用秒数格式:")
    time_ranges_sec = [(26100, 26400), (28800, 29100), (30000, 30300)]

    # 示例3: 混合格式
    print("3. 混合格式:")
    time_ranges_mixed = [("7:15", "7:20"), (28800, 29100), ("8:20", "8:25")]

    for i, ranges in enumerate([time_ranges_str, time_ranges_sec, time_ranges_mixed], 1):
        try:
            print(f"\n测试格式 {i}: {ranges}")
            segments = processor._create_time_segments(
                pd.DataFrame({'time': test_data['time'], 'efficiency': [75, 80, 78]}),
                custom_time_ranges=ranges
            )
            print(f"成功创建 {len(segments)} 个时间段")
            for seg in segments:
                print(f"  时段{seg['segment']}: {seg.get('time_start_str', seg['time_start'])}-{seg.get('time_end_str', seg['time_end'])}")
        except Exception as e:
            print(f"格式 {i} 处理失败: {e}")

if __name__ == "__main__":
    main()
    demo_custom_time_ranges()
