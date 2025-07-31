#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽取式吸附曲线完整数据处理与可视化算法
基于7.24数据.csv，实现从数据清洗到可视化的全流程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class AdsorptionCurveProcessor:
    """抽取式吸附曲线完整处理器"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        # 提取原始文件名（不含扩展名）
        self.base_filename = os.path.splitext(os.path.basename(data_file))[0]
        self.raw_data = None
        self.cleaned_data_ks = None
        self.cleaned_data_boxplot = None
        self.efficiency_data_ks = None
        self.efficiency_data_boxplot = None
        
    def load_data(self) -> bool:
        """加载原始数据 - 支持CSV、XLSX、XLS格式"""
        try:
            print("=== 加载原始数据 ===")

            # 获取文件扩展名
            file_extension = os.path.splitext(self.data_file)[1].lower()
            print(f"检测到文件格式: {file_extension}")

            # 根据文件扩展名选择相应的读取方法
            if file_extension == '.csv':
                self.raw_data = pd.read_csv(self.data_file, encoding='utf-8-sig')
                print("使用CSV格式加载数据")
            elif file_extension in ['.xlsx', '.xls']:
                # 尝试读取Excel文件，默认读取第一个工作表
                self.raw_data = pd.read_excel(self.data_file, engine='openpyxl' if file_extension == '.xlsx' else 'xlrd')
                print(f"使用Excel格式加载数据 (引擎: {'openpyxl' if file_extension == '.xlsx' else 'xlrd'})")
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}。支持的格式: .csv, .xlsx, .xls")

            # 转换时间列
            if '创建时间' in self.raw_data.columns:
                self.raw_data['创建时间'] = pd.to_datetime(self.raw_data['创建时间'])
                print(f"原始数据加载成功: {len(self.raw_data)} 条记录")
                print(f"时间范围: {self.raw_data['创建时间'].min()} 到 {self.raw_data['创建时间'].max()}")
            else:
                print(f"原始数据加载成功: {len(self.raw_data)} 条记录")
                print("警告: 未找到'创建时间'列，请检查数据格式")
                # 显示前几列的列名以便调试
                print(f"数据列名: {list(self.raw_data.columns[:10])}")

            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            print(f"请确保文件存在且格式正确")
            if file_extension in ['.xlsx', '.xls']:
                print("提示: Excel文件需要安装openpyxl或xlrd库")
                print("安装命令: pip install openpyxl xlrd")
            return False
    
    def basic_data_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """基础数据清洗"""
        print("\n=== 基础数据清洗 ===")
        original_count = len(data)
        
        # 1. 获取工作状态数据（风速值大于0.5）
        data = data[data['风管内风速值'] > 0.5].copy()
        print(f"1. 保留风速>0.5的数据: {len(data)} 条 (剔除 {original_count - len(data)} 条)")
        
        # 2. 根据进口0出口1列分别剔除0值
        before_zero_removal = len(data)
        # 当进口0出口1=0时，剔除进口voc为0的记录
        # 当进口0出口1=1时，剔除出口voc为0的记录
        inlet_mask = (data['进口0出口1'] == 0) & (data['进口voc'] > 0)
        outlet_mask = (data['进口0出口1'] == 1) & (data['出口voc'] > 0)
        data = data[inlet_mask | outlet_mask].copy()
        print(f"2. 根据进口0出口1列分别剔除相应VOC为0的数据: {len(data)} 条 (剔除 {before_zero_removal - len(data)} 条)")
        
        # 3. 剔除风量=0的数据
        before_flow_removal = len(data)
        data = data[data['风量'] > 0].copy()
        print(f"3. 剔除风量=0的数据: {len(data)} 条 (剔除 {before_flow_removal - len(data)} 条)")

        # 4. 剔除出口浓度大于等于进口浓度的记录
        before_concentration_removal = len(data)
        data = self._remove_invalid_concentration_pairs(data)
        print(f"4. 剔除出口浓度≥进口浓度的记录: {len(data)} 条 (剔除 {before_concentration_removal - len(data)} 条)")

        return data

    def _remove_invalid_concentration_pairs(self, data: pd.DataFrame) -> pd.DataFrame:
        """剔除出口浓度大于等于进口浓度的记录"""
        print("   正在检查进出口浓度配对...")

        # 分离进口和出口数据
        inlet_data = data[data['进口0出口1'] == 0].copy()
        outlet_data = data[data['进口0出口1'] == 1].copy()

        if len(inlet_data) == 0 or len(outlet_data) == 0:
            print("   警告: 缺少进口或出口数据，跳过浓度配对检查")
            return data

        # 按时间排序
        inlet_data = inlet_data.sort_values('创建时间')
        outlet_data = outlet_data.sort_values('创建时间')

        # 使用时间窗口匹配进出口数据
        valid_records = []
        time_window = pd.Timedelta(minutes=30)  # 30分钟时间窗口

        removed_count = 0

        # 检查每个进口记录
        for _, inlet_record in inlet_data.iterrows():
            inlet_time = inlet_record['创建时间']
            inlet_voc = inlet_record['进口voc']

            # 找到时间窗口内的出口记录
            outlet_candidates = outlet_data[
                (outlet_data['创建时间'] >= inlet_time - time_window) &
                (outlet_data['创建时间'] <= inlet_time + time_window)
            ]

            if len(outlet_candidates) > 0:
                # 找到最近的出口记录
                time_diffs = abs(outlet_candidates['创建时间'] - inlet_time)
                closest_outlet = outlet_candidates.loc[time_diffs.idxmin()]
                outlet_voc = closest_outlet['出口voc']

                # 检查浓度关系：只保留进口浓度 >= 出口浓度的记录
                if inlet_voc >= outlet_voc:
                    valid_records.append(inlet_record)
                    # 同时保留对应的出口记录
                    valid_records.append(closest_outlet)
                else:
                    removed_count += 2  # 进口和出口记录都被剔除
            else:
                # 没有匹配的出口记录，保留进口记录
                valid_records.append(inlet_record)

        # 检查剩余的出口记录（没有匹配进口记录的）
        for _, outlet_record in outlet_data.iterrows():
            outlet_time = outlet_record['创建时间']

            # 检查是否已经在valid_records中
            already_included = any(
                record['创建时间'] == outlet_time and record['进口0出口1'] == 1
                for record in valid_records
            )

            if not already_included:
                # 找到时间窗口内的进口记录
                inlet_candidates = inlet_data[
                    (inlet_data['创建时间'] >= outlet_time - time_window) &
                    (inlet_data['创建时间'] <= outlet_time + time_window)
                ]

                if len(inlet_candidates) == 0:
                    # 没有匹配的进口记录，保留出口记录
                    valid_records.append(outlet_record)

        if valid_records:
            result_data = pd.DataFrame(valid_records).drop_duplicates()
            print(f"   浓度配对检查完成，剔除了 {removed_count} 条记录")
            print(f"   剔除原因：出口浓度 ≥ 进口浓度")
            return result_data
        else:
            print("   警告: 浓度配对检查后无有效数据")
            return pd.DataFrame()

    def ks_test_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """K-S检验数据清洗"""
        print("\n=== K-S检验数据清洗 ===")
        
        # 分别处理进口和出口数据
        inlet_data = data[data['进口0出口1'] == 0].copy()
        outlet_data = data[data['进口0出口1'] == 1].copy()
        
        cleaned_data = []
        
        for data_type, subset in [('进口', inlet_data), ('出口', outlet_data)]:
            if len(subset) == 0:
                continue
                
            voc_column = '进口voc' if data_type == '进口' else '出口voc'
            voc_values = subset[voc_column].dropna()
            
            if len(voc_values) < 10:  # 数据量太少，跳过检验
                cleaned_data.append(subset)
                print(f"{data_type}数据量太少({len(voc_values)}条)，跳过K-S检验")
                continue
            
            # K-S检验正态性
            _, p_value = stats.kstest(voc_values, 'norm', args=(voc_values.mean(), voc_values.std()))
            print(f"{data_type}数据K-S检验 p值: {p_value:.4f}")
            
            if p_value > 0.05:
                # 正态分布，使用3σ准则
                mean_val = voc_values.mean()
                std_val = voc_values.std()
                threshold = 3 * std_val
                
                mask = np.abs(voc_values - mean_val) <= threshold
                cleaned_subset = subset[mask]
                removed_count = len(subset) - len(cleaned_subset)
                print(f"{data_type}数据正态分布，使用3σ准则: 保留{len(cleaned_subset)}条，剔除{removed_count}条")
                
            else:
                # 非正态分布，使用Z-score
                z_scores = np.abs(stats.zscore(voc_values))
                mask = z_scores <= 3
                cleaned_subset = subset[mask]
                removed_count = len(subset) - len(cleaned_subset)
                print(f"{data_type}数据非正态分布，使用Z-score: 保留{len(cleaned_subset)}条，剔除{removed_count}条")
            
            cleaned_data.append(cleaned_subset)
        
        # 合并清洗后的数据
        if cleaned_data:
            result = pd.concat(cleaned_data, ignore_index=True)
            result = result.sort_values('创建时间').reset_index(drop=True)
            print(f"K-S检验清洗完成: 最终保留 {len(result)} 条记录")
            return result
        else:
            return pd.DataFrame()
    
    def boxplot_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """箱型图异常值清洗"""
        print("\n=== 箱型图异常值清洗 ===")
        
        # 分别处理进口和出口数据
        inlet_data = data[data['进口0出口1'] == 0].copy()
        outlet_data = data[data['进口0出口1'] == 1].copy()
        
        cleaned_data = []
        
        for data_type, subset in [('进口', inlet_data), ('出口', outlet_data)]:
            if len(subset) == 0:
                continue
                
            voc_column = '进口voc' if data_type == '进口' else '出口voc'
            voc_values = subset[voc_column].dropna()
            
            if len(voc_values) < 4:  # 数据量太少，跳过清洗
                cleaned_data.append(subset)
                print(f"{data_type}数据量太少({len(voc_values)}条)，跳过箱型图清洗")
                continue
            
            # 计算四分位数
            Q1 = voc_values.quantile(0.25)
            Q3 = voc_values.quantile(0.75)
            IQR = Q3 - Q1
            
            # 计算异常值边界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 过滤异常值
            mask = (voc_values >= lower_bound) & (voc_values <= upper_bound)
            cleaned_subset = subset[mask]
            removed_count = len(subset) - len(cleaned_subset)
            
            print(f"{data_type}数据箱型图清洗: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
            print(f"  边界: [{lower_bound:.2f}, {upper_bound:.2f}], 保留{len(cleaned_subset)}条，剔除{removed_count}条")
            
            cleaned_data.append(cleaned_subset)
        
        # 合并清洗后的数据
        if cleaned_data:
            result = pd.concat(cleaned_data, ignore_index=True)
            result = result.sort_values('创建时间').reset_index(drop=True)
            print(f"箱型图清洗完成: 最终保留 {len(result)} 条记录")
            return result
        else:
            return pd.DataFrame()
    
    def calculate_efficiency_data(self, data: pd.DataFrame, method_name: str) -> Optional[pd.DataFrame]:
        """计算吸附效率数据"""
        print(f"\n=== 计算{method_name}吸附效率 ===")
        
        if len(data) == 0:
            print(f"警告: {method_name}数据为空")
            return None
        
        # 分离进出口数据
        inlet_data = data[data['进口0出口1'] == 0].copy()
        outlet_data = data[data['进口0出口1'] == 1].copy()
        
        print(f"进口数据: {len(inlet_data)} 条")
        print(f"出口数据: {len(outlet_data)} 条")
        
        if len(inlet_data) == 0 or len(outlet_data) == 0:
            print(f"警告: {method_name}缺少进口或出口数据")
            return None
        
        # 按时间排序
        inlet_data = inlet_data.sort_values('创建时间')
        outlet_data = outlet_data.sort_values('创建时间')
        
        # 计算效率数据
        efficiency_records = []
        
        # 获取所有时间点并排序
        all_times = sorted(data['创建时间'].unique())
        
        # 识别连续的时间段（间隔超过1小时认为是不同时间段）
        time_segments = []
        current_segment = [all_times[0]]
        
        for i in range(1, len(all_times)):
            time_diff = (all_times[i] - all_times[i-1]).total_seconds() / 60
            if time_diff > 60:  # 间隔超过1小时
                time_segments.append(current_segment)
                current_segment = [all_times[i]]
            else:
                current_segment.append(all_times[i])
        
        if current_segment:
            time_segments.append(current_segment)
        
        print(f"识别到 {len(time_segments)} 个时间段")
        
        # 为每个时间段计算效率
        start_time = data['创建时间'].min()
        
        for segment_idx, time_segment in enumerate(time_segments):
            segment_start = time_segment[0]
            segment_end = time_segment[-1]
            
            # 获取该时间段的数据
            segment_data = data[
                (data['创建时间'] >= segment_start) & 
                (data['创建时间'] <= segment_end)
            ]
            
            segment_inlet = segment_data[segment_data['进口0出口1'] == 0]
            segment_outlet = segment_data[segment_data['进口0出口1'] == 1]
            
            if len(segment_inlet) > 0 and len(segment_outlet) > 0:
                # 计算平均浓度
                avg_inlet = segment_inlet['进口voc'].mean()
                avg_outlet = segment_outlet['出口voc'].mean()
                
                # 根据算法要求计算效率
                if avg_inlet > avg_outlet:  # C0 > C1
                    efficiency = (avg_outlet / avg_inlet) * 100
                else:  # C0 < C1，使用上一个小于C1的C0
                    # 简化处理：如果进口浓度小于出口浓度，效率设为0
                    efficiency = 0.0
                
                # 限制效率不超过95%
                efficiency = min(efficiency, 95.0)
                
                # 计算时间坐标
                segment_mid_time = segment_start + (segment_end - segment_start) / 2
                time_minutes = (segment_mid_time - start_time).total_seconds() / 60
                
                efficiency_records.append({
                    'time': time_minutes,
                    'efficiency': efficiency,
                    'inlet_conc': avg_inlet,
                    'outlet_conc': avg_outlet,
                    'data_count': len(segment_data),
                    'window_start': segment_start,
                    'window_end': segment_end,
                    'segment_idx': segment_idx + 1
                })
                
                print(f"时段{segment_idx+1}: 进口={avg_inlet:.2f}, 出口={avg_outlet:.2f}, 效率={efficiency:.1f}%")
        
        if efficiency_records:
            efficiency_df = pd.DataFrame(efficiency_records)
            print(f"生成效率数据: {len(efficiency_df)} 个时间段")
            print(f"平均效率: {efficiency_df['efficiency'].mean():.2f}%")
            return efficiency_df
        else:
            print(f"无法生成{method_name}效率数据")
            return None

    def _create_time_segments(self, efficiency_data: pd.DataFrame, time_intervals: int = None) -> List[Dict]:
        """创建时间段数据 - 与基于最终版算法的可视化保持一致"""
        if len(efficiency_data) == 0:
            return []

        print(f"   原始效率数据点数: {len(efficiency_data)}")

        # 按时间排序
        efficiency_data_sorted = efficiency_data.sort_values('time').reset_index(drop=True)

        # 将数据分成16组，用于标记大时间段，但不能超过数据点数量
        target_groups = min(16, len(efficiency_data_sorted))
        group_size = max(1, len(efficiency_data_sorted) // target_groups)

        print(f"   将 {len(efficiency_data_sorted)} 个时间段分为 {target_groups} 个大组进行标记")
        print(f"   每组包含约 {group_size} 个时间段")

        # 存储所有数据点
        all_data_points = []
        group_info = []  # 存储大时间段信息

        # 先计算每个大时间段的信息
        for group_idx in range(target_groups):
            start_idx = group_idx * group_size
            if group_idx == target_groups - 1:
                end_idx = len(efficiency_data_sorted)
            else:
                end_idx = (group_idx + 1) * group_size

            group_data = efficiency_data_sorted.iloc[start_idx:end_idx]

            if len(group_data) > 0:
                # 找到时间段处于最中间位置的数据点
                middle_relative_idx = len(group_data) // 2
                middle_absolute_idx = start_idx + middle_relative_idx
                middle_point = efficiency_data_sorted.iloc[middle_relative_idx]

                # 获取时间范围
                first_window = group_data.iloc[0]
                last_window = group_data.iloc[-1]

                start_time_str = first_window['window_start'].strftime('%m-%d')
                end_time_str = last_window['window_end'].strftime('%m-%d')

                if start_time_str == end_time_str:
                    time_display = start_time_str
                else:
                    time_display = f"{start_time_str}~{end_time_str}"

                group_info.append({
                    'group_idx': group_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'middle_data_idx': middle_absolute_idx,
                    'time_display': time_display,
                    'middle_efficiency': middle_point['efficiency']
                })

        # 为所有数据点添加信息
        for i, row in efficiency_data_sorted.iterrows():
            # 判断这个点属于哪个大时间段
            group_idx = min(i // group_size, target_groups - 1)
            group = group_info[group_idx] if group_idx < len(group_info) else None

            # 判断是否是大时间段的中间位置点
            is_median_point = group and i == group['middle_data_idx']

            # 格式化单个时间段的时间显示
            individual_start = row['window_start'].strftime('%m-%d %H:%M')
            individual_end = row['window_end'].strftime('%H:%M')
            individual_time_display = f"{individual_start}-{individual_end}"

            point_data = {
                'segment': i + 1,  # 原始序号
                'group_idx': group_idx + 1,  # 所属大时间段
                'time_start': row['time'],
                'time_end': row['time'],
                'time_start_str': individual_start,
                'time_end_str': individual_end,
                'time_display': individual_time_display,
                'group_time_display': group['time_display'] if group else '',
                'efficiency': row['efficiency'],
                'inlet_conc': row.get('inlet_conc', 0),
                'outlet_conc': row.get('outlet_conc', 0),
                'data_count': row.get('data_count', 1),
                'is_median_point': is_median_point,
                'window_start': row['window_start'],
                'window_end': row['window_end']
            }

            all_data_points.append(point_data)

        print(f"   处理完成，生成 {len(all_data_points)} 个数据点")
        print(f"   其中 {sum(1 for p in all_data_points if p['is_median_point'])} 个为大时间段中位值点")

        return all_data_points

    def _create_final_visualization(self, segment_data: List[Dict], method_name: str) -> plt.Figure:
        """创建最终的可视化图像 - 与基于最终版算法的可视化保持一致"""
        if not segment_data:
            raise ValueError("没有时间段数据可用于可视化")

        # 创建图像，启用交互功能
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))

        # 提取数据
        efficiencies = [d['efficiency'] for d in segment_data]
        x_positions = list(range(1, len(segment_data) + 1))

        # 分离中位值点和普通点，过滤掉效率为0的点
        median_points = [(i+1, d['efficiency']) for i, d in enumerate(segment_data)
                        if d['is_median_point'] and d['efficiency'] > 0]
        normal_points = [(i+1, d['efficiency']) for i, d in enumerate(segment_data)
                        if not d['is_median_point'] and d['efficiency'] > 0]

        # 绘制所有数据点的连线（模糊处理）
        ax.plot(x_positions, efficiencies, 'b-', linewidth=1.5, alpha=0.5, label='效率曲线', zorder=2)

        # 绘制普通数据点（模糊处理，较小，用于鼠标悬停）
        if normal_points:
            normal_x, normal_y = zip(*normal_points)
            scatter_normal = ax.scatter(normal_x, normal_y, color='lightblue', s=60, zorder=3,
                          label='普通数据点', edgecolors='blue', linewidth=0.5, alpha=0.4)

        # 绘制中间位置点（清晰显示，较大，红色）
        if median_points:
            median_x, median_y = zip(*median_points)
            scatter_median = ax.scatter(median_x, median_y, color='red', s=180, zorder=6,
                          label='大时间段中间点', edgecolors='darkred', linewidth=2, alpha=0.9)

        # 只为大时间段中间位置点且效率大于0的点添加黄色标签
        for i, data in enumerate(segment_data):
            if data['is_median_point'] and data['efficiency'] > 0:
                x_pos = i + 1
                efficiency = data['efficiency']

                # 动态调整标签位置，避免与图例重叠
                if x_pos < len(segment_data) * 0.2:  # 左侧20%区域
                    offset_y = 25  # 向上偏移更多
                    offset_x = 15  # 向右偏移
                else:
                    offset_y = 20
                    offset_x = 0

                # 黄色标签显示效率
                ax.annotate(f'{efficiency:.1f}%',
                           xy=(x_pos, efficiency),
                           xytext=(offset_x, offset_y),
                           textcoords='offset points',
                           fontsize=11,
                           ha='center',
                           va='bottom',
                           fontweight='bold',
                           zorder=10,  # 确保标签在最上层
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='yellow',
                                   alpha=0.95,
                                   edgecolor='orange',
                                   linewidth=1.5),
                           arrowprops=dict(arrowstyle='->',
                                         connectionstyle='arc3,rad=0',
                                         color='orange',
                                         linewidth=1.5,
                                         alpha=0.8))

        # 添加交互式tooltip功能
        def on_hover(event):
            if event.inaxes == ax:
                # 找到最近的数据点
                if event.xdata is not None and event.ydata is not None:
                    distances = [(abs(event.xdata - (i+1)) + abs(event.ydata - d['efficiency']))
                               for i, d in enumerate(segment_data)]
                    min_idx = distances.index(min(distances))

                    # 如果距离足够近且不是中位值点且效率大于0，显示tooltip
                    if (distances[min_idx] < 2 and
                        not segment_data[min_idx]['is_median_point'] and
                        segment_data[min_idx]['efficiency'] > 0):
                        data = segment_data[min_idx]
                        tooltip_text = (f"时间段: {data['time_display']}\n"
                                      f"处理效率: {data['efficiency']:.1f}%\n"
                                      f"所属大组: 第{data['group_idx']}组")

                        # 清除之前的tooltip
                        for txt in ax.texts:
                            if hasattr(txt, 'is_tooltip'):
                                txt.remove()

                        # 动态调整tooltip位置，避免遮挡
                        tooltip_x_offset = 25 if min_idx < len(segment_data) * 0.8 else -80
                        tooltip_y_offset = 25 if data['efficiency'] < max(efficiencies) * 0.8 else -60

                        # 添加新的tooltip
                        tooltip = ax.annotate(tooltip_text,
                                            xy=(min_idx + 1, data['efficiency']),
                                            xytext=(tooltip_x_offset, tooltip_y_offset),
                                            textcoords='offset points',
                                            fontsize=9,
                                            ha='left' if tooltip_x_offset > 0 else 'right',
                                            va='bottom' if tooltip_y_offset > 0 else 'top',
                                            zorder=15,  # 最高层级
                                            bbox=dict(boxstyle='round,pad=0.4',
                                                    facecolor='lightgray',
                                                    alpha=0.95,
                                                    edgecolor='darkgray',
                                                    linewidth=1),
                                            arrowprops=dict(arrowstyle='->',
                                                          color='darkgray',
                                                          alpha=0.8,
                                                          linewidth=1))
                        tooltip.is_tooltip = True
                        fig.canvas.draw_idle()

        # 连接鼠标移动事件
        fig.canvas.mpl_connect('motion_notify_event', on_hover)

        # 设置坐标轴
        ax.set_xlabel('时间段序号 / 大时间段组', fontsize=16, fontweight='bold')
        ax.set_ylabel('处理效率 (%)', fontsize=16, fontweight='bold')
        ax.set_title(f'抽取式吸附曲线 - {method_name}完整数据点分析', fontsize=18, fontweight='bold', pad=20)

        # 设置x轴 - 适当稀疏显示刻度
        step = max(1, len(x_positions)//20)  # 计算步长
        sparse_ticks = x_positions[::step]  # 稀疏的刻度位置
        sparse_labels = [str(x) for x in sparse_ticks]  # 对应的标签

        ax.set_xticks(sparse_ticks)
        ax.set_xticklabels(sparse_labels, fontsize=10)
        ax.set_xlim(0.5, len(segment_data) + 0.5)

        # 添加大时间段的分组标识
        # 找到每个大时间段的边界和中心位置
        group_boundaries = {}
        for i, data in enumerate(segment_data):
            group_idx = data['group_idx']
            if group_idx not in group_boundaries:
                group_boundaries[group_idx] = {'start': i+1, 'end': i+1, 'display': data['group_time_display']}
            else:
                group_boundaries[group_idx]['end'] = i+1

        # 设置y轴范围，为下方标签留出空间
        if efficiencies:
            y_min = min(efficiencies) - 15
            y_max = max(efficiencies) + 10
            ax.set_ylim(y_min, y_max)

        # 在x轴下方添加大时间段标签
        for group_idx, bounds in group_boundaries.items():
            center_x = (bounds['start'] + bounds['end']) / 2
            ax.text(center_x, y_min + 5,  # 使用固定的y位置
                   f"组{group_idx}\n{bounds['display']}",
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

            # 添加分组分隔线
            if bounds['end'] < len(segment_data):
                ax.axvline(x=bounds['end'] + 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # 美化
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        # 将图例移到左上角，设置透明背景避免遮挡
        ax.legend(fontsize=12, loc='upper left', framealpha=0.8,
                 fancybox=True, shadow=True, ncol=1,
                 bbox_to_anchor=(0.02, 0.98))

        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()
        return fig

    def process_and_visualize(self):
        """完整的数据处理和可视化流程"""
        print("=== 抽取式吸附曲线完整数据处理与可视化 ===")
        print("="*60)

        # 创建输出文件夹
        cleaned_data_dir = "可视化项目/清洗后数据"
        visualization_dir = "可视化项目/可视化图像"

        os.makedirs(cleaned_data_dir, exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)

        print(f"清洗后数据将保存到: {cleaned_data_dir}")
        print(f"可视化图像将保存到: {visualization_dir}")

        # 1. 加载数据
        if not self.load_data():
            return

        # 2. 基础数据清洗
        basic_cleaned = self.basic_data_cleaning(self.raw_data)
        if len(basic_cleaned) == 0:
            print("基础清洗后无数据，程序结束")
            return

        # 3. K-S检验清洗
        print("\n" + "="*40)
        print("开始K-S检验数据清洗")
        self.cleaned_data_ks = self.ks_test_cleaning(basic_cleaned)

        # 4. 箱型图清洗
        print("\n" + "="*40)
        print("开始箱型图数据清洗")
        self.cleaned_data_boxplot = self.boxplot_cleaning(basic_cleaned)

        # 5. 计算效率数据
        if len(self.cleaned_data_ks) > 0:
            self.efficiency_data_ks = self.calculate_efficiency_data(self.cleaned_data_ks, "K-S检验")

        if len(self.cleaned_data_boxplot) > 0:
            self.efficiency_data_boxplot = self.calculate_efficiency_data(self.cleaned_data_boxplot, "箱型图")

        # 6. 创建可视化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # K-S检验可视化
        if self.efficiency_data_ks is not None and len(self.efficiency_data_ks) > 0:
            print("\n" + "="*40)
            print("创建K-S检验可视化")
            ks_segments = self._create_time_segments(self.efficiency_data_ks)
            if ks_segments:
                fig_ks = self._create_final_visualization(ks_segments, "K-S检验清洗")
                filename_ks = os.path.join(visualization_dir, f"{self.base_filename}_KS检验清洗_{timestamp}.png")
                fig_ks.savefig(filename_ks, dpi=300, bbox_inches='tight')
                print(f"K-S检验可视化图片已保存: {filename_ks}")
                plt.show()

        # 箱型图可视化
        if self.efficiency_data_boxplot is not None and len(self.efficiency_data_boxplot) > 0:
            print("\n" + "="*40)
            print("创建箱型图可视化")
            box_segments = self._create_time_segments(self.efficiency_data_boxplot)
            if box_segments:
                fig_box = self._create_final_visualization(box_segments, "箱型图清洗")
                filename_box = os.path.join(visualization_dir, f"{self.base_filename}_箱型图清洗_{timestamp}.png")
                fig_box.savefig(filename_box, dpi=300, bbox_inches='tight')
                print(f"箱型图可视化图片已保存: {filename_box}")
                plt.show()

        # 7. 保存清洗后的数据
        if len(self.cleaned_data_ks) > 0:
            ks_filename = os.path.join(cleaned_data_dir, f"{self.base_filename}_KS检验清洗_{timestamp}.csv")
            self.cleaned_data_ks.to_csv(ks_filename, index=False, encoding='utf-8-sig')
            print(f"K-S检验清洗数据已保存: {ks_filename}")

        if len(self.cleaned_data_boxplot) > 0:
            box_filename = os.path.join(cleaned_data_dir, f"{self.base_filename}_箱型图清洗_{timestamp}.csv")
            self.cleaned_data_boxplot.to_csv(box_filename, index=False, encoding='utf-8-sig')
            print(f"箱型图清洗数据已保存: {box_filename}")

        print("\n" + "="*60)
        print("数据处理与可视化完成！")


def main():
    """主函数"""
    print("抽取式吸附曲线完整数据处理与可视化算法")
    print("支持CSV、XLSX、XLS格式文件，实现从数据清洗到可视化的全流程")
    print("="*60)

    # 数据文件路径 - 支持多种格式
    data_file = "可视化项目/7.24.csv"  # 可以是 .csv, .xlsx, .xls 格式

    print(f"当前处理文件: {data_file}")
    print("支持的文件格式: CSV (.csv), Excel (.xlsx, .xls)")
    print("="*60)

    # 创建处理器并执行完整流程
    processor = AdsorptionCurveProcessor(data_file)
    processor.process_and_visualize()


if __name__ == "__main__":
    main()
