#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于最终版抽取式吸附曲线算法的可视化脚本
使用K-S检验和箱型图清洗后的数据进行可视化分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class AdsorptionCurveVisualizer:
    """抽取式吸附曲线可视化器"""
    
    def __init__(self):
        self.ks_data = None
        self.boxplot_data = None
        self.ks_efficiency_data = None
        self.boxplot_efficiency_data = None
        
    def load_data(self):
        """加载数据文件"""
        print("=== 加载数据文件 ===")
        
        # 使用绝对路径
        ks_file = r'D:\连微\7.24数据_KS检验清洗结果_修改版.csv'
        box_file = r'D:\连微\7.24数据_箱型图清洗结果_修改版.csv'
        
        try:
            # 加载K-S检验数据
            self.ks_data = pd.read_csv(ks_file, encoding='utf-8-sig')
            self.ks_data['创建时间'] = pd.to_datetime(self.ks_data['创建时间'])
            print(f"K-S检验数据加载成功: {len(self.ks_data)} 条记录")
            
            # 加载箱型图数据
            self.boxplot_data = pd.read_csv(box_file, encoding='utf-8-sig')
            self.boxplot_data['创建时间'] = pd.to_datetime(self.boxplot_data['创建时间'])
            print(f"箱型图数据加载成功: {len(self.boxplot_data)} 条记录")
            
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def calculate_efficiency_data(self, data, method_name):
        """计算效率数据"""
        print(f"\n=== 计算{method_name}效率数据 ===")

        # 分离进出口数据
        inlet_data = data[data['进口0出口1'] == 0].copy()
        outlet_data = data[data['进口0出口1'] == 1].copy()

        print(f"进口数据: {len(inlet_data)} 条")
        print(f"出口数据: {len(outlet_data)} 条")
        print(f"时间范围: {data['创建时间'].min()} 到 {data['创建时间'].max()}")

        if len(inlet_data) == 0 or len(outlet_data) == 0:
            print(f"警告: {method_name}缺少进口或出口数据")
            return None

        # 创建时间序列（以分钟为单位）
        start_time = data['创建时间'].min()
        end_time = data['创建时间'].max()
        total_duration = (end_time - start_time).total_seconds() / 60  # 总时长（分钟）

        print(f"数据总时长: {total_duration:.1f} 分钟")

        # 识别不连续的时间段并计算每个时间段的效率
        print("识别不连续时间段并计算效率")

        # 获取所有时间点并排序
        all_times = sorted(data['创建时间'].unique())
        print(f"共有 {len(all_times)} 个不同的时间点")

        # 识别时间段（间隔超过1小时的认为是不同时间段）
        time_segments = []
        current_segment = [all_times[0]]

        for i in range(1, len(all_times)):
            time_diff = (all_times[i] - all_times[i-1]).total_seconds() / 60  # 分钟
            if time_diff > 60:  # 间隔超过1小时，开始新时间段
                time_segments.append(current_segment)
                current_segment = [all_times[i]]
            else:
                current_segment.append(all_times[i])

        # 添加最后一个时间段
        if current_segment:
            time_segments.append(current_segment)

        print(f"识别到 {len(time_segments)} 个不连续时间段:")
        for i, segment in enumerate(time_segments):
            print(f"  时段{i+1}: {segment[0].strftime('%H:%M')} - {segment[-1].strftime('%H:%M')} ({len(segment)}个时间点)")

        efficiency_data = []

        # 为每个时间段计算效率
        for segment_idx, time_segment in enumerate(time_segments):
            segment_start = time_segment[0]
            segment_end = time_segment[-1]

            # 获取该时间段内的所有数据
            segment_data = data[
                (data['创建时间'] >= segment_start) &
                (data['创建时间'] <= segment_end)
            ]

            segment_inlet = segment_data[segment_data['进口0出口1'] == 0]
            segment_outlet = segment_data[segment_data['进口0出口1'] == 1]

            if len(segment_inlet) > 0 and len(segment_outlet) > 0:
                avg_inlet = segment_inlet['进口voc'].mean()
                avg_outlet = segment_outlet['出口voc'].mean()

                # 计算处理效率：(出口浓度/进口浓度)*100%
                if avg_inlet > 0:
                    efficiency = (avg_outlet / avg_inlet) * 100
                    # 确保效率不超过95%
                    efficiency = min(efficiency, 95.0)

                    # 使用时间段的中点作为时间坐标
                    segment_mid_time = segment_start + (segment_end - segment_start) / 2
                    time_minutes = (segment_mid_time - start_time).total_seconds() / 60

                    efficiency_data.append({
                        'time': time_minutes,
                        'efficiency': efficiency,
                        'inlet_conc': avg_inlet,
                        'outlet_conc': avg_outlet,
                        'data_count': len(segment_data),
                        'window_start': segment_start,
                        'window_end': segment_end,
                        'segment_idx': segment_idx + 1
                    })

                    print(f"时段{segment_idx+1} ({segment_start.strftime('%H:%M')}-{segment_end.strftime('%H:%M')}): "
                          f"进口={avg_inlet:.2f}, 出口={avg_outlet:.2f}, 效率={efficiency:.1f}%")

        if efficiency_data:
            efficiency_df = pd.DataFrame(efficiency_data)
            print(f"生成效率数据点: {len(efficiency_df)} 个")
            print(f"平均效率: {efficiency_df['efficiency'].mean():.2f}%")
            print(f"效率范围: {efficiency_df['efficiency'].min():.2f}% - {efficiency_df['efficiency'].max():.2f}%")
            return efficiency_df
        else:
            print(f"无法生成{method_name}效率数据")
            return None
    
    def _create_time_segments(self, efficiency_data: pd.DataFrame, time_intervals: int = None) -> List[Dict]:
        """创建时间段数据 - 展示所有数据点，标记大时间段的中位值"""
        if len(efficiency_data) == 0:
            return []

        print(f"   原始效率数据点数: {len(efficiency_data)}")

        # 按时间排序
        efficiency_data_sorted = efficiency_data.sort_values('time').reset_index(drop=True)

        # 将数据分成16组，用于标记大时间段
        target_groups = 16
        group_size = len(efficiency_data_sorted) // target_groups

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
                    'middle_data_idx': middle_absolute_idx,  # 改为中间位置的绝对索引
                    'time_display': time_display,
                    'middle_efficiency': middle_point['efficiency']  # 中间位置的效率
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
                'is_median_point': is_median_point,  # 标记是否为大时间段中位值
                'window_start': row['window_start'],
                'window_end': row['window_end']
            }

            all_data_points.append(point_data)

        print(f"   处理完成，生成 {len(all_data_points)} 个数据点")
        print(f"   其中 {sum(1 for p in all_data_points if p['is_median_point'])} 个为大时间段中位值点")

        return all_data_points
    
    def _create_final_visualization(self, segment_data: List[Dict], method_name: str) -> plt.Figure:
        """创建最终的可视化图像"""
        if not segment_data:
            raise ValueError("没有时间段数据可用于可视化")

        # 创建图像，启用交互功能
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))

        # 提取数据
        efficiencies = [d['efficiency'] for d in segment_data]
        x_positions = list(range(1, len(segment_data) + 1))

        # 分离中位值点和普通点
        median_points = [(i+1, d['efficiency']) for i, d in enumerate(segment_data) if d['is_median_point']]
        normal_points = [(i+1, d['efficiency']) for i, d in enumerate(segment_data) if not d['is_median_point']]

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

        # 只为大时间段中间位置点添加黄色标签
        for i, data in enumerate(segment_data):
            if data['is_median_point']:
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

                    # 如果距离足够近且不是中位值点，显示tooltip
                    if distances[min_idx] < 2 and not segment_data[min_idx]['is_median_point']:
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
    
    def create_visualizations(self):
        """创建可视化图表"""
        if not self.load_data():
            return
        
        # 计算效率数据
        if self.ks_data is not None:
            self.ks_efficiency_data = self.calculate_efficiency_data(self.ks_data, "K-S检验")
        
        if self.boxplot_data is not None:
            self.boxplot_efficiency_data = self.calculate_efficiency_data(self.boxplot_data, "箱型图")
        
        # 创建可视化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # K-S检验可视化
        if self.ks_efficiency_data is not None:
            ks_segments = self._create_time_segments(self.ks_efficiency_data)
            if ks_segments:
                fig_ks = self._create_final_visualization(ks_segments, "K-S检验清洗")
                filename_ks = f"抽取式吸附曲线_KS检验_{timestamp}.png"
                fig_ks.savefig(filename_ks, dpi=300, bbox_inches='tight')
                print(f"K-S检验可视化图片已保存: {filename_ks}")
                plt.show()
        
        # 箱型图可视化
        if self.boxplot_efficiency_data is not None:
            box_segments = self._create_time_segments(self.boxplot_efficiency_data)
            if box_segments:
                fig_box = self._create_final_visualization(box_segments, "箱型图清洗")
                filename_box = f"抽取式吸附曲线_箱型图_{timestamp}.png"
                fig_box.savefig(filename_box, dpi=300, bbox_inches='tight')
                print(f"箱型图可视化图片已保存: {filename_box}")
                plt.show()

def main():
    """主函数"""
    print("基于最终版算法的抽取式吸附曲线可视化")
    print("="*50)
    
    visualizer = AdsorptionCurveVisualizer()
    visualizer.create_visualizations()
    
    print("\n可视化完成!")

if __name__ == "__main__":
    main()
