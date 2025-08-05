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
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class WarningLevel(Enum):
    """预警等级"""
    GREEN = "绿色"      # 无需更换
    YELLOW = "黄色"     # 适时更换
    ORANGE = "橙色"     # 立即更换
    RED = "红色"        # 立即更换

@dataclass
class WarningEvent:
    """预警事件"""
    timestamp: float
    warning_level: WarningLevel
    breakthrough_ratio: float  # 穿透率 %
    efficiency: float         # 吸附效率 %
    reason: str              # 预警原因
    recommendation: str      # 建议措施
    predicted_saturation_time: Optional[float] = None  # 预计饱和时间

class LogisticWarningModel:
    """基于Logistic模型的预警系统"""

    def __init__(self,
                 breakthrough_start_threshold: float = 0.01,  # 穿透起始点阈值 1%
                 warning_ratio: float = 0.8,                 # 预警点比例 80%
                 saturation_threshold: float = 0.9):         # 预测饱和点阈值 90%
        """
        初始化预警模型

        参数:
            breakthrough_start_threshold: 穿透起始点阈值（基于实际数据检测）
            warning_ratio: 预警点比例（从起始到预测饱和的百分比）
            saturation_threshold: 预测饱和点阈值
        """
        self.breakthrough_start_threshold = breakthrough_start_threshold
        self.warning_ratio = warning_ratio
        self.saturation_threshold = saturation_threshold

        # 模型参数
        self.params = None
        self.fitted = False

        # 关键时间点
        self.breakthrough_start_time = None
        self.predicted_saturation_time = None
        self.warning_time = None

    @staticmethod
    def logistic_function(t, A, k, t0):
        """
        Logistic函数: C/C0 = A / (1 + exp(-k*(t-t0)))

        参数:
            t: 时间
            A: 最大穿透率（通常接近1）
            k: 增长率
            t0: 拐点时间
        """
        return A / (1 + np.exp(-k * (t - t0)))

    def _analyze_data_phases(self, t_valid: np.array, bt_valid: np.array) -> dict:
        """分析数据的不同阶段特征"""
        from scipy.ndimage import gaussian_filter1d

        # 平滑数据用于分析
        bt_smooth = gaussian_filter1d(bt_valid, sigma=max(1, len(bt_valid)//10))

        # 计算一阶和二阶导数
        first_derivative = np.gradient(bt_smooth, t_valid)
        second_derivative = np.gradient(first_derivative, t_valid)

        # 识别不同阶段
        phases = {
            'initial_phase': [],      # 初始缓慢增长阶段
            'growth_phase': [],       # 快速增长阶段
            'saturation_phase': []    # 接近饱和阶段
        }

        # 基于导数变化识别阶段
        max_growth_rate = np.max(first_derivative)
        growth_threshold = max_growth_rate * 0.3

        for i in range(len(bt_valid)):
            if first_derivative[i] < growth_threshold:
                if bt_valid[i] < np.median(bt_valid):
                    phases['initial_phase'].append(i)
                else:
                    phases['saturation_phase'].append(i)
            else:
                phases['growth_phase'].append(i)

        return phases

    def _calculate_dynamic_weights(self, t_valid: np.array, bt_valid: np.array) -> np.array:
        """基于数据特征计算动态权重"""
        weights = np.ones_like(bt_valid)
        phases = self._analyze_data_phases(t_valid, bt_valid)

        # 初始阶段权重：对确定穿透起始点很重要
        for i in phases['initial_phase']:
            weights[i] *= 3.0  # 初始阶段权重最高

        # 增长阶段权重：对确定增长率很重要
        for i in phases['growth_phase']:
            weights[i] *= 2.0

        # 饱和阶段权重：对确定最大值重要，但不如前两个阶段
        for i in phases['saturation_phase']:
            weights[i] *= 1.5

        # 基于数据质量调整权重
        # 数据变化较大的点权重降低（可能是噪声）
        if len(bt_valid) > 3:
            bt_diff = np.abs(np.diff(bt_valid))
            bt_diff_normalized = bt_diff / (np.mean(bt_diff) + 1e-8)

            # 对变化异常大的点降低权重
            for i in range(len(bt_diff)):
                if bt_diff_normalized[i] > 2.0:  # 变化超过平均值2倍
                    weights[i] *= 0.5
                    weights[i+1] *= 0.5

        print(f"动态权重统计: 最小={np.min(weights):.2f}, 最大={np.max(weights):.2f}, 平均={np.mean(weights):.2f}")
        return weights

    def _estimate_dynamic_growth_rate(self, t_valid: np.array, bt_valid: np.array) -> float:
        """基于数据动态估计增长率"""
        if len(bt_valid) < 5:
            return 0.0001

        # 计算局部增长率
        window_size = max(3, len(bt_valid) // 5)
        local_growth_rates = []

        for i in range(window_size, len(bt_valid) - window_size):
            # 在窗口内拟合线性增长
            window_t = t_valid[i-window_size:i+window_size]
            window_bt = bt_valid[i-window_size:i+window_size]

            if len(window_t) > 2:
                # 计算局部斜率
                dt = window_t[-1] - window_t[0]
                dbt = window_bt[-1] - window_bt[0]
                if dt > 0 and window_bt[0] > 0:
                    local_rate = dbt / (dt * window_bt[0])
                    if local_rate > 0:
                        local_growth_rates.append(local_rate)

        if local_growth_rates:
            # 使用中位数作为稳健估计
            k_estimate = np.median(local_growth_rates)
            # 限制在合理范围内
            k_estimate = np.clip(k_estimate, 0.000001, 0.1)
            print(f"动态增长率估计: {k_estimate:.6f}")
            return k_estimate
        else:
            return 0.0001

    def fit_model(self, time_data: np.array, breakthrough_ratio_data: np.array) -> bool:
        """
        拟合Logistic模型 - 使用动态权重和动态增长率

        参数:
            time_data: 时间数据（秒）
            breakthrough_ratio_data: 穿透率数据（出口浓度/进口浓度）

        返回:
            是否拟合成功
        """
        try:
            # 更合理的数据范围限制
            breakthrough_data = np.clip(breakthrough_ratio_data, 0.0001, 0.9999)

            # 过滤有效数据
            valid_mask = (breakthrough_data > 0) & (breakthrough_data < 1) & (time_data > 0)
            if np.sum(valid_mask) < 5:  # 至少需要5个数据点
                print("数据点不足，无法拟合Logistic模型")
                return False

            t_valid = time_data[valid_mask]
            bt_valid = breakthrough_data[valid_mask]

            print(f"数据分析: 时间范围 {t_valid.min():.1f}-{t_valid.max():.1f}s, 穿透率范围 {bt_valid.min():.3f}-{bt_valid.max():.3f}")

            # 动态参数估计
            A_init = min(0.98, np.max(bt_valid) * 1.05)  # 基于数据最大值，更保守的估计
            k_init = self._estimate_dynamic_growth_rate(t_valid, bt_valid)  # 动态增长率估计

            # 改进的拐点估计
            if len(bt_valid) > 5:
                from scipy.ndimage import gaussian_filter1d
                bt_smooth = gaussian_filter1d(bt_valid, sigma=max(1, len(bt_valid)//15))
                first_derivative = np.gradient(bt_smooth, t_valid)
                # 找到增长率最大的点作为拐点
                max_growth_idx = np.argmax(first_derivative)
                t0_init = t_valid[max_growth_idx]
            else:
                t0_init = np.median(t_valid)

            print(f"动态参数估计: A={A_init:.3f}, k={k_init:.6f}, t0={t0_init:.1f}")

            # 计算动态权重
            weights = self._calculate_dynamic_weights(t_valid, bt_valid)

            # 动态调整边界约束
            A_max = min(1.0, np.max(bt_valid) * 1.2)
            A_min = max(0.1, np.max(bt_valid) * 0.7)
            k_max = max(0.01, k_init * 10)  # 基于估计值动态调整
            k_min = max(0.000001, k_init * 0.1)

            lower_bounds = [A_min, k_min, 0]
            upper_bounds = [A_max, k_max, np.max(t_valid) * 2]

            print(f"动态边界: A[{A_min:.3f}, {A_max:.3f}], k[{k_min:.6f}, {k_max:.6f}]")

            # 多次拟合尝试，选择最佳结果
            best_params = None
            best_score = float('inf')

            for attempt in range(3):
                try:
                    # 每次尝试稍微调整初始参数
                    A_try = A_init * (0.9 + 0.2 * attempt / 2)
                    k_try = k_init * (0.5 + attempt)
                    t0_try = t0_init * (0.8 + 0.4 * attempt / 2)

                    params, covariance = curve_fit(
                        self.logistic_function,
                        t_valid, bt_valid,
                        p0=[A_try, k_try, t0_try],
                        bounds=(lower_bounds, upper_bounds),
                        sigma=1/weights,  # 动态权重
                        maxfev=8000
                    )

                    # 计算拟合质量
                    predicted = self.logistic_function(t_valid, *params)
                    weighted_residuals = (bt_valid - predicted) * weights
                    score = np.sum(weighted_residuals**2)

                    if score < best_score:
                        best_score = score
                        best_params = params

                except:
                    continue

            if best_params is None:
                raise ValueError("所有拟合尝试都失败")

            self.params = best_params
            self.fitted = True

            # 计算关键时间点
            self._calculate_key_timepoints(t_valid, bt_valid)

            # 评估拟合质量
            predicted = self.logistic_function(t_valid, *self.params)
            r_squared = 1 - np.sum((bt_valid - predicted)**2) / np.sum((bt_valid - np.mean(bt_valid))**2)

            print(f"动态Logistic模型拟合成功:")
            print(f"  参数: A={self.params[0]:.3f}, k={self.params[1]:.6f}, t0={self.params[2]:.1f}")
            print(f"  拟合质量 R²: {r_squared:.3f}")
            return True

        except Exception as e:
            print(f"动态Logistic模型拟合失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _calculate_key_timepoints(self, time_data: np.array, breakthrough_data: np.array = None):
        """计算关键时间点 - 基于实际数据和模型预测"""
        if not self.fitted:
            return

        A, k, t0 = self.params

        # 1. 计算穿透起始时间 - 基于实际数据中浓度开始产生的点
        if breakthrough_data is not None:
            # 在实际数据中找到第一个超过阈值的点
            breakthrough_indices = np.where(breakthrough_data >= self.breakthrough_start_threshold)[0]
            if len(breakthrough_indices) > 0:
                self.breakthrough_start_time = time_data[breakthrough_indices[0]]
                print(f"基于实际数据检测到穿透起始时间: {self.breakthrough_start_time:.1f}s")
            else:
                self.breakthrough_start_time = np.min(time_data)
                print(f"未检测到穿透起始点，使用数据起始时间: {self.breakthrough_start_time:.1f}s")
        else:
            # 如果没有实际数据，使用模型计算
            try:
                if A > self.breakthrough_start_threshold:
                    self.breakthrough_start_time = t0 - np.log(A / self.breakthrough_start_threshold - 1) / k
                    if self.breakthrough_start_time < 0:
                        self.breakthrough_start_time = np.min(time_data)
                else:
                    self.breakthrough_start_time = np.min(time_data)
            except:
                self.breakthrough_start_time = np.min(time_data)

        # 2. 计算实际饱和时间 - 预测曲线达到饱和的时间点
        # 找到预测曲线接近最大值（A）的95%的时间点作为实际饱和点
        actual_saturation_ratio = A * 0.95  # 取模型最大值的95%作为实际饱和点
        try:
            if A > actual_saturation_ratio:
                self.predicted_saturation_time = t0 - np.log(A / actual_saturation_ratio - 1) / k
            else:
                # 如果模型预测的最大穿透率很小，则外推
                self.predicted_saturation_time = np.max(time_data) * 1.5
        except:
            self.predicted_saturation_time = np.max(time_data) * 1.5

        # 3. 计算预警时间点 - 基于实际饱和点
        # 预警时间 = 穿透起始时间 + (实际饱和时间 - 穿透起始时间) * 0.8
        if self.breakthrough_start_time is not None and self.predicted_saturation_time is not None:
            time_span = self.predicted_saturation_time - self.breakthrough_start_time
            self.warning_time = self.breakthrough_start_time + time_span * self.warning_ratio
            print(f"实际饱和穿透率: {actual_saturation_ratio:.1%}")
            print(f"实际饱和时间: {self.predicted_saturation_time:.1f}s")

        print(f"关键时间点计算:")
        print(f"  穿透起始时间: {self.breakthrough_start_time:.1f}s")
        print(f"  拐点时间(t0): {t0:.1f}s")
        print(f"  实际饱和时间: {self.predicted_saturation_time:.1f}s")
        print(f"  预警时间: {self.warning_time:.1f}s (起始到饱和的{self.warning_ratio:.0%})")
        print(f"  起始到饱和时间跨度: {self.predicted_saturation_time - self.breakthrough_start_time:.1f}s")

    def predict_breakthrough(self, time_points: np.array) -> np.array:
        """预测指定时间点的穿透率"""
        if not self.fitted:
            return np.zeros_like(time_points)

        return self.logistic_function(time_points, *self.params)

    def get_warning_level(self, current_time: float, current_efficiency: float) -> WarningLevel:
        """
        根据当前时间和效率确定预警等级

        参数:
            current_time: 当前时间
            current_efficiency: 当前吸附效率

        返回:
            预警等级
        """
        current_breakthrough = (100 - current_efficiency) / 100

        # 基于穿透率的预警
        if current_breakthrough <= self.breakthrough_start_threshold:
            return WarningLevel.GREEN
        elif current_breakthrough >= self.saturation_threshold:
            return WarningLevel.RED

        # 基于时间的预警（如果模型已拟合）
        if self.fitted and self.warning_time is not None and self.predicted_saturation_time is not None:
            if current_time >= self.predicted_saturation_time:
                return WarningLevel.RED
            elif current_time >= self.warning_time:
                return WarningLevel.ORANGE
            elif current_breakthrough > self.breakthrough_start_threshold:
                return WarningLevel.YELLOW

        # 仅基于穿透率的预警
        if current_breakthrough > 0.8:  # 80%穿透率
            return WarningLevel.ORANGE
        elif current_breakthrough > self.breakthrough_start_threshold:
            return WarningLevel.YELLOW

        return WarningLevel.GREEN

    def generate_warning_event(self, current_time: float, current_efficiency: float) -> Optional[WarningEvent]:
        """生成预警事件"""
        level = self.get_warning_level(current_time, current_efficiency)

        if level == WarningLevel.GREEN:
            return None

        current_breakthrough = (100 - current_efficiency) / 100

        # 生成预警原因和建议
        if level == WarningLevel.YELLOW:
            reason = f"穿透率达到{current_breakthrough*100:.1f}%，已超过起始点阈值"
            recommendation = "建议开始准备更换活性炭，监控穿透率变化趋势"
        elif level == WarningLevel.ORANGE:
            if self.warning_time and current_time >= self.warning_time:
                reason = f"已达到预警时间点({self.warning_time:.1f}s)，穿透率{current_breakthrough*100:.1f}%"
            else:
                reason = f"穿透率达到{current_breakthrough*100:.1f}%，接近饱和状态"
            recommendation = "立即安排更换活性炭，设备处于非稳定运行状态"
        else:  # RED
            if self.predicted_saturation_time and current_time >= self.predicted_saturation_time:
                reason = f"已达到预测饱和时间({self.predicted_saturation_time:.1f}s)"
            else:
                reason = f"穿透率达到{current_breakthrough*100:.1f}%，活性炭已饱和"
            recommendation = "紧急更换活性炭！设备已无法正常净化VOCs"

        return WarningEvent(
            timestamp=current_time,
            warning_level=level,
            breakthrough_ratio=current_breakthrough * 100,
            efficiency=current_efficiency,
            reason=reason,
            recommendation=recommendation,
            predicted_saturation_time=self.predicted_saturation_time
        )

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

        # 预警系统
        self.warning_model = LogisticWarningModel()
        self.warning_events = []
        
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
        """计算吸附效率数据 - 修改为按需求文档要求处理"""
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
            time_diff = (all_times[i] - all_times[i-1]).total_seconds()
            if time_diff > 3600:  # 间隔超过1小时（3600秒）
                time_segments.append(current_segment)
                current_segment = [all_times[i]]
            else:
                current_segment.append(all_times[i])

        if current_segment:
            time_segments.append(current_segment)

        print(f"识别到 {len(time_segments)} 个不连续时间段")

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

                # 根据需求文档：计算出口浓度/进口浓度的比值（穿透率）
                if avg_inlet > 0:
                    breakthrough_ratio = avg_outlet / avg_inlet  # 穿透率 = 出口浓度/进口浓度
                    efficiency = (1 - breakthrough_ratio) * 100  # 效率 = (1 - 穿透率) * 100%
                else:
                    breakthrough_ratio = 0.0
                    efficiency = 100.0

                # 计算时间坐标（秒）
                segment_mid_time = segment_start + (segment_end - segment_start) / 2
                time_seconds = (segment_mid_time - start_time).total_seconds()

                efficiency_records.append({
                    'time': time_seconds,  # 改为秒
                    'efficiency': efficiency,
                    'breakthrough_ratio': breakthrough_ratio,  # 添加穿透率
                    'inlet_conc': avg_inlet,
                    'outlet_conc': avg_outlet,
                    'data_count': len(segment_data),
                    'window_start': segment_start,
                    'window_end': segment_end,
                    'segment_idx': segment_idx + 1
                })

                print(f"时段{segment_idx+1}: 进口={avg_inlet:.2f}, 出口={avg_outlet:.2f}, 穿透率={breakthrough_ratio:.3f}, 效率={efficiency:.1f}%")

        if efficiency_records:
            efficiency_df = pd.DataFrame(efficiency_records)
            print(f"生成效率数据: {len(efficiency_df)} 个时间段")
            print(f"平均效率: {efficiency_df['efficiency'].mean():.2f}%")
            print(f"平均穿透率: {efficiency_df['breakthrough_ratio'].mean():.3f}")
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

    def analyze_warning_system(self):
        """分析预警系统 - 修改为使用穿透率数据"""
        # 选择最佳的效率数据进行预警分析
        efficiency_data = None
        method_name = ""

        if self.efficiency_data_ks is not None and len(self.efficiency_data_ks) > 0:
            efficiency_data = self.efficiency_data_ks
            method_name = "K-S检验"
        elif self.efficiency_data_boxplot is not None and len(self.efficiency_data_boxplot) > 0:
            efficiency_data = self.efficiency_data_boxplot
            method_name = "箱型图"

        if efficiency_data is None or len(efficiency_data) == 0:
            print("无有效效率数据，跳过预警分析")
            return

        print(f"使用{method_name}清洗后的数据进行预警分析")
        print(f"效率数据点数: {len(efficiency_data)}")

        # 准备数据 - 使用穿透率而不是效率
        time_data = efficiency_data['time'].values
        breakthrough_ratio_data = efficiency_data['breakthrough_ratio'].values

        print(f"穿透率范围: {breakthrough_ratio_data.min():.3f} - {breakthrough_ratio_data.max():.3f}")

        # 拟合Logistic模型
        if self.warning_model.fit_model(time_data, breakthrough_ratio_data):
            print("Logistic模型拟合成功")

            # 生成预警事件 - 基于穿透率和预警时间点
            self.warning_events = []
            for _, row in efficiency_data.iterrows():
                # 检查是否达到预警点
                current_time = row['time']
                current_breakthrough = row['breakthrough_ratio']

                # 根据需求文档：当某时间段的出口浓度/进口浓度达到预警点时，推送预警信息
                if (self.warning_model.warning_time is not None and
                    current_time >= self.warning_model.warning_time):

                    # 计算预警时间点的预期穿透率
                    warning_breakthrough = self.warning_model.predict_breakthrough(
                        np.array([self.warning_model.warning_time]))[0]

                    if current_breakthrough >= warning_breakthrough:
                        event = WarningEvent(
                            timestamp=current_time,
                            warning_level=WarningLevel.ORANGE,
                            breakthrough_ratio=current_breakthrough * 100,
                            efficiency=row['efficiency'],
                            reason=f"达到预警时间点({self.warning_model.warning_time:.1f}s)，穿透率{current_breakthrough:.3f}达到预警阈值{warning_breakthrough:.3f}",
                            recommendation="建议立即更换活性炭，设备已达到预警状态",
                            predicted_saturation_time=self.warning_model.predicted_saturation_time
                        )
                        self.warning_events.append(event)

            print(f"生成预警事件: {len(self.warning_events)} 个")

            # 显示预警摘要
            self._display_warning_summary()

        else:
            print("Logistic模型拟合失败，无法进行预警分析")

    def _display_warning_summary(self):
        """显示预警摘要"""
        if not self.warning_events:
            print("✅ 当前无预警事件，设备运行正常")
            return

        print(f"\n⚠️  检测到 {len(self.warning_events)} 个预警事件:")

        # 按预警等级分类
        warning_counts = {}
        latest_event = None

        for event in self.warning_events:
            level = event.warning_level.value
            warning_counts[level] = warning_counts.get(level, 0) + 1

            if latest_event is None or event.timestamp > latest_event.timestamp:
                latest_event = event

        # 显示统计
        for level, count in warning_counts.items():
            print(f"  {level}: {count} 次")

        # 显示最新预警
        if latest_event:
            print(f"\n🚨 最新预警状态: {latest_event.warning_level.value}")
            print(f"   时间: {latest_event.timestamp:.1f}s")
            print(f"   穿透率: {latest_event.breakthrough_ratio:.1f}%")
            print(f"   吸附效率: {latest_event.efficiency:.1f}%")
            print(f"   原因: {latest_event.reason}")
            print(f"   建议: {latest_event.recommendation}")

            if latest_event.predicted_saturation_time:
                print(f"   预测饱和时间: {latest_event.predicted_saturation_time:.1f}s")

        # 显示关键时间点
        if self.warning_model.fitted:
            print(f"\n📊 关键时间点预测:")
            if self.warning_model.breakthrough_start_time:
                print(f"   穿透起始时间: {self.warning_model.breakthrough_start_time:.1f}s")
            if self.warning_model.warning_time:
                print(f"   预警时间: {self.warning_model.warning_time:.1f}s")
            if self.warning_model.predicted_saturation_time:
                print(f"   预测饱和时间: {self.warning_model.predicted_saturation_time:.1f}s")

    def create_warning_visualization(self, efficiency_data: pd.DataFrame, base_datetime=None) -> plt.Figure:
        """创建包含预警信息的可视化图表 - 创建类似抽取式吸附曲线的穿透曲线图

        参数:
            efficiency_data: 效率数据
            base_datetime: 基准时间，用于计算实际时间段。如果为None，使用默认时间
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('活性炭吸附穿透曲线分析与预警系统', fontsize=16, fontweight='bold')

        # 1. 主要穿透曲线图（类似您提供的图片样式）
        ax1 = axes[0]

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 设置基准时间
        from datetime import datetime, timedelta
        if base_datetime is None:
            base_datetime = datetime(2024, 7, 30, 0, 0, 0)  # 默认基准时间

        # 计算累积时间段时长（新的时间计算方式）
        def calculate_cumulative_segment_time(data):
            """计算累积的时间段时长"""
            cumulative_hours = []
            total_duration = 0

            for i in range(len(data)):
                if i == 0:
                    # 第一个数据点，假设时间段长度为下一个点的时间间隔
                    if len(data) > 1:
                        segment_duration = data.iloc[1]['time'] - data.iloc[0]['time']
                    else:
                        segment_duration = 1800  # 默认30分钟
                else:
                    # 计算当前点与前一个点的时间间隔作为当前时间段长度
                    segment_duration = data.iloc[i]['time'] - data.iloc[i-1]['time']

                total_duration += segment_duration
                cumulative_hours.append(total_duration / 3600)  # 转换为小时

            return cumulative_hours

        # 使用新的时间计算方式
        time_hours = calculate_cumulative_segment_time(efficiency_data)
        breakthrough_percent = efficiency_data['breakthrough_ratio'] * 100  # 转换为百分比

        ax1.plot(time_hours, breakthrough_percent, 'b-', linewidth=1.5,
                alpha=0.8, label='实际穿透率(出口浓度/进口浓度)')

        # 添加每个时间段对应的穿透率数据点（透明、微小的点）
        scatter_points = ax1.scatter(time_hours, breakthrough_percent,
                                   color='blue', s=15, alpha=0.4, zorder=3,
                                   edgecolors='darkblue', linewidth=0.5,
                                   label='时间段穿透率数据点')

        # 为散点图添加交互式标注功能（点击展示）
        def on_click(event):
            """鼠标点击时显示数据点信息"""
            if event.inaxes == ax1 and event.button == 1:  # 左键点击
                # 检查鼠标是否在散点附近
                min_distance = float('inf')
                closest_index = -1

                for i, (time_h, bt_pct) in enumerate(zip(time_hours, breakthrough_percent)):
                    # 计算鼠标位置与数据点的距离
                    distance = ((event.xdata - time_h) ** 2 + (event.ydata - bt_pct) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_index = i

                # 如果点击位置距离最近的数据点足够近，显示信息
                if closest_index >= 0 and min_distance < 2.0:  # 增加点击容忍度
                    i = closest_index
                    time_h = time_hours[i]
                    bt_pct = breakthrough_percent[i]

                    # 清除之前的标注
                    for annotation in getattr(ax1, '_click_annotations', []):
                        annotation.remove()
                    ax1._click_annotations = []

                    # 创建新的标注 - 按照清洗后图像的格式
                    time_seconds = efficiency_data.iloc[i]['time']

                    # 计算实际时间段的开始时间
                    start_time = base_datetime + timedelta(seconds=time_seconds)

                    # 计算时间段长度（基于相邻数据点的时间间隔）
                    if i < len(efficiency_data) - 1:
                        next_time_seconds = efficiency_data.iloc[i + 1]['time']
                        segment_duration = next_time_seconds - time_seconds
                    elif i > 0:
                        prev_time_seconds = efficiency_data.iloc[i - 1]['time']
                        segment_duration = time_seconds - prev_time_seconds
                    else:
                        segment_duration = 1800  # 默认30分钟

                    # 计算时间段的结束时间
                    end_time = start_time + timedelta(seconds=segment_duration)

                    # 格式化时间段显示（模仿图像中的格式）
                    date_str = start_time.strftime("%m-%d")
                    start_time_str = start_time.strftime("%H:%M")
                    end_time_str = end_time.strftime("%H:%M")
                    time_segment = f"{date_str} {start_time_str}-{end_time_str}"

                    # 计算累积时间段长度（小时）
                    cumulative_hours = time_h

                    # 创建标注文本（模仿清洗后图像的格式，并显示累积时间）
                    annotation_text = f"时间段: {time_segment}\n累积时长: {cumulative_hours:.2f}小时\n穿透率: {bt_pct:.1f}%"

                    annotation = ax1.annotate(
                        annotation_text,
                        xy=(time_h, bt_pct),
                        xytext=(20, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                                alpha=0.9, edgecolor='blue', linewidth=1),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                      color='blue', alpha=0.8, linewidth=1.5),
                        fontsize=10, ha='left', va='bottom',
                        zorder=15, fontweight='normal'
                    )

                    if not hasattr(ax1, '_click_annotations'):
                        ax1._click_annotations = []
                    ax1._click_annotations.append(annotation)
                    fig.canvas.draw_idle()
                else:
                    # 如果点击位置不在数据点附近，清除标注
                    for annotation in getattr(ax1, '_click_annotations', []):
                        annotation.remove()
                    ax1._click_annotations = []
                    fig.canvas.draw_idle()

        # 连接鼠标点击事件
        fig.canvas.mpl_connect('button_press_event', on_click)

        # 在数据点上添加红色圆点和黄色标签（类似原图，但减少显示频率）
        for i, (time_h, bt_pct) in enumerate(zip(time_hours, breakthrough_percent)):
            if i % max(1, len(time_hours)//15) == 0:  # 每隔一定间隔显示一个点
                ax1.scatter(time_h, bt_pct, color='red', s=80, zorder=5,
                           edgecolors='darkred', linewidth=1)
                ax1.annotate(f'{bt_pct:.1f}%',
                           xy=(time_h, bt_pct),
                           xytext=(0, 15), textcoords='offset points',
                           ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                           fontsize=9, fontweight='bold')

        # 如果有Logistic模型拟合结果，绘制预测曲线
        if self.warning_model.fitted:
            # 注意：预测曲线仍然基于原始时间计算，因为模型是基于原始时间训练的
            # 但显示时需要转换为累积时间段格式
            max_time = efficiency_data['time'].max()
            if self.warning_model.predicted_saturation_time:
                extend_time = max(max_time * 1.3, self.warning_model.predicted_saturation_time * 1.1)
            else:
                extend_time = max_time * 1.5

            time_smooth = np.linspace(efficiency_data['time'].min(), extend_time, 500)
            bt_smooth = self.warning_model.predict_breakthrough(time_smooth)

            # 将预测时间转换为累积时间段格式
            # 这里需要根据原始数据的时间间隔来估算累积时间
            avg_interval = np.mean(np.diff(efficiency_data['time']))  # 平均时间间隔
            cumulative_smooth_hours = []
            for t in time_smooth:
                # 估算到时间t为止的累积时间段数
                num_segments = (t - efficiency_data['time'].min()) / avg_interval
                cumulative_time = num_segments * avg_interval / 3600  # 转换为小时
                cumulative_smooth_hours.append(cumulative_time)

            bt_smooth_percent = bt_smooth * 100

            ax1.plot(cumulative_smooth_hours, bt_smooth_percent, 'g--', linewidth=2,
                    alpha=0.8, label='Logistic预测曲线')

            # 添加关键阈值线
            ax1.axhline(y=self.warning_model.breakthrough_start_threshold * 100,
                       color='green', linestyle='--', alpha=0.7,
                       label=f'穿透起始点({self.warning_model.breakthrough_start_threshold:.1%})')

            # 显示预测饱和阈值线
            ax1.axhline(y=self.warning_model.saturation_threshold * 100,
                       color='red', linestyle='--', alpha=0.7,
                       label=f'预测饱和阈值({self.warning_model.saturation_threshold:.1%})')

            # 标记关键时间点（转换为累积时间段格式）
            def convert_to_cumulative_time(original_time):
                """将原始时间转换为累积时间段格式"""
                if original_time <= efficiency_data['time'].min():
                    return 0
                # 估算到该时间点的累积时间段数
                avg_interval = np.mean(np.diff(efficiency_data['time']))
                num_segments = (original_time - efficiency_data['time'].min()) / avg_interval
                return num_segments * avg_interval / 3600  # 转换为小时

            if self.warning_model.breakthrough_start_time:
                start_time_h = convert_to_cumulative_time(self.warning_model.breakthrough_start_time)
                ax1.axvline(x=start_time_h, color='green', linestyle=':', alpha=0.8,
                           label='穿透起始时间')

            if self.warning_model.warning_time:
                warning_time_h = convert_to_cumulative_time(self.warning_model.warning_time)
                ax1.axvline(x=warning_time_h, color='orange', linestyle=':', alpha=0.8,
                           label='预警时间点(80%)')

                # 标记预警点（大橙色星号）- 显示在预测曲线上
                warning_breakthrough = self.warning_model.predict_breakthrough(
                    np.array([self.warning_model.warning_time]))[0] * 100
                ax1.scatter([warning_time_h], [warning_breakthrough],
                           color='orange', s=300, marker='*', zorder=10,
                           edgecolors='darkorange', linewidth=2,
                           label=f'预警点(穿透率:{warning_breakthrough:.1f}%)')

            if self.warning_model.predicted_saturation_time:
                sat_time_h = convert_to_cumulative_time(self.warning_model.predicted_saturation_time)
                ax1.axvline(x=sat_time_h, color='red', linestyle=':', alpha=0.8,
                           label='实际饱和时间')

        ax1.set_xlabel('累积运行时间 (小时)', fontsize=12)
        ax1.set_ylabel('穿透率 (%)', fontsize=12)
        ax1.set_title('穿透率变化趋势与预警分析 (基于累积时间段)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)  # 设置y轴范围

        # 2. 预警分析详情图
        ax2 = axes[1]

        # 创建预警分析的详细信息展示
        if self.warning_model.fitted:
            # 左侧：关键时间点信息
            info_text = "预警分析结果:\n\n"

            if self.warning_model.breakthrough_start_time:
                start_hours = self.warning_model.breakthrough_start_time / 3600
                info_text += f"穿透起始时间: {start_hours:.2f} 小时\n"

            if self.warning_model.warning_time:
                warning_hours = self.warning_model.warning_time / 3600
                info_text += f"预警时间点: {warning_hours:.2f} 小时\n"
                info_text += f"(从起始到饱和的{self.warning_model.warning_ratio:.0%})\n"

            if self.warning_model.predicted_saturation_time:
                sat_hours = self.warning_model.predicted_saturation_time / 3600
                info_text += f"实际饱和时间: {sat_hours:.2f} 小时\n"

                # 显示实际饱和点的穿透率
                if hasattr(self.warning_model, 'params') and self.warning_model.params is not None:
                    A, k, t0 = self.warning_model.params
                    sat_breakthrough = self.warning_model.predict_breakthrough(
                        np.array([self.warning_model.predicted_saturation_time]))[0]
                    info_text += f"实际饱和穿透率: {sat_breakthrough:.1%}\n"

            # 模型参数信息
            if hasattr(self.warning_model, 'params') and self.warning_model.params is not None:
                A, k, t0 = self.warning_model.params
                info_text += f"\nLogistic模型参数:\n"
                info_text += f"最大穿透率 A: {A:.3f}\n"
                info_text += f"增长率 k: {k:.6f}\n"
                info_text += f"拐点时间 t0: {t0/3600:.2f} 小时\n"

            if self.warning_events:
                info_text += f"\n预警事件数: {len(self.warning_events)}\n"
                latest_event = max(self.warning_events, key=lambda x: x.timestamp)
                latest_hours = latest_event.timestamp / 3600
                info_text += f"最新预警: {latest_event.warning_level.value}\n"
                info_text += f"预警时间: {latest_hours:.2f} 小时\n"
                info_text += f"穿透率: {latest_event.breakthrough_ratio:.1f}%"
            else:
                info_text += "\n✅ 暂无预警事件\n设备运行正常"

            ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

            # 右侧：绘制预警逻辑示意图
            if (self.warning_model.breakthrough_start_time and
                self.warning_model.warning_time and
                self.warning_model.predicted_saturation_time):

                # 时间轴示意图
                times_hours = [
                    self.warning_model.breakthrough_start_time / 3600,
                    self.warning_model.warning_time / 3600,
                    self.warning_model.predicted_saturation_time / 3600
                ]
                labels = ['穿透起始', '预警点(80%)', '预测饱和']
                colors = ['green', 'orange', 'red']

                # 绘制时间轴
                y_pos = 0.4
                x_start = 0.55
                x_end = 0.95

                # 时间轴线
                ax2.plot([x_start, x_end], [y_pos, y_pos],
                        transform=ax2.transAxes, color='black', linewidth=2)

                # 标记点
                for i, (time_h, label, color) in enumerate(zip(times_hours, labels, colors)):
                    x_pos = x_start + (x_end - x_start) * i / 2

                    # 绘制点
                    ax2.scatter([x_pos], [y_pos], s=200, c=color, alpha=0.8,
                               transform=ax2.transAxes, zorder=5, edgecolors='black', linewidth=2)

                    # 标签
                    ax2.text(x_pos, y_pos + 0.08, label, transform=ax2.transAxes,
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

                    # 时间值
                    ax2.text(x_pos, y_pos - 0.08, f'{time_h:.2f}h', transform=ax2.transAxes,
                            ha='center', va='top', fontsize=9)

                # 添加箭头和说明
                ax2.annotate('', xy=(x_start + (x_end - x_start) * 0.8, y_pos),
                           xytext=(x_start + (x_end - x_start) * 0.4, y_pos),
                           transform=ax2.transAxes,
                           arrowprops=dict(arrowstyle='->', lw=2, color='orange'))

                ax2.text(x_start + (x_end - x_start) * 0.6, y_pos + 0.15,
                        '80%时间点预警', transform=ax2.transAxes,
                        ha='center', va='bottom', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            ax2.set_title('预警分析与时间点详情', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, '模型拟合失败\n无法进行预警分析', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=16,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            ax2.set_title('预警分析状态', fontsize=14, fontweight='bold')
            ax2.axis('off')



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

        # 6. 预警分析
        print("\n" + "="*40)
        print("开始预警分析")
        self.analyze_warning_system()

        # 7. 创建可视化
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

        # 预警系统可视化
        if self.warning_events or self.warning_model.fitted:
            print("\n" + "="*40)
            print("创建预警系统可视化")

            # 选择最佳的效率数据
            efficiency_data = None
            if self.efficiency_data_ks is not None and len(self.efficiency_data_ks) > 0:
                efficiency_data = self.efficiency_data_ks
            elif self.efficiency_data_boxplot is not None and len(self.efficiency_data_boxplot) > 0:
                efficiency_data = self.efficiency_data_boxplot

            if efficiency_data is not None:
                fig_warning = self.create_warning_visualization(efficiency_data)
                filename_warning = os.path.join(visualization_dir, f"{self.base_filename}_预警系统_{timestamp}.png")
                fig_warning.savefig(filename_warning, dpi=300, bbox_inches='tight')
                print(f"预警系统可视化图片已保存: {filename_warning}")
                plt.show()

        # 8. 保存清洗后的数据
        if len(self.cleaned_data_ks) > 0:
            ks_filename = os.path.join(cleaned_data_dir, f"{self.base_filename}_KS检验清洗_{timestamp}.csv")
            self.cleaned_data_ks.to_csv(ks_filename, index=False, encoding='utf-8-sig')
            print(f"K-S检验清洗数据已保存: {ks_filename}")

        if len(self.cleaned_data_boxplot) > 0:
            box_filename = os.path.join(cleaned_data_dir, f"{self.base_filename}_箱型图清洗_{timestamp}.csv")
            self.cleaned_data_boxplot.to_csv(box_filename, index=False, encoding='utf-8-sig')
            print(f"箱型图清洗数据已保存: {box_filename}")

        # 9. 保存预警报告
        if self.warning_events or self.warning_model.fitted:
            self._save_warning_report(cleaned_data_dir, timestamp)

        print("\n" + "="*60)
        print("数据处理、可视化与预警分析完成！")

        # 显示最终预警摘要
        if self.warning_events:
            print("\n🚨 最终预警摘要:")
            latest_event = max(self.warning_events, key=lambda x: x.timestamp)
            print(f"   当前预警状态: {latest_event.warning_level.value}")
            print(f"   总预警事件数: {len(self.warning_events)}")
            if self.warning_model.predicted_saturation_time:
                print(f"   预测饱和时间: {self.warning_model.predicted_saturation_time:.1f}s")
        else:
            print("\n✅ 设备运行正常，无预警事件")

    def _save_warning_report(self, output_dir: str, timestamp: str):
        """保存预警报告"""
        report_filename = os.path.join(output_dir, f"{self.base_filename}_预警报告_{timestamp}.txt")

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("活性炭更换预警报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {self.data_file}\n\n")

            # Logistic模型信息
            if self.warning_model.fitted:
                f.write("Logistic模型拟合结果:\n")
                f.write(f"  参数: A={self.warning_model.params[0]:.3f}, k={self.warning_model.params[1]:.6f}, t0={self.warning_model.params[2]:.1f}\n")

                if self.warning_model.breakthrough_start_time:
                    f.write(f"  穿透起始时间: {self.warning_model.breakthrough_start_time:.1f}s\n")
                if self.warning_model.warning_time:
                    f.write(f"  预警时间: {self.warning_model.warning_time:.1f}s\n")
                if self.warning_model.predicted_saturation_time:
                    f.write(f"  预测饱和时间: {self.warning_model.predicted_saturation_time:.1f}s\n")
                f.write("\n")
            else:
                f.write("Logistic模型拟合失败\n\n")

            # 预警事件
            if self.warning_events:
                f.write(f"预警事件总数: {len(self.warning_events)}\n\n")

                # 按预警等级分类统计
                warning_counts = {}
                for event in self.warning_events:
                    level = event.warning_level.value
                    warning_counts[level] = warning_counts.get(level, 0) + 1

                f.write("预警等级统计:\n")
                for level, count in warning_counts.items():
                    f.write(f"  {level}: {count} 次\n")
                f.write("\n")

                # 详细预警事件
                f.write("详细预警事件:\n")
                f.write("-" * 40 + "\n")

                for i, event in enumerate(self.warning_events, 1):
                    f.write(f"\n事件 {i}:\n")
                    f.write(f"  时间: {event.timestamp:.1f}s\n")
                    f.write(f"  预警等级: {event.warning_level.value}\n")
                    f.write(f"  穿透率: {event.breakthrough_ratio:.1f}%\n")
                    f.write(f"  吸附效率: {event.efficiency:.1f}%\n")
                    f.write(f"  原因: {event.reason}\n")
                    f.write(f"  建议: {event.recommendation}\n")

                # 最新预警状态
                latest_event = max(self.warning_events, key=lambda x: x.timestamp)
                f.write(f"\n当前预警状态: {latest_event.warning_level.value}\n")
                f.write(f"最新预警时间: {latest_event.timestamp:.1f}s\n")

            else:
                f.write("✅ 无预警事件，设备运行正常\n")

            f.write("\n" + "=" * 50 + "\n")

        print(f"预警报告已保存: {report_filename}")


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
