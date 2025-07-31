"""
活性炭更换预警系统
基于穿透曲线分析的四色预警算法
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class WarningLevel(Enum):
    """预警等级"""
    GREEN = "绿色"      # 无需更换
    YELLOW = "黄色"     # 适时更换  
    ORANGE = "橙色"     # 立即更换
    RED = "红色"        # 立即更换

@dataclass
class WarningEvent:
    """预警事件"""
    timestamp: str
    warning_level: WarningLevel
    breakthrough_ratio: float  # 穿透率 %
    efficiency: float         # 吸附效率 %
    reason: str              # 预警原因
    recommendation: str      # 建议措施
    predicted_saturation_time: Optional[str] = None  # 预计饱和时间

class LogisticModel:
    """Logistic模型用于穿透曲线拟合和预测"""
    
    def __init__(self):
        self.params = None
        self.fitted = False
        
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
    
    def fit(self, time_data: np.array, breakthrough_data: np.array) -> bool:
        """
        拟合Logistic模型
        
        参数:
            time_data: 时间数据
            breakthrough_data: 穿透率数据 (0-1之间)
            
        返回:
            是否拟合成功
        """
        try:
            # 过滤有效数据
            valid_mask = (breakthrough_data > 0) & (breakthrough_data <= 1)
            if np.sum(valid_mask) < 5:  # 至少需要5个数据点
                return False
                
            t_valid = time_data[valid_mask]
            bt_valid = breakthrough_data[valid_mask]
            
            # 初始参数估计
            A_init = 1.0  # 最大穿透率
            k_init = 0.001  # 增长率
            t0_init = np.median(t_valid)  # 拐点时间
            
            # 拟合
            self.params, _ = curve_fit(
                self.logistic_function,
                t_valid, bt_valid,
                p0=[A_init, k_init, t0_init],
                bounds=([0.5, 0.0001, 0], [1.2, 0.1, np.max(t_valid)*2]),
                maxfev=2000
            )
            
            self.fitted = True
            return True
            
        except Exception as e:
            print(f"Logistic模型拟合失败: {e}")
            return False
    
    def predict(self, time_points: np.array) -> np.array:
        """预测指定时间点的穿透率"""
        if not self.fitted:
            return np.zeros_like(time_points)
        
        return self.logistic_function(time_points, *self.params)
    
    def find_saturation_time(self, saturation_threshold: float = 0.95) -> Optional[float]:
        """
        预测达到饱和阈值的时间
        
        参数:
            saturation_threshold: 饱和阈值（默认95%）
            
        返回:
            预计饱和时间，如果无法预测则返回None
        """
        if not self.fitted:
            return None
            
        A, k, t0 = self.params
        
        # 求解 saturation_threshold = A / (1 + exp(-k*(t-t0)))
        # 即 t = t0 - ln(A/saturation_threshold - 1) / k
        try:
            if A <= saturation_threshold:
                return None
            
            t_sat = t0 - np.log(A / saturation_threshold - 1) / k
            return t_sat if t_sat > 0 else None
            
        except:
            return None

class ActivatedCarbonWarningSystem:
    """活性炭更换预警系统"""
    
    def __init__(self, 
                 breakthrough_start_threshold: float = 0.05,  # 穿透起始点阈值 5%
                 warning_threshold: float = 0.80,            # 预警点阈值 80%
                 saturation_threshold: float = 0.95,         # 饱和点阈值 95%
                 min_continuous_points: int = 3):            # 最少连续点数
        """
        初始化预警系统
        
        参数:
            breakthrough_start_threshold: 穿透起始点阈值
            warning_threshold: 预警点阈值  
            saturation_threshold: 饱和点阈值
            min_continuous_points: 触发预警的最少连续点数
        """
        self.breakthrough_start_threshold = breakthrough_start_threshold
        self.warning_threshold = warning_threshold
        self.saturation_threshold = saturation_threshold
        self.min_continuous_points = min_continuous_points
        
        self.logistic_model = LogisticModel()
        self.warning_history: List[WarningEvent] = []
        
    def calculate_breakthrough_ratio(self, c0: float, c1: float) -> float:
        """
        计算穿透率
        
        参数:
            c0: 进口浓度
            c1: 出口浓度
            
        返回:
            穿透率 (0-1之间)
        """
        if c0 <= 0:
            return 0.0
        return min(c1 / c0, 1.0)
    
    def determine_warning_level(self, breakthrough_ratio: float) -> WarningLevel:
        """
        根据穿透率确定预警等级
        
        参数:
            breakthrough_ratio: 穿透率 (0-1之间)
            
        返回:
            预警等级
        """
        if breakthrough_ratio <= self.breakthrough_start_threshold:
            return WarningLevel.GREEN
        elif breakthrough_ratio <= self.warning_threshold:
            return WarningLevel.YELLOW
        elif breakthrough_ratio <= self.saturation_threshold:
            return WarningLevel.ORANGE
        else:
            return WarningLevel.RED
    
    def analyze_data(self, data: pd.DataFrame) -> Dict:
        """
        分析数据并生成预警
        
        参数:
            data: 包含时间、进口浓度(c0)、出口浓度(c1)的数据框
            
        返回:
            分析结果字典
        """
        if len(data) == 0:
            return {"warning_events": [], "current_status": WarningLevel.GREEN}
        
        # 计算穿透率
        data = data.copy()
        data['breakthrough_ratio'] = data.apply(
            lambda row: self.calculate_breakthrough_ratio(row['c0'], row['c1']), 
            axis=1
        )
        
        # 拟合Logistic模型
        time_data = data['time'].values
        breakthrough_data = data['breakthrough_ratio'].values
        
        model_fitted = self.logistic_model.fit(time_data, breakthrough_data)
        
        # 生成预警事件
        warning_events = []
        current_level = WarningLevel.GREEN
        
        for idx, row in data.iterrows():
            level = self.determine_warning_level(row['breakthrough_ratio'])
            
            # 检查是否需要生成预警事件
            if level != WarningLevel.GREEN:
                # 检查连续性
                if self._check_continuous_warning(data, idx, level):
                    reason, recommendation = self._generate_warning_content(
                        level, row['breakthrough_ratio'], row.get('efficiency', 0)
                    )
                    
                    # 预测饱和时间
                    predicted_time = None
                    if model_fitted:
                        sat_time = self.logistic_model.find_saturation_time()
                        if sat_time:
                            predicted_time = self._format_predicted_time(sat_time, row['time'])
                    
                    event = WarningEvent(
                        timestamp=str(row['time']),
                        warning_level=level,
                        breakthrough_ratio=row['breakthrough_ratio'] * 100,
                        efficiency=row.get('efficiency', 0),
                        reason=reason,
                        recommendation=recommendation,
                        predicted_saturation_time=predicted_time
                    )
                    
                    warning_events.append(event)
                    current_level = level
        
        # 返回分析结果
        result = {
            "warning_events": warning_events,
            "current_status": current_level,
            "model_fitted": model_fitted,
            "data_with_breakthrough": data,
            "logistic_params": self.logistic_model.params if model_fitted else None
        }
        
        return result
    
    def _check_continuous_warning(self, data: pd.DataFrame, current_idx: int, 
                                 level: WarningLevel) -> bool:
        """检查是否满足连续预警条件"""
        if current_idx < self.min_continuous_points - 1:
            return False
            
        # 检查前面几个点是否也达到相同或更高级别的预警
        start_idx = max(0, current_idx - self.min_continuous_points + 1)
        
        for i in range(start_idx, current_idx + 1):
            if i >= len(data):
                continue
            row_level = self.determine_warning_level(data.iloc[i]['breakthrough_ratio'])
            if row_level.value < level.value:  # 假设预警等级有数值顺序
                return False
                
        return True
    
    def _generate_warning_content(self, level: WarningLevel, 
                                breakthrough_ratio: float, efficiency: float) -> Tuple[str, str]:
        """生成预警原因和建议"""
        ratio_percent = breakthrough_ratio * 100
        
        if level == WarningLevel.YELLOW:
            reason = f"穿透率达到{ratio_percent:.1f}%，已超过起始点阈值"
            recommendation = "建议开始准备更换活性炭，监控穿透率变化趋势"
        elif level == WarningLevel.ORANGE:
            reason = f"穿透率达到{ratio_percent:.1f}%，接近饱和状态"
            recommendation = "立即安排更换活性炭，设备处于非稳定运行状态"
        else:  # RED
            reason = f"穿透率达到{ratio_percent:.1f}%，活性炭已饱和"
            recommendation = "紧急更换活性炭！设备已无法正常净化VOCs"
            
        return reason, recommendation
    
    def _format_predicted_time(self, predicted_time: float, current_time: float) -> str:
        """格式化预测时间"""
        time_diff = predicted_time - current_time
        if time_diff > 0:
            hours = time_diff / 3600  # 假设时间单位是秒
            return f"预计{hours:.1f}小时后达到饱和"
        else:
            return "已达到饱和状态"
