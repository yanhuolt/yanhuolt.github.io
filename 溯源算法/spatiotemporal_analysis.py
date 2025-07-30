"""
时空关联分析模块
实现GIS时空分析技术，生成浓度分布热力图，智能匹配污染事件与源头行动
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import datetime

from data_fusion import MonitoringData
from three_color_warning import WarningEvent, PollutionSource


@dataclass
class HeatmapData:
    """热力图数据结构"""
    timestamp: str
    pollutant: str
    x_grid: np.ndarray
    y_grid: np.ndarray
    concentration_grid: np.ndarray
    max_concentration: float
    min_concentration: float
    hotspots: List[Tuple[float, float, float]]  # (x, y, concentration)


@dataclass
class SpatialEvent:
    """空间事件结构"""
    event_id: str
    event_type: str  # construction, industrial, traffic
    location: Tuple[float, float, float]
    start_time: str
    end_time: Optional[str]
    intensity: float
    description: str
    responsible_unit: str


@dataclass
class CorrelationResult:
    """关联分析结果"""
    pollution_event: WarningEvent
    related_spatial_events: List[SpatialEvent]
    correlation_score: float
    distance_to_source: float
    time_correlation: float
    confidence_level: str  # high, medium, low


class SpatiotemporalAnalysis:
    """时空关联分析类"""
    
    def __init__(self, 
                 region_bounds: Tuple[float, float, float, float] = (-1000, 1000, -1000, 1000),
                 grid_resolution: int = 100):
        """
        初始化时空分析器
        
        Args:
            region_bounds: 区域边界 (x_min, x_max, y_min, y_max)
            grid_resolution: 网格分辨率
        """
        self.region_bounds = region_bounds
        self.grid_resolution = grid_resolution
        self.spatial_events: Dict[str, SpatialEvent] = {}
        
        # 创建网格
        x_min, x_max, y_min, y_max = region_bounds
        self.x_grid = np.linspace(x_min, x_max, grid_resolution)
        self.y_grid = np.linspace(y_min, y_max, grid_resolution)
        self.X_grid, self.Y_grid = np.meshgrid(self.x_grid, self.y_grid)
    
    def add_spatial_events(self, events: List[SpatialEvent]) -> None:
        """添加空间事件数据"""
        for event in events:
            self.spatial_events[event.event_id] = event
    
    def generate_concentration_heatmap(self, 
                                     monitoring_data: List[MonitoringData],
                                     pollutant: str,
                                     interpolation_method: str = 'cubic') -> HeatmapData:
        """
        生成污染物浓度分布热力图
        
        Args:
            monitoring_data: 监测数据
            pollutant: 污染物类型
            interpolation_method: 插值方法 ('linear', 'cubic', 'nearest')
            
        Returns:
            热力图数据
        """
        # 提取有效数据点
        points = []
        values = []
        
        for data in monitoring_data:
            concentration = getattr(data, pollutant, None)
            if concentration is not None and concentration >= 0:
                points.append([data.location[0], data.location[1]])
                values.append(concentration)
        
        if len(points) < 3:
            # 数据点太少，返回空热力图
            return HeatmapData(
                timestamp=monitoring_data[0].timestamp if monitoring_data else "",
                pollutant=pollutant,
                x_grid=self.X_grid,
                y_grid=self.Y_grid,
                concentration_grid=np.zeros_like(self.X_grid),
                max_concentration=0,
                min_concentration=0,
                hotspots=[]
            )
        
        points = np.array(points)
        values = np.array(values)
        
        # 网格插值
        grid_points = np.column_stack([self.X_grid.ravel(), self.Y_grid.ravel()])
        
        try:
            interpolated_values = griddata(
                points, values, grid_points, 
                method=interpolation_method, 
                fill_value=0
            )
            concentration_grid = interpolated_values.reshape(self.X_grid.shape)
        except:
            # 插值失败，使用最近邻方法
            interpolated_values = griddata(
                points, values, grid_points, 
                method='nearest', 
                fill_value=0
            )
            concentration_grid = interpolated_values.reshape(self.X_grid.shape)
        
        # 识别热点区域（浓度前10%的区域）
        threshold = np.percentile(concentration_grid[concentration_grid > 0], 90)
        hotspot_indices = np.where(concentration_grid >= threshold)
        
        hotspots = []
        for i, j in zip(hotspot_indices[0], hotspot_indices[1]):
            x = self.X_grid[i, j]
            y = self.Y_grid[i, j]
            conc = concentration_grid[i, j]
            hotspots.append((x, y, conc))
        
        # 按浓度排序，取前10个热点
        hotspots.sort(key=lambda x: x[2], reverse=True)
        hotspots = hotspots[:10]
        
        return HeatmapData(
            timestamp=monitoring_data[0].timestamp if monitoring_data else "",
            pollutant=pollutant,
            x_grid=self.X_grid,
            y_grid=self.Y_grid,
            concentration_grid=concentration_grid,
            max_concentration=float(np.max(concentration_grid)),
            min_concentration=float(np.min(concentration_grid)),
            hotspots=hotspots
        )
    
    def find_spatial_correlations(self, 
                                warning_event: WarningEvent,
                                time_window: int = 60,  # 时间窗口(分钟)
                                distance_threshold: float = 1000) -> CorrelationResult:
        """
        寻找污染事件与空间事件的关联
        
        Args:
            warning_event: 预警事件
            time_window: 时间关联窗口(分钟)
            distance_threshold: 距离阈值(米)
            
        Returns:
            关联分析结果
        """
        related_events = []
        
        # 解析预警事件时间
        try:
            warning_time = datetime.datetime.fromisoformat(warning_event.timestamp)
        except:
            warning_time = datetime.datetime.now()
        
        warning_location = np.array(warning_event.location[:2])  # 只考虑x,y坐标
        
        for event in self.spatial_events.values():
            # 时间关联检查
            try:
                event_start = datetime.datetime.fromisoformat(event.start_time)
                time_diff = abs((warning_time - event_start).total_seconds() / 60)  # 分钟
                
                if time_diff > time_window:
                    continue
            except:
                continue
            
            # 空间关联检查
            event_location = np.array(event.location[:2])
            distance = np.linalg.norm(warning_location - event_location)
            
            if distance <= distance_threshold:
                related_events.append(event)
        
        # 计算关联得分
        correlation_score = self._calculate_correlation_score(warning_event, related_events)
        
        # 计算到最近污染源的距离
        min_distance = float('inf')
        if related_events:
            distances = [
                np.linalg.norm(warning_location - np.array(event.location[:2]))
                for event in related_events
            ]
            min_distance = min(distances)
        
        # 时间关联度
        time_correlation = 1.0
        if related_events:
            time_diffs = []
            for event in related_events:
                try:
                    event_start = datetime.datetime.fromisoformat(event.start_time)
                    time_diff = abs((warning_time - event_start).total_seconds() / 60)
                    time_diffs.append(time_diff)
                except:
                    time_diffs.append(time_window)
            
            avg_time_diff = np.mean(time_diffs)
            time_correlation = max(0, 1 - avg_time_diff / time_window)
        
        # 置信度评估
        confidence_level = self._assess_confidence(correlation_score, min_distance, time_correlation)
        
        return CorrelationResult(
            pollution_event=warning_event,
            related_spatial_events=related_events,
            correlation_score=correlation_score,
            distance_to_source=min_distance,
            time_correlation=time_correlation,
            confidence_level=confidence_level
        )
    
    def _calculate_correlation_score(self, 
                                   warning_event: WarningEvent, 
                                   spatial_events: List[SpatialEvent]) -> float:
        """计算关联得分"""
        if not spatial_events:
            return 0.0
        
        score = 0.0
        
        # 基于事件类型的权重
        type_weights = {
            'construction': 0.8,  # 建筑工地
            'industrial': 0.9,    # 工业排放
            'traffic': 0.6,       # 交通污染
            'dust': 0.7,          # 扬尘
            'burning': 0.9        # 燃烧
        }
        
        # 基于污染物类型的关联性
        pollutant_correlations = {
            'PM25': {'construction': 0.9, 'industrial': 0.8, 'traffic': 0.7, 'dust': 0.9, 'burning': 0.9},
            'PM10': {'construction': 0.9, 'industrial': 0.7, 'traffic': 0.6, 'dust': 0.9, 'burning': 0.8},
            'NO2': {'industrial': 0.9, 'traffic': 0.9, 'construction': 0.3, 'burning': 0.7},
            'VOCS': {'industrial': 0.9, 'traffic': 0.6, 'construction': 0.2, 'burning': 0.5},
            'O3': {'industrial': 0.6, 'traffic': 0.7, 'construction': 0.2, 'burning': 0.4}
        }
        
        for event in spatial_events:
            # 基础权重
            base_weight = type_weights.get(event.event_type, 0.5)
            
            # 污染物关联权重
            pollutant_weight = pollutant_correlations.get(
                warning_event.pollutant, {}
            ).get(event.event_type, 0.5)
            
            # 强度权重
            intensity_weight = min(1.0, event.intensity / 10.0)
            
            # 综合得分
            event_score = base_weight * pollutant_weight * intensity_weight
            score += event_score
        
        # 归一化到0-1范围
        return min(1.0, score / len(spatial_events))
    
    def _assess_confidence(self, 
                          correlation_score: float, 
                          distance: float, 
                          time_correlation: float) -> str:
        """评估置信度"""
        # 综合评分
        distance_score = max(0, 1 - distance / 1000)  # 距离越近得分越高
        overall_score = (correlation_score + distance_score + time_correlation) / 3
        
        if overall_score >= 0.8:
            return "high"
        elif overall_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def batch_correlation_analysis(self, 
                                 warning_events: List[WarningEvent]) -> List[CorrelationResult]:
        """批量关联分析"""
        results = []
        
        for warning in warning_events:
            result = self.find_spatial_correlations(warning)
            results.append(result)
        
        return results
    
    def generate_hotspot_report(self, 
                              heatmap_data: HeatmapData,
                              correlation_results: List[CorrelationResult]) -> Dict[str, Any]:
        """
        生成热点区域报告
        
        Args:
            heatmap_data: 热力图数据
            correlation_results: 关联分析结果
            
        Returns:
            热点报告
        """
        # 热点统计
        hotspots_info = []
        for i, (x, y, conc) in enumerate(heatmap_data.hotspots):
            # 查找该热点附近的关联事件
            nearby_events = []
            for result in correlation_results:
                event_location = np.array(result.pollution_event.location[:2])
                hotspot_location = np.array([x, y])
                distance = np.linalg.norm(event_location - hotspot_location)
                
                if distance <= 200:  # 200米范围内
                    nearby_events.extend(result.related_spatial_events)
            
            hotspots_info.append({
                "hotspot_id": f"hotspot_{i+1}",
                "location": (x, y),
                "concentration": conc,
                "nearby_events_count": len(nearby_events),
                "event_types": list(set(event.event_type for event in nearby_events))
            })
        
        # 关联统计
        high_confidence_count = sum(1 for r in correlation_results if r.confidence_level == "high")
        medium_confidence_count = sum(1 for r in correlation_results if r.confidence_level == "medium")
        low_confidence_count = sum(1 for r in correlation_results if r.confidence_level == "low")
        
        # 事件类型统计
        event_type_counts = {}
        for result in correlation_results:
            for event in result.related_spatial_events:
                event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
        
        report = {
            "report_time": datetime.datetime.now().isoformat(),
            "pollutant": heatmap_data.pollutant,
            "analysis_time": heatmap_data.timestamp,
            "concentration_summary": {
                "max_concentration": heatmap_data.max_concentration,
                "min_concentration": heatmap_data.min_concentration,
                "hotspots_count": len(heatmap_data.hotspots)
            },
            "hotspots_detail": hotspots_info,
            "correlation_summary": {
                "total_correlations": len(correlation_results),
                "high_confidence": high_confidence_count,
                "medium_confidence": medium_confidence_count,
                "low_confidence": low_confidence_count
            },
            "event_type_distribution": event_type_counts,
            "recommendations": self._generate_recommendations(heatmap_data, correlation_results)
        }
        
        return report
    
    def _generate_recommendations(self, 
                                heatmap_data: HeatmapData,
                                correlation_results: List[CorrelationResult]) -> List[str]:
        """生成管控建议"""
        recommendations = []
        
        # 基于热点数量的建议
        if len(heatmap_data.hotspots) > 5:
            recommendations.append("检测到多个污染热点，建议启动区域联防联控措施")
        
        # 基于最高浓度的建议
        if heatmap_data.max_concentration > 150:  # 假设重度污染阈值
            recommendations.append("检测到重度污染区域，建议立即启动应急响应")
        
        # 基于关联分析的建议
        high_confidence_results = [r for r in correlation_results if r.confidence_level == "high"]
        
        if high_confidence_results:
            # 统计主要污染源类型
            source_types = {}
            for result in high_confidence_results:
                for event in result.related_spatial_events:
                    source_types[event.event_type] = source_types.get(event.event_type, 0) + 1
            
            # 针对主要污染源类型给出建议
            main_source = max(source_types.items(), key=lambda x: x[1])[0] if source_types else None
            
            if main_source == 'construction':
                recommendations.append("主要污染源为建筑工地，建议加强扬尘管控和洒水降尘")
            elif main_source == 'industrial':
                recommendations.append("主要污染源为工业排放，建议检查企业排放设施运行状况")
            elif main_source == 'traffic':
                recommendations.append("主要污染源为交通排放，建议优化交通组织和限制高排放车辆")
            elif main_source == 'burning':
                recommendations.append("检测到燃烧污染源，建议加强禁烧巡查和执法")
        
        # 基于时空分布的建议
        if len(correlation_results) > 10:
            recommendations.append("污染事件频发，建议建立常态化监管机制")
        
        return recommendations
    
    def visualize_heatmap(self, 
                         heatmap_data: HeatmapData,
                         save_path: Optional[str] = None) -> None:
        """
        可视化热力图
        
        Args:
            heatmap_data: 热力图数据
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 10))
        
        # 绘制热力图
        im = plt.contourf(
            heatmap_data.x_grid, 
            heatmap_data.y_grid, 
            heatmap_data.concentration_grid,
            levels=20,
            cmap='YlOrRd'
        )
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label(f'{heatmap_data.pollutant} 浓度 (μg/m³)', fontsize=12)
        
        # 标记热点
        for i, (x, y, conc) in enumerate(heatmap_data.hotspots):
            plt.plot(x, y, 'ro', markersize=8)
            plt.annotate(f'热点{i+1}\n{conc:.1f}', 
                        (x, y), xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.xlabel('X坐标 (m)', fontsize=12)
        plt.ylabel('Y坐标 (m)', fontsize=12)
        plt.title(f'{heatmap_data.pollutant} 浓度分布热力图\n时间: {heatmap_data.timestamp}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
