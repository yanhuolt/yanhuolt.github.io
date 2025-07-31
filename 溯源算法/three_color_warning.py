"""
三色预警系统模块
实现基于污染程度的黄色-橙色-红色动态预警机制
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import datetime
import json

from data_fusion import MonitoringData
from source_inversion import InversionResult


class WarningLevel(Enum):
    """预警等级"""
    GREEN = "绿色"      # 正常
    YELLOW = "黄色"     # 轻度污染
    ORANGE = "橙色"     # 中度污染  
    RED = "红色"        # 重度污染


@dataclass
class WarningThreshold:
    """预警阈值配置"""
    pollutant: str
    yellow_threshold: float    # 黄色预警阈值
    orange_threshold: float    # 橙色预警阈值
    red_threshold: float       # 红色预警阈值
    unit: str                  # 单位


@dataclass
class WarningEvent:
    """预警事件"""
    event_id: str
    timestamp: str
    station_id: str
    location: Tuple[float, float, float]
    pollutant: str
    concentration: float
    warning_level: WarningLevel
    threshold_exceeded: float
    duration: Optional[int] = None  # 持续时间(分钟)
    source_info: Optional[InversionResult] = None  # 溯源结果
    response_actions: List[str] = None  # 响应措施
    status: str = "active"  # active, resolved, escalated


@dataclass
class PollutionSource:
    """污染源信息"""
    source_id: str
    location: Tuple[float, float, float]
    source_type: str  # industrial, construction, traffic, etc.
    emission_rate: float
    warning_level: WarningLevel
    last_update: str
    responsible_unit: str
    contact_info: str


class ThreeColorWarningSystem:
    """三色预警系统"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化预警系统
        
        Args:
            config_file: 配置文件路径
        """
        self.warning_thresholds = self._load_default_thresholds()
        self.active_warnings: Dict[str, WarningEvent] = {}
        self.warning_history: List[WarningEvent] = []
        self.pollution_sources: Dict[str, PollutionSource] = {}
        
        if config_file:
            self._load_config(config_file)
    
    def _load_default_thresholds(self) -> Dict[str, WarningThreshold]:
        """加载默认预警阈值"""
        return {
            'pm25': WarningThreshold(
                pollutant='PM2.5',
                yellow_threshold=75.0,   # 轻度污染
                orange_threshold=115.0,  # 中度污染
                red_threshold=150.0,     # 重度污染
                unit='μg/m³'
            ),
            'pm10': WarningThreshold(
                pollutant='PM10',
                yellow_threshold=150.0,
                orange_threshold=250.0,
                red_threshold=350.0,
                unit='μg/m³'
            ),
            'o3': WarningThreshold(
                pollutant='O3',
                yellow_threshold=160.0,
                orange_threshold=200.0,
                red_threshold=300.0,
                unit='μg/m³'
            ),
            'no2': WarningThreshold(
                pollutant='NO2',
                yellow_threshold=80.0,
                orange_threshold=180.0,
                red_threshold=280.0,
                unit='μg/m³'
            ),
            'vocs': WarningThreshold(
                pollutant='VOCs',
                yellow_threshold=200.0,
                orange_threshold=400.0,
                red_threshold=600.0,
                unit='μg/m³'
            )
        }
    
    def _load_config(self, config_file: str) -> None:
        """从配置文件加载设置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 更新阈值配置
            if 'thresholds' in config:
                for pollutant, thresholds in config['thresholds'].items():
                    if pollutant in self.warning_thresholds:
                        self.warning_thresholds[pollutant].yellow_threshold = thresholds.get('yellow', 
                            self.warning_thresholds[pollutant].yellow_threshold)
                        self.warning_thresholds[pollutant].orange_threshold = thresholds.get('orange',
                            self.warning_thresholds[pollutant].orange_threshold)
                        self.warning_thresholds[pollutant].red_threshold = thresholds.get('red',
                            self.warning_thresholds[pollutant].red_threshold)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
    
    def evaluate_warning_level(self, pollutant: str, concentration: float) -> WarningLevel:
        """
        评估预警等级
        
        Args:
            pollutant: 污染物类型
            concentration: 浓度值
            
        Returns:
            预警等级
        """
        if pollutant not in self.warning_thresholds:
            return WarningLevel.GREEN
        
        threshold = self.warning_thresholds[pollutant]
        
        if concentration >= threshold.red_threshold:
            return WarningLevel.RED
        elif concentration >= threshold.orange_threshold:
            return WarningLevel.ORANGE
        elif concentration >= threshold.yellow_threshold:
            return WarningLevel.YELLOW
        else:
            return WarningLevel.GREEN
    
    def process_monitoring_data(self, data: List[MonitoringData]) -> List[WarningEvent]:
        """
        处理监测数据，生成预警事件
        
        Args:
            data: 监测数据列表
            
        Returns:
            新生成的预警事件列表
        """
        new_warnings = []
        
        for monitoring_data in data:
            # 检查各污染物
            pollutants = {
                'pm25': monitoring_data.pm25,
                'pm10': monitoring_data.pm10,
                'o3': monitoring_data.o3,
                'no2': monitoring_data.no2,
                'vocs': monitoring_data.vocs
            }
            
            for pollutant, concentration in pollutants.items():
                if concentration is None:
                    continue
                
                warning_level = self.evaluate_warning_level(pollutant, concentration)
                
                if warning_level != WarningLevel.GREEN:
                    # 生成预警事件
                    event_id = f"{monitoring_data.station_id}_{pollutant}_{monitoring_data.timestamp}"
                    
                    # 计算超标倍数
                    threshold = self.warning_thresholds[pollutant]
                    if warning_level == WarningLevel.YELLOW:
                        exceeded = concentration / threshold.yellow_threshold
                    elif warning_level == WarningLevel.ORANGE:
                        exceeded = concentration / threshold.orange_threshold
                    else:  # RED
                        exceeded = concentration / threshold.red_threshold
                    
                    warning_event = WarningEvent(
                        event_id=event_id,
                        timestamp=monitoring_data.timestamp,
                        station_id=monitoring_data.station_id,
                        location=monitoring_data.location,
                        pollutant=pollutant.upper(),
                        concentration=concentration,
                        warning_level=warning_level,
                        threshold_exceeded=exceeded,
                        response_actions=self._get_response_actions(warning_level, pollutant)
                    )
                    
                    new_warnings.append(warning_event)
                    self.active_warnings[event_id] = warning_event
                    self.warning_history.append(warning_event)
        
        return new_warnings
    
    def _get_response_actions(self, warning_level: WarningLevel, pollutant: str) -> List[str]:
        """获取响应措施"""
        actions = []
        
        if warning_level == WarningLevel.YELLOW:
            actions = [
                "加强监测频次",
                "通知相关责任单位",
                "启动应急预案一级响应",
                "增加巡查频次"
            ]
        elif warning_level == WarningLevel.ORANGE:
            actions = [
                "启动应急预案二级响应",
                "实施区域限产措施",
                "加强工地扬尘管控",
                "增加道路清扫频次",
                "通知执法部门现场检查"
            ]
        elif warning_level == WarningLevel.RED:
            actions = [
                "启动应急预案一级响应",
                "实施重点企业停产限产",
                "禁止建筑工地土石方作业",
                "实施机动车限行措施",
                "启动人工影响天气作业",
                "发布健康防护提示"
            ]
        
        # 根据污染物类型添加特定措施
        if pollutant == 'pm25' or pollutant == 'pm10':
            if warning_level in [WarningLevel.ORANGE, WarningLevel.RED]:
                actions.append("重点管控扬尘源")
                actions.append("加强工业企业排放监管")
        elif pollutant == 'vocs':
            actions.append("重点检查VOCs排放企业")
            actions.append("加强加油站、化工企业监管")
        elif pollutant == 'no2':
            actions.append("加强机动车尾气监管")
            actions.append("重点检查燃煤企业")
        
        return actions
    
    def update_pollution_sources(self, inversion_results: List[InversionResult]) -> None:
        """
        更新污染源信息
        
        Args:
            inversion_results: 溯源结果列表
        """
        for result in inversion_results:
            source_id = f"source_{result.source_x:.0f}_{result.source_y:.0f}"
            
            # 根据源强确定预警等级
            if result.emission_rate > 5.0:
                warning_level = WarningLevel.RED
            elif result.emission_rate > 2.0:
                warning_level = WarningLevel.ORANGE
            elif result.emission_rate > 0.5:
                warning_level = WarningLevel.YELLOW
            else:
                warning_level = WarningLevel.GREEN
            
            pollution_source = PollutionSource(
                source_id=source_id,
                location=(result.source_x, result.source_y, result.source_z),
                source_type="unknown",  # 需要进一步识别
                emission_rate=result.emission_rate,
                warning_level=warning_level,
                last_update=datetime.datetime.now().isoformat(),
                responsible_unit="待确定",
                contact_info="待确定"
            )
            
            self.pollution_sources[source_id] = pollution_source
    
    def generate_warning_report(self, time_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        生成预警报告
        
        Args:
            time_range: 时间范围 (start_time, end_time)
            
        Returns:
            预警报告
        """
        # 筛选时间范围内的预警
        if time_range:
            start_time, end_time = time_range
            filtered_warnings = [
                w for w in self.warning_history
                if start_time <= w.timestamp <= end_time
            ]
        else:
            filtered_warnings = self.warning_history
        
        # 统计分析
        total_warnings = len(filtered_warnings)
        level_counts = {level.value: 0 for level in WarningLevel}
        pollutant_counts = {}
        station_counts = {}
        
        for warning in filtered_warnings:
            level_counts[warning.warning_level.value] += 1
            pollutant_counts[warning.pollutant] = pollutant_counts.get(warning.pollutant, 0) + 1
            station_counts[warning.station_id] = station_counts.get(warning.station_id, 0) + 1
        
        # 活跃预警
        active_warnings_count = len(self.active_warnings)
        
        # 污染源统计
        source_level_counts = {level.value: 0 for level in WarningLevel}
        for source in self.pollution_sources.values():
            source_level_counts[source.warning_level.value] += 1
        
        report = {
            "report_time": datetime.datetime.now().isoformat(),
            "time_range": time_range,
            "summary": {
                "total_warnings": total_warnings,
                "active_warnings": active_warnings_count,
                "total_pollution_sources": len(self.pollution_sources)
            },
            "warning_statistics": {
                "by_level": level_counts,
                "by_pollutant": pollutant_counts,
                "by_station": station_counts
            },
            "pollution_source_statistics": {
                "by_level": source_level_counts
            },
            "active_warnings_detail": [
                {
                    "event_id": w.event_id,
                    "timestamp": w.timestamp,
                    "station_id": w.station_id,
                    "pollutant": w.pollutant,
                    "concentration": w.concentration,
                    "warning_level": w.warning_level.value,
                    "threshold_exceeded": w.threshold_exceeded,
                    "response_actions": w.response_actions
                }
                for w in self.active_warnings.values()
            ],
            "high_risk_sources": [
                {
                    "source_id": s.source_id,
                    "location": s.location,
                    "emission_rate": s.emission_rate,
                    "warning_level": s.warning_level.value,
                    "responsible_unit": s.responsible_unit
                }
                for s in self.pollution_sources.values()
                if s.warning_level in [WarningLevel.ORANGE, WarningLevel.RED]
            ]
        }
        
        return report
    
    def resolve_warning(self, event_id: str, resolution_note: str = "") -> bool:
        """
        解除预警
        
        Args:
            event_id: 预警事件ID
            resolution_note: 解除说明
            
        Returns:
            是否成功解除
        """
        if event_id in self.active_warnings:
            warning = self.active_warnings[event_id]
            warning.status = "resolved"
            if resolution_note:
                warning.response_actions.append(f"解除说明: {resolution_note}")
            
            del self.active_warnings[event_id]
            return True
        
        return False
    
    def escalate_warning(self, event_id: str, escalation_note: str = "") -> bool:
        """
        升级预警
        
        Args:
            event_id: 预警事件ID
            escalation_note: 升级说明
            
        Returns:
            是否成功升级
        """
        if event_id in self.active_warnings:
            warning = self.active_warnings[event_id]
            
            # 升级预警等级
            if warning.warning_level == WarningLevel.YELLOW:
                warning.warning_level = WarningLevel.ORANGE
            elif warning.warning_level == WarningLevel.ORANGE:
                warning.warning_level = WarningLevel.RED
            
            warning.status = "escalated"
            if escalation_note:
                warning.response_actions.append(f"升级说明: {escalation_note}")
            
            # 更新响应措施
            warning.response_actions.extend(
                self._get_response_actions(warning.warning_level, warning.pollutant.lower())
            )
            
            return True
        
        return False
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """获取实时预警状态"""
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "active_warnings_count": len(self.active_warnings),
            "warning_levels": {
                level.value: sum(1 for w in self.active_warnings.values() 
                               if w.warning_level == level)
                for level in WarningLevel
            },
            "pollution_sources_count": len(self.pollution_sources),
            "high_risk_sources_count": sum(
                1 for s in self.pollution_sources.values()
                if s.warning_level in [WarningLevel.ORANGE, WarningLevel.RED]
            )
        }
