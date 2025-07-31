"""
污染溯源系统主控制器
集成所有模块，提供统一的API接口，实现完整的AI污染溯源流程
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import datetime
import json
import time

# 导入各个模块
from gaussian_plume_model import GaussianPlumeModel, PollutionSource, MeteoData
from genetic_pattern_search import GeneticPatternSearchAlgorithm, GAParameters
from source_inversion import SourceInversion, SensorData, InversionResult
from data_fusion import DataFusion, MonitoringData, DataQualityReport
from three_color_warning import ThreeColorWarningSystem, WarningEvent, WarningLevel
from spatiotemporal_analysis import SpatiotemporalAnalysis, HeatmapData, CorrelationResult, SpatialEvent


@dataclass
class SystemConfig:
    """系统配置"""
    region_bounds: Tuple[float, float, float, float] = (-1000, 1000, -1000, 1000)
    grid_resolution: int = 100
    search_bounds: Dict[str, Tuple[float, float]] = None
    ga_parameters: GAParameters = None
    warning_config_file: Optional[str] = None
    enable_data_imputation: bool = True
    enable_data_normalization: bool = True
    response_time_threshold: float = 3.0  # 响应时间阈值(秒)


@dataclass
class TracingResult:
    """溯源结果"""
    timestamp: str
    computation_time: float
    data_quality_report: DataQualityReport
    inversion_results: List[InversionResult]
    warning_events: List[WarningEvent]
    heatmap_data: Dict[str, HeatmapData]
    correlation_results: List[CorrelationResult]
    system_performance: Dict[str, float]
    recommendations: List[str]


class PollutionTracingSystem:
    """污染溯源系统主类"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        初始化污染溯源系统
        
        Args:
            config: 系统配置
        """
        self.config = config or SystemConfig()
        
        # 初始化各个模块
        self._initialize_modules()
        
        # 系统状态
        self.is_running = False
        self.last_update_time = None
        self.processing_history = []
    
    def _initialize_modules(self) -> None:
        """初始化各个模块"""
        # 高斯烟羽模型
        self.gaussian_model = GaussianPlumeModel()
        
        # 数据融合模块
        self.data_fusion = DataFusion()
        
        # 污染源反算模块
        search_bounds = self.config.search_bounds or {
            'x': (self.config.region_bounds[0], self.config.region_bounds[1]),
            'y': (self.config.region_bounds[2], self.config.region_bounds[3]),
            'z': (0, 50),
            'q': (0.01, 10.0)
        }
        
        ga_params = self.config.ga_parameters or GAParameters(
            population_size=50,
            max_generations=1000,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_rate=0.2,
            temperature=1.0,
            convergence_threshold=1e-6
        )
        
        self.source_inversion = SourceInversion(search_bounds, ga_params)
        
        # 三色预警系统
        self.warning_system = ThreeColorWarningSystem(self.config.warning_config_file)
        
        # 时空关联分析
        self.spatiotemporal_analysis = SpatiotemporalAnalysis(
            self.config.region_bounds,
            self.config.grid_resolution
        )
    
    def process_real_time_data(self, 
                              monitoring_data_sources: Dict[str, Any],
                              spatial_events: Optional[List[SpatialEvent]] = None,
                              verbose: bool = False) -> TracingResult:
        """
        处理实时监测数据，执行完整的溯源分析流程
        
        Args:
            monitoring_data_sources: 多源监测数据
            spatial_events: 空间事件数据
            verbose: 是否输出详细信息
            
        Returns:
            溯源分析结果
        """
        start_time = time.time()
        
        if verbose:
            print("=== 开始污染溯源分析 ===")
            print(f"分析时间: {datetime.datetime.now().isoformat()}")
        
        try:
            # 1. 数据融合与预处理
            if verbose:
                print("\n1. 数据融合与预处理...")
            
            processed_data, quality_report = self.data_fusion.process_data_pipeline(
                monitoring_data_sources,
                self.config.enable_data_imputation,
                self.config.enable_data_normalization
            )
            
            if verbose:
                print(f"   处理数据量: {quality_report.total_records}")
                print(f"   站点数量: {len(quality_report.station_coverage)}")
            
            # 2. 三色预警分析
            if verbose:
                print("\n2. 三色预警分析...")
            
            warning_events = self.warning_system.process_monitoring_data(processed_data)
            
            if verbose:
                print(f"   生成预警事件: {len(warning_events)}")
                for event in warning_events[:3]:  # 显示前3个
                    print(f"   - {event.pollutant} {event.warning_level.value} "
                          f"浓度:{event.concentration:.1f} 站点:{event.station_id}")
            
            # 3. 污染源反算（针对预警事件）
            if verbose:
                print("\n3. 污染源反算...")
            
            inversion_results = []
            
            # 按时间和污染物分组处理预警事件
            warning_groups = self._group_warnings_for_inversion(warning_events, processed_data)
            
            for group_key, (group_warnings, group_data, meteo_data) in warning_groups.items():
                if verbose:
                    print(f"   处理组: {group_key}")
                
                # 转换为传感器数据格式
                sensor_data = []
                for data in group_data:
                    pollutant = group_key.split('_')[1]  # 从group_key提取污染物类型
                    # 确保污染物名称是小写的
                    pollutant_attr = pollutant.lower()
                    concentration = getattr(data, pollutant_attr, None)
                    
                    if concentration is not None:
                        sensor_data.append(SensorData(
                            sensor_id=data.station_id,
                            x=data.location[0],
                            y=data.location[1],
                            z=data.location[2],
                            concentration=concentration,
                            timestamp=data.timestamp
                        ))
                
                if len(sensor_data) >= 3:  # 至少需要3个传感器
                    result = self.source_inversion.invert_source(
                        sensor_data, meteo_data, verbose=False
                    )
                    inversion_results.append(result)
            
            if verbose:
                print(f"   完成反算: {len(inversion_results)}个污染源")
            
            # 4. 更新污染源信息
            self.warning_system.update_pollution_sources(inversion_results)
            
            # 5. 时空关联分析
            if verbose:
                print("\n4. 时空关联分析...")
            
            # 添加空间事件
            if spatial_events:
                self.spatiotemporal_analysis.add_spatial_events(spatial_events)
            
            # 生成热力图
            heatmap_data = {}
            pollutants = ['pm25', 'pm10', 'o3', 'no2', 'vocs']
            
            for pollutant in pollutants:
                heatmap = self.spatiotemporal_analysis.generate_concentration_heatmap(
                    processed_data, pollutant
                )
                if heatmap.max_concentration > 0:  # 只保存有数据的热力图
                    heatmap_data[pollutant] = heatmap
            
            # 关联分析
            correlation_results = self.spatiotemporal_analysis.batch_correlation_analysis(warning_events)
            
            if verbose:
                print(f"   生成热力图: {len(heatmap_data)}个污染物")
                print(f"   关联分析: {len(correlation_results)}个事件")
            
            # 6. 生成建议
            recommendations = self._generate_system_recommendations(
                quality_report, warning_events, inversion_results, correlation_results
            )
            
            # 7. 计算系统性能指标
            computation_time = time.time() - start_time
            performance = self._calculate_system_performance(
                computation_time, inversion_results, quality_report
            )
            
            # 创建结果对象
            result = TracingResult(
                timestamp=datetime.datetime.now().isoformat(),
                computation_time=computation_time,
                data_quality_report=quality_report,
                inversion_results=inversion_results,
                warning_events=warning_events,
                heatmap_data=heatmap_data,
                correlation_results=correlation_results,
                system_performance=performance,
                recommendations=recommendations
            )
            
            # 更新系统状态
            self.last_update_time = datetime.datetime.now()
            self.processing_history.append({
                'timestamp': result.timestamp,
                'computation_time': computation_time,
                'warnings_count': len(warning_events),
                'sources_found': len(inversion_results)
            })
            
            # 保持历史记录在合理范围内
            if len(self.processing_history) > 100:
                self.processing_history = self.processing_history[-100:]
            
            if verbose:
                print(f"\n=== 溯源分析完成 ===")
                print(f"总用时: {computation_time:.2f}秒")
                print(f"预警事件: {len(warning_events)}")
                print(f"识别污染源: {len(inversion_results)}")
                print(f"系统性能: {performance}")
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"溯源分析出错: {e}")
            
            # 返回错误结果
            return TracingResult(
                timestamp=datetime.datetime.now().isoformat(),
                computation_time=time.time() - start_time,
                data_quality_report=DataQualityReport(0, {}, {}, 0, ('', ''), []),
                inversion_results=[],
                warning_events=[],
                heatmap_data={},
                correlation_results=[],
                system_performance={'error': str(e)},
                recommendations=[f"系统处理出错: {e}"]
            )
    
    def _group_warnings_for_inversion(self, 
                                    warning_events: List[WarningEvent],
                                    monitoring_data: List[MonitoringData]) -> Dict[str, Tuple[List[WarningEvent], List[MonitoringData], MeteoData]]:
        """将预警事件分组用于反算"""
        groups = {}
        
        # 按时间窗口和污染物类型分组
        for warning in warning_events:
            # 创建分组键
            time_key = warning.timestamp[:16]  # 精确到分钟
            group_key = f"{time_key}_{warning.pollutant.lower()}"
            
            if group_key not in groups:
                # 找到对应时间的监测数据
                matching_data = [
                    data for data in monitoring_data
                    if data.timestamp.startswith(time_key[:13])  # 同一小时
                ]
                
                # 创建气象数据（使用第一个有效数据）
                meteo_data = None
                for data in matching_data:
                    if all([data.wind_speed, data.wind_direction, data.temperature, data.humidity, data.pressure]):
                        meteo_data = MeteoData(
                            wind_speed=data.wind_speed,
                            wind_direction=data.wind_direction,
                            temperature=data.temperature,
                            humidity=data.humidity,
                            pressure=data.pressure,
                            solar_radiation=300.0,  # 默认值
                            cloud_cover=0.5  # 默认值
                        )
                        break
                
                if meteo_data is None:
                    # 使用默认气象数据
                    meteo_data = MeteoData(
                        wind_speed=2.0,
                        wind_direction=180.0,
                        temperature=20.0,
                        humidity=60.0,
                        pressure=1013.25,
                        solar_radiation=300.0,
                        cloud_cover=0.5
                    )
                
                groups[group_key] = ([], matching_data, meteo_data)
            
            groups[group_key][0].append(warning)
        
        return groups
    
    def _generate_system_recommendations(self,
                                       quality_report: DataQualityReport,
                                       warning_events: List[WarningEvent],
                                       inversion_results: List[InversionResult],
                                       correlation_results: List[CorrelationResult]) -> List[str]:
        """生成系统建议"""
        recommendations = []
        
        # 数据质量建议
        if quality_report.total_records > 0:
            avg_missing_rate = np.mean(list(quality_report.missing_rate.values()))
            if avg_missing_rate > 0.2:
                recommendations.append("数据缺失率较高，建议检查监测设备运行状况")
        
        # 预警建议
        red_warnings = [w for w in warning_events if w.warning_level == WarningLevel.RED]
        if red_warnings:
            recommendations.append(f"检测到{len(red_warnings)}个红色预警，建议立即启动应急响应")
        
        orange_warnings = [w for w in warning_events if w.warning_level == WarningLevel.ORANGE]
        if orange_warnings:
            recommendations.append(f"检测到{len(orange_warnings)}个橙色预警，建议加强管控措施")
        
        # 溯源结果建议
        if inversion_results:
            avg_computation_time = np.mean([r.computation_time for r in inversion_results])
            if avg_computation_time > self.config.response_time_threshold:
                recommendations.append("溯源计算时间较长，建议优化算法参数或增加计算资源")
            
            high_emission_sources = [r for r in inversion_results if r.emission_rate > 2.0]
            if high_emission_sources:
                recommendations.append(f"识别到{len(high_emission_sources)}个高强度污染源，建议重点管控")
        
        # 关联分析建议
        high_confidence_correlations = [r for r in correlation_results if r.confidence_level == "high"]
        if high_confidence_correlations:
            recommendations.append(f"发现{len(high_confidence_correlations)}个高置信度污染源关联，建议优先处置")
        
        return recommendations
    
    def _calculate_system_performance(self,
                                    computation_time: float,
                                    inversion_results: List[InversionResult],
                                    quality_report: DataQualityReport) -> Dict[str, float]:
        """计算系统性能指标"""
        performance = {
            'total_computation_time': computation_time,
            'data_processing_efficiency': quality_report.total_records / max(1, computation_time),
            'average_inversion_time': np.mean([r.computation_time for r in inversion_results]) if inversion_results else 0,
            'success_rate': len(inversion_results) / max(1, len(inversion_results)),  # 简化计算
            'data_quality_score': 1 - np.mean(list(quality_report.missing_rate.values())) if quality_report.missing_rate else 1
        }
        
        return performance
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_running': self.is_running,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'processing_history_count': len(self.processing_history),
            'active_warnings_count': len(self.warning_system.active_warnings),
            'pollution_sources_count': len(self.warning_system.pollution_sources),
            'config': {
                'region_bounds': self.config.region_bounds,
                'grid_resolution': self.config.grid_resolution,
                'response_time_threshold': self.config.response_time_threshold
            }
        }
    
    def export_results(self, result: TracingResult, export_path: str) -> None:
        """导出分析结果"""
        export_data = {
            'timestamp': result.timestamp,
            'computation_time': result.computation_time,
            'data_quality': {
                'total_records': result.data_quality_report.total_records,
                'missing_rate': result.data_quality_report.missing_rate,
                'station_coverage': result.data_quality_report.station_coverage
            },
            'warning_events': [
                {
                    'event_id': w.event_id,
                    'timestamp': w.timestamp,
                    'station_id': w.station_id,
                    'pollutant': w.pollutant,
                    'concentration': w.concentration,
                    'warning_level': w.warning_level.value,
                    'location': w.location
                }
                for w in result.warning_events
            ],
            'inversion_results': [
                {
                    'source_location': (r.source_x, r.source_y, r.source_z),
                    'emission_rate': r.emission_rate,
                    'computation_time': r.computation_time,
                    'objective_value': r.objective_value
                }
                for r in result.inversion_results
            ],
            'system_performance': result.system_performance,
            'recommendations': result.recommendations
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
