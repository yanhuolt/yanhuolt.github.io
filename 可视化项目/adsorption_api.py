#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽取式吸附曲线数据处理API接口
基于完整数据处理与可视化算法，提供数据点坐标、标签和预警点信息
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import warnings
import os
warnings.filterwarnings('ignore')

# 导入原始算法的类和函数
from Adsorption_isotherm import (
    AdsorptionCurveProcessor,
    WarningLevel,
    WarningEvent,
    LogisticWarningModel
)

@dataclass
class DataPoint:
    """数据点信息"""
    x: float          # x轴坐标（时间）
    y: float          # y轴坐标（浓度或效率）
    label: str        # 数据点标签
    data_type: str    # 数据类型（原始/清洗后/拟合）

@dataclass
class WarningPoint:
    """预警点信息"""
    x: float                    # x轴坐标（时间）
    y: float                    # y轴坐标（浓度或效率）
    warning_level: WarningLevel # 预警等级
    reason: str                 # 预警原因
    recommendation: str         # 建议措施

@dataclass
class AdsorptionAnalysisResult:
    """吸附分析结果"""
    # 原始数据点
    raw_data_points: List[DataPoint]
    
    # 清洗后数据点
    cleaned_data_points_ks: List[DataPoint]
    cleaned_data_points_boxplot: List[DataPoint]
    
    # 效率数据点
    efficiency_data_points_ks: List[DataPoint]
    efficiency_data_points_boxplot: List[DataPoint]
    
    # 拟合曲线数据点
    fitted_curve_points: List[DataPoint]
    
    # 预警点
    warning_points: List[WarningPoint]
    
    # 统计信息
    statistics: Dict[str, Any]

class AdsorptionAPI:
    """吸附曲线预警系统API"""

    def __init__(self, data_file: str):
        """
        初始化API

        Args:
            data_file: 数据文件路径，支持CSV、XLSX、XLS格式
        """
        self.data_file = data_file
        self.processor = AdsorptionCurveProcessor(data_file)
        self.warning_result = None
    
    def analyze_warning_system(self) -> Dict[str, any]:
        """
        执行预警系统分析，返回预警相关的数据点和预警点

        Returns:
            Dict: 包含预警系统数据点、标签和预警点坐标的字典
        """
        print("=== 开始预警系统分析 ===")

        # 1. 加载数据
        if not self.processor.load_data():
            raise ValueError("数据加载失败")

        # 2. 基础数据清洗
        basic_cleaned = self.processor.basic_data_cleaning(self.processor.raw_data)
        if len(basic_cleaned) == 0:
            raise ValueError("基础清洗后无数据")

        # 3. K-S检验清洗（用于预警系统）
        self.processor.cleaned_data_ks = self.processor.ks_test_cleaning(basic_cleaned)

        # 4. 计算吸附效率数据（预警系统的核心数据）
        self.processor.efficiency_data_ks = self.processor.calculate_efficiency_data(
            self.processor.cleaned_data_ks, "K-S检验"
        )

        # 5. 预警分析 - 使用正确的方法名
        if self.processor.efficiency_data_ks is not None:
            # 调用处理器的预警分析方法
            self.processor.analyze_warning_system()
        else:
            self.processor.warning_events = []

        # 6. 提取预警系统相关数据
        result = self._extract_warning_data()

        self.warning_result = result
        print("=== 预警系统分析完成 ===")

        return result
    
    def _extract_warning_data(self) -> Dict[str, any]:
        """提取预警系统相关数据"""

        # 预警系统的时间段穿透率数据点
        data_points = []
        if self.processor.efficiency_data_ks is not None:
            for i, row in self.processor.efficiency_data_ks.iterrows():
                # 时间段编号（从1开始）
                time_segment = i + 1
                # 穿透率（转换为百分比）
                breakthrough_ratio = row.get('breakthrough_ratio', 0) * 100
                # 效率
                efficiency = row.get('efficiency', 0)
                # 进口浓度
                inlet_conc = row.get('inlet_concentration', 0)
                # 出口浓度
                outlet_conc = row.get('outlet_concentration', 0)

                data_points.append({
                    "x": time_segment,  # x轴：时间段
                    "y": breakthrough_ratio,  # y轴：穿透率%
                    "label": f"时段{time_segment}: 进口={inlet_conc:.2f}, 出口={outlet_conc:.2f}, 穿透率={breakthrough_ratio:.1f}%, 效率={efficiency:.1f}%"
                })

        # 预警时间点的穿透率（五角星标记的预警点）
        warning_point_breakthrough = None
        warning_time_segment = None

        if (self.processor.warning_model.fitted and
            self.processor.warning_model.warning_time is not None):

            # 获取预警时间点的穿透率
            warning_time = self.processor.warning_model.warning_time
            warning_breakthrough = self.processor.warning_model.predict_breakthrough(
                np.array([warning_time]))[0] * 100

            # 将预警时间转换为时间段（假设每个时间段对应一个索引）
            if self.processor.efficiency_data_ks is not None:
                # 找到最接近预警时间的时间段
                time_data = self.processor.efficiency_data_ks.get('time', range(len(self.processor.efficiency_data_ks)))
                if hasattr(time_data, '__iter__'):
                    closest_idx = min(range(len(time_data)),
                                    key=lambda i: abs(time_data.iloc[i] if hasattr(time_data, 'iloc') else time_data[i] - warning_time))
                    warning_time_segment = closest_idx + 1
                else:
                    warning_time_segment = int(warning_time)

            warning_point_breakthrough = warning_breakthrough

        return {
            "data_points": data_points,  # 所有时间段的穿透率数据点
            "warning_point": {  # 预警时间点的穿透率
                "time_segment": warning_time_segment,
                "breakthrough_rate": warning_point_breakthrough,
                "description": f"预警点(穿透率: {warning_point_breakthrough:.1f}%)" if warning_point_breakthrough else None
            },
            "statistics": {
                "total_data_points": len(data_points),
                "has_warning_point": warning_point_breakthrough is not None,
                "time_segments_range": {
                    "start": 1,
                    "end": len(data_points)
                },
                "breakthrough_range": {
                    "min": min([p["y"] for p in data_points]) if data_points else 0,
                    "max": max([p["y"] for p in data_points]) if data_points else 0
                }
            }
        }

    def _extract_data_points(self) -> AdsorptionAnalysisResult:
        """提取所有数据点信息"""
        
        # 原始数据点
        raw_points = []
        if self.processor.raw_data is not None:
            for _, row in self.processor.raw_data.iterrows():
                # 根据进口0出口1字段判断数据类型
                location = "进口" if row.get('进口0出口1', 1) == 0 else "出口"
                concentration = row.get('浓度(mg/m³)', row.get('出口浓度(mg/m³)', 0))
                time_val = row.get('时间(s)', row.get('时间', 0))

                raw_points.append(DataPoint(
                    x=float(time_val),
                    y=float(concentration),
                    label=f"原始数据({location}) t={time_val:.1f}s, C={concentration:.2f}mg/m³",
                    data_type="原始数据"
                ))
        
        # K-S检验清洗后数据点
        ks_points = []
        if self.processor.cleaned_data_ks is not None:
            for _, row in self.processor.cleaned_data_ks.iterrows():
                location = "进口" if row.get('进口0出口1', 1) == 0 else "出口"
                concentration = row.get('浓度(mg/m³)', row.get('出口浓度(mg/m³)', 0))
                time_val = row.get('时间(s)', row.get('时间', 0))

                ks_points.append(DataPoint(
                    x=float(time_val),
                    y=float(concentration),
                    label=f"K-S清洗({location}) t={time_val:.1f}s, C={concentration:.2f}mg/m³",
                    data_type="K-S清洗"
                ))

        # 箱型图清洗后数据点
        boxplot_points = []
        if self.processor.cleaned_data_boxplot is not None:
            for _, row in self.processor.cleaned_data_boxplot.iterrows():
                location = "进口" if row.get('进口0出口1', 1) == 0 else "出口"
                concentration = row.get('浓度(mg/m³)', row.get('出口浓度(mg/m³)', 0))
                time_val = row.get('时间(s)', row.get('时间', 0))

                boxplot_points.append(DataPoint(
                    x=float(time_val),
                    y=float(concentration),
                    label=f"箱型图清洗({location}) t={time_val:.1f}s, C={concentration:.2f}mg/m³",
                    data_type="箱型图清洗"
                ))
        
        # K-S效率数据点
        eff_ks_points = []
        if self.processor.efficiency_data_ks is not None:
            for _, row in self.processor.efficiency_data_ks.iterrows():
                eff_ks_points.append(DataPoint(
                    x=row['时间(s)'],
                    y=row['吸附效率(%)'],
                    label=f"K-S效率 t={row['时间(s)']:.1f}s, η={row['吸附效率(%)']:.1f}%",
                    data_type="K-S效率"
                ))
        
        # 箱型图效率数据点
        eff_boxplot_points = []
        if self.processor.efficiency_data_boxplot is not None:
            for _, row in self.processor.efficiency_data_boxplot.iterrows():
                eff_boxplot_points.append(DataPoint(
                    x=row['时间(s)'],
                    y=row['吸附效率(%)'],
                    label=f"箱型图效率 t={row['时间(s)']:.1f}s, η={row['吸附效率(%)']:.1f}%",
                    data_type="箱型图效率"
                ))
        
        # 拟合曲线数据点（如果有的话）
        fitted_points = []
        if hasattr(self.processor, 'fitted_data') and self.processor.fitted_data is not None:
            for _, row in self.processor.fitted_data.iterrows():
                fitted_points.append(DataPoint(
                    x=row['时间(s)'],
                    y=row['拟合浓度(mg/m³)'],
                    label=f"拟合曲线 t={row['时间(s)']:.1f}s, C={row['拟合浓度(mg/m³)']:.2f}mg/m³",
                    data_type="拟合曲线"
                ))
        
        # 预警点
        warning_points = []
        for event in self.processor.warning_events:
            # 找到对应时间点的浓度值
            y_value = 0
            if self.processor.efficiency_data_ks is not None:
                closest_row = self.processor.efficiency_data_ks.iloc[
                    (self.processor.efficiency_data_ks['时间(s)'] - event.timestamp).abs().argsort()[:1]
                ]
                if not closest_row.empty:
                    y_value = closest_row.iloc[0]['吸附效率(%)']
            
            warning_points.append(WarningPoint(
                x=event.timestamp,
                y=y_value,
                warning_level=event.warning_level,
                reason=event.reason,
                recommendation=event.recommendation
            ))
        
        # 统计信息
        statistics = self._calculate_statistics()
        
        return AdsorptionAnalysisResult(
            raw_data_points=raw_points,
            cleaned_data_points_ks=ks_points,
            cleaned_data_points_boxplot=boxplot_points,
            efficiency_data_points_ks=eff_ks_points,
            efficiency_data_points_boxplot=eff_boxplot_points,
            fitted_curve_points=fitted_points,
            warning_points=warning_points,
            statistics=statistics
        )
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """计算统计信息"""
        stats = {}
        
        if self.processor.raw_data is not None:
            stats['raw_data_count'] = len(self.processor.raw_data)
            stats['raw_data_time_range'] = {
                'start': self.processor.raw_data['时间(s)'].min(),
                'end': self.processor.raw_data['时间(s)'].max()
            }
            stats['raw_data_concentration_range'] = {
                'min': self.processor.raw_data['出口浓度(mg/m³)'].min(),
                'max': self.processor.raw_data['出口浓度(mg/m³)'].max(),
                'mean': self.processor.raw_data['出口浓度(mg/m³)'].mean()
            }
        
        if self.processor.cleaned_data_ks is not None:
            stats['ks_cleaned_count'] = len(self.processor.cleaned_data_ks)
            stats['ks_cleaning_ratio'] = len(self.processor.cleaned_data_ks) / len(self.processor.raw_data) if self.processor.raw_data is not None else 0
        
        if self.processor.cleaned_data_boxplot is not None:
            stats['boxplot_cleaned_count'] = len(self.processor.cleaned_data_boxplot)
            stats['boxplot_cleaning_ratio'] = len(self.processor.cleaned_data_boxplot) / len(self.processor.raw_data) if self.processor.raw_data is not None else 0
        
        stats['warning_count'] = len(self.processor.warning_events)
        stats['warning_levels'] = {}
        for event in self.processor.warning_events:
            level = event.warning_level.value
            stats['warning_levels'][level] = stats['warning_levels'].get(level, 0) + 1
        
        return stats
    
    def get_data_points_by_type(self, data_type: str) -> List[DataPoint]:
        """
        根据数据类型获取数据点
        
        Args:
            data_type: 数据类型 ("原始数据", "K-S清洗", "箱型图清洗", "K-S效率", "箱型图效率", "拟合曲线")
        
        Returns:
            List[DataPoint]: 指定类型的数据点列表
        """
        if self.analysis_result is None:
            raise ValueError("请先调用analyze()方法进行分析")
        
        type_mapping = {
            "原始数据": self.analysis_result.raw_data_points,
            "K-S清洗": self.analysis_result.cleaned_data_points_ks,
            "箱型图清洗": self.analysis_result.cleaned_data_points_boxplot,
            "K-S效率": self.analysis_result.efficiency_data_points_ks,
            "箱型图效率": self.analysis_result.efficiency_data_points_boxplot,
            "拟合曲线": self.analysis_result.fitted_curve_points
        }
        
        return type_mapping.get(data_type, [])
    
    def get_warning_points_by_level(self, warning_level: WarningLevel) -> List[WarningPoint]:
        """
        根据预警等级获取预警点
        
        Args:
            warning_level: 预警等级
        
        Returns:
            List[WarningPoint]: 指定等级的预警点列表
        """
        if self.analysis_result is None:
            raise ValueError("请先调用analyze()方法进行分析")
        
        return [point for point in self.analysis_result.warning_points 
                if point.warning_level == warning_level]
    
    def export_results_to_dict(self) -> Dict[str, Any]:
        """
        将分析结果导出为字典格式，便于JSON序列化
        
        Returns:
            Dict[str, Any]: 包含所有分析结果的字典
        """
        if self.analysis_result is None:
            raise ValueError("请先调用analyze()方法进行分析")
        
        result = self.analysis_result
        
        return {
            "raw_data_points": [
                {
                    "x": point.x,
                    "y": point.y,
                    "label": point.label,
                    "data_type": point.data_type
                } for point in result.raw_data_points
            ],
            "cleaned_data_points_ks": [
                {
                    "x": point.x,
                    "y": point.y,
                    "label": point.label,
                    "data_type": point.data_type
                } for point in result.cleaned_data_points_ks
            ],
            "cleaned_data_points_boxplot": [
                {
                    "x": point.x,
                    "y": point.y,
                    "label": point.label,
                    "data_type": point.data_type
                } for point in result.cleaned_data_points_boxplot
            ],
            "efficiency_data_points_ks": [
                {
                    "x": point.x,
                    "y": point.y,
                    "label": point.label,
                    "data_type": point.data_type
                } for point in result.efficiency_data_points_ks
            ],
            "efficiency_data_points_boxplot": [
                {
                    "x": point.x,
                    "y": point.y,
                    "label": point.label,
                    "data_type": point.data_type
                } for point in result.efficiency_data_points_boxplot
            ],
            "fitted_curve_points": [
                {
                    "x": point.x,
                    "y": point.y,
                    "label": point.label,
                    "data_type": point.data_type
                } for point in result.fitted_curve_points
            ],
            "warning_points": [
                {
                    "x": point.x,
                    "y": point.y,
                    "warning_level": point.warning_level.value,
                    "reason": point.reason,
                    "recommendation": point.recommendation
                } for point in result.warning_points
            ],
            "statistics": result.statistics
        }


def get_warning_system_data(data_file: str) -> Dict[str, any]:
    """
    获取预警系统数据的主要接口函数

    Args:
        data_file: 数据文件路径，支持CSV、XLSX、XLS格式

    Returns:
        Dict: 包含以下信息的字典:
            - data_points: 时间段穿透率数据点列表，每个点包含 x(时间段), y(穿透率%), label(描述)
            - warning_point: 预警时间点的穿透率信息
            - statistics: 统计信息
            - success: 是否成功
    """
    try:
        # 创建API实例并分析
        api = AdsorptionAPI(data_file)
        result = api.analyze_warning_system()

        return {
            "success": True,
            "data_points": result["data_points"],  # 所有时间段的穿透率数据点
            "warning_point": result["warning_point"],  # 预警时间点的穿透率
            "statistics": result["statistics"]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data_points": [],
            "warning_point": {
                "time_segment": None,
                "breakthrough_rate": None,
                "description": None
            },
            "statistics": {}
        }


def create_adsorption_api(data_file: str) -> AdsorptionAPI:
    """
    创建吸附曲线分析API实例

    Args:
        data_file: 数据文件路径

    Returns:
        AdsorptionAPI: API实例
    """
    return AdsorptionAPI(data_file)


def analyze_adsorption_data(data_file: str) -> Dict[str, Any]:
    """
    一键分析吸附数据，返回所有数据点坐标、标签和预警点信息

    Args:
        data_file: 数据文件路径，支持CSV、XLSX、XLS格式

    Returns:
        Dict[str, Any]: 包含以下信息的字典:
            - all_data_points: 所有数据点的x,y坐标和标签
            - warning_points: 预警点的x,y坐标和相关信息
            - statistics: 统计信息
            - success: 是否成功
    """
    try:
        # 创建API实例并分析
        api = AdsorptionAPI(data_file)
        result = api.analyze()

        # 整理所有数据点
        all_data_points = []

        # 添加原始数据点
        for point in result.raw_data_points:
            all_data_points.append({
                "x": point.x,
                "y": point.y,
                "label": point.label,
                "type": "原始数据",
                "data_category": "concentration"  # 浓度数据
            })

        # 添加K-S清洗数据点
        for point in result.cleaned_data_points_ks:
            all_data_points.append({
                "x": point.x,
                "y": point.y,
                "label": point.label,
                "type": "K-S清洗",
                "data_category": "concentration"
            })

        # 添加箱型图清洗数据点
        for point in result.cleaned_data_points_boxplot:
            all_data_points.append({
                "x": point.x,
                "y": point.y,
                "label": point.label,
                "type": "箱型图清洗",
                "data_category": "concentration"
            })

        # 添加K-S效率数据点
        for point in result.efficiency_data_points_ks:
            all_data_points.append({
                "x": point.x,
                "y": point.y,
                "label": point.label,
                "type": "K-S效率",
                "data_category": "efficiency"  # 效率数据
            })

        # 添加箱型图效率数据点
        for point in result.efficiency_data_points_boxplot:
            all_data_points.append({
                "x": point.x,
                "y": point.y,
                "label": point.label,
                "type": "箱型图效率",
                "data_category": "efficiency"
            })

        # 添加拟合曲线数据点
        for point in result.fitted_curve_points:
            all_data_points.append({
                "x": point.x,
                "y": point.y,
                "label": point.label,
                "type": "拟合曲线",
                "data_category": "fitted"
            })

        # 整理预警点信息
        warning_points = []
        for point in result.warning_points:
            warning_points.append({
                "x": point.x,
                "y": point.y,
                "warning_level": point.warning_level.value,
                "reason": point.reason,
                "recommendation": point.recommendation,
                "color_code": {
                    "绿色": "#00FF00",
                    "黄色": "#FFFF00",
                    "橙色": "#FFA500",
                    "红色": "#FF0000"
                }.get(point.warning_level.value, "#808080")
            })

        return {
            "success": True,
            "all_data_points": all_data_points,
            "warning_points": warning_points,
            "statistics": result.statistics,
            "data_summary": {
                "total_points": len(all_data_points),
                "warning_count": len(warning_points),
                "data_types": list(set([p["type"] for p in all_data_points])),
                "time_range": {
                    "min": min([p["x"] for p in all_data_points]) if all_data_points else 0,
                    "max": max([p["x"] for p in all_data_points]) if all_data_points else 0
                }
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "all_data_points": [],
            "warning_points": [],
            "statistics": {},
            "data_summary": {}
        }


# 示例使用
if __name__ == "__main__":
    print("=== 预警系统API示例 ===")

    # 使用主要接口函数
    data_file = "可视化项目/7.24.csv"
    result = get_warning_system_data(data_file)

    if result["success"]:
        print("✅ 预警系统分析成功")

        # 显示时间段穿透率数据点
        data_points = result["data_points"]
        print(f"\n📊 时间段穿透率数据点: {len(data_points)} 个")
        if data_points:
            print("前5个数据点:")
            for i, point in enumerate(data_points[:5]):
                print(f"  时段{point['x']}: 穿透率={point['y']:.1f}%")
                print(f"    标签: {point['label']}")

        # 显示预警时间点的穿透率
        warning_point = result["warning_point"]
        print(f"\n⭐ 预警时间点信息:")
        if warning_point["breakthrough_rate"] is not None:
            print(f"  时间段: {warning_point['time_segment']}")
            print(f"  穿透率: {warning_point['breakthrough_rate']:.1f}%")
            print(f"  描述: {warning_point['description']}")
        else:
            print("  无预警点")

        # 显示统计信息
        stats = result["statistics"]
        print(f"\n📈 统计信息:")
        print(f"  数据点总数: {stats['total_data_points']}")
        print(f"  是否有预警点: {stats['has_warning_point']}")
        print(f"  时间段范围: {stats['time_segments_range']['start']} - {stats['time_segments_range']['end']}")
        print(f"  穿透率范围: {stats['breakthrough_range']['min']:.1f}% - {stats['breakthrough_range']['max']:.1f}%")

        # 返回格式示例
        print(f"\n📋 返回数据格式:")
        print(f"  data_points: 列表，每个元素包含 x(时间段), y(穿透率%), label(描述)")
        print(f"  warning_point: 字典，包含预警时间点的穿透率信息")
        print(f"  statistics: 字典，包含统计信息")

        # 提取坐标用于绘图
        x_coords = [p['x'] for p in data_points]  # 时间段
        y_coords = [p['y'] for p in data_points]  # 穿透率%
        labels = [p['label'] for p in data_points]  # 标签

        print(f"\n🎯 可用于绘图的数据:")
        print(f"  X坐标(时间段): {x_coords[:10]}...")  # 显示前10个
        print(f"  Y坐标(穿透率%): {y_coords[:10]}...")  # 显示前10个
        print(f"  预警点穿透率: {warning_point['breakthrough_rate']:.1f}%" if warning_point['breakthrough_rate'] else "无预警点")

    else:
        print(f"❌ 预警系统分析失败: {result['error']}")

    print("\n=== API调用完成 ===")
