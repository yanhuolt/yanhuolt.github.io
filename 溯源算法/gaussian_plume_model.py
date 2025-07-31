"""
高斯烟羽模型实现
基于浙江大学学报论文中的高斯烟羽模型，用于计算大气污染物扩散的理论浓度
"""

import numpy as np
import math
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class MeteoData:
    """气象数据结构"""
    wind_speed: float  # 风速 (m/s)
    wind_direction: float  # 风向 (度)
    temperature: float  # 温度 (°C)
    humidity: float  # 湿度 (%)
    pressure: float  # 气压 (hPa)
    solar_radiation: float  # 太阳辐射强度
    cloud_cover: float  # 云量 (0-1)


@dataclass
class PollutionSource:
    """污染源数据结构"""
    x: float  # x坐标 (m)
    y: float  # y坐标 (m)
    z: float  # 高度 (m)
    emission_rate: float  # 排放源强 (g/s)


class AtmosphericStability:
    """大气稳定度分类"""
    
    # 大气稳定度等级
    STABILITY_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F']
    
    @staticmethod
    def get_stability_class(wind_speed: float, solar_radiation: float, cloud_cover: float) -> str:
        """
        根据风速、太阳辐射强度和云量确定大气稳定度等级
        
        Args:
            wind_speed: 风速 (m/s)
            solar_radiation: 太阳辐射强度 (W/m²)
            cloud_cover: 云量 (0-1)
            
        Returns:
            大气稳定度等级 ('A', 'B', 'C', 'D', 'E', 'F')
        """
        # 白天条件判断
        if solar_radiation > 500:  # 强太阳辐射
            if wind_speed < 2:
                return 'A'
            elif wind_speed < 3:
                return 'A'
            elif wind_speed < 5:
                return 'B'
            elif wind_speed < 6:
                return 'C'
            else:
                return 'D'
        elif solar_radiation > 200:  # 中等太阳辐射
            if wind_speed < 2:
                return 'A'
            elif wind_speed < 3:
                return 'B'
            elif wind_speed < 5:
                return 'B'
            elif wind_speed < 6:
                return 'C'
            else:
                return 'D'
        elif solar_radiation > 50:  # 弱太阳辐射
            if wind_speed < 2:
                return 'B'
            elif wind_speed < 3:
                return 'C'
            elif wind_speed < 5:
                return 'C'
            elif wind_speed < 6:
                return 'D'
            else:
                return 'D'
        else:  # 夜晚条件
            if cloud_cover >= 0.5:  # 多云
                if wind_speed < 2:
                    return 'E'
                elif wind_speed < 3:
                    return 'D'
                elif wind_speed < 5:
                    return 'D'
                else:
                    return 'D'
            else:  # 晴朗
                if wind_speed < 2:
                    return 'F'
                elif wind_speed < 3:
                    return 'E'
                elif wind_speed < 5:
                    return 'D'
                else:
                    return 'D'


class DiffusionCoefficient:
    """扩散系数计算"""
    
    # 扩散系数参数表
    DIFFUSION_PARAMS = {
        'A': {'ay': 0.22, 'by': 0.0001, 'cy': -0.5, 'az': 0.20, 'bz': 0.0, 'cz': 0.0},
        'B': {'ay': 0.16, 'by': 0.0001, 'cy': -0.5, 'az': 0.12, 'bz': 0.0, 'cz': 0.0},
        'C': {'ay': 0.11, 'by': 0.0001, 'cy': -0.5, 'az': 0.08, 'bz': 0.0002, 'cz': -0.5},
        'D': {'ay': 0.08, 'by': 0.0001, 'cy': -0.5, 'az': 0.06, 'bz': 0.0015, 'cz': -0.5},
        'E': {'ay': 0.06, 'by': 0.0001, 'cy': -0.5, 'az': 0.03, 'bz': 0.0003, 'cz': -1.0},
        'F': {'ay': 0.04, 'by': 0.0001, 'cy': -0.5, 'az': 0.016, 'bz': 0.0003, 'cz': -1.0}
    }
    
    @staticmethod
    def calculate_sigma_y(stability_class: str, distance: float) -> float:
        """计算水平扩散系数σy"""
        params = DiffusionCoefficient.DIFFUSION_PARAMS[stability_class]
        return params['ay'] * distance * (1 + params['by'] * distance) ** params['cy']
    
    @staticmethod
    def calculate_sigma_z(stability_class: str, distance: float) -> float:
        """计算垂直扩散系数σz"""
        params = DiffusionCoefficient.DIFFUSION_PARAMS[stability_class]
        return params['az'] * distance * (1 + params['bz'] * distance) ** params['cz']


class GaussianPlumeModel:
    """高斯烟羽模型"""
    
    def __init__(self):
        self.stability_calculator = AtmosphericStability()
        self.diffusion_calculator = DiffusionCoefficient()
    
    def calculate_concentration(self, 
                              source: PollutionSource,
                              receptor_x: float,
                              receptor_y: float, 
                              receptor_z: float,
                              meteo: MeteoData) -> float:
        """
        计算受体点的污染物浓度
        
        Args:
            source: 污染源信息
            receptor_x, receptor_y, receptor_z: 受体点坐标
            meteo: 气象数据
            
        Returns:
            污染物浓度 (μg/m³)
        """
        # 计算相对坐标
        dx = receptor_x - source.x
        dy = receptor_y - source.y
        dz = receptor_z - source.z
        
        # 风向转换（将风向转换为坐标系方向）
        wind_rad = math.radians(meteo.wind_direction)
        
        # 坐标系转换到风向坐标系
        x_wind = dx * math.cos(wind_rad) + dy * math.sin(wind_rad)
        y_wind = -dx * math.sin(wind_rad) + dy * math.cos(wind_rad)
        
        # 只考虑下风向
        if x_wind <= 0:
            return 0.0
        
        # 确定大气稳定度
        stability_class = self.stability_calculator.get_stability_class(
            meteo.wind_speed, meteo.solar_radiation, meteo.cloud_cover
        )
        
        # 计算扩散系数
        sigma_y = self.diffusion_calculator.calculate_sigma_y(stability_class, x_wind)
        sigma_z = self.diffusion_calculator.calculate_sigma_z(stability_class, x_wind)
        
        # 避免除零错误
        if sigma_y <= 0 or sigma_z <= 0 or meteo.wind_speed <= 0:
            return 0.0
        
        # 高斯烟羽公式
        # ρ(x,y,z) = (q / (2π * u * σy * σz)) * exp(-y²/(2σy²)) * 
        #            [exp(-(z-H)²/(2σz²)) + exp(-(z+H)²/(2σz²))]
        
        coeff = source.emission_rate / (2 * math.pi * meteo.wind_speed * sigma_y * sigma_z)
        
        # y方向扩散项
        y_term = math.exp(-0.5 * (y_wind / sigma_y) ** 2)
        
        # z方向扩散项（考虑地面反射）
        z_term1 = math.exp(-0.5 * ((receptor_z - source.z) / sigma_z) ** 2)
        z_term2 = math.exp(-0.5 * ((receptor_z + source.z) / sigma_z) ** 2)
        z_term = z_term1 + z_term2
        
        concentration = coeff * y_term * z_term
        
        # 转换单位：g/m³ -> μg/m³
        return concentration * 1e6
    
    def calculate_concentration_field(self,
                                    source: PollutionSource,
                                    x_range: Tuple[float, float],
                                    y_range: Tuple[float, float],
                                    z_height: float,
                                    grid_size: int,
                                    meteo: MeteoData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算浓度场
        
        Args:
            source: 污染源
            x_range: x坐标范围 (min, max)
            y_range: y坐标范围 (min, max)
            z_height: 计算高度
            grid_size: 网格大小
            meteo: 气象数据
            
        Returns:
            (X网格, Y网格, 浓度场)
        """
        x = np.linspace(x_range[0], x_range[1], grid_size)
        y = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        concentration_field = np.zeros_like(X)
        
        for i in range(grid_size):
            for j in range(grid_size):
                concentration_field[i, j] = self.calculate_concentration(
                    source, X[i, j], Y[i, j], z_height, meteo
                )
        
        return X, Y, concentration_field
