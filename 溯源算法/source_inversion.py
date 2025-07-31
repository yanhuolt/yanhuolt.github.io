"""
污染源反算模块
基于高斯烟羽模型和遗传-模式搜索算法的污染源位置和强度反算
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

from gaussian_plume_model import GaussianPlumeModel, PollutionSource, MeteoData
from genetic_pattern_search import GeneticPatternSearchAlgorithm, GAParameters, Individual


@dataclass
class SensorData:
    """传感器数据结构"""
    sensor_id: str
    x: float  # 传感器x坐标 (m)
    y: float  # 传感器y坐标 (m)
    z: float  # 传感器高度 (m)
    concentration: float  # 观测浓度 (μg/m³)
    timestamp: str  # 时间戳


@dataclass
class InversionResult:
    """反算结果结构"""
    source_x: float  # 污染源x坐标 (m)
    source_y: float  # 污染源y坐标 (m)
    source_z: float  # 污染源高度 (m)
    emission_rate: float  # 排放源强 (g/s)
    objective_value: float  # 目标函数值
    computation_time: float  # 计算时间 (s)
    convergence_history: List[float]  # 收敛历史
    position_error: float  # 位置误差 (m)
    emission_error: float  # 源强相对误差 (%)


class SourceInversion:
    """污染源反算类"""
    
    def __init__(self, 
                 search_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 ga_parameters: Optional[GAParameters] = None):
        """
        初始化污染源反算器
        
        Args:
            search_bounds: 搜索边界 {'x': (min, max), 'y': (min, max), 'z': (min, max), 'q': (min, max)}
            ga_parameters: 遗传算法参数
        """
        self.gaussian_model = GaussianPlumeModel()
        
        # 默认搜索边界
        self.search_bounds = search_bounds or {
            'x': (-500, 500),    # x坐标范围 (m)
            'y': (-500, 500),    # y坐标范围 (m)
            'z': (0, 50),        # 高度范围 (m)
            'q': (0.01, 10.0)    # 源强范围 (g/s)
        }
        
        # 默认遗传算法参数
        self.ga_params = ga_parameters or GAParameters(
            population_size=50,
            max_generations=1000,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_rate=0.2,
            temperature=1.0,
            convergence_threshold=1e-6
        )
        
        self.optimizer = GeneticPatternSearchAlgorithm(self.ga_params)
    
    def create_objective_function(self, 
                                sensor_data: List[SensorData], 
                                meteo_data: MeteoData) -> callable:
        """
        创建目标函数
        
        Args:
            sensor_data: 传感器观测数据列表
            meteo_data: 气象数据
            
        Returns:
            目标函数
        """
        def objective_function(genes: np.ndarray) -> float:
            """
            目标函数：理论浓度与观测浓度的误差平方和
            f(x,y,z,q) = Σ(ρ_mea^i - ρ_cal^i)²
            
            Args:
                genes: [x, y, z, q] - 污染源位置和源强
                
            Returns:
                目标函数值
            """
            source_x, source_y, source_z, emission_rate = genes
            
            # 创建污染源对象
            source = PollutionSource(
                x=source_x,
                y=source_y, 
                z=source_z,
                emission_rate=emission_rate
            )
            
            # 计算误差平方和
            error_sum = 0.0
            for sensor in sensor_data:
                # 计算理论浓度
                theoretical_conc = self.gaussian_model.calculate_concentration(
                    source=source,
                    receptor_x=sensor.x,
                    receptor_y=sensor.y,
                    receptor_z=sensor.z,
                    meteo=meteo_data
                )
                
                # 累加误差平方
                error = sensor.concentration - theoretical_conc
                error_sum += error ** 2
            
            return error_sum
        
        return objective_function
    
    def invert_source(self, 
                     sensor_data: List[SensorData],
                     meteo_data: MeteoData,
                     true_source: Optional[PollutionSource] = None,
                     verbose: bool = False) -> InversionResult:
        """
        执行污染源反算
        
        Args:
            sensor_data: 传感器观测数据
            meteo_data: 气象数据
            true_source: 真实污染源（用于计算误差，可选）
            verbose: 是否输出详细信息
            
        Returns:
            反算结果
        """
        start_time = time.time()
        
        if verbose:
            print("开始污染源反算...")
            print(f"传感器数量: {len(sensor_data)}")
            print(f"气象条件: 风速={meteo_data.wind_speed}m/s, 风向={meteo_data.wind_direction}°")
        
        # 创建目标函数
        objective_func = self.create_objective_function(sensor_data, meteo_data)
        
        # 设置搜索边界
        bounds = [
            self.search_bounds['x'],
            self.search_bounds['y'],
            self.search_bounds['z'],
            self.search_bounds['q']
        ]
        
        # 执行优化
        best_individual, convergence_history = self.optimizer.optimize(
            objective_func=objective_func,
            bounds=bounds,
            verbose=verbose
        )
        
        computation_time = time.time() - start_time
        
        # 计算误差（如果提供了真实源）
        position_error = 0.0
        emission_error = 0.0
        
        if true_source is not None:
            # 位置误差（欧几里得距离）
            dx = best_individual.genes[0] - true_source.x
            dy = best_individual.genes[1] - true_source.y
            dz = best_individual.genes[2] - true_source.z
            position_error = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # 源强相对误差
            if true_source.emission_rate > 0:
                emission_error = abs(best_individual.genes[3] - true_source.emission_rate) / true_source.emission_rate * 100
        
        # 创建结果对象
        result = InversionResult(
            source_x=best_individual.genes[0],
            source_y=best_individual.genes[1],
            source_z=best_individual.genes[2],
            emission_rate=best_individual.genes[3],
            objective_value=best_individual.objective_value,
            computation_time=computation_time,
            convergence_history=convergence_history,
            position_error=position_error,
            emission_error=emission_error
        )
        
        if verbose:
            print(f"\n反算完成!")
            print(f"计算时间: {computation_time:.2f}秒")
            print(f"反算结果:")
            print(f"  位置: ({result.source_x:.2f}, {result.source_y:.2f}, {result.source_z:.2f})")
            print(f"  源强: {result.emission_rate:.4f} g/s")
            print(f"  目标函数值: {result.objective_value:.6f}")
            
            if true_source is not None:
                print(f"误差分析:")
                print(f"  位置误差: {position_error:.2f} m")
                print(f"  源强相对误差: {emission_error:.2f}%")
        
        return result
    
    def batch_inversion(self, 
                       sensor_data_list: List[List[SensorData]],
                       meteo_data_list: List[MeteoData],
                       verbose: bool = False) -> List[InversionResult]:
        """
        批量反算
        
        Args:
            sensor_data_list: 多个时刻的传感器数据
            meteo_data_list: 对应的气象数据
            verbose: 是否输出详细信息
            
        Returns:
            反算结果列表
        """
        results = []
        
        for i, (sensor_data, meteo_data) in enumerate(zip(sensor_data_list, meteo_data_list)):
            if verbose:
                print(f"\n处理第{i+1}/{len(sensor_data_list)}个时刻的数据...")
            
            result = self.invert_source(sensor_data, meteo_data, verbose=verbose)
            results.append(result)
        
        return results
    
    def validate_performance(self, 
                           test_cases: List[Dict],
                           verbose: bool = False) -> Dict[str, float]:
        """
        验证算法性能
        
        Args:
            test_cases: 测试用例列表，每个包含 {'true_source', 'sensor_data', 'meteo_data'}
            verbose: 是否输出详细信息
            
        Returns:
            性能统计 {'avg_time', 'avg_position_error', 'avg_emission_error', 'success_rate'}
        """
        if verbose:
            print("开始性能验证...")
        
        times = []
        position_errors = []
        emission_errors = []
        success_count = 0
        
        for i, test_case in enumerate(test_cases):
            if verbose:
                print(f"\n测试用例 {i+1}/{len(test_cases)}")
            
            result = self.invert_source(
                sensor_data=test_case['sensor_data'],
                meteo_data=test_case['meteo_data'],
                true_source=test_case['true_source'],
                verbose=verbose
            )
            
            times.append(result.computation_time)
            position_errors.append(result.position_error)
            emission_errors.append(result.emission_error)
            
            # 成功标准：位置误差<10m，源强误差<20%
            if result.position_error < 10.0 and result.emission_error < 20.0:
                success_count += 1
        
        performance = {
            'avg_time': np.mean(times),
            'avg_position_error': np.mean(position_errors),
            'avg_emission_error': np.mean(emission_errors),
            'success_rate': success_count / len(test_cases) * 100
        }
        
        if verbose:
            print(f"\n性能统计:")
            print(f"平均计算时间: {performance['avg_time']:.2f}秒")
            print(f"平均位置误差: {performance['avg_position_error']:.2f}m")
            print(f"平均源强误差: {performance['avg_emission_error']:.2f}%")
            print(f"成功率: {performance['success_rate']:.1f}%")
        
        return performance
