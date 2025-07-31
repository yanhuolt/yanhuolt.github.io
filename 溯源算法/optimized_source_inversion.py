"""
优化版污染源反算模块
集成优化版遗传算法、缓存机制和可视化功能
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

from gaussian_plume_model import GaussianPlumeModel, PollutionSource, MeteoData
from optimized_genetic_algorithm import OptimizedGeneticPatternSearch, AdaptiveGAParameters, OptimizedIndividual

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class OptimizedSensorData:
    """优化版传感器数据结构"""
    sensor_id: str
    x: float
    y: float
    z: float
    concentration: float
    timestamp: str
    uncertainty: float = 0.1  # 测量不确定性
    weight: float = 1.0  # 数据权重


@dataclass
class OptimizedInversionResult:
    """优化版反算结果结构"""
    source_x: float
    source_y: float
    source_z: float
    emission_rate: float
    objective_value: float
    computation_time: float
    convergence_history: List[float]
    position_error: float
    emission_error: float
    confidence_interval: Dict[str, Tuple[float, float]]  # 置信区间
    performance_metrics: Dict[str, float]  # 性能指标


class OptimizedSourceInversion:
    """优化版污染源反算类"""
    
    def __init__(self, 
                 search_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 ga_parameters: Optional[AdaptiveGAParameters] = None):
        """
        初始化优化版污染源反算器
        
        Args:
            search_bounds: 搜索边界
            ga_parameters: 优化版遗传算法参数
        """
        self.gaussian_model = GaussianPlumeModel()
        
        # 默认搜索边界
        self.search_bounds = search_bounds or {
            'x': (-1000, 1000),
            'y': (-1000, 1000),
            'z': (0, 100),
            'q': (0.001, 50.0)
        }
        
        # 优化版遗传算法参数
        self.ga_params = ga_parameters or AdaptiveGAParameters(
            population_size=100,  # 增大种群
            max_generations=2000,  # 增加代数
            initial_crossover_rate=0.8,
            initial_mutation_rate=0.1,
            elite_rate=0.15,
            temperature=1.0,
            convergence_threshold=1e-8,  # 更严格的收敛条件
            use_parallel=True,
            use_cache=True,
            cache_size=20000
        )
        
        self.optimizer = OptimizedGeneticPatternSearch(self.ga_params)
        
        # 性能统计
        self.evaluation_count = 0
        self.cache_hit_count = 0
    
    def setup_objective_function_data(self,
                                    sensor_data: List[OptimizedSensorData],
                                    meteo_data: MeteoData) -> None:
        """
        设置目标函数所需的数据

        Args:
            sensor_data: 传感器数据
            meteo_data: 气象数据
        """
        # 预计算传感器位置和权重
        self.sensor_positions = [(s.x, s.y, s.z) for s in sensor_data]
        self.sensor_concentrations = np.array([s.concentration for s in sensor_data])
        self.sensor_weights = np.array([s.weight / (s.uncertainty + 1e-10) for s in sensor_data])
        self.sensor_weights = self.sensor_weights / np.sum(self.sensor_weights)  # 归一化权重
        self.sensor_data_list = sensor_data
        self.meteo_data = meteo_data

        # 气象数据哈希（用于缓存）
        self.meteo_hash = hash((meteo_data.wind_speed, meteo_data.wind_direction,
                              meteo_data.temperature, meteo_data.pressure))

    def weighted_objective_function(self, genes: np.ndarray) -> float:
        """
        加权目标函数：考虑测量不确定性的加权误差平方和
        """
        self.evaluation_count += 1

        # 检查缓存
        if self.optimizer.cache:
            cached_value = self.optimizer.cache.get(genes, self.sensor_positions, self.meteo_hash)
            if cached_value is not None:
                self.cache_hit_count += 1
                return cached_value

        source_x, source_y, source_z, emission_rate = genes

        # 边界检查
        if not (self.search_bounds['x'][0] <= source_x <= self.search_bounds['x'][1] and
                self.search_bounds['y'][0] <= source_y <= self.search_bounds['y'][1] and
                self.search_bounds['z'][0] <= source_z <= self.search_bounds['z'][1] and
                self.search_bounds['q'][0] <= emission_rate <= self.search_bounds['q'][1]):
            return 1e10  # 超出边界的惩罚

        # 创建污染源对象
        source = PollutionSource(
            x=source_x,
            y=source_y,
            z=source_z,
            emission_rate=emission_rate
        )

        # 向量化计算理论浓度
        theoretical_concentrations = np.zeros(len(self.sensor_data_list))

        for i, sensor in enumerate(self.sensor_data_list):
            try:
                theoretical_concentrations[i] = self.gaussian_model.calculate_concentration(
                    source=source,
                    receptor_x=sensor.x,
                    receptor_y=sensor.y,
                    receptor_z=sensor.z,
                    meteo=self.meteo_data
                )
            except Exception:
                theoretical_concentrations[i] = 0.0

        # 计算加权误差平方和
        errors = self.sensor_concentrations - theoretical_concentrations
        weighted_errors = errors * self.sensor_weights
        objective_value = np.sum(weighted_errors ** 2)

        # 添加正则化项（防止过拟合）
        regularization = 1e-6 * (emission_rate ** 2)
        objective_value += regularization

        # 缓存结果
        if self.optimizer.cache:
            self.optimizer.cache.set(genes, self.sensor_positions, self.meteo_hash, objective_value)

        return objective_value
    
    def monte_carlo_uncertainty_analysis(self, 
                                       best_solution: OptimizedIndividual,
                                       sensor_data: List[OptimizedSensorData],
                                       meteo_data: MeteoData,
                                       n_samples: int = 1000) -> Dict[str, Tuple[float, float]]:
        """
        蒙特卡洛不确定性分析
        
        Args:
            best_solution: 最优解
            sensor_data: 传感器数据
            meteo_data: 气象数据
            n_samples: 采样次数
            
        Returns:
            参数置信区间
        """
        print("执行蒙特卡洛不确定性分析...")
        
        # 设置目标函数数据
        self.setup_objective_function_data(sensor_data, meteo_data)

        # 生成扰动样本
        samples = []
        
        for _ in range(n_samples):
            # 在最优解附近生成随机扰动
            perturbation = np.random.normal(0, 0.1, size=len(best_solution.genes))
            perturbed_genes = best_solution.genes + perturbation * best_solution.genes
            
            # 边界处理
            bounds = [self.search_bounds['x'], self.search_bounds['y'], 
                     self.search_bounds['z'], self.search_bounds['q']]
            for i in range(len(perturbed_genes)):
                perturbed_genes[i] = np.clip(perturbed_genes[i], bounds[i][0], bounds[i][1])
            
            # 计算目标函数值
            obj_value = self.weighted_objective_function(perturbed_genes)
            
            # 只保留合理的解
            if obj_value < best_solution.objective_value * 2:
                samples.append(perturbed_genes)
        
        if len(samples) < 10:
            print("警告: 有效样本数量不足，不确定性分析可能不准确")
            return {}
        
        samples = np.array(samples)
        
        # 计算置信区间
        confidence_intervals = {}
        param_names = ['x', 'y', 'z', 'q']
        
        for i, param in enumerate(param_names):
            values = samples[:, i]
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            confidence_intervals[param] = (ci_lower, ci_upper)
        
        return confidence_intervals
    
    def invert_source(self, 
                     sensor_data: List[OptimizedSensorData],
                     meteo_data: MeteoData,
                     true_source: Optional[PollutionSource] = None,
                     verbose: bool = False,
                     enable_visualization: bool = False,
                     uncertainty_analysis: bool = False) -> OptimizedInversionResult:
        """
        执行优化版污染源反算
        
        Args:
            sensor_data: 传感器观测数据
            meteo_data: 气象数据
            true_source: 真实污染源（用于计算误差）
            verbose: 是否输出详细信息
            enable_visualization: 是否启用可视化
            uncertainty_analysis: 是否进行不确定性分析
            
        Returns:
            优化版反算结果
        """
        start_time = time.time()
        self.evaluation_count = 0
        self.cache_hit_count = 0
        
        if verbose:
            print("开始优化版污染源反算...")
            print(f"传感器数量: {len(sensor_data)}")
            print(f"气象条件: 风速={meteo_data.wind_speed}m/s, 风向={meteo_data.wind_direction}°")
            print(f"搜索范围: x{self.search_bounds['x']}, y{self.search_bounds['y']}, "
                  f"z{self.search_bounds['z']}, q{self.search_bounds['q']}")
        
        # 数据预处理：计算权重
        concentrations = [s.concentration for s in sensor_data]
        max_conc = max(concentrations) if concentrations else 1.0
        
        for sensor in sensor_data:
            # 根据浓度大小调整权重（高浓度点权重更大）
            sensor.weight = (sensor.concentration / max_conc) ** 0.5
        
        # 设置目标函数数据
        self.setup_objective_function_data(sensor_data, meteo_data)
        
        # 设置搜索边界
        bounds = [
            self.search_bounds['x'],
            self.search_bounds['y'],
            self.search_bounds['z'],
            self.search_bounds['q']
        ]
        
        # 执行优化
        best_individual, convergence_history = self.optimizer.optimize(
            objective_func=self.weighted_objective_function,
            bounds=bounds,
            verbose=verbose,
            enable_visualization=enable_visualization
        )
        
        computation_time = time.time() - start_time
        
        # 计算性能指标
        cache_hit_rate = self.cache_hit_count / max(self.evaluation_count, 1) * 100
        evaluations_per_second = self.evaluation_count / computation_time
        
        performance_metrics = {
            'total_evaluations': self.evaluation_count,
            'cache_hit_rate': cache_hit_rate,
            'evaluations_per_second': evaluations_per_second,
            'convergence_generations': len(convergence_history)
        }
        
        # 计算误差（如果提供了真实源）
        position_error = 0.0
        emission_error = 0.0
        
        if true_source is not None:
            dx = best_individual.genes[0] - true_source.x
            dy = best_individual.genes[1] - true_source.y
            dz = best_individual.genes[2] - true_source.z
            position_error = np.sqrt(dx**2 + dy**2 + dz**2)
            
            if true_source.emission_rate > 0:
                emission_error = abs(best_individual.genes[3] - true_source.emission_rate) / true_source.emission_rate * 100
        
        # 不确定性分析
        confidence_intervals = {}
        if uncertainty_analysis:
            confidence_intervals = self.monte_carlo_uncertainty_analysis(
                best_individual, sensor_data, meteo_data
            )
        
        if verbose:
            print(f"\n反算完成!")
            print(f"计算时间: {computation_time:.2f}秒")
            print(f"目标函数评估次数: {self.evaluation_count}")
            print(f"缓存命中率: {cache_hit_rate:.1f}%")
            print(f"评估速度: {evaluations_per_second:.1f} 次/秒")
            print(f"反算结果: x={best_individual.genes[0]:.2f}m, y={best_individual.genes[1]:.2f}m, "
                  f"z={best_individual.genes[2]:.2f}m, q={best_individual.genes[3]:.4f}g/s")
            if true_source:
                print(f"位置误差: {position_error:.2f}m")
                print(f"源强误差: {emission_error:.2f}%")
        
        return OptimizedInversionResult(
            source_x=best_individual.genes[0],
            source_y=best_individual.genes[1],
            source_z=best_individual.genes[2],
            emission_rate=best_individual.genes[3],
            objective_value=best_individual.objective_value,
            computation_time=computation_time,
            convergence_history=convergence_history,
            position_error=position_error,
            emission_error=emission_error,
            confidence_interval=confidence_intervals,
            performance_metrics=performance_metrics
        )
