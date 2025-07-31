"""
遗传-模式搜索算法实现
基于浙江大学学报论文中的GA-PS混合优化算法，用于污染源反算
"""

import numpy as np
import random
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
import time


@dataclass
class Individual:
    """个体类，表示一个解"""
    genes: np.ndarray  # 基因 [x, y, z, q] - 位置和源强
    fitness: float = 0.0  # 适应度
    objective_value: float = float('inf')  # 目标函数值


@dataclass
class GAParameters:
    """遗传算法参数"""
    population_size: int = 50  # 种群大小
    max_generations: int = 1000  # 最大迭代次数
    crossover_rate: float = 0.8  # 交叉率 P_C
    mutation_rate: float = 0.1  # 变异率 P_M
    elite_rate: float = 0.2  # 精英保留率 P_E
    temperature: float = 1.0  # 适应度函数温度参数 T
    convergence_threshold: float = 1e-6  # 收敛阈值


class PatternSearch:
    """模式搜索算法"""
    
    def __init__(self, step_size: float = 1.0, contraction_factor: float = 0.5):
        self.step_size = step_size
        self.contraction_factor = contraction_factor
        self.min_step_size = 1e-6
    
    def search(self, 
               individual: Individual, 
               objective_func: Callable,
               bounds: List[Tuple[float, float]]) -> Individual:
        """
        模式搜索优化
        
        Args:
            individual: 当前个体
            objective_func: 目标函数
            bounds: 变量边界 [(x_min, x_max), (y_min, y_max), (z_min, z_max), (q_min, q_max)]
            
        Returns:
            优化后的个体
        """
        current_individual = Individual(genes=individual.genes.copy())
        current_individual.objective_value = objective_func(current_individual.genes)
        
        step_size = self.step_size
        
        while step_size > self.min_step_size:
            improved = False
            
            # 对每个维度进行搜索
            for i in range(len(current_individual.genes)):
                # 正方向搜索
                test_genes = current_individual.genes.copy()
                test_genes[i] += step_size
                
                # 检查边界约束
                if bounds[i][0] <= float(test_genes[i]) <= bounds[i][1]:
                    test_value = objective_func(test_genes)
                    if test_value < current_individual.objective_value:
                        current_individual.genes = test_genes
                        current_individual.objective_value = test_value
                        improved = True
                        continue
                
                # 负方向搜索
                test_genes = current_individual.genes.copy()
                test_genes[i] -= step_size
                
                # 检查边界约束
                if bounds[i][0] <= float(test_genes[i]) <= bounds[i][1]:
                    test_value = objective_func(test_genes)
                    if test_value < current_individual.objective_value:
                        current_individual.genes = test_genes
                        current_individual.objective_value = test_value
                        improved = True
            
            if not improved:
                step_size *= self.contraction_factor
        
        return current_individual


class GeneticPatternSearchAlgorithm:
    """遗传-模式搜索混合算法"""
    
    def __init__(self, parameters: GAParameters):
        self.params = parameters
        self.pattern_search = PatternSearch()
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.convergence_history: List[float] = []
    
    def initialize_population(self, bounds: List[Tuple[float, float]]) -> None:
        """初始化种群"""
        self.population = []
        for _ in range(self.params.population_size):
            genes = np.array([
                random.uniform(bounds[i][0], bounds[i][1]) 
                for i in range(len(bounds))
            ])
            individual = Individual(genes=genes)
            self.population.append(individual)
    
    def calculate_fitness(self, objective_func: Callable) -> None:
        """计算种群适应度"""
        # 计算目标函数值
        for individual in self.population:
            individual.objective_value = objective_func(individual.genes)
        
        # 计算适应度 γ = exp(-|f|/T) / Σexp(-|f|/T)
        exp_values = [np.exp(-abs(ind.objective_value) / self.params.temperature) 
                     for ind in self.population]
        sum_exp = sum(exp_values)
        
        for i, individual in enumerate(self.population):
            individual.fitness = exp_values[i] / sum_exp if sum_exp > 0 else 0.0
    
    def roulette_wheel_selection(self) -> Individual:
        """轮盘赌选择"""
        total_fitness = sum(ind.fitness for ind in self.population)
        if total_fitness == 0:
            return random.choice(self.population)
        
        r = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        
        for individual in self.population:
            cumulative_fitness += individual.fitness
            if cumulative_fitness >= r:
                return individual
        
        return self.population[-1]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作"""
        if random.random() > self.params.crossover_rate:
            return Individual(genes=parent1.genes.copy()), Individual(genes=parent2.genes.copy())
        
        # 算术交叉
        alpha = random.random()
        child1_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        child2_genes = (1 - alpha) * parent1.genes + alpha * parent2.genes
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    def mutate(self, individual: Individual, bounds: List[Tuple[float, float]]) -> Individual:
        """变异操作"""
        mutated_genes = individual.genes.copy()
        
        for i in range(len(mutated_genes)):
            if random.random() < self.params.mutation_rate:
                # 高斯变异
                mutation_range = (bounds[i][1] - bounds[i][0]) * 0.1
                mutation = np.random.normal(0, mutation_range)
                mutated_genes[i] += mutation
                
                # 边界处理
                mutated_genes[i] = np.clip(mutated_genes[i], bounds[i][0], bounds[i][1])
        
        return Individual(genes=mutated_genes)
    
    def optimize(self, 
                objective_func: Callable,
                bounds: List[Tuple[float, float]],
                verbose: bool = False) -> Tuple[Individual, List[float]]:
        """
        执行遗传-模式搜索优化
        
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            verbose: 是否输出详细信息
            
        Returns:
            (最优个体, 收敛历史)
        """
        start_time = time.time()
        
        # 初始化种群
        self.initialize_population(bounds)
        self.convergence_history = []
        
        for generation in range(self.params.max_generations):
            # 计算适应度
            self.calculate_fitness(objective_func)
            
            # 排序种群
            self.population.sort(key=lambda x: x.objective_value)
            
            # 更新最优个体
            if self.best_individual is None or self.population[0].objective_value < self.best_individual.objective_value:
                self.best_individual = Individual(genes=self.population[0].genes.copy())
                self.best_individual.objective_value = self.population[0].objective_value
            
            self.convergence_history.append(self.best_individual.objective_value)
            
            # 收敛检查
            if generation > 10:
                recent_improvement = abs(self.convergence_history[-10] - self.convergence_history[-1])
                if recent_improvement < self.params.convergence_threshold:
                    if verbose:
                        print(f"算法在第{generation}代收敛")
                    break
            
            # 生成新种群
            new_population = []
            
            # 精英保留
            elite_count = int(self.params.population_size * self.params.elite_rate)
            new_population.extend([Individual(genes=ind.genes.copy()) for ind in self.population[:elite_count]])
            
            # 生成子代
            while len(new_population) < self.params.population_size:
                parent1 = self.roulette_wheel_selection()
                parent2 = self.roulette_wheel_selection()
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, bounds)
                child2 = self.mutate(child2, bounds)
                
                new_population.extend([child1, child2])
            
            # 截断到指定大小
            new_population = new_population[:self.params.population_size]
            
            # 对最差个体应用模式搜索
            # 先计算所有个体的目标函数值
            for ind in new_population:
                if not hasattr(ind, 'objective_value') or ind.objective_value is None:
                    ind.objective_value = objective_func(ind.genes)

            worst_individual = max(new_population, key=lambda x: x.objective_value)
            improved_individual = self.pattern_search.search(worst_individual, objective_func, bounds)
            
            # 替换最差个体
            worst_index = new_population.index(worst_individual)
            new_population[worst_index] = improved_individual
            
            self.population = new_population
            
            if verbose and generation % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"第{generation}代: 最优值={self.best_individual.objective_value:.6f}, "
                      f"用时={elapsed_time:.2f}秒")
        
        total_time = time.time() - start_time
        if verbose:
            print(f"优化完成，总用时: {total_time:.2f}秒")
            print(f"最优解: x={self.best_individual.genes[0]:.2f}, "
                  f"y={self.best_individual.genes[1]:.2f}, "
                  f"z={self.best_individual.genes[2]:.2f}, "
                  f"q={self.best_individual.genes[3]:.4f}")
            print(f"目标函数值: {self.best_individual.objective_value:.6f}")
        
        return self.best_individual, self.convergence_history
