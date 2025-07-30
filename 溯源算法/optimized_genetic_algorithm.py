"""
优化版遗传-模式搜索算法
包含并行计算、缓存机制、自适应参数调整等性能优化
"""

import numpy as np
import random
from typing import List, Tuple, Callable, Optional, Dict
from dataclasses import dataclass, field
import time
import multiprocessing as mp
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class OptimizedIndividual:
    """优化版个体类"""
    genes: np.ndarray
    fitness: float = 0.0
    objective_value: float = float('inf')
    age: int = 0  # 个体年龄，用于多样性维护
    evaluation_count: int = 0  # 评估次数


@dataclass
class AdaptiveGAParameters:
    """自适应遗传算法参数"""
    population_size: int = 50
    max_generations: int = 1000
    initial_crossover_rate: float = 0.8
    initial_mutation_rate: float = 0.1
    elite_rate: float = 0.2
    temperature: float = 1.0
    convergence_threshold: float = 1e-6
    
    # 自适应参数
    crossover_rate_range: Tuple[float, float] = (0.6, 0.9)
    mutation_rate_range: Tuple[float, float] = (0.05, 0.3)
    diversity_threshold: float = 0.1
    stagnation_threshold: int = 50
    
    # 并行计算参数
    use_parallel: bool = True
    n_processes: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    
    # 缓存参数
    use_cache: bool = True
    cache_size: int = 10000


class GaussianPlumeCache:
    """高斯烟羽模型计算缓存"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def _hash_key(self, genes: np.ndarray, sensor_positions: List[Tuple], meteo_hash: int) -> str:
        """生成缓存键"""
        genes_rounded = np.round(genes, decimals=3)  # 降低精度以增加缓存命中率
        return f"{genes_rounded.tobytes()}_{hash(tuple(sensor_positions))}_{meteo_hash}"
    
    def get(self, genes: np.ndarray, sensor_positions: List[Tuple], meteo_hash: int) -> Optional[float]:
        """获取缓存值"""
        key = self._hash_key(genes, sensor_positions, meteo_hash)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, genes: np.ndarray, sensor_positions: List[Tuple], meteo_hash: int, value: float):
        """设置缓存值"""
        if len(self.cache) >= self.max_size:
            # 删除最少使用的缓存项
            least_used_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[least_used_key]
            del self.access_count[least_used_key]
        
        key = self._hash_key(genes, sensor_positions, meteo_hash)
        self.cache[key] = value
        self.access_count[key] = 1


# 全局函数，用于并行计算（避免序列化问题）
_global_objective_func = None
_global_bounds = None

def set_global_objective_function(objective_func, bounds):
    """设置全局目标函数和边界"""
    global _global_objective_func, _global_bounds
    _global_objective_func = objective_func
    _global_bounds = bounds

def parallel_fitness_evaluation(individual):
    """并行适应度评估函数"""
    global _global_objective_func, _global_bounds

    if _global_objective_func is None or _global_bounds is None:
        return float('inf')

    # 边界检查
    for i, gene in enumerate(individual.genes):
        if not (_global_bounds[i][0] <= gene <= _global_bounds[i][1]):
            return float('inf')  # 超出边界的个体给予极大惩罚

    try:
        objective_value = _global_objective_func(individual.genes)
        return objective_value
    except Exception as e:
        return float('inf')  # 计算错误时给予极大惩罚


class OptimizedPatternSearch:
    """优化版模式搜索算法"""
    
    def __init__(self, step_size: float = 1.0, contraction_factor: float = 0.5):
        self.step_size = step_size
        self.contraction_factor = contraction_factor
        self.min_step_size = 1e-6
        self.success_history = []  # 成功历史，用于自适应调整
    
    def adaptive_search(self, 
                       individual: OptimizedIndividual, 
                       objective_func: Callable,
                       bounds: List[Tuple[float, float]]) -> OptimizedIndividual:
        """自适应模式搜索"""
        current_individual = OptimizedIndividual(genes=individual.genes.copy())
        current_individual.objective_value = objective_func(current_individual.genes)
        
        # 根据历史成功率调整步长
        if len(self.success_history) > 10:
            success_rate = sum(self.success_history[-10:]) / 10
            if success_rate > 0.7:
                step_size = self.step_size * 1.2  # 增大步长
            elif success_rate < 0.3:
                step_size = self.step_size * 0.8  # 减小步长
            else:
                step_size = self.step_size
        else:
            step_size = self.step_size
        
        iteration_count = 0
        max_iterations = 100
        
        while step_size > self.min_step_size and iteration_count < max_iterations:
            improved = False
            
            # 坐标搜索方向（包括对角线方向）
            directions = []
            n_vars = len(current_individual.genes)
            
            # 坐标轴方向
            for i in range(n_vars):
                direction = np.zeros(n_vars)
                direction[i] = 1
                directions.append(direction)
                directions.append(-direction)
            
            # 对角线方向（提高搜索效率）
            if n_vars <= 4:  # 避免维度爆炸
                for i in range(n_vars):
                    for j in range(i+1, n_vars):
                        direction = np.zeros(n_vars)
                        direction[i] = 1
                        direction[j] = 1
                        directions.append(direction / np.linalg.norm(direction))
                        directions.append(-direction / np.linalg.norm(direction))
            
            for direction in directions:
                test_genes = current_individual.genes + step_size * direction
                
                # 边界检查
                valid = True
                for i in range(len(test_genes)):
                    if not (bounds[i][0] <= test_genes[i] <= bounds[i][1]):
                        valid = False
                        break
                
                if valid:
                    test_value = objective_func(test_genes)
                    if test_value < current_individual.objective_value:
                        current_individual.genes = test_genes
                        current_individual.objective_value = test_value
                        improved = True
                        break
            
            if not improved:
                step_size *= self.contraction_factor
            
            iteration_count += 1
        
        # 记录搜索成功率
        self.success_history.append(1 if improved else 0)
        if len(self.success_history) > 50:
            self.success_history.pop(0)
        
        return current_individual


class OptimizedGeneticPatternSearch:
    """优化版遗传-模式搜索算法"""
    
    def __init__(self, parameters: AdaptiveGAParameters):
        self.params = parameters
        self.pattern_search = OptimizedPatternSearch()
        self.population: List[OptimizedIndividual] = []
        self.best_individual: Optional[OptimizedIndividual] = None
        self.convergence_history: List[float] = []
        self.diversity_history: List[float] = []
        self.cache = GaussianPlumeCache(self.params.cache_size) if self.params.use_cache else None
        
        # 自适应参数
        self.current_crossover_rate = self.params.initial_crossover_rate
        self.current_mutation_rate = self.params.initial_mutation_rate
        self.stagnation_counter = 0
        
        # 可视化相关
        self.visualization_queue = queue.Queue()
        self.visualization_thread = None
        self.fig = None
        self.axes = None
    
    def calculate_population_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population) < 2:
            return 0.0
        
        distances = []
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                dist = np.linalg.norm(self.population[i].genes - self.population[j].genes)
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def adaptive_parameter_adjustment(self, generation: int):
        """自适应参数调整"""
        diversity = self.calculate_population_diversity()
        self.diversity_history.append(diversity)
        
        # 检查停滞
        if len(self.convergence_history) > 10:
            recent_improvement = abs(self.convergence_history[-10] - self.convergence_history[-1])
            if recent_improvement < self.params.convergence_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        # 调整交叉率和变异率
        if diversity < self.params.diversity_threshold or self.stagnation_counter > self.params.stagnation_threshold:
            # 增加变异率，降低交叉率，促进探索
            self.current_mutation_rate = min(self.params.mutation_rate_range[1], 
                                           self.current_mutation_rate * 1.1)
            self.current_crossover_rate = max(self.params.crossover_rate_range[0],
                                            self.current_crossover_rate * 0.9)
        else:
            # 恢复正常参数
            self.current_mutation_rate = max(self.params.mutation_rate_range[0],
                                           self.current_mutation_rate * 0.95)
            self.current_crossover_rate = min(self.params.crossover_rate_range[1],
                                            self.current_crossover_rate * 1.05)
    
    def initialize_population(self, bounds: List[Tuple[float, float]]) -> None:
        """初始化种群"""
        self.population = []
        
        # 使用拉丁超立方采样提高初始种群质量
        n_vars = len(bounds)
        n_samples = self.params.population_size
        
        # 简化的拉丁超立方采样
        samples = np.random.random((n_samples, n_vars))
        for i in range(n_vars):
            samples[:, i] = np.random.permutation(samples[:, i])
        
        for i in range(n_samples):
            genes = np.array([
                bounds[j][0] + samples[i, j] * (bounds[j][1] - bounds[j][0])
                for j in range(n_vars)
            ])
            individual = OptimizedIndividual(genes=genes)
            self.population.append(individual)
    
    def parallel_fitness_calculation(self, objective_func: Callable, bounds: List[Tuple[float, float]]) -> None:
        """并行计算适应度"""
        if self.params.use_parallel and len(self.population) > 1:
            # 设置全局变量
            set_global_objective_function(objective_func, bounds)

            # 使用进程池并行计算
            with mp.Pool(processes=self.params.n_processes) as pool:
                objective_values = pool.map(parallel_fitness_evaluation, self.population)

            # 更新个体的目标函数值
            for i, obj_val in enumerate(objective_values):
                self.population[i].objective_value = obj_val
                self.population[i].evaluation_count += 1
        else:
            # 串行计算
            for individual in self.population:
                individual.objective_value = objective_func(individual.genes)
                individual.evaluation_count += 1

        # 计算适应度
        exp_values = [np.exp(-abs(ind.objective_value) / self.params.temperature)
                     for ind in self.population]
        sum_exp = sum(exp_values)

        for i, individual in enumerate(self.population):
            individual.fitness = exp_values[i] / sum_exp if sum_exp > 0 else 0.0

    def tournament_selection(self, tournament_size: int = 3) -> OptimizedIndividual:
        """锦标赛选择（比轮盘赌选择更高效）"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return min(tournament, key=lambda x: x.objective_value)

    def enhanced_crossover(self, parent1: OptimizedIndividual, parent2: OptimizedIndividual) -> Tuple[OptimizedIndividual, OptimizedIndividual]:
        """增强交叉操作（混合多种交叉方式）"""
        if random.random() > self.current_crossover_rate:
            return OptimizedIndividual(genes=parent1.genes.copy()), OptimizedIndividual(genes=parent2.genes.copy())

        # 随机选择交叉方式
        crossover_type = random.choice(['arithmetic', 'blend', 'simulated_binary'])

        if crossover_type == 'arithmetic':
            # 算术交叉
            alpha = random.random()
            child1_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
            child2_genes = (1 - alpha) * parent1.genes + alpha * parent2.genes

        elif crossover_type == 'blend':
            # 混合交叉
            alpha = 0.5
            for i in range(len(parent1.genes)):
                min_val = min(parent1.genes[i], parent2.genes[i])
                max_val = max(parent1.genes[i], parent2.genes[i])
                range_val = max_val - min_val

                low = min_val - alpha * range_val
                high = max_val + alpha * range_val

                child1_genes = parent1.genes.copy()
                child2_genes = parent2.genes.copy()
                child1_genes[i] = random.uniform(low, high)
                child2_genes[i] = random.uniform(low, high)

        else:  # simulated_binary
            # 模拟二进制交叉
            eta = 2.0  # 分布指数
            child1_genes = parent1.genes.copy()
            child2_genes = parent2.genes.copy()

            for i in range(len(parent1.genes)):
                if random.random() <= 0.5:
                    if abs(parent1.genes[i] - parent2.genes[i]) > 1e-14:
                        u = random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                        child1_genes[i] = 0.5 * ((1 + beta) * parent1.genes[i] + (1 - beta) * parent2.genes[i])
                        child2_genes[i] = 0.5 * ((1 - beta) * parent1.genes[i] + (1 + beta) * parent2.genes[i])

        return OptimizedIndividual(genes=child1_genes), OptimizedIndividual(genes=child2_genes)

    def adaptive_mutation(self, individual: OptimizedIndividual, bounds: List[Tuple[float, float]], generation: int) -> OptimizedIndividual:
        """自适应变异操作"""
        mutated_genes = individual.genes.copy()

        # 自适应变异强度
        mutation_strength = self.current_mutation_rate * (1 - generation / self.params.max_generations) ** 0.5

        for i in range(len(mutated_genes)):
            if random.random() < mutation_strength:
                # 选择变异类型
                mutation_type = random.choice(['gaussian', 'uniform', 'polynomial'])

                if mutation_type == 'gaussian':
                    # 高斯变异
                    sigma = (bounds[i][1] - bounds[i][0]) * 0.1 * mutation_strength
                    mutation = np.random.normal(0, sigma)
                    mutated_genes[i] += mutation

                elif mutation_type == 'uniform':
                    # 均匀变异
                    range_val = (bounds[i][1] - bounds[i][0]) * 0.2 * mutation_strength
                    mutation = random.uniform(-range_val, range_val)
                    mutated_genes[i] += mutation

                else:  # polynomial
                    # 多项式变异
                    eta = 20.0
                    u = random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                    mutated_genes[i] += delta * (bounds[i][1] - bounds[i][0]) * mutation_strength

                # 边界处理
                mutated_genes[i] = np.clip(mutated_genes[i], bounds[i][0], bounds[i][1])

        return OptimizedIndividual(genes=mutated_genes)

    def start_visualization(self):
        """启动实时可视化"""
        def visualization_worker():
            # 设置matplotlib为非交互式后端，避免线程警告
            import matplotlib
            matplotlib.use('Agg')  # 使用非GUI后端

            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
            self.fig.suptitle('遗传算法优化过程实时监控', fontsize=16)

            # 初始化子图
            self.axes[0, 0].set_title('收敛曲线')
            self.axes[0, 0].set_xlabel('代数')
            self.axes[0, 0].set_ylabel('目标函数值')
            self.axes[0, 0].grid(True)

            self.axes[0, 1].set_title('种群多样性')
            self.axes[0, 1].set_xlabel('代数')
            self.axes[0, 1].set_ylabel('多样性指标')
            self.axes[0, 1].grid(True)

            self.axes[1, 0].set_title('适应度分布')
            self.axes[1, 0].set_xlabel('个体编号')
            self.axes[1, 0].set_ylabel('适应度')
            self.axes[1, 0].grid(True)

            self.axes[1, 1].set_title('参数自适应调整')
            self.axes[1, 1].set_xlabel('代数')
            self.axes[1, 1].set_ylabel('参数值')
            self.axes[1, 1].grid(True)

            plt.tight_layout()
            plt.show(block=False)

            # 实时更新循环
            while True:
                try:
                    data = self.visualization_queue.get(timeout=1)
                    if data is None:  # 结束信号
                        break

                    generation, best_value, diversity, fitness_values, crossover_rate, mutation_rate = data

                    # 更新收敛曲线
                    self.axes[0, 0].clear()
                    self.axes[0, 0].plot(self.convergence_history, 'b-', linewidth=2)
                    self.axes[0, 0].set_title(f'收敛曲线 (第{generation}代)')
                    self.axes[0, 0].set_xlabel('代数')
                    self.axes[0, 0].set_ylabel('目标函数值')
                    self.axes[0, 0].grid(True)

                    # 更新多样性曲线
                    self.axes[0, 1].clear()
                    self.axes[0, 1].plot(self.diversity_history, 'g-', linewidth=2)
                    self.axes[0, 1].set_title(f'种群多样性 (当前: {diversity:.4f})')
                    self.axes[0, 1].set_xlabel('代数')
                    self.axes[0, 1].set_ylabel('多样性指标')
                    self.axes[0, 1].grid(True)

                    # 更新适应度分布
                    self.axes[1, 0].clear()
                    self.axes[1, 0].bar(range(len(fitness_values)), fitness_values, alpha=0.7)
                    self.axes[1, 0].set_title('当前种群适应度分布')
                    self.axes[1, 0].set_xlabel('个体编号')
                    self.axes[1, 0].set_ylabel('适应度')
                    self.axes[1, 0].grid(True)

                    # 更新参数调整
                    self.axes[1, 1].clear()
                    generations = list(range(len(self.convergence_history)))
                    crossover_rates = [crossover_rate] * len(generations)
                    mutation_rates = [mutation_rate] * len(generations)

                    self.axes[1, 1].plot(generations, crossover_rates, 'r-', label='交叉率', linewidth=2)
                    self.axes[1, 1].plot(generations, mutation_rates, 'b-', label='变异率', linewidth=2)
                    self.axes[1, 1].set_title('参数自适应调整')
                    self.axes[1, 1].set_xlabel('代数')
                    self.axes[1, 1].set_ylabel('参数值')
                    self.axes[1, 1].legend()
                    self.axes[1, 1].grid(True)

                    plt.tight_layout()
                    plt.pause(0.01)

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"可视化更新错误: {e}")
                    break

        self.visualization_thread = threading.Thread(target=visualization_worker)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()

    def stop_visualization(self):
        """停止可视化"""
        if self.visualization_thread:
            self.visualization_queue.put(None)  # 发送结束信号
            self.visualization_thread.join(timeout=2)
            if self.fig:
                plt.close(self.fig)

    def optimize(self,
                objective_func: Callable,
                bounds: List[Tuple[float, float]],
                verbose: bool = False,
                enable_visualization: bool = False) -> Tuple[OptimizedIndividual, List[float]]:
        """
        执行优化版遗传-模式搜索优化

        Args:
            objective_func: 目标函数
            bounds: 变量边界
            verbose: 是否输出详细信息
            enable_visualization: 是否启用实时可视化

        Returns:
            (最优个体, 收敛历史)
        """
        start_time = time.time()

        if verbose:
            print("开始优化版遗传-模式搜索算法...")
            print(f"种群大小: {self.params.population_size}")
            print(f"最大代数: {self.params.max_generations}")
            print(f"并行计算: {'启用' if self.params.use_parallel else '禁用'} ({self.params.n_processes}进程)")
            print(f"缓存机制: {'启用' if self.params.use_cache else '禁用'}")

        # 启动可视化
        if enable_visualization:
            self.start_visualization()

        # 初始化种群
        self.initialize_population(bounds)

        # 主优化循环
        for generation in range(self.params.max_generations):
            # 计算适应度（并行）
            self.parallel_fitness_calculation(objective_func, bounds)

            # 排序种群
            self.population.sort(key=lambda x: x.objective_value)

            # 更新最优个体
            if self.best_individual is None or self.population[0].objective_value < self.best_individual.objective_value:
                self.best_individual = OptimizedIndividual(genes=self.population[0].genes.copy())
                self.best_individual.objective_value = self.population[0].objective_value
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            self.convergence_history.append(self.best_individual.objective_value)

            # 自适应参数调整
            self.adaptive_parameter_adjustment(generation)

            # 收敛检查
            if generation > 20:
                recent_improvement = abs(self.convergence_history[-20] - self.convergence_history[-1])
                if recent_improvement < self.params.convergence_threshold:
                    if verbose:
                        print(f"算法在第{generation}代收敛")
                    break

            # 生成新种群
            new_population = []

            # 精英保留
            elite_count = int(self.params.population_size * self.params.elite_rate)
            for i in range(elite_count):
                elite = OptimizedIndividual(genes=self.population[i].genes.copy())
                elite.age = self.population[i].age + 1
                new_population.append(elite)

            # 生成子代
            while len(new_population) < self.params.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                child1, child2 = self.enhanced_crossover(parent1, parent2)
                child1 = self.adaptive_mutation(child1, bounds, generation)
                child2 = self.adaptive_mutation(child2, bounds, generation)

                new_population.extend([child1, child2])

            # 截断到指定大小
            new_population = new_population[:self.params.population_size]

            # 对部分个体应用模式搜索（避免计算开销过大）
            pattern_search_count = max(1, int(self.params.population_size * 0.1))
            for i in range(pattern_search_count):
                if i < len(new_population):
                    # 计算目标函数值（如果还没有）
                    if new_population[i].objective_value == float('inf'):
                        new_population[i].objective_value = objective_func(new_population[i].genes)

                    improved_individual = self.pattern_search.adaptive_search(
                        new_population[i], objective_func, bounds
                    )
                    new_population[i] = improved_individual

            self.population = new_population

            # 更新可视化
            if enable_visualization:
                diversity = self.calculate_population_diversity()
                fitness_values = [ind.fitness for ind in self.population]

                try:
                    self.visualization_queue.put((
                        generation,
                        self.best_individual.objective_value,
                        diversity,
                        fitness_values,
                        self.current_crossover_rate,
                        self.current_mutation_rate
                    ), timeout=0.1)
                except queue.Full:
                    pass  # 如果队列满了就跳过这次更新

            # 输出进度
            if verbose and generation % 50 == 0:
                elapsed_time = time.time() - start_time
                diversity = self.calculate_population_diversity()
                print(f"第{generation}代: 最优值={self.best_individual.objective_value:.6f}, "
                      f"多样性={diversity:.4f}, 用时={elapsed_time:.2f}秒")
                print(f"  当前参数: 交叉率={self.current_crossover_rate:.3f}, "
                      f"变异率={self.current_mutation_rate:.3f}")

        total_time = time.time() - start_time

        if verbose:
            print(f"\n优化完成!")
            print(f"总用时: {total_time:.2f}秒")
            print(f"最优解: {self.best_individual.genes}")
            print(f"最优值: {self.best_individual.objective_value:.6f}")
            print(f"总评估次数: {sum(ind.evaluation_count for ind in self.population)}")

        # 停止可视化
        if enable_visualization:
            self.stop_visualization()

        return self.best_individual, self.convergence_history
