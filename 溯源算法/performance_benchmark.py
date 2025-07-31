"""
污染源溯源算法性能基准测试
对比原始算法和优化算法的性能表现
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from gaussian_plume_model import GaussianPlumeModel, PollutionSource, MeteoData
from optimized_source_inversion import OptimizedSourceInversion, OptimizedSensorData, AdaptiveGAParameters
from source_inversion import SourceInversion, SensorData

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self):
        self.gaussian_model = GaussianPlumeModel()
        self.results = []
    
    def generate_test_case(self, 
                          n_sensors: int,
                          source_position: Tuple[float, float, float],
                          emission_rate: float,
                          wind_speed: float,
                          wind_direction: float,
                          noise_level: float = 0.05) -> Tuple[PollutionSource, MeteoData, List]:
        """
        生成测试用例
        
        Args:
            n_sensors: 传感器数量
            source_position: 污染源位置 (x, y, z)
            emission_rate: 排放源强
            wind_speed: 风速
            wind_direction: 风向
            noise_level: 噪声水平
            
        Returns:
            (污染源, 气象数据, 传感器数据)
        """
        # 创建污染源
        source = PollutionSource(
            x=source_position[0],
            y=source_position[1],
            z=source_position[2],
            emission_rate=emission_rate
        )
        
        # 创建气象数据
        meteo = MeteoData(
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            temperature=20.0,
            pressure=101325.0,
            humidity=60.0,
            solar_radiation=500.0,
            cloud_cover=0.3
        )
        
        # 生成传感器位置（随机分布）
        sensor_positions = []
        
        # 在污染源周围生成传感器
        for _ in range(n_sensors):
            # 随机距离和角度
            distance = np.random.uniform(50, 500)
            angle = np.random.uniform(0, 2 * np.pi)
            
            x = source.x + distance * np.cos(angle)
            y = source.y + distance * np.sin(angle)
            z = np.random.uniform(1, 10)  # 传感器高度
            
            sensor_positions.append((x, y, z))
        
        # 生成观测数据
        sensor_data = []
        for i, (x, y, z) in enumerate(sensor_positions):
            # 计算理论浓度
            theoretical_conc = self.gaussian_model.calculate_concentration(
                source, x, y, z, meteo
            )
            
            # 添加噪声
            noise = np.random.normal(0, noise_level * theoretical_conc)
            observed_conc = max(0, theoretical_conc + noise)
            
            sensor_data.append({
                'id': f'S{i+1:03d}',
                'x': x,
                'y': y,
                'z': z,
                'concentration': observed_conc,
                'uncertainty': noise_level * theoretical_conc
            })
        
        return source, meteo, sensor_data
    
    def run_original_algorithm(self, sensor_data_dict: List[Dict], meteo: MeteoData, true_source: PollutionSource) -> Dict:
        """运行原始算法"""
        # 转换数据格式
        sensor_data = [
            SensorData(
                sensor_id=s['id'],
                x=s['x'],
                y=s['y'],
                z=s['z'],
                concentration=s['concentration'],
                timestamp="2024-01-01 12:00:00"
            ) for s in sensor_data_dict
        ]
        
        # 创建反算器
        inverter = SourceInversion()
        
        # 执行反算
        start_time = time.time()
        result = inverter.invert_source(sensor_data, meteo, true_source, verbose=False)
        computation_time = time.time() - start_time
        
        return {
            'algorithm': 'Original',
            'computation_time': computation_time,
            'position_error': result.position_error,
            'emission_error': result.emission_error,
            'objective_value': result.objective_value,
            'convergence_generations': len(result.convergence_history),
            'cache_hit_rate': 0.0,
            'evaluations_per_second': 0.0
        }
    
    def run_optimized_algorithm(self, sensor_data_dict: List[Dict], meteo: MeteoData, true_source: PollutionSource) -> Dict:
        """运行优化算法"""
        # 转换数据格式
        sensor_data = [
            OptimizedSensorData(
                sensor_id=s['id'],
                x=s['x'],
                y=s['y'],
                z=s['z'],
                concentration=s['concentration'],
                timestamp="2024-01-01 12:00:00",
                uncertainty=s['uncertainty']
            ) for s in sensor_data_dict
        ]
        
        # 配置优化参数
        params = AdaptiveGAParameters(
            population_size=60,
            max_generations=1000,
            use_parallel=True,
            use_cache=True,
            cache_size=10000
        )
        
        # 创建反算器
        inverter = OptimizedSourceInversion(ga_parameters=params)
        
        # 执行反算
        start_time = time.time()
        result = inverter.invert_source(sensor_data, meteo, true_source, verbose=False)
        computation_time = time.time() - start_time
        
        return {
            'algorithm': 'Optimized',
            'computation_time': computation_time,
            'position_error': result.position_error,
            'emission_error': result.emission_error,
            'objective_value': result.objective_value,
            'convergence_generations': result.performance_metrics.get('convergence_generations', 0),
            'cache_hit_rate': result.performance_metrics.get('cache_hit_rate', 0),
            'evaluations_per_second': result.performance_metrics.get('evaluations_per_second', 0)
        }
    
    def run_single_test(self, test_params: Dict) -> List[Dict]:
        """运行单个测试"""
        print(f"运行测试: {test_params['name']}")
        
        # 生成测试用例
        source, meteo, sensor_data = self.generate_test_case(
            n_sensors=test_params['n_sensors'],
            source_position=test_params['source_position'],
            emission_rate=test_params['emission_rate'],
            wind_speed=test_params['wind_speed'],
            wind_direction=test_params['wind_direction'],
            noise_level=test_params['noise_level']
        )
        
        results = []
        
        # 运行原始算法
        try:
            original_result = self.run_original_algorithm(sensor_data, meteo, source)
            original_result.update({
                'test_name': test_params['name'],
                'n_sensors': test_params['n_sensors'],
                'noise_level': test_params['noise_level']
            })
            results.append(original_result)
        except Exception as e:
            print(f"原始算法测试失败: {e}")
        
        # 运行优化算法
        try:
            optimized_result = self.run_optimized_algorithm(sensor_data, meteo, source)
            optimized_result.update({
                'test_name': test_params['name'],
                'n_sensors': test_params['n_sensors'],
                'noise_level': test_params['noise_level']
            })
            results.append(optimized_result)
        except Exception as e:
            print(f"优化算法测试失败: {e}")
        
        return results
    
    def run_benchmark_suite(self) -> pd.DataFrame:
        """运行完整的基准测试套件"""
        print("开始性能基准测试...")
        print("="*60)
        
        # 定义测试用例
        test_cases = [
            {
                'name': '小规模测试',
                'n_sensors': 15,
                'source_position': (100, 150, 20),
                'emission_rate': 1.5,
                'wind_speed': 3.0,
                'wind_direction': 180,
                'noise_level': 0.05
            },
            {
                'name': '中等规模测试',
                'n_sensors': 30,
                'source_position': (200, 250, 25),
                'emission_rate': 2.5,
                'wind_speed': 4.0,
                'wind_direction': 225,
                'noise_level': 0.05
            },
            {
                'name': '大规模测试',
                'n_sensors': 50,
                'source_position': (150, 200, 30),
                'emission_rate': 3.0,
                'wind_speed': 5.0,
                'wind_direction': 270,
                'noise_level': 0.05
            },
            {
                'name': '高噪声测试',
                'n_sensors': 25,
                'source_position': (120, 180, 22),
                'emission_rate': 2.0,
                'wind_speed': 3.5,
                'wind_direction': 200,
                'noise_level': 0.15
            },
            {
                'name': '低风速测试',
                'n_sensors': 25,
                'source_position': (180, 220, 25),
                'emission_rate': 2.2,
                'wind_speed': 1.5,
                'wind_direction': 190,
                'noise_level': 0.05
            }
        ]
        
        # 运行所有测试
        all_results = []
        for test_case in test_cases:
            test_results = self.run_single_test(test_case)
            all_results.extend(test_results)
        
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        return df
    
    def analyze_results(self, df: pd.DataFrame) -> None:
        """分析测试结果"""
        print("\n" + "="*60)
        print("性能分析结果")
        print("="*60)
        
        # 创建输出目录
        output_dir = "溯源算法/性能测试结果"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 总体统计
        print("\n1. 总体性能统计:")
        summary = df.groupby('algorithm').agg({
            'computation_time': ['mean', 'std'],
            'position_error': ['mean', 'std'],
            'emission_error': ['mean', 'std'],
            'objective_value': ['mean', 'std']
        }).round(3)
        
        print(summary)
        
        # 2. 性能提升分析
        print("\n2. 性能提升分析:")
        
        # 按测试用例分组比较
        for test_name in df['test_name'].unique():
            test_data = df[df['test_name'] == test_name]
            if len(test_data) == 2:  # 确保有两个算法的结果
                original = test_data[test_data['algorithm'] == 'Original'].iloc[0]
                optimized = test_data[test_data['algorithm'] == 'Optimized'].iloc[0]
                
                time_improvement = (original['computation_time'] - optimized['computation_time']) / original['computation_time'] * 100
                pos_improvement = (original['position_error'] - optimized['position_error']) / original['position_error'] * 100
                emission_improvement = (original['emission_error'] - optimized['emission_error']) / original['emission_error'] * 100
                
                print(f"\n{test_name}:")
                print(f"  时间提升: {time_improvement:.1f}%")
                print(f"  位置精度提升: {pos_improvement:.1f}%")
                print(f"  源强精度提升: {emission_improvement:.1f}%")
        
        # 3. 可视化结果
        self.create_performance_plots(df, output_dir)
        
        # 4. 保存详细结果
        df.to_csv(os.path.join(output_dir, "详细测试结果.csv"), index=False, encoding='utf-8-sig')
        summary.to_csv(os.path.join(output_dir, "性能统计摘要.csv"), encoding='utf-8-sig')
        
        print(f"\n详细结果已保存至: {output_dir}")
    
    def create_performance_plots(self, df: pd.DataFrame, output_dir: str) -> None:
        """创建性能对比图表"""
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
        # 1. 计算时间对比
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 计算时间对比
        sns.barplot(data=df, x='test_name', y='computation_time', hue='algorithm', ax=axes[0, 0])
        axes[0, 0].set_title('计算时间对比', fontweight='bold')
        axes[0, 0].set_ylabel('计算时间 (秒)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 位置误差对比
        sns.barplot(data=df, x='test_name', y='position_error', hue='algorithm', ax=axes[0, 1])
        axes[0, 1].set_title('位置误差对比', fontweight='bold')
        axes[0, 1].set_ylabel('位置误差 (米)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 源强误差对比
        sns.barplot(data=df, x='test_name', y='emission_error', hue='algorithm', ax=axes[1, 0])
        axes[1, 0].set_title('源强误差对比', fontweight='bold')
        axes[1, 0].set_ylabel('源强误差 (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 目标函数值对比
        sns.barplot(data=df, x='test_name', y='objective_value', hue='algorithm', ax=axes[1, 1])
        axes[1, 1].set_title('目标函数值对比', fontweight='bold')
        axes[1, 1].set_ylabel('目标函数值')
        axes[1, 1].set_yscale('log')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "性能对比图表.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 性能提升雷达图
        self.create_improvement_radar_chart(df, output_dir)
    
    def create_improvement_radar_chart(self, df: pd.DataFrame, output_dir: str) -> None:
        """创建性能提升雷达图"""
        import matplotlib.patches as patches
        
        # 计算平均性能提升
        improvements = []
        metrics = ['computation_time', 'position_error', 'emission_error']
        metric_names = ['计算时间', '位置精度', '源强精度']
        
        for metric in metrics:
            original_mean = df[df['algorithm'] == 'Original'][metric].mean()
            optimized_mean = df[df['algorithm'] == 'Optimized'][metric].mean()
            
            if original_mean > 0:
                improvement = (original_mean - optimized_mean) / original_mean * 100
                improvements.append(max(0, improvement))  # 只显示正向提升
            else:
                improvements.append(0)
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # 角度设置
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        improvements += improvements[:1]  # 闭合数据
        
        # 绘制雷达图
        ax.plot(angles, improvements, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, improvements, alpha=0.25, color='blue')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, max(improvements) * 1.1)
        
        # 添加网格
        ax.grid(True)
        
        plt.title('算法优化性能提升雷达图', size=16, fontweight='bold', pad=20)
        plt.savefig(os.path.join(output_dir, "性能提升雷达图.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    print("污染源溯源算法性能基准测试")
    print("="*60)
    
    # 创建基准测试器
    benchmark = PerformanceBenchmark()
    
    try:
        # 运行基准测试
        results_df = benchmark.run_benchmark_suite()
        
        # 分析结果
        benchmark.analyze_results(results_df)
        
        print("\n基准测试完成！")
        
    except Exception as e:
        print(f"基准测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
