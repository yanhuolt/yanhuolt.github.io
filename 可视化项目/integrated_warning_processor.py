"""
集成预警处理器
结合吸附曲线分析和活性炭更换预警功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入现有算法和预警系统
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from activated_carbon_warning_system import (
    ActivatedCarbonWarningSystem, 
    WarningLevel, 
    WarningEvent
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class IntegratedWarningProcessor:
    """集成预警处理器"""
    
    def __init__(self, switch_interval: int = 8):
        """
        初始化处理器
        
        参数:
            switch_interval: 进出口切换间隔（秒）
        """
        self.switch_interval = switch_interval
        self.warning_system = ActivatedCarbonWarningSystem()
        
        # 数据存储
        self.raw_data = None
        self.cleaned_data = None
        self.efficiency_data = None
        self.warning_results = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据文件"""
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("不支持的文件格式")
                
            print(f"成功加载数据: {len(data)} 行")
            return data
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return pd.DataFrame()
    
    def clean_data(self, data: pd.DataFrame, wind_speed_threshold: float = 0.5) -> pd.DataFrame:
        """数据清洗"""
        print("开始数据清洗...")
        
        # 基本清洗
        cleaned = data.copy()
        
        # 1. 移除空值
        cleaned = cleaned.dropna()
        
        # 2. 移除风速过低的数据
        if 'wind_speed' in cleaned.columns:
            cleaned = cleaned[cleaned['wind_speed'] >= wind_speed_threshold]
        
        # 3. 移除浓度为0的数据
        if 'concentration' in cleaned.columns:
            cleaned = cleaned[cleaned['concentration'] > 0]
        
        # 4. 移除异常值（使用IQR方法）
        if 'concentration' in cleaned.columns:
            Q1 = cleaned['concentration'].quantile(0.25)
            Q3 = cleaned['concentration'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            cleaned = cleaned[
                (cleaned['concentration'] >= lower_bound) & 
                (cleaned['concentration'] <= upper_bound)
            ]
        
        print(f"清洗后数据: {len(cleaned)} 行 (移除了 {len(data) - len(cleaned)} 行)")
        return cleaned
    
    def calculate_efficiency(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算吸附效率"""
        print("计算吸附效率...")
        
        # 分离进出口数据
        inlet_data = data[data['inlet_outlet'] == 0].sort_values('time')
        outlet_data = data[data['inlet_outlet'] == 1].sort_values('time')
        
        efficiency_records = []
        
        for _, outlet_row in outlet_data.iterrows():
            outlet_time = outlet_row['time']
            c1 = outlet_row['concentration']
            
            # 查找对应的进口浓度
            c0 = self._find_corresponding_inlet(inlet_data, outlet_time, c1)
            
            if c0 is not None and c0 > 0:
                efficiency = (1 - c1 / c0) * 100
                efficiency_records.append({
                    'time': outlet_time,
                    'c0': c0,
                    'c1': c1,
                    'efficiency': efficiency
                })
        
        result = pd.DataFrame(efficiency_records)
        print(f"计算得到 {len(result)} 个效率数据点")
        return result
    
    def _find_corresponding_inlet(self, inlet_data: pd.DataFrame, 
                                outlet_time: float, outlet_conc: float) -> Optional[float]:
        """查找对应的进口浓度"""
        if len(inlet_data) == 0:
            return None
        
        # 查找时间最接近的进口数据
        time_diff = np.abs(inlet_data['time'] - outlet_time)
        
        if len(time_diff) > 0:
            closest_idx = time_diff.idxmin()
            
            if time_diff.loc[closest_idx] <= self.switch_interval:
                inlet_conc = inlet_data.loc[closest_idx, 'concentration']
                
                # 确保进口浓度 >= 出口浓度
                if inlet_conc >= outlet_conc:
                    return inlet_conc
                else:
                    # 查找上一个大于出口浓度的进口数据
                    valid_inlet = inlet_data[
                        (inlet_data['concentration'] >= outlet_conc) & 
                        (inlet_data['time'] <= outlet_time)
                    ]
                    
                    if len(valid_inlet) > 0:
                        return valid_inlet.iloc[-1]['concentration']
        
        return None
    
    def analyze_with_warning(self, efficiency_data: pd.DataFrame) -> Dict:
        """结合预警系统分析数据"""
        print("执行预警分析...")
        
        if len(efficiency_data) == 0:
            return {"warning_events": [], "current_status": WarningLevel.GREEN}
        
        # 使用预警系统分析
        warning_results = self.warning_system.analyze_data(efficiency_data)
        
        print(f"当前预警状态: {warning_results['current_status'].value}")
        print(f"生成预警事件: {len(warning_results['warning_events'])} 个")
        
        return warning_results
    
    def create_warning_visualization(self, efficiency_data: pd.DataFrame, 
                                   warning_results: Dict) -> plt.Figure:
        """创建包含预警信息的可视化图表"""
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('活性炭吸附效率分析与预警系统', fontsize=16, fontweight='bold')
        
        # 获取数据
        data_with_bt = warning_results.get('data_with_breakthrough', efficiency_data)
        
        # 1. 吸附效率趋势图
        ax1 = axes[0, 0]
        if len(efficiency_data) > 0:
            ax1.plot(efficiency_data['time'], efficiency_data['efficiency'], 
                    'b-', linewidth=2, label='吸附效率')
            ax1.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='效率警戒线(80%)')
            ax1.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='效率危险线(60%)')
        
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('吸附效率 (%)')
        ax1.set_title('吸附效率变化趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 穿透率趋势图
        ax2 = axes[0, 1]
        if 'breakthrough_ratio' in data_with_bt.columns:
            breakthrough_percent = data_with_bt['breakthrough_ratio'] * 100
            ax2.plot(data_with_bt['time'], breakthrough_percent, 
                    'r-', linewidth=2, label='穿透率')
            
            # 添加预警阈值线
            ax2.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='穿透起始点(5%)')
            ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='预警点(80%)')
            ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='饱和点(95%)')
            
            # 如果有Logistic模型拟合结果，绘制拟合曲线
            if warning_results.get('model_fitted', False):
                model = self.warning_system.logistic_model
                time_smooth = np.linspace(data_with_bt['time'].min(), 
                                        data_with_bt['time'].max(), 200)
                bt_smooth = model.predict(time_smooth) * 100
                ax2.plot(time_smooth, bt_smooth, 'g--', linewidth=2, 
                        alpha=0.8, label='Logistic拟合曲线')
        
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('穿透率 (%)')
        ax2.set_title('穿透率变化趋势')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 预警状态分布
        ax3 = axes[1, 0]
        warning_events = warning_results.get('warning_events', [])
        
        if warning_events:
            # 统计各预警等级的数量
            warning_counts = {}
            for event in warning_events:
                level = event.warning_level.value
                warning_counts[level] = warning_counts.get(level, 0) + 1
            
            # 创建饼图
            colors = {'绿色': 'green', '黄色': 'yellow', '橙色': 'orange', '红色': 'red'}
            pie_colors = [colors.get(level, 'gray') for level in warning_counts.keys()]
            
            ax3.pie(warning_counts.values(), labels=warning_counts.keys(), 
                   colors=pie_colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('预警等级分布')
        else:
            ax3.text(0.5, 0.5, '暂无预警事件', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('预警等级分布')
        
        # 4. 预警时间线
        ax4 = axes[1, 1]
        if warning_events:
            # 按时间排序预警事件
            sorted_events = sorted(warning_events, key=lambda x: float(x.timestamp))
            
            times = [float(event.timestamp) for event in sorted_events]
            levels = [event.warning_level.value for event in sorted_events]
            
            # 创建颜色映射
            level_colors = {'绿色': 'green', '黄色': 'yellow', '橙色': 'orange', '红色': 'red'}
            colors = [level_colors.get(level, 'gray') for level in levels]
            
            ax4.scatter(times, range(len(times)), c=colors, s=100, alpha=0.7)
            
            # 设置y轴标签
            ax4.set_yticks(range(len(times)))
            ax4.set_yticklabels([f"事件{i+1}" for i in range(len(times))])
            
        ax4.set_xlabel('时间 (s)')
        ax4.set_title('预警事件时间线')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_warning_report(self, warning_results: Dict) -> str:
        """生成预警报告"""
        report = []
        report.append("=" * 60)
        report.append("活性炭更换预警报告")
        report.append("=" * 60)
        
        # 当前状态
        current_status = warning_results.get('current_status', WarningLevel.GREEN)
        report.append(f"\n当前预警状态: {current_status.value}")
        
        # 预警事件
        warning_events = warning_results.get('warning_events', [])
        if warning_events:
            report.append(f"\n预警事件总数: {len(warning_events)}")
            report.append("\n详细预警信息:")
            report.append("-" * 40)
            
            for i, event in enumerate(warning_events[-5:], 1):  # 显示最近5个事件
                report.append(f"\n事件 {i}:")
                report.append(f"  时间: {event.timestamp}")
                report.append(f"  预警等级: {event.warning_level.value}")
                report.append(f"  穿透率: {event.breakthrough_ratio:.1f}%")
                report.append(f"  吸附效率: {event.efficiency:.1f}%")
                report.append(f"  原因: {event.reason}")
                report.append(f"  建议: {event.recommendation}")
                if event.predicted_saturation_time:
                    report.append(f"  预测: {event.predicted_saturation_time}")
        else:
            report.append("\n✅ 暂无预警事件，设备运行正常")
        
        # 模型信息
        if warning_results.get('model_fitted', False):
            report.append(f"\n📊 Logistic模型拟合: 成功")
            params = warning_results.get('logistic_params')
            if params is not None:
                report.append(f"  模型参数: A={params[0]:.3f}, k={params[1]:.6f}, t0={params[2]:.1f}")
        else:
            report.append(f"\n📊 Logistic模型拟合: 失败（数据不足或质量较差）")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def process_complete_workflow(self, file_path: str, 
                                wind_speed_threshold: float = 0.5) -> Dict:
        """完整的处理工作流程"""
        print("开始完整的预警分析工作流程...")
        
        # 1. 加载数据
        self.raw_data = self.load_data(file_path)
        if len(self.raw_data) == 0:
            return {"error": "数据加载失败"}
        
        # 2. 数据清洗
        self.cleaned_data = self.clean_data(self.raw_data, wind_speed_threshold)
        if len(self.cleaned_data) == 0:
            return {"error": "数据清洗后无有效数据"}
        
        # 3. 计算效率
        self.efficiency_data = self.calculate_efficiency(self.cleaned_data)
        if len(self.efficiency_data) == 0:
            return {"error": "无法计算吸附效率"}
        
        # 4. 预警分析
        self.warning_results = self.analyze_with_warning(self.efficiency_data)
        
        # 5. 创建可视化
        fig = self.create_warning_visualization(self.efficiency_data, self.warning_results)
        
        # 6. 生成报告
        report = self.generate_warning_report(self.warning_results)
        
        return {
            "raw_data": self.raw_data,
            "cleaned_data": self.cleaned_data,
            "efficiency_data": self.efficiency_data,
            "warning_results": self.warning_results,
            "visualization": fig,
            "report": report
        }
