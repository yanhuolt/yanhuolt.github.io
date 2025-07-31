"""
可视化工具模块
解决中文字体显示问题，提供专业的空气质量预报可视化功能
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_font():
    """
    设置中文字体显示
    """
    # 尝试多种中文字体
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',          # 宋体
        'KaiTi',           # 楷体
        'FangSong',        # 仿宋
        'DejaVu Sans',     # 备用字体
        'Arial Unicode MS' # Mac系统字体
    ]
    
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 选择第一个可用的中文字体
    selected_font = 'DejaVu Sans'  # 默认字体
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 配置matplotlib
    plt.rcParams['font.sans-serif'] = [selected_font]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    print(f"使用字体: {selected_font}")
    return selected_font

def plot_forecast_results(system_output, current_data, save_path=None):
    """
    绘制预报结果综合图表
    参数:
        system_output: 系统输出结果
        current_data: 当前观测数据
        save_path: 保存路径（可选）
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 预报时间序列
    predictions = system_output['predictions']
    recent_pm25 = current_data['PM2.5'].values[-24:]  # 最近24小时
    recent_hours = list(range(-24, 0))
    pred_hours = list(range(1, len(predictions) + 1))
    
    ax1.plot(recent_hours, recent_pm25, 'b-', label='历史观测', linewidth=2)
    ax1.plot(pred_hours, predictions, 'r-', label='预测值', linewidth=2)
    
    # 置信区间
    if 'confidence_intervals' in system_output:
        ci = system_output['confidence_intervals']
        ax1.fill_between(pred_hours, ci['lower'], ci['upper'], 
                        alpha=0.3, color='red', label='置信区间')
    
    # 污染等级线
    ax1.axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='轻度污染')
    ax1.axhline(y=115, color='red', linestyle='--', alpha=0.7, label='中度污染')
    ax1.axhline(y=150, color='purple', linestyle='--', alpha=0.7, label='重度污染')
    
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('PM2.5 浓度 (μg/m³)')
    ax1.set_title('PM2.5浓度预报')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 预警等级分布
    if 'warnings' in system_output:
        warning_levels = [w['pollution_level'] for w in system_output['warnings']]
        level_counts = [warning_levels.count(i) for i in range(6)]
        level_names = ['优', '良', '轻度', '中度', '重度', '严重']
        colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon']
        
        bars = ax2.bar(level_names, level_counts, color=colors, alpha=0.7)
        ax2.set_xlabel('污染等级')
        ax2.set_ylabel('小时数')
        ax2.set_title('预警等级分布')
        
        # 添加数值标签
        for bar, count in zip(bars, level_counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
    
    # 3. 训练历史
    if 'training_history' in system_output:
        history = system_output['training_history']
        if 'train_losses' in history and 'val_losses' in history:
            epochs = range(len(history['train_losses']))
            ax3.plot(epochs, history['train_losses'], label='训练损失', color='blue')
            ax3.plot(epochs, history['val_losses'], label='验证损失', color='red')
            ax3.set_xlabel('训练轮数')
            ax3.set_ylabel('损失值')
            ax3.set_title('模型训练历史')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # 4. 成因分析饼图
    if system_output.get('pollution_detected', False) and 'cause_analysis' in system_output:
        contributions = system_output['cause_analysis']['comprehensive_assessment']
        factor_names = ['气象因子', '排放因子', '传输因子', '二次生成']
        factor_values = [
            contributions['meteorological'], 
            contributions['emission'],
            contributions['transport'], 
            contributions['secondary']
        ]
        
        # 过滤掉值为0的因子
        non_zero_factors = [(name, value) for name, value in zip(factor_names, factor_values) if value > 0]
        
        if non_zero_factors:
            names, values = zip(*non_zero_factors)
            colors_pie = ['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(names)]
            
            wedges, texts, autotexts = ax4.pie(values, labels=names, autopct='%1.1f%%', 
                                              startangle=90, colors=colors_pie)
            ax4.set_title('污染成因贡献分析')
        else:
            ax4.text(0.5, 0.5, '未检测到污染过程', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('污染成因分析')
    else:
        ax4.text(0.5, 0.5, '未检测到污染过程', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('污染成因分析')
    
    plt.tight_layout()
    
    if save_path:
        # 确保保存目录存在
        import os
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    plt.show()

def plot_simple_forecast(predictions, recent_data, target_col='PM2.5', save_path=None):
    """
    绘制简单的预报图表
    参数:
        predictions: 预测结果
        recent_data: 最近观测数据
        target_col: 目标污染物
        save_path: 保存路径（可选）
    """
    # 设置中文字体
    setup_chinese_font()
    
    plt.figure(figsize=(12, 6))
    
    # 历史数据
    recent_values = recent_data[target_col].values[-24:]  # 最近24小时
    recent_hours = list(range(-24, 0))
    
    # 预测数据
    pred_hours = list(range(1, len(predictions) + 1))
    
    plt.plot(recent_hours, recent_values, 'b-', label='历史观测', linewidth=2, marker='o', markersize=3)
    plt.plot(pred_hours, predictions, 'r-', label='预测值', linewidth=2, marker='s', markersize=3)
    
    # 污染等级参考线
    plt.axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='轻度污染(75)')
    plt.axhline(y=115, color='red', linestyle='--', alpha=0.7, label='中度污染(115)')
    plt.axhline(y=150, color='purple', linestyle='--', alpha=0.7, label='重度污染(150)')
    
    plt.xlabel('时间 (小时)')
    plt.ylabel(f'{target_col} 浓度 (μg/m³)')
    plt.title(f'{target_col}浓度预报')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加垂直分割线
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.text(0, plt.ylim()[1]*0.9, '预报起始', ha='center', va='top', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        # 确保保存目录存在
        import os
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    plt.show()

def test_font_display():
    """
    测试中文字体显示
    """
    setup_chinese_font()
    
    plt.figure(figsize=(8, 6))
    
    # 测试数据
    x = np.arange(5)
    y = [20, 35, 30, 35, 27]
    labels = ['优', '良', '轻度污染', '中度污染', '重度污染']
    colors = ['green', 'yellow', 'orange', 'red', 'purple']
    
    bars = plt.bar(x, y, color=colors, alpha=0.7)
    plt.xticks(x, labels)
    plt.xlabel('空气质量等级')
    plt.ylabel('小时数')
    plt.title('中文字体显示测试')
    
    # 添加数值标签
    for bar, value in zip(bars, y):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(value), ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("中文字体测试完成")

if __name__ == "__main__":
    # 运行字体测试
    test_font_display()
