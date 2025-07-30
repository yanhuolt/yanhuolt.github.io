"""
图表生成插件
支持将ECharts配置转换为静态图片文件
"""

import json
import os
import uuid
import base64
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib import rcParams
import seaborn as sns
from datetime import datetime
import logging

# 设置中文字体支持
import platform
import matplotlib.font_manager as fm

def setup_chinese_font():
    """设置中文字体支持 - 增强版"""
    system = platform.system()
    print(f"🔍 检测系统: {system}")

    # 清除matplotlib字体缓存
    try:
        import matplotlib
        cache_dir = matplotlib.get_cachedir()
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.cache')]
            if cache_files:
                print(f"🗑️  清除matplotlib字体缓存: {len(cache_files)}个文件")
                for cache_file in cache_files:
                    try:
                        os.remove(os.path.join(cache_dir, cache_file))
                    except:
                        pass
            # 重新加载字体管理器
            fm._rebuild()
    except Exception as e:
        print(f"⚠️  清除字体缓存失败: {e}")

    # 定义各系统的中文字体优先级列表
    if system == "Windows":
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == "Darwin":  # macOS
        chinese_fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti', 'STSong']
    else:  # Linux/Ubuntu
        chinese_fonts = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # Google Noto字体
            'Source Han Sans CN',   # 思源黑体
            'AR PL UKai CN',        # 文鼎楷体
            'AR PL UMing CN',       # 文鼎明体
            'DejaVu Sans',          # 备用字体
            'Liberation Sans'       # 备用字体
        ]

    # 获取所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"📊 系统共有 {len(available_fonts)} 个字体")

    # 查找中文字体
    found_chinese_fonts = []
    for font in chinese_fonts:
        if font in available_fonts:
            found_chinese_fonts.append(font)
            print(f"✅ 找到中文字体: {font}")

    if not found_chinese_fonts:
        print("❌ 未找到任何中文字体!")
        print("🔍 可用字体示例:")
        for font in available_fonts[:10]:
            print(f"  - {font}")

        if system == "Linux":
            print("\n💡 Ubuntu系统安装中文字体:")
            print("sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei")
            print("sudo fc-cache -fv")

        # 使用备用字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return 'DejaVu Sans'

    # 使用第一个找到的中文字体
    selected_font = found_chinese_fonts[0]
    print(f"🎯 选择字体: {selected_font}")

    # 配置matplotlib
    plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

    # 验证字体设置
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试中文', fontsize=12)
        plt.close(fig)
        print(f"✅ 字体设置成功: {selected_font}")
    except Exception as e:
        print(f"⚠️  字体验证失败: {e}")

    return selected_font

# 初始化中文字体
setup_chinese_font()

logger = logging.getLogger(__name__)

class ChartGenerator:
    """图表生成器类"""
    
    def __init__(self, output_dir: str = "/tmp/charts"):
        """
        初始化图表生成器

        Args:
            output_dir: 图片输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 重新设置中文字体支持
        print("🎨 初始化图表生成器...")
        self.chinese_font = setup_chinese_font()

        # 设置matplotlib样式
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                print("⚠️  使用默认matplotlib样式")

        try:
            sns.set_palette("husl")
        except:
            print("⚠️  seaborn调色板设置失败")

        # 设置图表默认样式
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 12
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.edgecolor'] = 'none'

        print(f"✅ 图表生成器初始化完成，使用字体: {self.chinese_font}")
    
    def generate_chart_from_echarts(self, echarts_config: Dict[str, Any], 
                                  chart_id: str = None, 
                                  format: str = 'png',
                                  dpi: int = 300) -> str:
        """
        根据ECharts配置生成图表图片
        
        Args:
            echarts_config: ECharts配置字典
            chart_id: 图表ID，如果为None则自动生成
            format: 图片格式 ('png' 或 'jpeg')
            dpi: 图片分辨率
            
        Returns:
            生成的图片文件路径
        """
        if chart_id is None:
            chart_id = f"chart_{uuid.uuid4().hex[:8]}"
        
        try:
            # 解析ECharts配置
            chart_type = self._detect_chart_type(echarts_config)
            
            # 根据图表类型生成对应的matplotlib图表
            if chart_type == 'line':
                fig_path = self._generate_line_chart(echarts_config, chart_id, format, dpi)
            elif chart_type == 'bar':
                fig_path = self._generate_bar_chart(echarts_config, chart_id, format, dpi)
            elif chart_type == 'pie':
                fig_path = self._generate_pie_chart(echarts_config, chart_id, format, dpi)
            elif chart_type == 'scatter':
                fig_path = self._generate_scatter_chart(echarts_config, chart_id, format, dpi)
            else:
                # 默认生成折线图
                fig_path = self._generate_line_chart(echarts_config, chart_id, format, dpi)
            
            logger.info(f"成功生成图表: {fig_path}")
            return fig_path
            
        except Exception as e:
            logger.error(f"生成图表失败: {str(e)}")
            raise
    
    def _detect_chart_type(self, config: Dict[str, Any]) -> str:
        """检测图表类型"""
        if 'series' in config and len(config['series']) > 0:
            series_type = config['series'][0].get('type', 'line')
            return series_type
        return 'line'
    
    def _generate_line_chart(self, config: Dict[str, Any], chart_id: str,
                           format: str, dpi: int) -> str:
        """生成折线图"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # 确保使用中文字体
        plt.rcParams['font.sans-serif'] = [self.chinese_font] + plt.rcParams['font.sans-serif']

        # 获取数据
        x_data = config.get('xAxis', {}).get('data', [])
        series_list = config.get('series', [])

        # 绘制多条线
        for i, series in enumerate(series_list):
            y_data = series.get('data', [])
            label = series.get('name', f'系列{i+1}')
            ax.plot(x_data, y_data, marker='o', label=label, linewidth=2, markersize=6)

        # 设置标题和标签
        title = config.get('title', {}).get('text', '折线图')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, fontproperties=self.chinese_font)

        x_label = config.get('xAxis', {}).get('name', '时间/类别')
        y_label = config.get('yAxis', {}).get('name', '数值')

        if x_data:
            ax.set_xlabel(x_label, fontsize=12, fontproperties=self.chinese_font)
        ax.set_ylabel(y_label, fontsize=12, fontproperties=self.chinese_font)

        # 设置x轴标签字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.chinese_font)

        # 设置图例
        if len(series_list) > 1:
            legend = ax.legend(loc='best', frameon=True, shadow=True)
            for text in legend.get_texts():
                text.set_fontproperties(self.chinese_font)

        # 美化图表
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 旋转x轴标签以避免重叠
        if len(x_data) > 10:
            plt.xticks(rotation=45)

        plt.tight_layout()

        # 保存图片
        filename = f"{chart_id}.{format}"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return filepath
    
    def _generate_bar_chart(self, config: Dict[str, Any], chart_id: str,
                          format: str, dpi: int) -> str:
        """生成柱状图"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # 确保使用中文字体
        plt.rcParams['font.sans-serif'] = [self.chinese_font] + plt.rcParams['font.sans-serif']

        # 获取数据
        x_data = config.get('xAxis', {}).get('data', [])
        series_list = config.get('series', [])

        # 计算柱子位置
        x_pos = np.arange(len(x_data))
        bar_width = 0.8 / len(series_list) if len(series_list) > 1 else 0.8

        # 绘制多组柱子
        for i, series in enumerate(series_list):
            y_data = series.get('data', [])
            label = series.get('name', f'系列{i+1}')
            offset = (i - len(series_list)/2 + 0.5) * bar_width
            ax.bar(x_pos + offset, y_data, bar_width, label=label, alpha=0.8)

        # 设置标题和标签
        title = config.get('title', {}).get('text', '柱状图')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, fontproperties=self.chinese_font)

        x_label = config.get('xAxis', {}).get('name', '类别')
        y_label = config.get('yAxis', {}).get('name', '数值')

        ax.set_xlabel(x_label, fontsize=12, fontproperties=self.chinese_font)
        ax.set_ylabel(y_label, fontsize=12, fontproperties=self.chinese_font)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_data, fontproperties=self.chinese_font)

        # 设置y轴标签字体
        for label in ax.get_yticklabels():
            label.set_fontproperties(self.chinese_font)

        # 设置图例
        if len(series_list) > 1:
            legend = ax.legend(loc='best', frameon=True, shadow=True)
            for text in legend.get_texts():
                text.set_fontproperties(self.chinese_font)

        # 美化图表
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 旋转x轴标签
        if len(x_data) > 8:
            plt.xticks(rotation=45)

        plt.tight_layout()

        # 保存图片
        filename = f"{chart_id}.{format}"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return filepath
    
    def _generate_pie_chart(self, config: Dict[str, Any], chart_id: str,
                          format: str, dpi: int) -> str:
        """生成饼图"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # 确保使用中文字体
        plt.rcParams['font.sans-serif'] = [self.chinese_font] + plt.rcParams['font.sans-serif']

        # 获取数据
        series = config.get('series', [{}])[0]
        data = series.get('data', [])

        # 提取标签和数值
        labels = []
        values = []
        for item in data:
            if isinstance(item, dict):
                labels.append(item.get('name', ''))
                values.append(item.get('value', 0))
            else:
                labels.append(str(item))
                values.append(item)

        # 绘制饼图
        colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90,
                                         explode=[0.05] * len(values))

        # 设置标签字体
        for text in texts:
            text.set_fontproperties(self.chinese_font)
            text.set_fontsize(11)

        # 美化百分比文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        # 设置标题
        title = config.get('title', {}).get('text', '饼图')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, fontproperties=self.chinese_font)

        plt.tight_layout()

        # 保存图片
        filename = f"{chart_id}.{format}"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return filepath
    
    def _generate_scatter_chart(self, config: Dict[str, Any], chart_id: str,
                              format: str, dpi: int) -> str:
        """生成散点图"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # 确保使用中文字体
        plt.rcParams['font.sans-serif'] = [self.chinese_font] + plt.rcParams['font.sans-serif']

        # 获取数据
        series_list = config.get('series', [])

        # 绘制散点
        for i, series in enumerate(series_list):
            data = series.get('data', [])
            label = series.get('name', f'系列{i+1}')

            # 提取x, y坐标
            x_vals = []
            y_vals = []
            for point in data:
                if isinstance(point, list) and len(point) >= 2:
                    x_vals.append(point[0])
                    y_vals.append(point[1])

            ax.scatter(x_vals, y_vals, label=label, alpha=0.7, s=60)

        # 设置标题和标签
        title = config.get('title', {}).get('text', '散点图')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, fontproperties=self.chinese_font)

        x_label = config.get('xAxis', {}).get('name', 'X轴')
        y_label = config.get('yAxis', {}).get('name', 'Y轴')

        ax.set_xlabel(x_label, fontsize=12, fontproperties=self.chinese_font)
        ax.set_ylabel(y_label, fontsize=12, fontproperties=self.chinese_font)

        # 设置坐标轴标签字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.chinese_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(self.chinese_font)

        # 设置图例
        if len(series_list) > 1:
            legend = ax.legend(loc='best', frameon=True, shadow=True)
            for text in legend.get_texts():
                text.set_fontproperties(self.chinese_font)

        # 美化图表
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # 保存图片
        filename = f"{chart_id}.{format}"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return filepath

# 全局图表生成器实例
chart_generator = ChartGenerator()

def generate_chart_image(echarts_config: Dict[str, Any], 
                        chart_id: str = None,
                        format: str = 'png',
                        dpi: int = 300) -> str:
    """
    便捷函数：生成图表图片
    
    Args:
        echarts_config: ECharts配置
        chart_id: 图表ID
        format: 图片格式
        dpi: 分辨率
        
    Returns:
        图片文件路径
    """
    return chart_generator.generate_chart_from_echarts(
        echarts_config, chart_id, format, dpi
    )
