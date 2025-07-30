"""
Markdown处理模块
支持图表标记解析和图片插入
"""

import re
import json
import os
import uuid
import logging
from typing import Dict, List, Tuple, Any
from chart_generator import generate_chart_image

logger = logging.getLogger(__name__)

class MarkdownProcessor:
    """Markdown处理器，支持图表标记解析"""
    
    def __init__(self, chart_output_dir: str = "/tmp/charts"):
        """
        初始化处理器
        
        Args:
            chart_output_dir: 图表输出目录
        """
        self.chart_output_dir = chart_output_dir
        os.makedirs(chart_output_dir, exist_ok=True)
        
        # 图表标记正则表达式
        # 匹配格式: ![图表名称](chart_id) 或 ![图表名称](echarts:配置JSON)
        self.chart_pattern = re.compile(r'!\[([^\]]*)\]\((chart_[^)]+|echarts:[^)]+)\)')
        
        # 存储生成的图片信息
        self.generated_images = {}
    
    def process_markdown_with_charts(self, markdown_content: str, 
                                   chart_configs: Dict[str, Dict[str, Any]] = None) -> Tuple[str, Dict[str, str]]:
        """
        处理包含图表标记的markdown内容
        
        Args:
            markdown_content: 原始markdown内容
            chart_configs: 图表配置字典 {chart_id: echarts_config}
            
        Returns:
            (处理后的markdown内容, 图片路径映射字典)
        """
        if chart_configs is None:
            chart_configs = {}
        
        # 查找所有图表标记
        chart_matches = self.chart_pattern.findall(markdown_content)
        image_paths = {}
        
        for chart_title, chart_ref in chart_matches:
            try:
                if chart_ref.startswith('echarts:'):
                    # 直接从标记中解析ECharts配置
                    config_json = chart_ref[8:]  # 移除 'echarts:' 前缀
                    echarts_config = json.loads(config_json)
                    chart_id = f"chart_{uuid.uuid4().hex[:8]}"
                elif chart_ref.startswith('chart_'):
                    # 从配置字典中获取配置
                    chart_id = chart_ref
                    if chart_id not in chart_configs:
                        logger.warning(f"未找到图表配置: {chart_id}")
                        continue
                    echarts_config = chart_configs[chart_id]
                else:
                    logger.warning(f"无效的图表引用: {chart_ref}")
                    continue
                
                # 生成图片
                image_path = generate_chart_image(echarts_config, chart_id)
                image_paths[chart_id] = image_path
                
                # 替换markdown中的图表标记为图片标记
                old_pattern = f'![{chart_title}]({chart_ref})'
                new_pattern = f'![{chart_title}](file://{image_path})'
                markdown_content = markdown_content.replace(old_pattern, new_pattern)
                
                logger.info(f"成功处理图表: {chart_id} -> {image_path}")
                
            except Exception as e:
                logger.error(f"处理图表失败 {chart_ref}: {str(e)}")
                continue
        
        self.generated_images = image_paths
        return markdown_content, image_paths
    
    def extract_chart_configs_from_markdown(self, markdown_content: str) -> Dict[str, Dict[str, Any]]:
        """
        从markdown内容中提取ECharts配置
        支持代码块格式的图表配置
        
        Args:
            markdown_content: markdown内容
            
        Returns:
            图表配置字典
        """
        chart_configs = {}
        
        # 匹配echarts代码块
        echarts_pattern = re.compile(r'```echarts\s*\n(.*?)\n```', re.DOTALL)
        matches = echarts_pattern.findall(markdown_content)
        
        for i, config_str in enumerate(matches):
            try:
                config = json.loads(config_str.strip())
                chart_id = f"chart_{i+1}"
                chart_configs[chart_id] = config
                
                # 替换代码块为图表标记
                old_block = f'```echarts\n{config_str}\n```'
                chart_title = config.get('title', {}).get('text', f'图表{i+1}')
                new_mark = f'![{chart_title}](chart_{i+1})'
                markdown_content = markdown_content.replace(old_block, new_mark)
                
            except json.JSONDecodeError as e:
                logger.error(f"解析ECharts配置失败: {str(e)}")
                continue
        
        return chart_configs
    
    def clean_temp_images(self):
        """清理临时生成的图片文件"""
        for chart_id, image_path in self.generated_images.items():
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"清理临时图片: {image_path}")
            except Exception as e:
                logger.error(f"清理图片失败 {image_path}: {str(e)}")
        
        self.generated_images.clear()

def process_markdown_for_charts(markdown_content: str, 
                              chart_configs: Dict[str, Dict[str, Any]] = None) -> Tuple[str, Dict[str, str]]:
    """
    便捷函数：处理包含图表的markdown内容
    
    Args:
        markdown_content: 原始markdown内容
        chart_configs: 图表配置字典
        
    Returns:
        (处理后的markdown内容, 图片路径映射)
    """
    processor = MarkdownProcessor()
    
    # 如果没有提供配置，尝试从markdown中提取
    if chart_configs is None:
        chart_configs = processor.extract_chart_configs_from_markdown(markdown_content)
    
    return processor.process_markdown_with_charts(markdown_content, chart_configs)

# 示例使用
if __name__ == "__main__":
    # 测试markdown内容
    test_markdown = """
# 数据分析报告

## 温度趋势分析

下图显示了炉膛温度的变化趋势：

```echarts
{
  "title": { "text": "炉膛温度变化趋势" },
  "tooltip": { "trigger": "axis" },
  "xAxis": { 
    "type": "category", 
    "data": ["1月", "2月", "3月", "4月", "5月", "6月"] 
  },
  "yAxis": { "type": "value" },
  "series": [{
    "type": "line",
    "name": "温度",
    "data": [850, 860, 855, 870, 865, 875]
  }]
}
```

## 污染物排放统计

![污染物排放对比](chart_emission)

## 结论

数据显示温度控制良好。
"""
    
    # 额外的图表配置
    extra_configs = {
        "chart_emission": {
            "title": { "text": "污染物排放对比" },
            "tooltip": { "trigger": "axis" },
            "xAxis": { 
                "type": "category", 
                "data": ["PM", "NOx", "SO2", "HCl", "CO"] 
            },
            "yAxis": { "type": "value" },
            "series": [{
                "type": "bar",
                "name": "排放量",
                "data": [15, 25, 18, 12, 30]
            }]
        }
    }
    
    # 处理markdown
    processed_md, image_paths = process_markdown_for_charts(test_markdown, extra_configs)
    
    print("处理后的Markdown:")
    print(processed_md)
    print("\n生成的图片:")
    for chart_id, path in image_paths.items():
        print(f"{chart_id}: {path}")
