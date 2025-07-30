"""
Word文档生成器
支持图片插入的增强版markdown转word功能
"""

import os
import re
import uuid
import logging
from typing import Dict, List, Optional
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import markdown
from bs4 import BeautifulSoup
from markdown_processor import MarkdownProcessor

logger = logging.getLogger(__name__)

class WordGenerator:
    """Word文档生成器，支持图片插入"""
    
    def __init__(self, output_dir: str = "/var/www/files"):
        """
        初始化Word生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.markdown_processor = MarkdownProcessor()
    
    def convert_markdown_to_docx(self, markdown_content: str, 
                                chart_configs: Dict[str, Dict] = None,
                                filename: str = None) -> str:
        """
        将markdown内容转换为Word文档，支持图片插入
        
        Args:
            markdown_content: markdown内容
            chart_configs: 图表配置字典
            filename: 输出文件名，如果为None则自动生成
            
        Returns:
            生成的Word文档路径
        """
        if filename is None:
            filename = f"doc_{uuid.uuid4().hex[:8]}.docx"
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # 处理markdown中的图表标记
            processed_md, image_paths = self.markdown_processor.process_markdown_with_charts(
                markdown_content, chart_configs
            )
            
            # 创建Word文档
            doc = Document()
            
            # 设置文档样式
            self._setup_document_styles(doc)
            
            # 解析markdown为HTML
            html_content = markdown.markdown(processed_md, extensions=['tables', 'fenced_code'])
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 转换HTML元素为Word元素
            self._convert_html_to_docx(doc, soup, image_paths)
            
            # 保存文档
            doc.save(output_path)
            
            # 清理临时图片
            self.markdown_processor.clean_temp_images()
            
            logger.info(f"成功生成Word文档: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"生成Word文档失败: {str(e)}")
            raise
    
    def _setup_document_styles(self, doc: Document):
        """设置文档样式"""
        # 设置默认字体
        style = doc.styles['Normal']
        font = style.font
        font.name = '宋体'
        font.size = Pt(12)

        # 设置段落格式
        paragraph_format = style.paragraph_format
        paragraph_format.space_after = Pt(6)  # 段后间距
        paragraph_format.line_spacing = 1.15  # 行间距

        # 设置标题样式
        for i in range(1, 7):  # 支持更多级别的标题
            try:
                heading_style = doc.styles[f'Heading {i}']
                heading_font = heading_style.font
                heading_font.name = '黑体'
                heading_font.size = Pt(18 - i * 2)  # 调整字号
                heading_font.bold = True

                # 设置标题段落格式
                heading_paragraph = heading_style.paragraph_format
                heading_paragraph.space_before = Pt(12)
                heading_paragraph.space_after = Pt(6)
                heading_paragraph.keep_with_next = True
            except KeyError:
                # 如果样式不存在，跳过
                continue
    
    def _convert_html_to_docx(self, doc: Document, soup: BeautifulSoup, 
                            image_paths: Dict[str, str]):
        """将HTML内容转换为Word文档元素"""
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table', 'img', 'ul', 'ol']):
            if element.name.startswith('h'):
                # 处理标题
                level = int(element.name[1])
                heading = doc.add_heading(element.get_text().strip(), level=level)
                heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
            elif element.name == 'p':
                # 处理段落
                paragraph = doc.add_paragraph()
                self._process_paragraph_content(paragraph, element)
                
            elif element.name == 'table':
                # 处理表格
                self._add_table_to_doc(doc, element)
                
            elif element.name == 'img':
                # 处理图片
                self._add_image_to_doc(doc, element, image_paths)
                
            elif element.name in ['ul', 'ol']:
                # 处理列表
                self._add_list_to_doc(doc, element)
    
    def _process_paragraph_content(self, paragraph, p_element):
        """处理段落内容，支持粗体、斜体等格式"""
        # 设置段落基本格式
        paragraph.paragraph_format.space_after = Pt(6)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.15
        paragraph.paragraph_format.first_line_indent = Pt(0)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

        # 如果段落为空，添加一个空的run以确保格式一致
        if not p_element.contents:
            run = paragraph.add_run("")
            self._set_run_font(run, is_normal=True)
            return

        for content in p_element.contents:
            if hasattr(content, 'name'):
                text = content.get_text()
                if not text.strip():
                    continue

                if content.name == 'strong' or content.name == 'b':
                    run = paragraph.add_run(text)
                    self._set_run_font(run, is_bold=True)
                elif content.name == 'em' or content.name == 'i':
                    run = paragraph.add_run(text)
                    self._set_run_font(run, is_italic=True)
                elif content.name == 'code':
                    run = paragraph.add_run(text)
                    self._set_run_font(run, is_code=True)
                else:
                    run = paragraph.add_run(text)
                    self._set_run_font(run, is_normal=True)
            else:
                # 纯文本内容
                text = str(content).strip()
                if text:
                    run = paragraph.add_run(text)
                    self._set_run_font(run, is_normal=True)

    def _set_run_font(self, run, is_bold=False, is_italic=False, is_code=False, is_normal=False):
        """统一设置文本运行的字体格式"""
        if is_code:
            run.font.name = 'Courier New'
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(0x2F, 0x4F, 0x4F)  # 深灰色
        else:
            run.font.name = '宋体'
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)  # 黑色

        if is_bold:
            run.bold = True
        if is_italic:
            run.italic = True
    
    def _add_table_to_doc(self, doc: Document, table_element):
        """添加表格到文档"""
        rows = table_element.find_all('tr')
        if not rows:
            return

        # 获取列数
        first_row = rows[0]
        cols = len(first_row.find_all(['th', 'td']))

        # 创建表格
        table = doc.add_table(rows=len(rows), cols=cols)
        table.style = 'Table Grid'
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # 填充表格内容
        for i, row in enumerate(rows):
            cells = row.find_all(['th', 'td'])
            for j, cell in enumerate(cells):
                if j < len(table.rows[i].cells):
                    table_cell = table.rows[i].cells[j]
                    cell_text = cell.get_text().strip()

                    # 清空单元格默认内容
                    table_cell.text = ""

                    # 添加段落和文本
                    paragraph = table_cell.paragraphs[0]
                    run = paragraph.add_run(cell_text)

                    # 设置字体
                    run.font.name = '宋体'
                    run.font.size = Pt(11)
                    run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)

                    # 表头加粗
                    if cell.name == 'th':
                        run.bold = True
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    else:
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

                    # 设置段落格式
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.line_spacing = 1.0
    
    def _add_image_to_doc(self, doc: Document, img_element, image_paths: Dict[str, str]):
        """添加图片到文档"""
        src = img_element.get('src', '')
        alt_text = img_element.get('alt', '图片')
        
        # 处理file://路径
        if src.startswith('file://'):
            image_path = src[7:]  # 移除file://前缀
            
            if os.path.exists(image_path):
                try:
                    # 添加图片标题
                    caption_paragraph = doc.add_paragraph()
                    caption_run = caption_paragraph.add_run(alt_text)
                    caption_run.bold = True
                    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    # 添加图片
                    paragraph = doc.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
                    
                    # 设置图片大小（最大宽度6英寸）
                    run.add_picture(image_path, width=Inches(6))
                    
                    logger.info(f"成功插入图片: {image_path}")
                    
                except Exception as e:
                    logger.error(f"插入图片失败 {image_path}: {str(e)}")
                    # 如果图片插入失败，添加文本说明
                    doc.add_paragraph(f"[图片: {alt_text}]")
            else:
                logger.warning(f"图片文件不存在: {image_path}")
                doc.add_paragraph(f"[图片: {alt_text} - 文件不存在]")
        else:
            # 其他类型的图片引用
            doc.add_paragraph(f"[图片: {alt_text}]")
    
    def _add_list_to_doc(self, doc: Document, list_element):
        """添加列表到文档"""
        list_items = list_element.find_all('li')

        for item in list_items:
            paragraph = doc.add_paragraph()

            # 设置列表样式
            if list_element.name == 'ul':
                # 无序列表
                paragraph.style = 'List Bullet'
            else:
                # 有序列表
                paragraph.style = 'List Number'

            # 添加文本并设置字体
            run = paragraph.add_run(item.get_text().strip())
            run.font.name = '宋体'
            run.font.size = Pt(12)

            # 设置段落格式
            paragraph.paragraph_format.space_after = Pt(3)
            paragraph.paragraph_format.line_spacing = 1.15

def convert_markdown_to_word_with_charts(markdown_content: str,
                                       chart_configs: Dict[str, Dict] = None,
                                       output_dir: str = "/var/www/files",
                                       filename: str = None) -> str:
    """
    便捷函数：将包含图表的markdown转换为Word文档
    
    Args:
        markdown_content: markdown内容
        chart_configs: 图表配置字典
        output_dir: 输出目录
        filename: 文件名
        
    Returns:
        生成的Word文档路径
    """
    generator = WordGenerator(output_dir)
    return generator.convert_markdown_to_docx(markdown_content, chart_configs, filename)

# 测试代码
if __name__ == "__main__":
    test_markdown = """
# 垃圾焚烧企业数据分析报告

## 1. 数据概览

| 指标 | 数值 | 单位 |
|------|------|------|
| 处理量 | 1200 | 吨 |
| 运行时间 | 720 | 小时 |
| 平均温度 | 865 | ℃ |

## 2. 温度趋势分析

![炉膛温度变化趋势](chart_temp)

温度控制在**正常范围**内，符合环保要求。

## 3. 结论

- 设备运行稳定
- 环保指标达标
- 建议继续保持现有操作水平
"""
    
    chart_configs = {
        "chart_temp": {
            "title": { "text": "炉膛温度变化趋势" },
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
    }
    
    # 生成Word文档
    output_path = convert_markdown_to_word_with_charts(
        test_markdown, 
        chart_configs,
        output_dir="/tmp",
        filename="test_report.docx"
    )
    
    print(f"生成的Word文档: {output_path}")
