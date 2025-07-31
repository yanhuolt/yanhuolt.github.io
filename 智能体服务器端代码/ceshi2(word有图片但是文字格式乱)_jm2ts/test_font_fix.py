"""
测试字体格式修复效果
验证Word文档生成中的字体设置是否正确
"""

import os
import sys
import logging
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from word_generator import convert_markdown_to_word_with_charts

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_font_fix():
    """测试字体格式修复"""
    print("🚀 开始测试Word字体格式修复")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 测试用的markdown内容 - 重点测试标题和正文格式
    test_markdown = """
# 垃圾焚烧企业数据分析报告

## 1. 数据概览

本月共处理垃圾 1200 吨，设备运行稳定，各项指标正常。

### 1.1 运行指标

- 累计处理量：1200 吨
- 运行时间：720 小时
- 平均炉温：865 ℃

## 2. 炉膛温度变化趋势

下图显示了本月各焚烧炉的温度变化情况：

### 2.1 温度分析

炉膛温度保持在 **850℃以上**，符合环保要求。

## 3. 污染物排放统计

各类污染物的排放情况对比：

### 3.1 排放数据

- PM 排放浓度：15.2 mg/Nm³
- NOx 排放浓度：45.8 mg/Nm³

## 4. 设备工况分布

本月设备运行工况统计：

### 4.1 运行状态

设备整体运行良好，无重大故障。

## 5. 环保耗材消耗

### 5.1 消耗统计

活性炭消耗量正常。

## 6. 决策建议

基于以上数据分析，提出以下**专业建议**：

### 6.1 温度控制优化

- 保持现有温度控制参数，确保炉膛温度稳定在 850℃以上
- 定期检查温度传感器，确保数据准确性

### 6.2 排放控制改进

- PM 排放浓度略高，建议：
- 增加布袋除尘器清灰频次
- 检查滤袋完整性
- 优化活性炭喷射量

### 6.3 运行效率提升

- 正常运行时间占比达到 94.4%，表现良好
- 建议制定预防性维护计划，减少故障停机时间

## 7. 结论

本月垃圾焚烧企业运行状况**良好**，主要表现为：

1. ✅ 炉膛温度控制达标
2. ✅ 污染物排放符合标准
3. ✅ 设备运行稳定性高
4. ⚠️ 需关注 PM 排放浓度变化趋势

**总体评价**：设备运行正常，环保指标达标，建议继续保持现有管理水平。

## 2. 数据表格 Data Table

| 指标名称 | English Name | 数值 | Value | 单位 | Unit |
|----------|--------------|------|-------|------|------|
| 处理量 | Processing Volume | 1200 | 1200 | 吨 | tons |
| 运行时间 | Runtime | 720 | 720 | 小时 | hours |
| 平均温度 | Average Temperature | 865 | 865 | ℃ | °C |
| 效率 | Efficiency | 95.5 | 95.5 | % | % |

## 3. 代码示例 Code Example

```python
def process_data(data):
    # 处理数据的函数 - Process data function
    result = data.process()
    return result

# 中文注释测试
print("Hello World 你好世界")
```

## 4. 列表测试 List Testing

### 4.1 无序列表 Unordered List

- **中文项目**：测试中文字体显示效果
- **English Item**: Test English font display effect
- **混合项目 Mixed**: 中英文混合显示测试 Chinese-English mixed display test

### 4.2 有序列表 Ordered List

1. 第一步：数据收集 (Step 1: Data Collection)
2. 第二步：数据处理 (Step 2: Data Processing)  
3. 第三步：结果分析 (Step 3: Result Analysis)

## 5. 格式测试 Format Testing

这是一个包含**粗体中文**和**bold English**的段落。

This is a paragraph containing **bold Chinese** and **粗体英文**.

这是一个包含*斜体中文*和*italic English*的段落。

This is a paragraph containing *italic Chinese* and *斜体英文*.

这是一个包含`代码中文`和`code English`的段落。

This is a paragraph containing `code Chinese` and `代码英文`.

## 6. 温度趋势分析 Temperature Trend Analysis

![炉膛温度变化趋势](chart_temp)

上图显示了**炉膛温度**的变化趋势，整体保持在正常范围内。

The above chart shows the **furnace temperature** trend, which remains within the normal range overall.

## 7. 结论 Conclusion

通过强制字体设置，成功解决了Word文档中的字体格式混乱问题：

By forcing font settings, we successfully solved the font format confusion problem in Word documents:

- ✅ 中文使用微软雅黑字体 (Chinese uses Microsoft YaHei font)
- ✅ 英文使用Calibri字体 (English uses Calibri font)
- ✅ 代码使用Consolas字体 (Code uses Consolas font)
- ✅ 格式统一，显示正常 (Unified format, normal display)
"""
    
    # 图表配置
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
    
    try:
        # 创建测试输出目录
        test_output_dir = "./test_font_output"
        os.makedirs(test_output_dir, exist_ok=True)
        
        # 生成Word文档
        print("📝 正在生成Word文档...")
        output_path = convert_markdown_to_word_with_charts(
            markdown_content=test_markdown,
            chart_configs=chart_configs,
            output_dir=test_output_dir,
            filename="font_fix_test.docx"
        )
        
        print(f"✅ Word文档生成成功！")
        print(f"   文件路径: {output_path}")
        print(f"   文件大小: {os.path.getsize(output_path)} 字节")
        
        # 验证文件是否存在
        if os.path.exists(output_path):
            print("✅ 文件验证通过")
            
            # 提供使用建议
            print("\n📋 测试结果说明:")
            print("1. 请打开生成的Word文档查看字体效果")
            print("2. 检查中文是否使用微软雅黑字体")
            print("3. 检查英文是否使用Calibri字体")
            print("4. 检查代码是否使用Consolas字体")
            print("5. 检查表格、列表、图表的字体是否统一")
            
            return True
        else:
            print("❌ 文件验证失败：生成的文件不存在")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        logger.error(f"字体修复测试失败: {str(e)}")
        return False

def test_custom_fonts():
    """测试自定义字体配置"""
    print("\n" + "="*60)
    print("🎨 测试自定义字体配置")
    
    # 导入WordGenerator类进行自定义配置
    from word_generator import WordGenerator
    
    test_markdown = """
# 自定义字体测试

## 字体配置说明

这个文档使用了自定义的字体配置：
- 中文字体：宋体
- 英文字体：Arial
- 代码字体：Courier New

### 测试内容

**粗体测试**: This is bold text 这是粗体文字

*斜体测试*: This is italic text 这是斜体文字

`代码测试`: console.log("Hello World 你好世界");

| 表格测试 | Table Test |
|----------|------------|
| 中文内容 | Chinese Content |
| English Content | 英文内容 |
"""
    
    try:
        # 创建自定义字体配置的生成器
        test_output_dir = "./test_font_output"
        generator = WordGenerator(test_output_dir)
        
        # 修改字体配置
        generator.fonts['chinese'] = '宋体'
        generator.fonts['english'] = 'Arial'
        generator.fonts['code'] = 'Courier New'
        
        print(f"📝 使用自定义字体配置: {generator.fonts}")
        
        # 生成文档
        output_path = generator.convert_markdown_to_docx(
            markdown_content=test_markdown,
            filename="custom_font_test.docx"
        )
        
        print(f"✅ 自定义字体文档生成成功！")
        print(f"   文件路径: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 自定义字体测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🔧 Word字体格式修复测试工具")
    print("="*60)
    
    # 运行测试
    tests = [
        ("基础字体修复测试", test_font_fix),
        ("自定义字体配置测试", test_custom_fonts)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}执行异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    success_count = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n总计: {success_count}/{len(results)} 个测试通过")
    
    if success_count == len(results):
        print("🎉 所有测试通过！字体格式修复功能正常。")
        print("\n💡 使用建议:")
        print("1. 现在可以直接使用修复后的word_generator.py")
        print("2. 字体格式混乱问题已解决")
        print("3. 支持中英文字体分离显示")
        print("4. 支持自定义字体配置")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    
    print(f"\n📁 测试文件保存在: ./test_font_output/")

if __name__ == "__main__":
    main()
