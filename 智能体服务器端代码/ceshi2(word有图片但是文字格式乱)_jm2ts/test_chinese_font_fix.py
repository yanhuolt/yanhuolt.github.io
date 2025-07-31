"""
测试中文字体修复效果的脚本
"""

import requests
import json

# 服务器配置
SERVER_URL = "http://192.168.0.109:8089"

def test_chinese_font_fix():
    """测试中文字体修复效果"""
    print("测试中文字体修复效果...")
    
    # 准备包含中文的测试数据
    markdown_content = """
# 垃圾焚烧企业数据分析报告

## 1. 数据概览

本月共处理垃圾 **1200吨**，设备运行稳定，各项指标正常。

| 指标名称 | 数值 | 单位 | 标准限值 | 达标情况 |
|----------|------|------|----------|----------|
| 累计处理量 | 1200 | 吨 | - | 正常 |
| 运行时间 | 720 | 小时 | - | 正常 |
| 平均炉温 | 865 | ℃ | ≥850 | **达标** |
| PM排放浓度 | 15.2 | mg/Nm³ | ≤30 | **达标** |
| NOx排放浓度 | 45.8 | mg/Nm³ | ≤300 | **达标** |

## 2. 炉膛温度变化趋势

下图显示了本月各焚烧炉的温度变化情况：

![炉膛温度变化趋势](chart_temperature)

**分析结论**：
- 1号炉平均温度为 *862℃*，运行稳定
- 2号炉平均温度为 *868℃*，略高于1号炉
- 两台焚烧炉温度均保持在850℃以上，符合环保要求

## 3. 污染物排放统计

各类污染物的排放情况对比：

![污染物排放对比](chart_emissions)

**重点关注**：
1. **颗粒物(PM)**：排放浓度为15.2 mg/Nm³，低于标准限值
2. **氮氧化物(NOx)**：排放浓度为45.8 mg/Nm³，控制良好
3. **二氧化硫(SO2)**：排放浓度为12.3 mg/Nm³，达标排放
4. **氯化氢(HCl)**：排放浓度为8.5 mg/Nm³，符合要求
5. **一氧化碳(CO)**：排放浓度为28.7 mg/Nm³，在正常范围内

## 4. 设备工况分布

本月设备运行工况统计：

![设备工况分布](chart_status)

## 5. 环保耗材消耗

主要环保耗材使用情况：

- **活性炭**：日均消耗 *125公斤*
- **石灰**：日均消耗 *280公斤*  
- **氨水**：日均消耗 *45升*

## 6. 决策建议

基于以上数据分析，提出以下**专业建议**：

### 6.1 温度控制优化
- 保持现有温度控制参数，确保炉膛温度稳定在850℃以上
- 定期检查温度传感器，确保数据准确性

### 6.2 排放控制改进
- PM排放浓度略高，建议：
  1. 增加布袋除尘器清灰频次
  2. 检查滤袋完整性
  3. 优化活性炭喷射量

### 6.3 运行效率提升
- 正常运行时间占比达到94.4%，表现良好
- 建议制定预防性维护计划，减少故障停机时间

## 7. 结论

本月垃圾焚烧企业运行状况**良好**，主要表现为：

1. ✅ 炉膛温度控制达标
2. ✅ 污染物排放符合标准
3. ✅ 设备运行稳定性高
4. ⚠️ 需关注PM排放浓度变化趋势

**总体评价**：设备运行正常，环保指标达标，建议继续保持现有管理水平。
"""
    
    # 图表配置 - 包含中文标签
    chart_configs = {
        "chart_temperature": {
            "title": {"text": "炉膛温度变化趋势"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "name": "时间周期",
                "data": ["第一周", "第二周", "第三周", "第四周"]
            },
            "yAxis": {"type": "value", "name": "温度(℃)"},
            "series": [{
                "type": "line",
                "name": "1号焚烧炉",
                "data": [855, 862, 858, 870],
                "smooth": True
            }, {
                "type": "line", 
                "name": "2号焚烧炉",
                "data": [850, 865, 860, 875],
                "smooth": True
            }]
        },
        
        "chart_emissions": {
            "title": {"text": "污染物排放对比"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "name": "污染物类型",
                "data": ["颗粒物", "氮氧化物", "二氧化硫", "氯化氢", "一氧化碳"]
            },
            "yAxis": {"type": "value", "name": "浓度(mg/Nm³)"},
            "series": [{
                "type": "bar",
                "name": "实际排放浓度",
                "data": [15.2, 45.8, 12.3, 8.5, 28.7],
                "itemStyle": {"color": "#5470c6"}
            }, {
                "type": "bar",
                "name": "标准限值", 
                "data": [30, 300, 100, 60, 100],
                "itemStyle": {"color": "#ff6b6b"}
            }]
        },
        
        "chart_status": {
            "title": {"text": "设备工况分布"},
            "tooltip": {"trigger": "item"},
            "series": [{
                "type": "pie",
                "radius": "60%",
                "data": [
                    {"name": "正常运行", "value": 680},
                    {"name": "启动过程", "value": 15},
                    {"name": "停机维护", "value": 20},
                    {"name": "故障处理", "value": 5}
                ],
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }]
        }
    }
    
    # 准备请求数据
    request_data = {
        "markdown": markdown_content,
        "charts": chart_configs
    }
    
    # 发送请求
    response = requests.post(
        f"{SERVER_URL}/office/word/convert_with_charts",
        json=request_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        download_url = response.text
        print(f"✅ 中文字体修复测试成功！")
        print(f"📄 Word文档下载链接: {download_url}")
        print("\n🔍 请检查生成的文档：")
        print("1. 图表中的中文标签是否正常显示（不是方框）")
        print("2. 正文字体是否统一为宋体12号")
        print("3. 标题字体是否为黑体且加粗")
        print("4. 表格格式是否整齐统一")
        print("5. 列表项字体是否一致")
        return True
    else:
        print(f"❌ 测试失败: {response.status_code} - {response.text}")
        return False

if __name__ == "__main__":
    print("🚀 开始测试中文字体修复效果...\n")
    
    success = test_chinese_font_fix()
    
    print("\n" + "="*60)
    if success:
        print("🎉 测试完成！请下载Word文档检查修复效果。")
        print("\n📋 检查清单：")
        print("□ 图表标题、坐标轴标签、图例中的中文正常显示")
        print("□ 正文段落字体统一（宋体12号）")
        print("□ 标题层级清晰（黑体，不同字号）")
        print("□ 表格内容格式一致")
        print("□ 列表项字体统一")
        print("□ 粗体、斜体等格式正确应用")
    else:
        print("❌ 测试失败，请检查服务器状态和配置。")
