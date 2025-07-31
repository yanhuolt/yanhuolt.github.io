#!/usr/bin/env python3
"""
Ubuntu系统中文字体修复测试脚本
专门用于测试在Ubuntu/Linux环境下的中文字体显示问题
"""

import requests
import json
import os
import sys
import subprocess

# 服务器配置
SERVER_URL = "http://localhost:8089"  # 根据实际情况修改

def check_ubuntu_fonts():
    """检查Ubuntu系统中文字体安装情况"""
    print("🔍 检查Ubuntu系统中文字体...")
    
    try:
        # 检查fc-list命令是否可用
        result = subprocess.run(['fc-list', ':lang=zh'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            fonts = result.stdout.strip().split('\n')
            chinese_fonts = [f for f in fonts if f.strip()]
            
            print(f"📊 找到 {len(chinese_fonts)} 个中文字体:")
            for i, font in enumerate(chinese_fonts[:10]):  # 只显示前10个
                print(f"  {i+1}. {font.split(':')[0].split('/')[-1]}")
            
            if len(chinese_fonts) > 10:
                print(f"  ... 还有 {len(chinese_fonts) - 10} 个字体")
            
            return len(chinese_fonts) > 0
        else:
            print("❌ fc-list命令执行失败")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ fc-list命令超时")
        return False
    except FileNotFoundError:
        print("❌ 未找到fc-list命令，请安装fontconfig")
        print("sudo apt-get install fontconfig")
        return False
    except Exception as e:
        print(f"❌ 检查字体时出错: {e}")
        return False

def install_fonts_guide():
    """显示字体安装指导"""
    print("\n💡 Ubuntu中文字体安装指导:")
    print("="*50)
    print("1. 安装基本中文字体包:")
    print("   sudo apt-get update")
    print("   sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei")
    print("")
    print("2. 安装更多中文字体:")
    print("   sudo apt-get install fonts-arphic-ukai fonts-arphic-uming")
    print("   sudo apt-get install fonts-noto-cjk")
    print("")
    print("3. 刷新字体缓存:")
    print("   sudo fc-cache -fv")
    print("")
    print("4. 验证安装:")
    print("   fc-list :lang=zh")
    print("")
    print("5. 使用提供的安装脚本:")
    print("   sudo bash install_ubuntu_fonts.sh")

def test_matplotlib_fonts():
    """测试matplotlib字体支持"""
    print("\n🎨 测试matplotlib中文字体支持...")
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # 获取所有字体
        all_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 查找中文字体
        chinese_fonts = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei', 
            'Noto Sans CJK SC',
            'Source Han Sans CN',
            'AR PL UKai CN',
            'AR PL UMing CN'
        ]
        
        found_fonts = []
        for font in chinese_fonts:
            if font in all_fonts:
                found_fonts.append(font)
                print(f"✅ matplotlib可用字体: {font}")
        
        if not found_fonts:
            print("❌ matplotlib未找到中文字体")
            print("🔍 可用字体示例:")
            for font in all_fonts[:10]:
                print(f"  - {font}")
            return False
        
        # 测试字体渲染
        plt.rcParams['font.sans-serif'] = found_fonts
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label='测试数据')
        ax.set_title('中文字体测试图表')
        ax.set_xlabel('时间轴')
        ax.set_ylabel('数值轴')
        ax.legend()
        
        # 保存测试图片
        test_path = '/tmp/ubuntu_font_test.png'
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_path):
            print(f"✅ 字体测试成功，图片保存至: {test_path}")
            return True
        else:
            print("❌ 字体测试失败")
            return False
            
    except ImportError as e:
        print(f"❌ 导入matplotlib失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 字体测试出错: {e}")
        return False

def test_chart_service():
    """测试图表服务"""
    print("\n🚀 测试图表生成服务...")
    
    # 测试数据 - 包含中文
    test_data = {
        "markdown": """
# Ubuntu中文字体测试报告

## 测试图表

下面是测试图表，用于验证中文字体显示效果：

![温度趋势图](chart_test)

**测试结论**：如果图表中的中文正常显示，说明字体修复成功。
""",
        "charts": {
            "chart_test": {
                "title": {"text": "Ubuntu中文字体测试图表"},
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
            }
        }
    }
    
    try:
        response = requests.post(
            f"{SERVER_URL}/office/word/convert_with_charts",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            download_url = response.text.strip('"')
            print(f"✅ 服务测试成功!")
            print(f"📄 Word文档下载链接: {download_url}")
            return True
        else:
            print(f"❌ 服务测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器: {SERVER_URL}")
        print("请确保服务正在运行: python app.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return False
    except Exception as e:
        print(f"❌ 服务测试出错: {e}")
        return False

def main():
    """主函数"""
    print("🐧 Ubuntu中文字体修复测试工具")
    print("="*50)
    
    # 检查系统字体
    fonts_ok = check_ubuntu_fonts()
    
    if not fonts_ok:
        install_fonts_guide()
        print("\n⚠️  请先安装中文字体，然后重新运行此脚本")
        return False
    
    # 测试matplotlib
    matplotlib_ok = test_matplotlib_fonts()
    
    if not matplotlib_ok:
        print("\n⚠️  matplotlib字体支持有问题")
        print("尝试清除matplotlib缓存:")
        print("rm -rf ~/.cache/matplotlib")
        return False
    
    # 测试服务
    service_ok = test_chart_service()
    
    print("\n" + "="*50)
    print("📋 测试结果总结:")
    print(f"  系统字体: {'✅ 正常' if fonts_ok else '❌ 异常'}")
    print(f"  matplotlib: {'✅ 正常' if matplotlib_ok else '❌ 异常'}")
    print(f"  图表服务: {'✅ 正常' if service_ok else '❌ 异常'}")
    
    if fonts_ok and matplotlib_ok and service_ok:
        print("\n🎉 所有测试通过！中文字体修复成功！")
        print("\n📝 请检查生成的Word文档:")
        print("1. 图表标题是否显示中文而不是方框")
        print("2. 坐标轴标签是否正常显示中文")
        print("3. 图例文字是否正常显示")
        print("4. Word文档正文字体是否统一")
        return True
    else:
        print("\n❌ 部分测试失败，请根据上述提示进行修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
