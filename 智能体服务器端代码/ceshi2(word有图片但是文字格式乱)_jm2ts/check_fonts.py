"""
检查系统中文字体的脚本
"""

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import platform
import os

def check_system_fonts():
    """检查系统可用字体"""
    print(f"操作系统: {platform.system()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"Matplotlib版本: {plt.matplotlib.__version__}")
    print("-" * 50)
    
    # 获取所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 常用中文字体列表
    chinese_fonts = {
        'Windows': ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong'],
        'Darwin': ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti'],
        'Linux': ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans', 'Liberation Sans']
    }
    
    system = platform.system()
    target_fonts = chinese_fonts.get(system, chinese_fonts['Linux'])
    
    print("🔍 检查中文字体支持:")
    found_fonts = []
    
    for font in target_fonts:
        if font in available_fonts:
            print(f"✅ {font} - 可用")
            found_fonts.append(font)
        else:
            print(f"❌ {font} - 不可用")
    
    print(f"\n📊 找到 {len(found_fonts)} 个中文字体")
    
    if not found_fonts:
        print("\n⚠️  警告: 未找到合适的中文字体!")
        print("建议安装中文字体包:")
        if system == "Windows":
            print("- 通常Windows系统自带SimHei、Microsoft YaHei等字体")
        elif system == "Darwin":
            print("- macOS通常自带Arial Unicode MS等字体")
        else:
            print("- Ubuntu/Debian: sudo apt-get install fonts-wqy-microhei")
            print("- CentOS/RHEL: sudo yum install wqy-microhei-fonts")
    
    return found_fonts

def test_font_rendering():
    """测试字体渲染效果"""
    print("\n🎨 测试字体渲染效果...")
    
    # 设置中文字体
    found_fonts = check_system_fonts()
    
    if not found_fonts:
        print("❌ 无法进行字体渲染测试，请先安装中文字体")
        return False
    
    # 使用第一个找到的字体
    font_name = found_fonts[0]
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    categories = ['颗粒物', '氮氧化物', '二氧化硫', '氯化氢', '一氧化碳']
    values = [15.2, 45.8, 12.3, 8.5, 28.7]
    
    # 绘制柱状图
    bars = ax.bar(categories, values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
    
    # 设置标题和标签
    ax.set_title('污染物排放浓度测试图表', fontsize=16, fontweight='bold')
    ax.set_xlabel('污染物类型', fontsize=12)
    ax.set_ylabel('浓度 (mg/Nm³)', fontsize=12)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}', ha='center', va='bottom', fontsize=10)
    
    # 旋转x轴标签
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存测试图片
    test_image_path = "/tmp/font_test.png"
    plt.savefig(test_image_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    if os.path.exists(test_image_path):
        print(f"✅ 字体渲染测试成功!")
        print(f"📁 测试图片保存在: {test_image_path}")
        print(f"🔤 使用字体: {font_name}")
        return True
    else:
        print("❌ 字体渲染测试失败")
        return False

def get_font_recommendations():
    """获取字体安装建议"""
    system = platform.system()
    
    print(f"\n💡 {system} 系统字体安装建议:")
    
    if system == "Windows":
        print("""
Windows系统通常自带以下中文字体:
- SimHei (黑体)
- Microsoft YaHei (微软雅黑)
- SimSun (宋体)
- KaiTi (楷体)

如果缺少字体，可以:
1. 从控制面板 > 字体 中安装
2. 下载字体文件(.ttf)到 C:\\Windows\\Fonts\\
""")
    
    elif system == "Darwin":
        print("""
macOS系统通常自带以下中文字体:
- Arial Unicode MS
- PingFang SC
- Heiti SC
- STHeiti

如果需要更多字体:
1. 使用字体册应用安装
2. 下载字体文件到 ~/Library/Fonts/
""")
    
    else:  # Linux
        print("""
Linux系统中文字体安装:

Ubuntu/Debian:
sudo apt-get update
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

CentOS/RHEL:
sudo yum install wqy-microhei-fonts wqy-zenhei-fonts

或者手动安装:
1. 下载字体文件(.ttf)
2. 复制到 ~/.fonts/ 或 /usr/share/fonts/
3. 运行 fc-cache -fv 刷新字体缓存
""")

if __name__ == "__main__":
    print("🔍 系统字体检查工具")
    print("=" * 50)
    
    # 检查字体
    fonts = check_system_fonts()
    
    # 测试渲染
    if fonts:
        test_font_rendering()
    
    # 提供建议
    get_font_recommendations()
    
    print("\n" + "=" * 50)
    print("✨ 检查完成!")
    
    if fonts:
        print(f"推荐使用字体: {fonts[0]}")
    else:
        print("⚠️  请安装中文字体后重新运行此脚本")
