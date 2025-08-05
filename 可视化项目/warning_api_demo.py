#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预警系统API使用示例
演示如何获取预警系统的数据点坐标、标签和预警点信息
"""

from adsorption_api import get_warning_system_data
import json

def demo_warning_api():
    """演示预警系统API的使用"""
    print("=== 预警系统API使用演示 ===")
    
    # 数据文件路径
    data_file = "可视化项目/7.24.csv"  # 请替换为您的实际数据文件路径
    
    # 调用主要接口函数
    result = get_warning_system_data(data_file)
    
    if result["success"]:
        print("✅ 预警系统分析成功！")
        
        # 1. 获取数据点信息
        data_points = result["data_points"]
        print(f"\n📊 预警系统数据点信息:")
        print(f"   总数量: {len(data_points)} 个")
        
        if data_points:
            print(f"   前3个数据点:")
            for i, point in enumerate(data_points[:3]):
                print(f"     点{i+1}: x={point['x']:.1f}, y={point['y']:.1f}, 标签='{point['label']}'")
            
            # 提取所有x,y坐标
            x_coordinates = [point['x'] for point in data_points]
            y_coordinates = [point['y'] for point in data_points]
            labels = [point['label'] for point in data_points]
            
            print(f"   X坐标范围: {min(x_coordinates):.1f} - {max(x_coordinates):.1f}")
            print(f"   Y坐标范围: {min(y_coordinates):.1f} - {max(y_coordinates):.1f}")
        
        # 2. 获取预警点信息
        warning_points = result["warning_points"]
        print(f"\n⚠️ 预警点信息:")
        print(f"   总数量: {len(warning_points)} 个")
        
        if warning_points:
            for i, point in enumerate(warning_points):
                print(f"   预警点{i+1}:")
                print(f"     坐标: x={point['x']:.1f}, y={point['y']:.1f}")
                print(f"     预警等级: {point['warning_level']}")
                print(f"     预警原因: {point['reason']}")
                print(f"     建议措施: {point['recommendation']}")
                print(f"     穿透率: {point['breakthrough_ratio']:.1f}%")
                print(f"     吸附效率: {point['efficiency']:.1f}%")
                print()
            
            # 提取预警点坐标
            warning_x = [point['x'] for point in warning_points]
            warning_y = [point['y'] for point in warning_points]
            
            print(f"   预警点X坐标: {warning_x}")
            print(f"   预警点Y坐标: {warning_y}")
        else:
            print("   无预警点")
        
        # 3. 统计信息
        stats = result["statistics"]
        print(f"\n📈 统计信息:")
        print(f"   数据点总数: {stats['total_data_points']}")
        print(f"   预警点总数: {stats['total_warning_points']}")
        print(f"   时间范围: {stats['time_range']['start']:.1f}s - {stats['time_range']['end']:.1f}s")
        print(f"   效率范围: {stats['efficiency_range']['min']:.1f}% - {stats['efficiency_range']['max']:.1f}%")
        
        # 4. 返回结构化数据
        return {
            "data_points_coordinates": [(p['x'], p['y']) for p in data_points],
            "data_points_labels": [p['label'] for p in data_points],
            "warning_points_coordinates": [(p['x'], p['y']) for p in warning_points],
            "warning_points_info": [
                {
                    "coordinates": (p['x'], p['y']),
                    "level": p['warning_level'],
                    "reason": p['reason'],
                    "recommendation": p['recommendation']
                } for p in warning_points
            ],
            "statistics": stats
        }
        
    else:
        print(f"❌ 预警系统分析失败: {result['error']}")
        return None

def extract_coordinates_only(data_file: str):
    """
    仅提取坐标信息的简化函数
    
    Args:
        data_file: 数据文件路径
    
    Returns:
        Dict: 包含数据点和预警点坐标的字典
    """
    result = get_warning_system_data(data_file)
    
    if result["success"]:
        data_points = result["data_points"]
        warning_points = result["warning_points"]
        
        return {
            "success": True,
            "data_x": [p['x'] for p in data_points],
            "data_y": [p['y'] for p in data_points],
            "data_labels": [p['label'] for p in data_points],
            "warning_x": [p['x'] for p in warning_points],
            "warning_y": [p['y'] for p in warning_points],
            "warning_levels": [p['warning_level'] for p in warning_points]
        }
    else:
        return {
            "success": False,
            "error": result["error"],
            "data_x": [],
            "data_y": [],
            "data_labels": [],
            "warning_x": [],
            "warning_y": [],
            "warning_levels": []
        }

def save_results_to_json(data_file: str, output_file: str = "warning_system_results.json"):
    """
    将预警系统结果保存为JSON文件
    
    Args:
        data_file: 输入数据文件路径
        output_file: 输出JSON文件路径
    """
    result = get_warning_system_data(data_file)
    
    # 保存完整结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"预警系统结果已保存到: {output_file}")
    return result

if __name__ == "__main__":
    # 运行演示
    demo_result = demo_warning_api()
    
    if demo_result:
        print("\n" + "="*50)
        print("📋 API返回的数据结构:")
        print(f"  数据点坐标: {len(demo_result['data_points_coordinates'])} 个")
        print(f"  数据点标签: {len(demo_result['data_points_labels'])} 个")
        print(f"  预警点坐标: {len(demo_result['warning_points_coordinates'])} 个")
        print(f"  预警点信息: {len(demo_result['warning_points_info'])} 个")
        
        # 显示前几个坐标
        if demo_result['data_points_coordinates']:
            print(f"\n前3个数据点坐标: {demo_result['data_points_coordinates'][:3]}")
        
        if demo_result['warning_points_coordinates']:
            print(f"预警点坐标: {demo_result['warning_points_coordinates']}")
    
    print("\n" + "="*50)
    print("🔧 简化版本示例:")
    
    # 演示简化版本
    simple_result = extract_coordinates_only("可视化项目/7.24.csv")
    
    if simple_result["success"]:
        print(f"  数据点X坐标: {len(simple_result['data_x'])} 个")
        print(f"  数据点Y坐标: {len(simple_result['data_y'])} 个")
        print(f"  预警点X坐标: {simple_result['warning_x']}")
        print(f"  预警点Y坐标: {simple_result['warning_y']}")
        print(f"  预警等级: {simple_result['warning_levels']}")
    else:
        print(f"  简化版本失败: {simple_result['error']}")
    
    # 保存结果到JSON文件
    print("\n" + "="*50)
    print("💾 保存结果到JSON文件:")
    save_results_to_json("可视化项目/7.24.csv", "可视化项目/warning_results.json")
    
    print("\n🎉 演示完成！")
    print("\n📖 使用说明:")
    print("1. 调用 get_warning_system_data(data_file) 获取完整预警系统数据")
    print("2. 返回的数据包含:")
    print("   - data_points: 预警系统数据点的x,y坐标和标签")
    print("   - warning_points: 预警点的x,y坐标和预警信息")
    print("   - statistics: 统计信息")
    print("3. 可以直接提取坐标用于绘图或其他用途")
