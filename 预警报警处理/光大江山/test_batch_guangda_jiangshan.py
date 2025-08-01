#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试光大江山批量处理器
"""

import os
import sys
from datetime import datetime

def test_batch_processor():
    """测试批量处理器"""
    print("=== 测试光大江山批量处理器 ===")
    
    try:
        # 导入批量处理器
        from batch_process_guangda_jiangshan import BatchProcessorGuangdaJiangshan
        print("✓ 批量处理器导入成功")
        
        # 创建实例
        processor = BatchProcessorGuangdaJiangshan()
        print("✓ 批量处理器实例创建成功")
        
        # 检查数据文件夹
        folder_path = "数据下/数据下/光大（江山）"
        if os.path.exists(folder_path):
            print(f"✓ 数据文件夹存在: {folder_path}")
            
            # 列出文件夹中的文件
            files = os.listdir(folder_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            xlsx_files = [f for f in files if f.endswith('.xlsx')]
            
            print(f"  - CSV文件: {len(csv_files)} 个")
            print(f"  - XLSX文件: {len(xlsx_files)} 个")
            
            if csv_files:
                print("  CSV文件列表:")
                for f in csv_files[:5]:  # 只显示前5个
                    print(f"    - {f}")
                if len(csv_files) > 5:
                    print(f"    ... 还有 {len(csv_files) - 5} 个文件")
            
            if xlsx_files:
                print("  XLSX文件列表:")
                for f in xlsx_files[:5]:  # 只显示前5个
                    print(f"    - {f}")
                if len(xlsx_files) > 5:
                    print(f"    ... 还有 {len(xlsx_files) - 5} 个文件")
                    
            # 测试日期解析
            if csv_files or xlsx_files:
                test_file = csv_files[0] if csv_files else xlsx_files[0]
                test_date = processor.parse_date_from_filename(test_file)
                if test_date:
                    print(f"✓ 日期解析测试成功: {test_file} -> {test_date.strftime('%Y-%m-%d')}")
                else:
                    print(f"✗ 日期解析测试失败: {test_file}")
            
            print("\n=== 测试成功 ===")
            print("可以运行批量处理:")
            print("python batch_process_guangda_jiangshan.py")
            
        else:
            print(f"✗ 数据文件夹不存在: {folder_path}")
            print("请检查文件夹路径是否正确")
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_processor()
