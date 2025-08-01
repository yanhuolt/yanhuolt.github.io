#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复缺少'预警/报警区分'列的记录
"""

import re

def fix_missing_columns():
    """修复缺少'预警/报警区分'列的记录"""
    
    file_path = "shishi_data_yujing_gz.py"
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有缺少'预警/报警区分'列的记录
    # 匹配模式：以'预警/报警事件': 结尾但没有'预警/报警区分'的记录
    pattern = r"(\s+)'预警/报警事件': ([^}]+)\n(\s+)\})"
    
    def replacement(match):
        indent = match.group(1)
        event_content = match.group(2)
        closing_indent = match.group(3)
        
        # 检查是否已经有'预警/报警区分'列
        if "'预警/报警区分'" in event_content:
            return match.group(0)  # 已经有了，不需要修改
        
        # 添加'预警/报警区分'列
        return f"{indent}'预警/报警事件': {event_content},\n{indent}'预警/报警区分': '预警'\n{closing_indent}})"
    
    # 执行替换
    new_content = re.sub(pattern, replacement, content)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("修复完成！")

if __name__ == "__main__":
    fix_missing_columns()
