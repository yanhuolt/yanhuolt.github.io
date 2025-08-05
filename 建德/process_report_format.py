#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据副本报告输出格式处理数据：
1. 数据清洗（负值、空值、极值）
2. 按照格式要求计算各项指标
3. 输出标准格式的CSV报告
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def detect_outliers_iqr(data):
    """使用箱型图方法检测异常值"""
    if len(data) < 4:
        return data.min(), data.max()
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return lower_bound, upper_bound

def clean_column_data(series, preserve_negative=False):
    """清洗单列数据"""
    cleaned_series = series.copy()

    # 转换为数值类型
    cleaned_series = pd.to_numeric(cleaned_series, errors='coerce')

    # 处理负值（除了特殊列）
    if not preserve_negative:
        # 只清洗明显的负值，但保留接近0的小负值（可能是测量误差）
        cleaned_series[cleaned_series < -1] = np.nan

    # 去除异常值 - 使用更宽松的标准
    valid_data = cleaned_series.dropna()
    if len(valid_data) > 10:  # 需要更多数据点才进行异常值检测
        lower_bound, upper_bound = detect_outliers_iqr(valid_data)

        # 使用更宽松的异常值范围（2.5倍IQR而不是1.5倍）
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR

        outlier_mask = (cleaned_series < lower_bound) | (cleaned_series > upper_bound)
        outlier_count = outlier_mask.sum()

        # 只有当异常值比例不超过5%时才清洗
        if outlier_count / len(cleaned_series) <= 0.05:
            cleaned_series[outlier_mask] = np.nan

    return cleaned_series

def calculate_corrected_concentration(measured_conc, measured_o2):
    """计算标准状态下的浓度"""
    # ρ（标准）=ρ（实测）*10/(21-ρ（实测O2））
    corrected = measured_conc * 10 / (21 - measured_o2)
    return corrected

def process_daily_data(df, date_str):
    """处理单日数据"""
    print(f"\n处理日期: {date_str}")
    print(f"  数据行数: {len(df)}")

    # 数据清洗
    df_clean = df.copy()
    
    # 定义列映射（根据实际数据列名调整）
    column_mapping = {
        '入炉垃圾量': 'B',
        '炉膛温度': ['I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'],  # 9个温度点
        '省煤器出口温度': 'X',
        '消石灰用量': 'T',
        '雾化器浆液流量': 'W',
        '半干法反应塔温度': 'Y',
        '湿法烧碱流量': 'AB',
        '湿式洗涤塔温度': 'AC',
        '减湿液PH值': 'AA',
        'SNCR氨水流量': 'AD',
        'SCR氨水流量': 'AH',
        'SCR系统温度': 'AI',
        '除尘器灰斗温度': 'AP',
        '除尘器压差': 'AO',  # 保留负值
        '活性炭用量': 'AL',
        '实测烟尘': 'AU',
        '实测SO2': 'AV',
        '实测NOx': 'AW',
        '实测CO': 'AX',
        '实测HCL': 'AY',
        '实测O2': 'AT',
        'NH3浓度': 'AG'
    }
    
    # 清洗数据
    for col in df_clean.columns:
        if col in ['除尘器压差', '烟气压力']:  # 压差列保留负值
            df_clean[col] = clean_column_data(df_clean[col], preserve_negative=True)
        else:
            df_clean[col] = clean_column_data(df_clean[col], preserve_negative=False)
    
    # 计算各项指标
    results = {}
    
    # 1. 入炉垃圾量：最大值-最小值（如果差值太小，使用平均值）
    if '入炉垃圾量' in df_clean.columns:
        garbage_data = df_clean['入炉垃圾量'].dropna()
        if len(garbage_data) > 0:
            max_val = garbage_data.max()
            min_val = garbage_data.min()
            diff_val = max_val - min_val
            # 如果差值太小（可能是累计量没有变化），使用平均值作为参考
            if diff_val < 1:
                results['入炉垃圾量'] = garbage_data.mean()
                print(f"    入炉垃圾量: 差值过小({diff_val:.2f})，使用平均值 {results['入炉垃圾量']:.2f}")
            else:
                results['入炉垃圾量'] = diff_val
                print(f"    入炉垃圾量: 最大-最小 = {results['入炉垃圾量']:.2f}")
        else:
            results['入炉垃圾量'] = np.nan
    
    # 2. 炉膛日平均温度：上部、中部、下部烟气温度的9个测点平均
    furnace_temp_cols = [
        '上部烟气温度左', '上部烟气温度中', '上部烟气温度右',
        '中部烟气温度左', '中部烟气温度中', '中部烟气温度右',
        '下部烟气温度左', '下部烟气温度中', '下部烟气温度右'
    ]
    existing_furnace_cols = [col for col in furnace_temp_cols if col in df_clean.columns]
    if existing_furnace_cols:
        # 计算每行的平均温度，然后计算日平均
        temp_data = df_clean[existing_furnace_cols].mean(axis=1).dropna()
        results['炉膛日平均温度'] = temp_data.mean() if len(temp_data) > 0 else np.nan
        print(f"  炉膛温度计算: 使用 {len(existing_furnace_cols)} 个测点, 日平均 {results['炉膛日平均温度']:.2f}°C")
    else:
        results['炉膛日平均温度'] = np.nan
        print(f"  炉膛温度计算: 未找到温度测点")
    
    # 3. 省煤器出口日平均温度
    if '省煤器出口温度' in df_clean.columns:
        results['省煤器出口日平均温度'] = df_clean['省煤器出口温度'].mean()
        print(f"  省煤器出口温度: {results['省煤器出口日平均温度']:.2f}°C")
    else:
        results['省煤器出口日平均温度'] = np.nan
    
    # 4. 消石灰用量：最大值-最小值
    if '消石灰累计' in df_clean.columns:
        lime_data = df_clean['消石灰累计'].dropna()
        if len(lime_data) > 0:
            max_val = lime_data.max()
            min_val = lime_data.min()
            diff_val = max_val - min_val
            if diff_val < 0.1:  # 消石灰用量差值很小时使用平均值
                results['消石灰用量'] = lime_data.mean()
                print(f"    消石灰用量: 差值过小({diff_val:.2f})，使用平均值 {results['消石灰用量']:.2f}")
            else:
                results['消石灰用量'] = diff_val
                print(f"    消石灰用量: 最大-最小 = {results['消石灰用量']:.2f}")
        else:
            results['消石灰用量'] = np.nan
    
    # 5. 雾化器浆液日平均流量
    if '雾化器浆液流量' in df_clean.columns:
        results['雾化器浆液日平均流量'] = df_clean['雾化器浆液流量'].mean()
    
    # 6. 半干法反应塔平均温度
    if '反应塔温度' in df_clean.columns:
        results['半干法反应塔平均温度'] = df_clean['反应塔温度'].mean()
    
    # 7. 湿法烧碱日平均流量
    if '湿法烧碱供应流量' in df_clean.columns:
        results['湿法烧碱日平均流量'] = df_clean['湿法烧碱供应流量'].mean()
    
    # 8. 湿式洗涤塔平均反应温度
    if '湿法洗涤塔温度' in df_clean.columns:
        results['湿式洗涤塔平均反应温度'] = df_clean['湿法洗涤塔温度'].mean()
    
    # 9. 减湿液平均PH值
    if '减湿液PH值' in df_clean.columns:
        results['减湿液平均PH值'] = df_clean['减湿液PH值'].mean()
    
    # 10. SNCR分配柜氨水日平均流量
    if 'SNCR分配柜氨水流量' in df_clean.columns:
        results['SNCR分配柜氨水日平均流量'] = df_clean['SNCR分配柜氨水流量'].mean()
    
    # 11. SCR氨水平均流量
    if '1#SCR氨水流量' in df_clean.columns:
        results['SCR氨水平均流量'] = df_clean['1#SCR氨水流量'].mean()
    
    # 12. SCR系统平均反应温度
    if 'SCR系统温度' in df_clean.columns:
        results['SCR系统平均反应温度'] = df_clean['SCR系统温度'].mean()
    
    # 13. 除尘器灰斗平均温度
    if '除尘器灰斗温度' in df_clean.columns:
        results['除尘器灰斗平均温度'] = df_clean['除尘器灰斗温度'].mean()
    
    # 14. 除尘器日平均压差
    if '除尘器差压' in df_clean.columns:
        results['除尘器日平均压差'] = df_clean['除尘器差压'].mean()
    
    # 15. 活性炭用量：最大值-最小值
    if '活性炭称重累计' in df_clean.columns:
        carbon_data = df_clean['活性炭称重累计'].dropna()
        if len(carbon_data) > 0:
            max_val = carbon_data.max()
            min_val = carbon_data.min()
            diff_val = max_val - min_val
            if diff_val < 0.1:  # 活性炭用量差值很小时使用平均值
                results['活性炭用量'] = carbon_data.mean()
                print(f"    活性炭用量: 差值过小({diff_val:.2f})，使用平均值 {results['活性炭用量']:.2f}")
            else:
                results['活性炭用量'] = diff_val
                print(f"    活性炭用量: 最大-最小 = {results['活性炭用量']:.2f}")
        else:
            results['活性炭用量'] = np.nan
    
    # 16-20. 烟气排放浓度（需要标准化计算）
    if all(col in df_clean.columns for col in ['烟气氧量', '烟气烟尘']):
        measured_dust = df_clean['烟气烟尘'].dropna()
        measured_o2 = df_clean['烟气氧量'].dropna()
        if len(measured_dust) > 0 and len(measured_o2) > 0:
            # 确保数据长度一致
            min_len = min(len(measured_dust), len(measured_o2))
            corrected_dust = calculate_corrected_concentration(
                measured_dust.iloc[:min_len], measured_o2.iloc[:min_len]
            )
            results['烟尘平均浓度'] = corrected_dust.mean()
    
    # SO2浓度
    if all(col in df_clean.columns for col in ['烟气氧量', 'SO2浓度']):
        measured_so2 = df_clean['SO2浓度'].dropna()
        measured_o2 = df_clean['烟气氧量'].dropna()
        if len(measured_so2) > 0 and len(measured_o2) > 0:
            min_len = min(len(measured_so2), len(measured_o2))
            corrected_so2 = calculate_corrected_concentration(
                measured_so2.iloc[:min_len], measured_o2.iloc[:min_len]
            )
            results['SO2平均浓度'] = corrected_so2.mean()
    
    # NOX浓度
    if all(col in df_clean.columns for col in ['烟气氧量', 'NOX浓度']):
        measured_nox = df_clean['NOX浓度'].dropna()
        measured_o2 = df_clean['烟气氧量'].dropna()
        if len(measured_nox) > 0 and len(measured_o2) > 0:
            min_len = min(len(measured_nox), len(measured_o2))
            corrected_nox = calculate_corrected_concentration(
                measured_nox.iloc[:min_len], measured_o2.iloc[:min_len]
            )
            results['NOX平均浓度'] = corrected_nox.mean()
    
    # CO浓度
    if all(col in df_clean.columns for col in ['烟气氧量', 'CO浓度']):
        measured_co = df_clean['CO浓度'].dropna()
        measured_o2 = df_clean['烟气氧量'].dropna()
        if len(measured_co) > 0 and len(measured_o2) > 0:
            min_len = min(len(measured_co), len(measured_o2))
            corrected_co = calculate_corrected_concentration(
                measured_co.iloc[:min_len], measured_o2.iloc[:min_len]
            )
            results['CO平均浓度'] = corrected_co.mean()
    
    # HCL浓度
    if all(col in df_clean.columns for col in ['烟气氧量', 'HCL浓度']):
        measured_hcl = df_clean['HCL浓度'].dropna()
        measured_o2 = df_clean['烟气氧量'].dropna()
        if len(measured_hcl) > 0 and len(measured_o2) > 0:
            min_len = min(len(measured_hcl), len(measured_o2))
            corrected_hcl = calculate_corrected_concentration(
                measured_hcl.iloc[:min_len], measured_o2.iloc[:min_len]
            )
            results['HCL平均浓度'] = corrected_hcl.mean()
    
    # NH3浓度
    if 'NH3浓度' in df_clean.columns:
        results['NH3平均浓度'] = df_clean['NH3浓度'].mean()
    
    return results

def process_single_file(file_path):
    """处理单个_process.xlsx文件"""
    print(f"处理文件: {file_path.name}")

    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"  读取成功: {len(df)} 行, {len(df.columns)} 列")

        if len(df) == 0:
            print(f"  ⚠️ 文件为空，跳过")
            return None

    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return None

    # 转换时间列
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])

    # 从文件路径提取日期信息
    import re

    # 先尝试从文件路径中提取年月信息
    path_str = str(file_path)
    year_month_match = re.search(r'(\d{4})年(\d+)月', path_str)
    file_name_match = re.search(r'(\d+)\.(\d+)', file_path.name)

    if year_month_match and file_name_match:
        year, month = year_month_match.groups()
        _, day = file_name_match.groups()
        date_str = f"{year}年{month}月{day}日"
    elif file_name_match:
        # 如果只能从文件名提取，默认使用2025年
        month, day = file_name_match.groups()
        date_str = f"2025年{month}月{day}日"
    else:
        # 如果无法从文件名提取，使用数据中的第一个日期
        first_date = df[time_col].iloc[0].date()
        date_str = first_date.strftime('%Y年%m月%d日')

    print(f"  提取日期: {date_str}")

    # 处理该文件的数据（整个文件作为一天的数据）
    results = process_daily_data(df, date_str)
    results['时间'] = date_str

    return results

def main():
    """主函数 - 批量处理所有_process.xlsx文件"""
    print("=== 批量处理建德数据报告 ===")

    # 查找所有_process.xlsx文件
    data_folder = Path("建德/建德数据")

    if not data_folder.exists():
        print(f"❌ 数据文件夹不存在: {data_folder}")
        return

    process_files = list(data_folder.rglob("*_process.xlsx"))

    if not process_files:
        print("❌ 未找到任何_process.xlsx文件")
        return

    print(f"📁 找到 {len(process_files)} 个处理后的文件")

    # 批量处理所有文件
    all_results = []
    success_count = 0
    fail_count = 0

    for file_path in sorted(process_files):
        result = process_single_file(file_path)
        if result is not None:
            all_results.append(result)
            success_count += 1
        else:
            fail_count += 1

    if not all_results:
        print("❌ 没有成功处理的文件")
        return

    # 创建输出DataFrame
    output_df = pd.DataFrame(all_results)

    # 重新排列列顺序以匹配模板
    column_order = [
        '时间', '入炉垃圾量', '炉膛日平均温度', '省煤器出口日平均温度', '消石灰用量',
        '雾化器浆液日平均流量', '半干法反应塔平均温度', '湿法烧碱日平均流量',
        '湿式洗涤塔平均反应温度', '减湿液平均PH值', 'SNCR分配柜氨水日平均流量',
        'SCR氨水平均流量', 'SCR系统平均反应温度', '除尘器灰斗平均温度',
        '除尘器日平均压差', '活性炭用量', '烟尘平均浓度', 'SO2平均浓度',
        'NOX平均浓度', 'CO平均浓度', 'HCL平均浓度', 'NH3平均浓度'
    ]

    # 确保所有列都存在
    for col in column_order:
        if col not in output_df.columns:
            output_df[col] = np.nan

    output_df = output_df[column_order]

    # 按时间排序
    output_df = output_df.sort_values('时间')

    # 保存结果
    output_file = "建德/建德数据汇总报告.csv"
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✅ 批量处理完成！")
    print(f"📊 处理统计:")
    print(f"  - 成功处理: {success_count} 个文件")
    print(f"  - 处理失败: {fail_count} 个文件")
    print(f"  - 汇总数据: {len(output_df)} 天")
    print(f"  - 输出文件: {output_file}")

    # 显示前几行结果
    print(f"\n📋 前5行汇总结果:")
    print(output_df.head().to_string(index=False))

    # 统计数据完整性
    print(f"\n📈 数据完整性统计:")
    for col in column_order[1:]:  # 跳过时间列
        valid_count = output_df[col].notna().sum()
        total_count = len(output_df)
        percentage = (valid_count / total_count) * 100 if total_count > 0 else 0
        if percentage < 100:
            print(f"  {col}: {valid_count}/{total_count} ({percentage:.1f}%)")

    print(f"\n🎉 汇总报告已生成，包含 {len(output_df)} 天的数据！")

if __name__ == "__main__":
    main()
