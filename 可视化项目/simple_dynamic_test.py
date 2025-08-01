#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的动态拟合测试
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def logistic_function(t, A, k, t0):
    """Logistic函数"""
    return A / (1 + np.exp(-k * (t - t0)))

def create_test_data():
    """创建测试数据"""
    # 48小时数据
    time_hours = np.arange(0, 48, 1)
    time_seconds = time_hours * 3600
    
    breakthrough_ratios = []
    for t in time_hours:
        if t < 20:  # 前20小时，穿透率很低
            ratio = 0.005 + 0.002 * t + np.random.normal(0, 0.002)
        elif t < 35:  # 20-35小时，开始增长
            ratio = 0.045 + 0.01 * (t - 20) + np.random.normal(0, 0.005)
        else:  # 35小时后，快速增长
            ratio = 0.195 + 0.6 * (1 - np.exp(-0.3 * (t - 35))) + np.random.normal(0, 0.01)
        
        ratio = max(0.001, min(0.85, ratio))
        breakthrough_ratios.append(ratio)
    
    return time_seconds, np.array(breakthrough_ratios)

def analyze_data_phases(t_valid, bt_valid):
    """分析数据阶段"""
    from scipy.ndimage import gaussian_filter1d
    
    bt_smooth = gaussian_filter1d(bt_valid, sigma=max(1, len(bt_valid)//10))
    first_derivative = np.gradient(bt_smooth, t_valid)
    
    phases = {'initial_phase': [], 'growth_phase': [], 'saturation_phase': []}
    
    max_growth_rate = np.max(first_derivative)
    growth_threshold = max_growth_rate * 0.3
    
    for i in range(len(bt_valid)):
        if first_derivative[i] < growth_threshold:
            if bt_valid[i] < np.median(bt_valid):
                phases['initial_phase'].append(i)
            else:
                phases['saturation_phase'].append(i)
        else:
            phases['growth_phase'].append(i)
    
    return phases

def calculate_dynamic_weights(t_valid, bt_valid):
    """计算动态权重"""
    weights = np.ones_like(bt_valid)
    phases = analyze_data_phases(t_valid, bt_valid)
    
    # 不同阶段权重
    for i in phases['initial_phase']:
        weights[i] *= 3.0
    for i in phases['growth_phase']:
        weights[i] *= 2.0
    for i in phases['saturation_phase']:
        weights[i] *= 1.5
    
    # 基于数据质量调整
    if len(bt_valid) > 3:
        bt_diff = np.abs(np.diff(bt_valid))
        bt_diff_normalized = bt_diff / (np.mean(bt_diff) + 1e-8)
        
        for i in range(len(bt_diff)):
            if bt_diff_normalized[i] > 2.0:
                weights[i] *= 0.5
                weights[i+1] *= 0.5
    
    return weights

def estimate_dynamic_growth_rate(t_valid, bt_valid):
    """估计动态增长率"""
    if len(bt_valid) < 5:
        return 0.0001
    
    window_size = max(3, len(bt_valid) // 5)
    local_growth_rates = []
    
    for i in range(window_size, len(bt_valid) - window_size):
        window_t = t_valid[i-window_size:i+window_size]
        window_bt = bt_valid[i-window_size:i+window_size]
        
        if len(window_t) > 2:
            dt = window_t[-1] - window_t[0]
            dbt = window_bt[-1] - window_bt[0]
            if dt > 0 and window_bt[0] > 0:
                local_rate = dbt / (dt * window_bt[0])
                if local_rate > 0:
                    local_growth_rates.append(local_rate)
    
    if local_growth_rates:
        k_estimate = np.median(local_growth_rates)
        k_estimate = np.clip(k_estimate, 0.000001, 0.1)
        return k_estimate
    else:
        return 0.0001

def test_dynamic_fitting():
    """测试动态拟合"""
    print("开始动态拟合测试...")
    
    # 创建测试数据
    time_data, breakthrough_data = create_test_data()
    
    print(f"数据范围: 时间 0-{time_data[-1]/3600:.1f}小时, 穿透率 {breakthrough_data.min():.3f}-{breakthrough_data.max():.3f}")
    
    # 动态参数估计
    A_init = min(0.98, np.max(breakthrough_data) * 1.05)
    k_init = estimate_dynamic_growth_rate(time_data, breakthrough_data)
    
    # 拐点估计
    from scipy.ndimage import gaussian_filter1d
    bt_smooth = gaussian_filter1d(breakthrough_data, sigma=max(1, len(breakthrough_data)//15))
    first_derivative = np.gradient(bt_smooth, time_data)
    max_growth_idx = np.argmax(first_derivative)
    t0_init = time_data[max_growth_idx]
    
    print(f"动态参数估计: A={A_init:.3f}, k={k_init:.6f}, t0={t0_init:.1f}s ({t0_init/3600:.2f}h)")
    
    # 计算动态权重
    weights = calculate_dynamic_weights(time_data, breakthrough_data)
    print(f"权重统计: 最小={np.min(weights):.2f}, 最大={np.max(weights):.2f}, 平均={np.mean(weights):.2f}")
    
    # 拟合
    try:
        A_max = min(1.0, np.max(breakthrough_data) * 1.2)
        A_min = max(0.1, np.max(breakthrough_data) * 0.7)
        k_max = max(0.01, k_init * 10)
        k_min = max(0.000001, k_init * 0.1)
        
        params, _ = curve_fit(
            logistic_function,
            time_data, breakthrough_data,
            p0=[A_init, k_init, t0_init],
            bounds=([A_min, k_min, 0], [A_max, k_max, np.max(time_data) * 2]),
            sigma=1/weights,
            maxfev=8000
        )
        
        A, k, t0 = params
        print(f"拟合成功: A={A:.3f}, k={k:.6f}, t0={t0:.1f}s ({t0/3600:.2f}h)")
        
        # 计算拟合质量
        predicted = logistic_function(time_data, *params)
        r_squared = 1 - np.sum((breakthrough_data - predicted)**2) / np.sum((breakthrough_data - np.mean(breakthrough_data))**2)
        mse = np.mean((breakthrough_data - predicted)**2)
        
        print(f"拟合质量: R²={r_squared:.3f}, MSE={mse:.6f}")
        
        # 绘图
        plt.figure(figsize=(12, 8))
        time_hours = time_data / 3600
        
        plt.subplot(2, 1, 1)
        plt.scatter(time_hours, breakthrough_data * 100, alpha=0.7, label='实际数据')
        
        # 预测曲线
        time_extended = np.linspace(0, np.max(time_data) * 1.2, 500)
        predicted_extended = logistic_function(time_extended, *params)
        plt.plot(time_extended / 3600, predicted_extended * 100, 'r-', linewidth=2, label='动态拟合曲线')
        
        plt.xlabel('时间 (小时)')
        plt.ylabel('穿透率 (%)')
        plt.title(f'动态权重Logistic拟合 (R²={r_squared:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 权重分布
        plt.subplot(2, 1, 2)
        plt.plot(time_hours, weights, 'g-', marker='o', markersize=4, label='动态权重')
        plt.xlabel('时间 (小时)')
        plt.ylabel('权重')
        plt.title('动态权重分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dynamic_fitting_test.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 dynamic_fitting_test.png")
        
        return True
        
    except Exception as e:
        print(f"拟合失败: {e}")
        return False

if __name__ == "__main__":
    success = test_dynamic_fitting()
    if success:
        print("\n✓ 动态拟合测试成功")
    else:
        print("\n✗ 动态拟合测试失败")
