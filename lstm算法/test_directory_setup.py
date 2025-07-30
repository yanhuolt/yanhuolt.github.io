"""
测试目录设置和文件保存功能
"""

import os
import torch
from datetime import datetime

def test_directory_setup():
    """测试目录创建和文件保存路径"""
    print("=== 测试LSTM算法目录设置 ===\n")
    
    # 创建必要的目录
    checkpoints_dir = "lstm算法/checkpoints"
    visualization_dir = "lstm算法/可视化结果"
    
    print("1. 创建目录...")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    print(f"✅ 模型保存目录: {checkpoints_dir}")
    print(f"✅ 可视化保存目录: {visualization_dir}")
    
    # 测试文件路径生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(checkpoints_dir, f"best_lstm_model_{timestamp}.pth")
    viz_save_path = os.path.join(visualization_dir, f"LSTM预报结果_{timestamp}.png")
    summary_path = os.path.join(visualization_dir, f"预报系统总结_{timestamp}.txt")
    
    print(f"\n2. 生成的文件路径:")
    print(f"   模型文件: {model_save_path}")
    print(f"   可视化文件: {viz_save_path}")
    print(f"   总结文件: {summary_path}")
    
    # 测试目录是否可写
    print(f"\n3. 测试目录权限...")
    try:
        # 测试模型目录
        test_model_file = os.path.join(checkpoints_dir, "test.txt")
        with open(test_model_file, 'w') as f:
            f.write("test")
        os.remove(test_model_file)
        print(f"✅ 模型目录可写: {checkpoints_dir}")
        
        # 测试可视化目录
        test_viz_file = os.path.join(visualization_dir, "test.txt")
        with open(test_viz_file, 'w') as f:
            f.write("test")
        os.remove(test_viz_file)
        print(f"✅ 可视化目录可写: {visualization_dir}")
        
    except Exception as e:
        print(f"❌ 目录权限测试失败: {e}")
        return False
    
    # 检查PyTorch环境
    print(f"\n4. PyTorch环境检查:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA设备: {torch.cuda.get_device_name()}")
    
    print(f"\n✅ 目录设置测试完成！")
    print(f"现在可以运行 test_pytorch_forecast_system.py 进行完整测试")
    
    return True

if __name__ == "__main__":
    test_directory_setup()
