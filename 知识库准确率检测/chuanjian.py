import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data():
    """生成更详细的虚拟监测数据用于演示"""
    # 创建时间序列
    start_date = datetime.now() - timedelta(days=7)
    dates = [start_date + timedelta(hours=i) for i in range(168)]  # 一周的小时数据
    
    # 创建DataFrame
    df = pd.DataFrame()
    df['时间'] = [d.strftime('%Y-%m-%d %H:%M') for d in dates]
    
    # 生成烟气排放数据
    # SO2 - 大部分时间合格，有几个峰值超标
    df['SO2_小时'] = [60 + 20*np.sin(i/12) + np.random.normal(0, 10) for i in range(len(dates))]
    df.loc[np.random.choice(df.index, 5), 'SO2_小时'] = np.random.uniform(110, 150, 5)  # 添加几个超标点
    
    # NOx - 大部分时间合格
    df['NOx_小时'] = [200 + 30*np.sin(i/24) + np.random.normal(0, 15) for i in range(len(dates))]
    df.loc[np.random.choice(df.index, 3), 'NOx_小时'] = np.random.uniform(260, 300, 3)  # 添加几个超标点
    
    # CO - 全部合格
    df['CO_小时'] = [50 + 20*np.sin(i/12) + np.random.normal(0, 10) for i in range(len(dates))]
    
    # HCl - 全部合格但接近限值
    df['HCl_小时'] = [45 + 10*np.sin(i/24) + np.random.normal(0, 5) for i in range(len(dates))]
    
    # 颗粒物 - 部分超标
    df['颗粒物_小时'] = [25 + 5*np.sin(i/12) + np.random.normal(0, 3) for i in range(len(dates))]
    df.loc[np.random.choice(df.index, 10), '颗粒物_小时'] = np.random.uniform(32, 40, 10)  # 添加几个超标点
    
    # 二噁英类 - 全部合格
    df['二噁英类'] = [0.05 + 0.02*np.sin(i/24) + np.random.normal(0, 0.01) for i in range(len(dates))]
    
    # 炉温控制数据
    # 炉温 - 部分低于要求
    df['炉温'] = [870 + 20*np.sin(i/24) + np.random.normal(0, 15) for i in range(len(dates))]
    df.loc[np.random.choice(df.index, 8), '炉温'] = np.random.uniform(820, 845, 8)  # 添加几个低温点
    
    # 灰渣热灼减率 - 全部合格
    df['灰渣热灼减率'] = [3.5 + 1.0*np.sin(i/24) + np.random.normal(0, 0.5) for i in range(len(dates))]
    
    return df

# 生成示例数据
sample_data = generate_sample_data()

# 将数据保存为CSV，用于演示
sample_data.to_csv("焚烧发电厂监测数据示例.csv", index=False)

# 打印数据样例
print(sample_data.head())