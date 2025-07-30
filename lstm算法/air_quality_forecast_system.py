"""
基于PyTorch LSTM的空气质量预报预警和污染成因分析系统
结合气象、传输路径等因素，提供重点点位的空气质量预报预警服务
在污染过程中分析成因并提出针对性管控建议
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AirQualityDataset(Dataset):
    """空气质量数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMForecastModel(nn.Module):
    """LSTM预报模型"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
        super(LSTMForecastModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步的输出
        last_output = attn_out[:, -1, :]
        
        # 全连接层
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

class AirQualityForecastSystem:
    """
    基于LSTM的空气质量预报预警系统
    功能：
    1. 多因子空气质量预报
    2. 污染预警分级
    3. 污染成因分析
    4. 管控建议生成
    """
    
    def __init__(self, sequence_length=24, forecast_horizon=72, device=None, model_save_path=None):
        """
        初始化预报系统
        参数:
            sequence_length: 输入序列长度（小时）
            forecast_horizon: 预报时长（小时）
            device: 计算设备
            model_save_path: 模型保存路径
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = model_save_path if model_save_path else 'best_model.pth'

        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_target = MinMaxScaler()
        self.feature_names = []
        self.is_trained = False
        
        # 污染等级阈值（基于国标）
        self.aqi_thresholds = {
            'PM2.5': [35, 75, 115, 150, 250, 350],  # 优良轻中重严重
            'PM10': [50, 150, 250, 350, 420, 500],
            'O3': [100, 160, 215, 265, 800, 1000],
            'NO2': [40, 80, 180, 280, 565, 750],
            'SO2': [50, 150, 475, 800, 1600, 2100],
            'CO': [2, 4, 14, 24, 36, 48]
        }
        
        # 预警等级
        self.warning_levels = {
            0: '优', 1: '良', 2: '轻度污染', 
            3: '中度污染', 4: '重度污染', 5: '严重污染'
        }
        
        # 管控措施建议
        self.control_measures = {
            2: ['建议减少户外活动', '关注敏感人群健康'],
            3: ['减少户外运动', '工业企业限产20%', '机动车限行'],
            4: ['停止户外活动', '工业企业限产50%', '建筑工地停工', '机动车单双号限行'],
            5: ['全面停止户外活动', '工业企业停产', '全面停工停课', '机动车禁行']
        }

    def prepare_features(self, data, exclude_target=None):
        """
        准备多维特征数据
        包括：污染物浓度、气象要素、时间特征、空间特征
        参数:
            data: 输入数据
            exclude_target: 要排除的目标列
        """
        features = []

        # 1. 污染物浓度特征
        pollutant_cols = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
        for col in pollutant_cols:
            if col in data.columns and col != exclude_target:
                features.append(col)
        
        # 2. 气象要素特征
        weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 
                       'wind_direction', 'precipitation', 'visibility']
        for col in weather_cols:
            if col in data.columns:
                features.append(col)
        
        # 3. 时间特征
        if 'datetime' in data.columns:
            data['hour'] = pd.to_datetime(data['datetime']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['datetime']).dt.dayofweek
            data['month'] = pd.to_datetime(data['datetime']).dt.month
            data['season'] = pd.to_datetime(data['datetime']).dt.month % 12 // 3 + 1
            features.extend(['hour', 'day_of_week', 'month', 'season'])
        
        # 4. 气象稳定度特征
        if 'wind_speed' in data.columns and 'temperature' in data.columns:
            data['stability_index'] = self.calculate_stability_index(
                data['wind_speed'], data['temperature']
            )
            features.append('stability_index')
        
        # 5. 传输路径特征
        if 'wind_direction' in data.columns and 'wind_speed' in data.columns:
            data['wind_u'] = data['wind_speed'] * np.cos(np.radians(data['wind_direction']))
            data['wind_v'] = data['wind_speed'] * np.sin(np.radians(data['wind_direction']))
            features.extend(['wind_u', 'wind_v'])
        
        # 6. 边界层高度特征
        if 'temperature' in data.columns and 'wind_speed' in data.columns:
            data['mixing_height'] = self.estimate_mixing_height(
                data['temperature'], data['wind_speed']
            )
            features.append('mixing_height')
        
        self.feature_names = features
        return data[features]

    def calculate_stability_index(self, wind_speed, temperature):
        """计算大气稳定度指数"""
        temp_gradient = temperature.diff().fillna(0)
        stability = temp_gradient / (wind_speed + 0.1)**2
        return stability.fillna(0)

    def estimate_mixing_height(self, temperature, wind_speed):
        """估算混合层高度"""
        base_height = 500
        temp_factor = (temperature - temperature.mean()) * 10
        wind_factor = wind_speed * 50
        mixing_height = base_height + temp_factor + wind_factor
        return np.clip(mixing_height, 100, 3000)

    def create_sequences(self, data, target_col='PM2.5'):
        """
        创建LSTM训练序列
        参数:
            data: 特征数据
            target_col: 目标预测列
        返回:
            X: 输入序列 [samples, sequence_length, features]
            y: 目标序列 [samples, forecast_horizon]
        """
        if target_col not in data.columns:
            raise ValueError(f"目标列 '{target_col}' 不在数据中")

        features = data.drop(columns=[target_col])
        target = data[target_col].values  # 转换为numpy数组

        X, y = [], []

        min_length = self.sequence_length + self.forecast_horizon
        if len(data) < min_length:
            raise ValueError(f"数据长度 {len(data)} 小于所需的最小长度 {min_length}")

        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # 输入序列
            X.append(features.iloc[i:i + self.sequence_length].values)
            # 目标序列 - 直接从numpy数组切片，确保是1维
            y.append(target[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])

        X = np.array(X)  # [samples, sequence_length, features]
        y = np.array(y)  # [samples, forecast_horizon]

        print(f"创建序列完成: X.shape={X.shape}, y.shape={y.shape}")

        # 确保y是正确的2维形状
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif len(y.shape) == 3:
            y = y.reshape(y.shape[0], -1)

        print(f"调整后y形状: {y.shape}")
        return X, y

    def create_sequences_fixed(self, data, target_col='PM2.5'):
        """
        修正的序列创建方法，确保维度正确
        """
        if target_col not in data.columns:
            raise ValueError(f"目标列 '{target_col}' 不在数据中")

        features = data.drop(columns=[target_col])
        target = data[target_col].values

        X, y = [], []

        min_length = self.sequence_length + self.forecast_horizon
        if len(data) < min_length:
            raise ValueError(f"数据长度 {len(data)} 小于所需的最小长度 {min_length}")

        print(f"开始创建序列，数据长度: {len(data)}, 特征数: {features.shape[1]}")

        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # 输入序列 [sequence_length, features]
            X.append(features.iloc[i:i + self.sequence_length].values)

            # 目标序列 [forecast_horizon] - 注意这里只取一维
            target_seq = target[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]

            # 确保目标序列长度正确
            if len(target_seq) == self.forecast_horizon:
                y.append(target_seq)  # 直接添加1维数组
            else:
                # 移除对应的X样本
                X.pop()

        X = np.array(X)  # [samples, sequence_length, features]
        y = np.array(y)  # [samples, forecast_horizon]

        print(f"修正序列创建完成: X.shape={X.shape}, y.shape={y.shape}")
        print(f"期望的forecast_horizon: {self.forecast_horizon}")

        # 确保y是2维的
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif len(y.shape) == 3:
            # 如果是3维，压缩到2维
            y = y.reshape(y.shape[0], -1)

        print(f"最终y形状: {y.shape}")

        return X, y

    def train_model(self, data, target_col='PM2.5', validation_split=0.2, 
                   epochs=100, batch_size=32, learning_rate=0.001):
        """
        训练LSTM模型
        参数:
            data: 训练数据
            target_col: 目标预测列
            validation_split: 验证集比例
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        print("开始训练LSTM空气质量预报模型...")
        
        # 准备特征（排除目标列）
        features_data = self.prepare_features(data, exclude_target=target_col)
        print(f"特征数据形状: {features_data.shape}")
        print(f"特征列: {self.feature_names}")
        print(f"排除的目标列: {target_col}")

        # 数据标准化
        features_scaled = self.scaler_features.fit_transform(features_data)
        target_data = data[target_col].values.reshape(-1, 1)
        target_scaled = self.scaler_target.fit_transform(target_data)

        print(f"标准化后特征形状: {features_scaled.shape}")
        print(f"标准化后目标形状: {target_scaled.shape}")

        # 重新组合数据
        scaled_data = pd.DataFrame(
            np.column_stack([features_scaled, target_scaled.flatten()]),
            columns=self.feature_names + [target_col]
        )

        print(f"组合后数据形状: {scaled_data.shape}")
        
        # 创建序列
        X, y = self.create_sequences(scaled_data, target_col)
        print(f"训练数据形状: X={X.shape}, y={y.shape}")

        # 确保y的形状正确
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # 如果y的第二维不等于forecast_horizon，需要调整
        if y.shape[1] != self.forecast_horizon:
            print(f"警告: y的形状 {y.shape} 与预期的forecast_horizon {self.forecast_horizon} 不匹配")
            # 重新创建序列，确保正确的forecast_horizon
            X, y = self.create_sequences_fixed(scaled_data, target_col)
            print(f"修正后训练数据形状: X={X.shape}, y={y.shape}")

        # 分割训练和验证数据
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"验证集形状: X_val={X_val.shape}, y_val={y_val.shape}")
        print(f"期望的forecast_horizon: {self.forecast_horizon}")

        # 再次检查y的维度
        if y_train.shape[1] != self.forecast_horizon:
            raise ValueError(f"y_train的第二维 {y_train.shape[1]} 不等于forecast_horizon {self.forecast_horizon}")

        # 创建数据集和数据加载器
        train_dataset = AirQualityDataset(X_train, y_train)
        val_dataset = AirQualityDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 构建模型
        input_size = X.shape[2]
        self.model = LSTMForecastModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            output_size=self.forecast_horizon,
            dropout=0.2
        ).to(self.device)
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 训练历史
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("开始训练...")
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            batch_count = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # 在第一个batch时打印调试信息
                if epoch == 0 and batch_count == 0:
                    print(f"第一个batch - batch_X形状: {batch_X.shape}, batch_y形状: {batch_y.shape}")

                optimizer.zero_grad()
                outputs = self.model(batch_X)

                # 在第一个batch时打印输出形状
                if epoch == 0 and batch_count == 0:
                    print(f"模型输出形状: {outputs.shape}")
                    print(f"期望目标形状: {batch_y.shape}")

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                batch_count += 1
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型到指定路径
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"保存最佳模型至: {self.model_save_path}")
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if patience_counter >= 15:
                print("早停触发")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.is_trained = True

        print(f"模型训练完成！最佳模型已保存至: {self.model_save_path}")
        return {'train_losses': train_losses, 'val_losses': val_losses}

    def predict_air_quality(self, recent_data, target_col='PM2.5'):
        """
        预测空气质量
        参数:
            recent_data: 最近的观测数据（至少sequence_length长度）
            target_col: 预测目标
        返回:
            predictions: 预测结果
            confidence_intervals: 置信区间
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train_model方法")

        # 准备特征（排除目标列）
        features_data = self.prepare_features(recent_data, exclude_target=target_col)

        # 标准化
        features_scaled = self.scaler_features.transform(features_data)

        # 取最后sequence_length个时间步
        X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        X = torch.FloatTensor(X).to(self.device)

        # 预测
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X).cpu().numpy()

        # 反标准化
        predictions = self.scaler_target.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()

        # 计算置信区间（基于模型不确定性）
        # 使用Monte Carlo Dropout进行不确定性估计
        self.model.train()  # 启用dropout
        mc_predictions = []

        with torch.no_grad():
            for _ in range(50):  # Monte Carlo采样
                pred = self.model(X).cpu().numpy()
                pred_unscaled = self.scaler_target.inverse_transform(
                    pred.reshape(-1, 1)
                ).flatten()
                mc_predictions.append(pred_unscaled)

        mc_predictions = np.array(mc_predictions)
        prediction_std = np.std(mc_predictions, axis=0)

        confidence_intervals = {
            'lower': predictions - 1.96 * prediction_std,
            'upper': predictions + 1.96 * prediction_std
        }

        return predictions, confidence_intervals

    def generate_warnings(self, predictions, pollutant='PM2.5'):
        """
        生成预警信息
        参数:
            predictions: 预测浓度值
            pollutant: 污染物类型
        返回:
            warnings: 预警信息列表
        """
        warnings = []
        thresholds = self.aqi_thresholds.get(pollutant, self.aqi_thresholds['PM2.5'])

        for i, pred in enumerate(predictions):
            # 确定污染等级
            level = 0
            for j, threshold in enumerate(thresholds):
                if pred > threshold:
                    level = j + 1
                else:
                    break

            # 生成预警信息
            warning_info = {
                'hour': i + 1,
                'predicted_concentration': pred,
                'pollution_level': level,
                'level_name': self.warning_levels[level],
                'warning_message': self.generate_warning_message(level, pred, pollutant),
                'control_measures': self.control_measures.get(level, [])
            }
            warnings.append(warning_info)

        return warnings

    def generate_warning_message(self, level, concentration, pollutant):
        """生成预警消息"""
        if level <= 1:
            return f"{pollutant}浓度{concentration:.1f}μg/m³，空气质量{self.warning_levels[level]}"
        elif level == 2:
            return f"{pollutant}浓度{concentration:.1f}μg/m³，轻度污染，建议减少户外活动"
        elif level == 3:
            return f"{pollutant}浓度{concentration:.1f}μg/m³，中度污染，建议启动应急响应"
        elif level == 4:
            return f"{pollutant}浓度{concentration:.1f}μg/m³，重度污染，启动红色预警"
        else:
            return f"{pollutant}浓度{concentration:.1f}μg/m³，严重污染，启动最高级别应急响应"

    def analyze_pollution_causes(self, data, pollution_period):
        """
        分析污染成因
        参数:
            data: 历史数据
            pollution_period: 污染时段 [start_time, end_time]
        返回:
            cause_analysis: 成因分析结果
        """
        start_time, end_time = pollution_period

        # 提取污染时段数据
        pollution_data = data[
            (data['datetime'] >= start_time) & (data['datetime'] <= end_time)
        ]

        # 提取对比时段数据（污染前7天同时段）
        compare_start = start_time - timedelta(days=7)
        compare_end = end_time - timedelta(days=7)
        compare_data = data[
            (data['datetime'] >= compare_start) & (data['datetime'] <= compare_end)
        ]

        cause_analysis = {
            'meteorological_factors': self.analyze_meteorological_factors(
                pollution_data, compare_data
            ),
            'emission_factors': self.analyze_emission_factors(
                pollution_data, compare_data
            ),
            'transport_factors': self.analyze_transport_factors(
                pollution_data, compare_data
            ),
            'secondary_formation': self.analyze_secondary_formation(
                pollution_data, compare_data
            )
        }

        # 综合评估各因子贡献
        cause_analysis['comprehensive_assessment'] = self.assess_factor_contributions(
            cause_analysis
        )

        return cause_analysis

    def analyze_meteorological_factors(self, pollution_data, compare_data):
        """分析气象因子影响"""
        analysis = {}

        # 风速影响
        if 'wind_speed' in pollution_data.columns:
            pollution_wind = pollution_data['wind_speed'].mean()
            compare_wind = compare_data['wind_speed'].mean()

            # 避免除零错误
            if compare_wind != 0:
                wind_change = (pollution_wind - compare_wind) / compare_wind * 100
            else:
                wind_change = 0

            analysis['wind_speed'] = {
                'pollution_period': pollution_wind,
                'compare_period': compare_wind,
                'change_percent': wind_change,
                'impact': '不利' if wind_change < -20 else '中性' if abs(wind_change) < 20 else '有利',
                'description': self.get_wind_impact_description(wind_change)
            }

        # 湿度影响
        if 'humidity' in pollution_data.columns:
            pollution_humidity = pollution_data['humidity'].mean()
            compare_humidity = compare_data['humidity'].mean()

            # 避免除零错误
            if compare_humidity != 0:
                humidity_change = (pollution_humidity - compare_humidity) / compare_humidity * 100
            else:
                humidity_change = 0

            analysis['humidity'] = {
                'pollution_period': pollution_humidity,
                'compare_period': compare_humidity,
                'change_percent': humidity_change,
                'impact': '不利' if humidity_change > 15 else '中性' if abs(humidity_change) < 15 else '有利',
                'description': self.get_humidity_impact_description(humidity_change)
            }

        # 混合层高度影响
        if 'mixing_height' in pollution_data.columns:
            pollution_height = pollution_data['mixing_height'].mean()
            compare_height = compare_data['mixing_height'].mean()

            # 避免除零错误
            if compare_height != 0:
                height_change = (pollution_height - compare_height) / compare_height * 100
            else:
                height_change = 0

            analysis['mixing_height'] = {
                'pollution_period': pollution_height,
                'compare_period': compare_height,
                'change_percent': height_change,
                'impact': '不利' if height_change < -30 else '中性' if abs(height_change) < 30 else '有利',
                'description': self.get_mixing_height_impact_description(height_change)
            }

        return analysis

    def analyze_emission_factors(self, pollution_data, compare_data):
        """分析排放因子影响"""
        analysis = {}

        # 一次污染物分析
        primary_pollutants = ['NO2', 'SO2', 'CO']

        for pollutant in primary_pollutants:
            if pollutant in pollution_data.columns:
                pollution_conc = pollution_data[pollutant].mean()
                compare_conc = compare_data[pollutant].mean()

                # 避免除零错误
                if compare_conc != 0:
                    change_percent = (pollution_conc - compare_conc) / compare_conc * 100
                else:
                    change_percent = 0

                analysis[pollutant] = {
                    'pollution_period': pollution_conc,
                    'compare_period': compare_conc,
                    'change_percent': change_percent,
                    'emission_impact': self.assess_emission_impact(change_percent),
                    'source_indication': self.get_emission_source_indication(pollutant, change_percent)
                }

        return analysis

    def analyze_transport_factors(self, pollution_data, compare_data):
        """分析传输因子影响"""
        analysis = {}

        # 风向分析
        if 'wind_direction' in pollution_data.columns:
            pollution_wind_dir = pollution_data['wind_direction'].mean()

            analysis['wind_direction'] = {
                'pollution_period': pollution_wind_dir,
                'transport_potential': self.assess_transport_potential(pollution_wind_dir),
                'source_region': self.identify_potential_source_region(pollution_wind_dir)
            }

        # 风速持续性分析
        if 'wind_speed' in pollution_data.columns:
            low_wind_hours = len(pollution_data[pollution_data['wind_speed'] < 2])
            total_hours = len(pollution_data)

            # 避免除零错误
            if total_hours > 0:
                stagnation_ratio = low_wind_hours / total_hours
            else:
                stagnation_ratio = 0

            analysis['stagnation'] = {
                'low_wind_hours': low_wind_hours,
                'total_hours': total_hours,
                'stagnation_ratio': stagnation_ratio,
                'transport_capacity': '弱' if stagnation_ratio > 0.6 else '中等' if stagnation_ratio > 0.3 else '强'
            }

        return analysis

    def analyze_secondary_formation(self, pollution_data, compare_data):
        """分析二次生成因子"""
        analysis = {}

        # O3与前体物关系分析
        if all(col in pollution_data.columns for col in ['O3', 'NO2']):
            o3_pollution = pollution_data['O3'].mean()
            no2_pollution = pollution_data['NO2'].mean()

            # O3/NO2比值
            pollution_ratio = o3_pollution / (no2_pollution + 1)

            analysis['O3_formation'] = {
                'O3_concentration': o3_pollution,
                'NO2_concentration': no2_pollution,
                'ratio': pollution_ratio,
                'formation_regime': self.identify_o3_formation_regime(pollution_ratio),
                'secondary_contribution': self.assess_secondary_o3_contribution(o3_pollution)
            }

        return analysis

    def assess_factor_contributions(self, cause_analysis):
        """综合评估各因子贡献度"""
        contributions = {
            'meteorological': 0,
            'emission': 0,
            'transport': 0,
            'secondary': 0
        }

        # 气象因子贡献评估
        met_factors = cause_analysis['meteorological_factors']
        met_score = 0
        for factor_name, data in met_factors.items():
            if data['impact'] == '不利':
                met_score += 25
            elif data['impact'] == '中性':
                met_score += 10
        contributions['meteorological'] = min(met_score, 100)

        # 排放因子贡献评估
        emission_factors = cause_analysis['emission_factors']
        emission_score = 0
        for factor_name, data in emission_factors.items():
            if 'change_percent' in data and data['change_percent'] > 20:
                emission_score += 30
            elif 'change_percent' in data and data['change_percent'] > 10:
                emission_score += 15
        contributions['emission'] = min(emission_score, 100)

        # 传输因子贡献评估
        transport_factors = cause_analysis['transport_factors']
        if 'stagnation' in transport_factors:
            stagnation_ratio = transport_factors['stagnation']['stagnation_ratio']
            contributions['transport'] = min(stagnation_ratio * 100, 100)

        # 二次生成贡献评估
        secondary_factors = cause_analysis['secondary_formation']
        secondary_score = 0
        for factor_name, data in secondary_factors.items():
            if 'secondary_contribution' in data and data['secondary_contribution'] == '高':
                secondary_score += 40
            elif 'secondary_contribution' in data and data['secondary_contribution'] == '中等':
                secondary_score += 20
        contributions['secondary'] = min(secondary_score, 100)

        # 归一化贡献度
        total_score = sum(contributions.values())
        if total_score > 0:
            contributions = {k: v/total_score*100 for k, v in contributions.items()}

        return contributions

    def generate_control_recommendations(self, cause_analysis, pollution_level):
        """
        生成针对性管控建议
        参数:
            cause_analysis: 成因分析结果
            pollution_level: 污染等级
        返回:
            recommendations: 管控建议
        """
        recommendations = {
            'immediate_measures': [],
            'source_control': [],
            'meteorological_response': [],
            'long_term_measures': [],
            'priority_level': pollution_level
        }

        contributions = cause_analysis['comprehensive_assessment']

        # 基于污染等级的基础措施
        if pollution_level >= 3:
            recommendations['immediate_measures'].extend([
                '启动重污染天气应急响应',
                '发布健康防护提示',
                '加强空气质量监测频次'
            ])

        # 基于排放因子的管控建议
        if contributions['emission'] > 30:
            emission_factors = cause_analysis['emission_factors']

            # 工业源管控
            if 'SO2' in emission_factors and emission_factors['SO2']['change_percent'] > 20:
                recommendations['source_control'].extend([
                    '加强工业企业SO2排放监管',
                    '重点行业限产30-50%',
                    '燃煤电厂超低排放改造'
                ])

            # 机动车源管控
            if 'NO2' in emission_factors and emission_factors['NO2']['change_percent'] > 15:
                recommendations['source_control'].extend([
                    '实施机动车限行措施',
                    '加强柴油车排放检查',
                    '推广新能源车辆使用'
                ])

        # 基于气象条件的应对措施
        if contributions['meteorological'] > 25:
            met_factors = cause_analysis['meteorological_factors']

            if 'wind_speed' in met_factors and met_factors['wind_speed']['impact'] == '不利':
                recommendations['meteorological_response'].extend([
                    '静稳天气条件下加强污染源管控',
                    '适时开展人工增雨作业',
                    '重点区域实施更严格的排放限制'
                ])

        # 基于传输因子的管控建议
        if contributions['transport'] > 30:
            transport_factors = cause_analysis['transport_factors']

            if 'wind_direction' in transport_factors:
                source_region = transport_factors['wind_direction']['source_region']
                recommendations['source_control'].extend([
                    f'加强{source_region}方向污染源管控',
                    '启动区域联防联控机制',
                    '协调上风向地区减排措施'
                ])

        # 长期措施建议
        recommendations['long_term_measures'] = [
            '完善空气质量监测网络',
            '建立污染源动态清单',
            '推进产业结构优化升级',
            '加强区域协同治理',
            '提升科学治污水平'
        ]

        return recommendations

    # 辅助方法
    def get_wind_impact_description(self, wind_change):
        """风速影响描述"""
        if wind_change < -30:
            return "风速显著减弱，大气扩散条件明显恶化"
        elif wind_change < -15:
            return "风速减弱，不利于污染物扩散"
        elif wind_change > 30:
            return "风速显著增强，有利于污染物扩散"
        elif wind_change > 15:
            return "风速增强，改善大气扩散条件"
        else:
            return "风速变化不大，扩散条件基本稳定"

    def get_humidity_impact_description(self, humidity_change):
        """湿度影响描述"""
        if humidity_change > 20:
            return "湿度显著增加，有利于二次颗粒物生成"
        elif humidity_change > 10:
            return "湿度增加，可能促进颗粒物吸湿增长"
        elif humidity_change < -20:
            return "湿度显著降低，抑制二次颗粒物生成"
        else:
            return "湿度变化对污染影响较小"

    def get_mixing_height_impact_description(self, height_change):
        """混合层高度影响描述"""
        if height_change < -40:
            return "混合层高度显著降低，垂直扩散能力严重受限"
        elif height_change < -20:
            return "混合层高度降低，不利于污染物垂直扩散"
        elif height_change > 40:
            return "混合层高度显著增加，有利于污染物垂直扩散"
        else:
            return "混合层高度变化对扩散影响较小"

    def assess_emission_impact(self, change_percent):
        """评估排放影响"""
        if change_percent > 30:
            return '显著增加'
        elif change_percent > 15:
            return '明显增加'
        elif change_percent > 5:
            return '轻微增加'
        elif change_percent < -15:
            return '明显减少'
        else:
            return '基本稳定'

    def get_emission_source_indication(self, pollutant, change_percent):
        """获取排放源指示"""
        source_map = {
            'NO2': '机动车、工业燃烧',
            'SO2': '燃煤、工业过程',
            'CO': '机动车、工业燃烧'
        }

        if change_percent > 20:
            return f"{source_map.get(pollutant, '相关源')}排放显著增加"
        elif change_percent > 10:
            return f"{source_map.get(pollutant, '相关源')}排放有所增加"
        else:
            return f"{source_map.get(pollutant, '相关源')}排放相对稳定"

    def assess_transport_potential(self, wind_direction):
        """评估传输潜力"""
        if 315 <= wind_direction <= 360 or 0 <= wind_direction <= 45:
            return "高传输潜力"
        elif 270 <= wind_direction <= 315:
            return "中等传输潜力"
        else:
            return "低传输潜力"

    def identify_potential_source_region(self, wind_direction):
        """识别潜在污染源区域"""
        if 315 <= wind_direction <= 360 or 0 <= wind_direction <= 45:
            return "北部/东北部"
        elif 45 < wind_direction <= 135:
            return "东部/东南部"
        elif 135 < wind_direction <= 225:
            return "南部/西南部"
        else:
            return "西部/西北部"

    def identify_o3_formation_regime(self, o3_no2_ratio):
        """识别臭氧生成机制"""
        if o3_no2_ratio > 10:
            return "VOCs限制型"
        elif o3_no2_ratio > 5:
            return "过渡型"
        else:
            return "NOx限制型"

    def assess_secondary_o3_contribution(self, o3_concentration):
        """评估二次臭氧贡献"""
        if o3_concentration > 160:
            return "高"
        elif o3_concentration > 100:
            return "中等"
        else:
            return "低"

    def run_forecast_system(self, historical_data, current_data, target_pollutant='PM2.5'):
        """
        运行完整的预报预警系统
        参数:
            historical_data: 历史训练数据
            current_data: 当前观测数据
            target_pollutant: 目标污染物
        返回:
            system_output: 系统输出结果
        """
        print("=== 启动空气质量预报预警系统 ===")

        # 1. 训练预报模型
        print("1. 训练LSTM预报模型...")
        history = self.train_model(historical_data, target_pollutant)

        # 2. 生成预报
        print("2. 生成空气质量预报...")
        predictions, confidence_intervals = self.predict_air_quality(
            current_data, target_pollutant
        )

        # 3. 生成预警信息
        print("3. 生成预警信息...")
        warnings = self.generate_warnings(predictions, target_pollutant)

        # 4. 检查是否存在污染过程
        max_level = max([w['pollution_level'] for w in warnings])

        system_output = {
            'forecast_time': datetime.now(),
            'target_pollutant': target_pollutant,
            'predictions': predictions.tolist(),
            'confidence_intervals': confidence_intervals,
            'warnings': warnings,
            'max_pollution_level': max_level,
            'training_history': history
        }

        # 5. 如果预测有污染过程，进行成因分析
        if max_level >= 2:  # 轻度污染及以上
            print("4. 检测到污染过程，进行成因分析...")

            # 确定污染时段
            pollution_start = datetime.now()
            pollution_end = pollution_start + timedelta(hours=self.forecast_horizon)

            # 成因分析
            cause_analysis = self.analyze_pollution_causes(
                historical_data, [pollution_start, pollution_end]
            )

            # 生成管控建议
            print("5. 生成管控建议...")
            control_recommendations = self.generate_control_recommendations(
                cause_analysis, max_level
            )

            system_output.update({
                'pollution_detected': True,
                'cause_analysis': cause_analysis,
                'control_recommendations': control_recommendations
            })
        else:
            system_output['pollution_detected'] = False

        print("=== 预报预警系统运行完成 ===")
        return system_output
