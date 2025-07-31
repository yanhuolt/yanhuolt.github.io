"""
数据融合与预处理模块
实现多源监测数据的融合、清洗、补全和归一化处理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MonitoringData:
    """监测数据结构"""
    timestamp: str
    station_id: str
    station_type: str  # 'national', 'township', 'micro', 'dust'
    location: Tuple[float, float, float]  # (x, y, z)
    pm25: Optional[float] = None  # PM2.5浓度 (μg/m³)
    pm10: Optional[float] = None  # PM10浓度 (μg/m³)
    o3: Optional[float] = None    # O3浓度 (μg/m³)
    no2: Optional[float] = None   # NO2浓度 (μg/m³)
    vocs: Optional[float] = None  # VOCs浓度 (μg/m³)
    temperature: Optional[float] = None  # 温度 (°C)
    humidity: Optional[float] = None     # 湿度 (%)
    wind_speed: Optional[float] = None   # 风速 (m/s)
    wind_direction: Optional[float] = None  # 风向 (度)
    pressure: Optional[float] = None     # 气压 (hPa)


@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_records: int
    missing_rate: Dict[str, float]
    outlier_rate: Dict[str, float]
    duplicate_rate: float
    time_coverage: Tuple[str, str]
    station_coverage: List[str]


class DataFusion:
    """数据融合与预处理类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()
        self.mlp_models = {}  # 存储各污染物的MLP补全模型
        self.data_quality_thresholds = {
            'pm25_max': 500,    # PM2.5最大合理值
            'pm10_max': 1000,   # PM10最大合理值
            'o3_max': 400,      # O3最大合理值
            'no2_max': 200,     # NO2最大合理值
            'vocs_max': 1000,   # VOCs最大合理值
            'temp_range': (-20, 50),    # 温度范围
            'humidity_range': (0, 100), # 湿度范围
            'wind_speed_max': 30,       # 最大风速
            'pressure_range': (900, 1100)  # 气压范围
        }
    
    def load_monitoring_data(self, data_sources: Dict[str, Any]) -> List[MonitoringData]:
        """
        加载多源监测数据
        
        Args:
            data_sources: 数据源字典，包含不同类型监测站的数据
            
        Returns:
            统一格式的监测数据列表
        """
        all_data = []
        
        for source_type, data in data_sources.items():
            if isinstance(data, pd.DataFrame):
                for _, row in data.iterrows():
                    monitoring_data = MonitoringData(
                        timestamp=str(row.get('timestamp', '')),
                        station_id=str(row.get('station_id', '')),
                        station_type=source_type,
                        location=(
                            float(row.get('x', 0)),
                            float(row.get('y', 0)),
                            float(row.get('z', 0))
                        ),
                        pm25=self._safe_float(row.get('pm25')),
                        pm10=self._safe_float(row.get('pm10')),
                        o3=self._safe_float(row.get('o3')),
                        no2=self._safe_float(row.get('no2')),
                        vocs=self._safe_float(row.get('vocs')),
                        temperature=self._safe_float(row.get('temperature')),
                        humidity=self._safe_float(row.get('humidity')),
                        wind_speed=self._safe_float(row.get('wind_speed')),
                        wind_direction=self._safe_float(row.get('wind_direction')),
                        pressure=self._safe_float(row.get('pressure'))
                    )
                    all_data.append(monitoring_data)
            elif isinstance(data, list):
                # 处理MonitoringData对象列表
                for item in data:
                    if isinstance(item, MonitoringData):
                        all_data.append(item)
                    else:
                        # 处理字典格式的数据
                        monitoring_data = MonitoringData(
                            timestamp=str(item.get('timestamp', '')),
                            station_id=str(item.get('station_id', '')),
                            station_type=source_type,
                            location=(
                                float(item.get('x', 0)),
                                float(item.get('y', 0)),
                                float(item.get('z', 0))
                            ),
                            pm25=self._safe_float(item.get('pm25')),
                            pm10=self._safe_float(item.get('pm10')),
                            o3=self._safe_float(item.get('o3')),
                            no2=self._safe_float(item.get('no2')),
                            vocs=self._safe_float(item.get('vocs')),
                            temperature=self._safe_float(item.get('temperature')),
                            humidity=self._safe_float(item.get('humidity')),
                            wind_speed=self._safe_float(item.get('wind_speed')),
                            wind_direction=self._safe_float(item.get('wind_direction')),
                            pressure=self._safe_float(item.get('pressure'))
                        )
                        all_data.append(monitoring_data)
        
        return all_data
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """安全转换为浮点数"""
        try:
            if pd.isna(value) or value is None:
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def data_quality_check(self, data: List[MonitoringData]) -> DataQualityReport:
        """
        数据质量检查
        
        Args:
            data: 监测数据列表
            
        Returns:
            数据质量报告
        """
        if not data:
            return DataQualityReport(0, {}, {}, 0, ('', ''), [])
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame([
            {
                'timestamp': d.timestamp,
                'station_id': d.station_id,
                'pm25': d.pm25,
                'pm10': d.pm10,
                'o3': d.o3,
                'no2': d.no2,
                'vocs': d.vocs,
                'temperature': d.temperature,
                'humidity': d.humidity,
                'wind_speed': d.wind_speed,
                'wind_direction': d.wind_direction,
                'pressure': d.pressure
            }
            for d in data
        ])
        
        total_records = len(df)
        
        # 计算缺失率
        missing_rate = {}
        for col in ['pm25', 'pm10', 'o3', 'no2', 'vocs', 'temperature', 
                   'humidity', 'wind_speed', 'wind_direction', 'pressure']:
            missing_rate[col] = df[col].isna().sum() / total_records
        
        # 计算异常值率
        outlier_rate = {}
        for col, max_val in [('pm25', self.data_quality_thresholds['pm25_max']),
                            ('pm10', self.data_quality_thresholds['pm10_max']),
                            ('o3', self.data_quality_thresholds['o3_max']),
                            ('no2', self.data_quality_thresholds['no2_max']),
                            ('vocs', self.data_quality_thresholds['vocs_max']),
                            ('wind_speed', self.data_quality_thresholds['wind_speed_max'])]:
            if col in df.columns:
                outliers = (df[col] < 0) | (df[col] > max_val)
                outlier_rate[col] = outliers.sum() / total_records
        
        # 范围检查
        for col, (min_val, max_val) in [('temperature', self.data_quality_thresholds['temp_range']),
                                       ('humidity', self.data_quality_thresholds['humidity_range']),
                                       ('pressure', self.data_quality_thresholds['pressure_range'])]:
            if col in df.columns:
                outliers = (df[col] < min_val) | (df[col] > max_val)
                outlier_rate[col] = outliers.sum() / total_records
        
        # 重复率
        duplicate_rate = df.duplicated().sum() / total_records
        
        # 时间覆盖
        timestamps = df['timestamp'].dropna()
        time_coverage = (timestamps.min(), timestamps.max()) if len(timestamps) > 0 else ('', '')
        
        # 站点覆盖
        station_coverage = df['station_id'].unique().tolist()
        
        return DataQualityReport(
            total_records=total_records,
            missing_rate=missing_rate,
            outlier_rate=outlier_rate,
            duplicate_rate=duplicate_rate,
            time_coverage=time_coverage,
            station_coverage=station_coverage
        )
    
    def clean_data(self, data: List[MonitoringData]) -> List[MonitoringData]:
        """
        数据清洗：去除异常值、重复值等
        
        Args:
            data: 原始监测数据
            
        Returns:
            清洗后的数据
        """
        cleaned_data = []
        
        for d in data:
            # 检查异常值
            if self._is_valid_data(d):
                cleaned_data.append(d)
        
        # 去重（基于时间戳和站点ID）
        seen = set()
        unique_data = []
        for d in cleaned_data:
            key = (d.timestamp, d.station_id)
            if key not in seen:
                seen.add(key)
                unique_data.append(d)
        
        return unique_data
    
    def _is_valid_data(self, data: MonitoringData) -> bool:
        """检查数据是否有效"""
        # 检查污染物浓度
        if data.pm25 is not None and (data.pm25 < 0 or data.pm25 > self.data_quality_thresholds['pm25_max']):
            return False
        if data.pm10 is not None and (data.pm10 < 0 or data.pm10 > self.data_quality_thresholds['pm10_max']):
            return False
        if data.o3 is not None and (data.o3 < 0 or data.o3 > self.data_quality_thresholds['o3_max']):
            return False
        if data.no2 is not None and (data.no2 < 0 or data.no2 > self.data_quality_thresholds['no2_max']):
            return False
        if data.vocs is not None and (data.vocs < 0 or data.vocs > self.data_quality_thresholds['vocs_max']):
            return False
        
        # 检查气象参数
        temp_min, temp_max = self.data_quality_thresholds['temp_range']
        if data.temperature is not None and (data.temperature < temp_min or data.temperature > temp_max):
            return False
        
        hum_min, hum_max = self.data_quality_thresholds['humidity_range']
        if data.humidity is not None and (data.humidity < hum_min or data.humidity > hum_max):
            return False
        
        if data.wind_speed is not None and (data.wind_speed < 0 or data.wind_speed > self.data_quality_thresholds['wind_speed_max']):
            return False
        
        pres_min, pres_max = self.data_quality_thresholds['pressure_range']
        if data.pressure is not None and (data.pressure < pres_min or data.pressure > pres_max):
            return False
        
        return True
    
    def train_mlp_imputation_models(self, data: List[MonitoringData]) -> None:
        """
        训练MLP神经网络数据补全模型
        
        Args:
            data: 训练数据
        """
        # 转换为DataFrame
        df = pd.DataFrame([
            {
                'pm25': d.pm25, 'pm10': d.pm10, 'o3': d.o3, 'no2': d.no2, 'vocs': d.vocs,
                'temperature': d.temperature, 'humidity': d.humidity,
                'wind_speed': d.wind_speed, 'wind_direction': d.wind_direction,
                'pressure': d.pressure
            }
            for d in data
        ])
        
        # 为每个污染物训练MLP模型
        pollutants = ['pm25', 'pm10', 'o3', 'no2', 'vocs']
        
        for target in pollutants:
            # 选择特征（其他污染物和气象参数）
            features = [col for col in df.columns if col != target]
            
            # 获取完整数据（目标变量和特征都不为空）
            complete_mask = df[target].notna()
            for feature in features:
                complete_mask &= df[feature].notna()
            
            if complete_mask.sum() < 50:  # 数据太少，跳过
                continue
            
            X = df.loc[complete_mask, features].values
            y = df.loc[complete_mask, target].values
            
            if len(X) > 0:
                # 训练MLP模型
                mlp = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.2
                )
                
                try:
                    mlp.fit(X, y)
                    self.mlp_models[target] = {
                        'model': mlp,
                        'features': features,
                        'scaler': StandardScaler().fit(X)
                    }
                except:
                    continue
    
    def impute_missing_data(self, data: List[MonitoringData]) -> List[MonitoringData]:
        """
        使用MLP模型补全缺失数据
        
        Args:
            data: 包含缺失值的数据
            
        Returns:
            补全后的数据
        """
        if not self.mlp_models:
            return data
        
        imputed_data = []
        
        for d in data:
            new_data = MonitoringData(
                timestamp=d.timestamp,
                station_id=d.station_id,
                station_type=d.station_type,
                location=d.location,
                pm25=d.pm25,
                pm10=d.pm10,
                o3=d.o3,
                no2=d.no2,
                vocs=d.vocs,
                temperature=d.temperature,
                humidity=d.humidity,
                wind_speed=d.wind_speed,
                wind_direction=d.wind_direction,
                pressure=d.pressure
            )
            
            # 为每个缺失的污染物进行预测
            data_dict = {
                'pm25': d.pm25, 'pm10': d.pm10, 'o3': d.o3, 'no2': d.no2, 'vocs': d.vocs,
                'temperature': d.temperature, 'humidity': d.humidity,
                'wind_speed': d.wind_speed, 'wind_direction': d.wind_direction,
                'pressure': d.pressure
            }
            
            for target, model_info in self.mlp_models.items():
                if data_dict[target] is None:  # 如果该污染物缺失
                    # 检查特征是否完整
                    features_available = True
                    feature_values = []
                    
                    for feature in model_info['features']:
                        if data_dict[feature] is None:
                            features_available = False
                            break
                        feature_values.append(data_dict[feature])
                    
                    if features_available:
                        try:
                            # 标准化特征
                            X = np.array(feature_values).reshape(1, -1)
                            X_scaled = model_info['scaler'].transform(X)
                            
                            # 预测
                            predicted_value = model_info['model'].predict(X_scaled)[0]
                            
                            # 更新数据
                            setattr(new_data, target, max(0, predicted_value))  # 确保非负
                        except:
                            continue
            
            imputed_data.append(new_data)
        
        return imputed_data
    
    def normalize_data(self, data: List[MonitoringData]) -> List[MonitoringData]:
        """
        数据归一化处理
        
        Args:
            data: 原始数据
            
        Returns:
            归一化后的数据
        """
        if not data:
            return data
        
        # 提取数值数据
        values = []
        for d in data:
            values.append([
                d.pm25 or 0, d.pm10 or 0, d.o3 or 0, d.no2 or 0, d.vocs or 0,
                d.temperature or 0, d.humidity or 0, d.wind_speed or 0, d.pressure or 0
            ])
        
        values = np.array(values)
        
        # 归一化
        normalized_values = self.normalizer.fit_transform(values)
        
        # 更新数据
        normalized_data = []
        for i, d in enumerate(data):
            new_data = MonitoringData(
                timestamp=d.timestamp,
                station_id=d.station_id,
                station_type=d.station_type,
                location=d.location,
                pm25=normalized_values[i, 0] if d.pm25 is not None else None,
                pm10=normalized_values[i, 1] if d.pm10 is not None else None,
                o3=normalized_values[i, 2] if d.o3 is not None else None,
                no2=normalized_values[i, 3] if d.no2 is not None else None,
                vocs=normalized_values[i, 4] if d.vocs is not None else None,
                temperature=normalized_values[i, 5] if d.temperature is not None else None,
                humidity=normalized_values[i, 6] if d.humidity is not None else None,
                wind_speed=normalized_values[i, 7] if d.wind_speed is not None else None,
                wind_direction=d.wind_direction,  # 风向不归一化
                pressure=normalized_values[i, 8] if d.pressure is not None else None
            )
            normalized_data.append(new_data)
        
        return normalized_data
    
    def process_data_pipeline(self, 
                            data_sources: Dict[str, Any],
                            enable_imputation: bool = True,
                            enable_normalization: bool = True) -> Tuple[List[MonitoringData], DataQualityReport]:
        """
        完整的数据处理流水线
        
        Args:
            data_sources: 多源数据
            enable_imputation: 是否启用数据补全
            enable_normalization: 是否启用数据归一化
            
        Returns:
            (处理后的数据, 数据质量报告)
        """
        # 1. 加载数据
        raw_data = self.load_monitoring_data(data_sources)
        
        # 2. 数据质量检查
        quality_report = self.data_quality_check(raw_data)
        
        # 3. 数据清洗
        cleaned_data = self.clean_data(raw_data)
        
        # 4. 数据补全
        if enable_imputation:
            self.train_mlp_imputation_models(cleaned_data)
            imputed_data = self.impute_missing_data(cleaned_data)
        else:
            imputed_data = cleaned_data
        
        # 5. 数据归一化
        if enable_normalization:
            final_data = self.normalize_data(imputed_data)
        else:
            final_data = imputed_data
        
        return final_data, quality_report
