#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块
负责数据清洗、路网构建、特征提取和数据格式转换
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger('TrafficPrediction')

# 数据加载函数
def load_dataset(dataset_name, **kwargs):
    """
    加载数据集 - 强制使用真实数据集
    
    Args:
        dataset_name: 数据集名称，支持 'SZBZ'、'Los-loop'
        **kwargs: 其他参数，可传入 config_file 指定配置文件路径
    
    Returns:
        pandas.DataFrame: 加载的数据集
    """
    # 尝试导入数据配置模块
    try:
        from data_config import DataConfig
        data_config = DataConfig()
        
        # 加载配置文件
        config_file = kwargs.get('config_file', None)
        config = data_config.load_config(config_file)
        
        # 强制使用真实数据集，忽略enabled设置
        real_data_config = config.get('real_data_config', {})
        logger.info('强制使用真实数据集')
        return load_real_dataset(real_data_config)
    except ImportError:
        logger.error('未能导入数据配置模块，无法加载真实数据集')
        raise ImportError('必须使用数据配置模块加载真实数据集')
    except Exception as e:
        logger.error(f'加载配置时出错: {e}')
        raise Exception(f'无法加载真实数据集: {e}')


def load_real_dataset(config: dict):
    """
    加载真实数据集
    
    Args:
        config: 真实数据集配置
        
    Returns:
        pandas.DataFrame: 加载并处理后的真实数据集
    """
    import pandas as pd
    import os
    
    # 获取配置参数
    data_file = config.get('data_file', '')
    has_header = config.get('has_header', True)
    delimiter = config.get('delimiter', ',')
    field_mapping = config.get('field_mapping', {})
    time_format = config.get('time_format', '%Y-%m-%d %H:%M:%S')
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        raise FileNotFoundError(f'真实数据集文件不存在: {data_file}')
    
    logger.info(f'加载真实数据集: {data_file}')
    
    # 根据文件扩展名选择不同的读取方法
    file_ext = os.path.splitext(data_file)[1].lower()
    
    if file_ext == '.h5' or file_ext == '.hdf5':
        # 读取HDF5格式文件
        try:
            df = pd.read_hdf(data_file)
            logger.info(f'HDF5文件读取成功，数据形状: {df.shape}')
        except Exception as e:
            logger.error(f'读取HDF5文件时出错: {e}')
            # 尝试使用键名读取
            try:
                store = pd.HDFStore(data_file)
                keys = store.keys()
                logger.info(f'HDF5文件中包含的键: {keys}')
                if keys:
                    # 使用第一个键读取数据
                    df = store[keys[0]]
                    logger.info(f'使用键 {keys[0]} 读取数据成功')
                else:
                    raise ValueError('HDF5文件中没有找到数据键')
            except Exception as e2:
                logger.error(f'尝试使用键名读取HDF5文件时出错: {e2}')
                raise
        
        # 检查是否为时间序列矩阵格式（DatetimeIndex作为索引，列名为节点ID）
        if isinstance(df.index, pd.DatetimeIndex) and len(df.columns) > 1:
            logger.info('检测到时间序列矩阵格式，转换为长表格格式')
            # 将矩阵格式转换为长表格格式
            df_long = df.reset_index()
            df_long = pd.melt(df_long, id_vars=['index'], var_name='node_id', value_name='speed')
            df_long.rename(columns={'index': 'timestamp'}, inplace=True)
            df = df_long
        elif 'timestamp' not in df.columns:
            raise ValueError('HDF5文件格式不符合要求，缺少timestamp列或不是时间序列矩阵格式')
    else:
        # 读取CSV格式文件
        if has_header:
            df = pd.read_csv(data_file, delimiter=delimiter)
        else:
            # 如果没有表头，使用默认列名
            default_columns = list(field_mapping.values())
            df = pd.read_csv(data_file, delimiter=delimiter, header=None, names=default_columns)
    
    # 重命名列名，使其符合系统期望的格式
    rename_mapping = {v: k for k, v in field_mapping.items() if v != k}
    if rename_mapping:
        df.rename(columns=rename_mapping, inplace=True)
    
    # 确保时间戳列是 datetime 类型
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format=time_format)
        except ValueError:
            # 如果指定格式失败，尝试自动解析
            df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
    
    # 如果缺少某些必要字段，尝试计算或填充
    required_fields = ['timestamp', 'node_id', 'speed']
    for field in required_fields:
        if field not in df.columns:
            raise ValueError(f'数据集中缺少必要字段: {field}')
    
    # 如果缺少衍生字段，尝试计算
    if 'hour' not in df.columns and 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        
    if 'day_of_week' not in df.columns and 'timestamp' in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
    if 'is_weekend' not in df.columns and 'day_of_week' in df.columns:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 天气编码转换
    if 'weather' in df.columns and 'weather_mapping' in config:
        weather_mapping = config['weather_mapping']
        # 使用模糊匹配来处理天气字段可能的不一致
        try:
            from fuzzywuzzy import fuzz
            def map_weather(weather_str):
                best_match = None
                best_score = 0
                for key, value in weather_mapping.items():
                    score = fuzz.ratio(str(weather_str).lower(), key.lower())
                    if score > best_score:
                        best_score = score
                        best_match = value
                return best_match if best_score > 50 else 0  # 默认返回晴天
            
            df['weather_code'] = df['weather'].apply(map_weather)
        except ImportError:
            logger.warning('未能导入 fuzzywuzzy 库，使用简单映射')
            df['weather_code'] = df['weather'].apply(lambda x: weather_mapping.get(str(x).lower(), 0))
    
    logger.info(f'成功加载真实数据集，共 {len(df)} 条记录，{df["node_id"].nunique()} 个节点')
    
    return df

# 生成模拟数据
def generate_simulation_data(dataset_name):
    """生成模拟交通数据"""
    np.random.seed(42)
    
    # 模拟道路网络
    num_nodes = 50  # 模拟50个道路节点
    num_time_points = 10080  # 两周的数据，每5分钟一个点: 2*7*24*12=4032
    
    # 生成时间序列
    start_date = datetime(2023, 1, 1)
    time_points = [start_date + timedelta(minutes=5*i) for i in range(num_time_points)]
    
    # 生成基础速度数据（带有时序模式）
    speeds = np.zeros((num_nodes, num_time_points))
    for i in range(num_nodes):
        # 基础速度（40-80 km/h之间随机）
        base_speed = 40 + 40 * np.random.random()
        # 添加周模式
        weekly_pattern = 10 * np.sin(np.array(range(num_time_points)) * 2 * np.pi / (7*24*12))
        # 添加日模式
        daily_pattern = 15 * np.sin(np.array(range(num_time_points)) * 2 * np.pi / (24*12))
        # 添加随机波动
        random_noise = 5 * np.random.randn(num_time_points)
        # 组合所有因素
        speeds[i] = base_speed + weekly_pattern + daily_pattern + random_noise
        # 确保速度为正值
        speeds[i] = np.maximum(10, speeds[i])
    
    # 创建数据框
    data = []
    for node_id in range(num_nodes):
        for t_idx, timestamp in enumerate(time_points):
            # 模拟外部属性
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = 1 if timestamp.month in [1, 2, 5, 10] and timestamp.day <= 3 else 0
            weather = np.random.choice(['sunny', 'rainy', 'cloudy', 'foggy'], p=[0.6, 0.2, 0.15, 0.05])
            temperature = 15 + 15 * np.random.random()  # 15-30度
            
            # 模拟交通流量（与速度负相关）
            flow = 1000 - 10 * speeds[node_id, t_idx] + 200 * np.random.random()
            flow = max(100, flow)
            
            # 模拟拥堵指数（基于速度和流量）
            congestion_index = min(10, max(1, 10 - speeds[node_id, t_idx]/8 + flow/200))
            
            data.append({
                'timestamp': timestamp,
                'node_id': f'node_{node_id}',
                'speed': speeds[node_id, t_idx],
                'flow': flow,
                'congestion_index': congestion_index,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
                'weather': weather,
                'temperature': temperature
            })
    
    df = pd.DataFrame(data)
    
    # 保存模拟数据
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    df.to_csv(os.path.join(data_dir, f'{dataset_name}_simulated_data.csv'), index=False)
    logger.info(f'模拟数据已保存至: {os.path.join(data_dir, f"{dataset_name}_simulated_data.csv")}')
    
    # 生成邻接矩阵（模拟道路网络连接）
    adj_matrix = generate_adjacency_matrix(num_nodes)
    np.save(os.path.join(data_dir, f'{dataset_name}_adj_matrix.npy'), adj_matrix)
    logger.info(f'邻接矩阵已保存至: {os.path.join(data_dir, f"{dataset_name}_adj_matrix.npy")}')
    
    return df

# 生成邻接矩阵
def generate_adjacency_matrix(num_nodes):
    """生成模拟的道路网络邻接矩阵"""
    # 创建一个随机图作为道路网络
    G = nx.erdos_renyi_graph(n=num_nodes, p=0.15, seed=42)
    # 转换为邻接矩阵
    adj_matrix = nx.to_numpy_array(G)
    # 确保图是强连通的
    if not nx.is_strongly_connected(nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)):
        # 添加额外的边以确保连通性
        for i in range(num_nodes-1):
            adj_matrix[i][i+1] = 1
            adj_matrix[i+1][i] = 1
    # 归一化邻接矩阵
    degrees = np.sum(adj_matrix, axis=1)
    degrees[degrees == 0] = 1  # 避免除零
    normalized_adj = adj_matrix / degrees.reshape(-1, 1)
    
    return normalized_adj

# 数据预处理主函数
def preprocess_data(dataset_name, config_file=None):
    """数据预处理主流程"""
    logger.info(f'开始数据预处理: {dataset_name}')
    
    # 1. 加载数据
    df = load_dataset(dataset_name, config_file=config_file)
    
    # 2. 数据清洗
    logger.info('执行数据清洗...')
    df = clean_data(df)
    
    # 3. 特征工程
    logger.info('执行特征工程...')
    df = feature_engineering(df)
    
    # 4. 构建时空序列数据
    logger.info('构建时空序列数据...')
    timesteps = 12  # 历史12个时间步（1小时）
    X, y, external_features = create_spatio_temporal_sequences(df, timesteps)
    
    # 5. 数据标准化
    logger.info('执行数据标准化...')
    X, y, external_features, scalers = normalize_data(X, y, external_features)
    
    # 6. 加载邻接矩阵
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    adj_matrix_path = os.path.join(data_dir, f'{dataset_name}_adj_matrix.npy')
    
    # 如果找不到.npy格式的邻接矩阵，尝试从配置中获取或使用默认路径
    if not os.path.exists(adj_matrix_path) and config_file:
        # 尝试从配置中获取邻接矩阵路径
        try:
            from data_config import DataConfig
            data_config = DataConfig()
            config = data_config.load_config(config_file)
            adj_matrix_path = config.get('real_data_config', {}).get('adj_matrix_file', '')
            if adj_matrix_path and os.path.exists(adj_matrix_path):
                logger.info(f'从配置中获取邻接矩阵路径: {adj_matrix_path}')
            else:
                # 使用默认的pickle格式邻接矩阵路径
                adj_matrix_path = os.path.join(data_dir, f'adj_{dataset_name}.pkl')
        except Exception as e:
            logger.error(f'加载配置获取邻接矩阵路径时出错: {e}')
            adj_matrix_path = os.path.join(data_dir, f'adj_{dataset_name}.pkl')
    
    # 根据文件扩展名选择不同的加载方法
    file_ext = os.path.splitext(adj_matrix_path)[1].lower()
    
    if file_ext == '.pkl':
        # 加载pickle格式的邻接矩阵
        try:
            import pickle
            with open(adj_matrix_path, 'rb') as f:
                adj_matrix = pickle.load(f)
            logger.info(f'成功加载pickle格式邻接矩阵，形状: {adj_matrix.shape}')
        except Exception as e:
            logger.error(f'加载pickle格式邻接矩阵时出错: {e}')
            # 如果加载失败，生成一个默认的邻接矩阵
            logger.warning('生成默认邻接矩阵...')
            # 获取节点数量
            nodes = df['node_id'].unique()
            num_nodes = len(nodes)
            adj_matrix = generate_adjacency_matrix(num_nodes)
    else:
        # 加载npy格式的邻接矩阵
        try:
            adj_matrix = np.load(adj_matrix_path)
            logger.info(f'成功加载npy格式邻接矩阵，形状: {adj_matrix.shape}')
        except Exception as e:
            logger.error(f'加载npy格式邻接矩阵时出错: {e}')
            # 如果加载失败，生成一个默认的邻接矩阵
            logger.warning('生成默认邻接矩阵...')
            # 获取节点数量
            nodes = df['node_id'].unique()
            num_nodes = len(nodes)
            adj_matrix = generate_adjacency_matrix(num_nodes)
    
    # 7. 保存处理后的数据
    save_preprocessed_data(X, y, external_features, adj_matrix, scalers, dataset_name)
    
    logger.info('数据预处理完成!')

# 数据清洗
def clean_data(df):
    """数据清洗"""
    # 移除异常值
    df = df[(df['speed'] >= 5) & (df['speed'] <= 120)]
    
    # 如果存在flow列，也对其进行异常值处理
    if 'flow' in df.columns:
        df = df[(df['flow'] >= 0) & (df['flow'] <= 2000)]
    
    # 填充缺失值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 按时间排序
    df = df.sort_values(['node_id', 'timestamp'])
    
    return df

# 特征工程
def feature_engineering(df):
    """特征工程"""
    # 确保timestamp列存在
    if 'timestamp' not in df.columns:
        raise ValueError('数据集中缺少timestamp列')
    
    # 确保node_id列存在
    if 'node_id' not in df.columns:
        raise ValueError('数据集中缺少node_id列')
    
    # 确保speed列存在
    if 'speed' not in df.columns:
        raise ValueError('数据集中缺少speed列')
    
    # 创建或填充hour列
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    
    # 创建或填充day_of_week列
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 创建或填充is_weekend列
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 创建或填充is_holiday列（简单规则：假设1月、2月、5月、10月的前3天为假期）
    if 'is_holiday' not in df.columns:
        df['is_holiday'] = df['timestamp'].apply(lambda x: 1 if x.month in [1, 2, 5, 10] and x.day <= 3 else 0)
    
    # 将时间戳转换为正弦和余弦编码
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 如果有weather列，进行编码
    if 'weather' in df.columns:
        weather_encoding = {
            'sunny': 0,
            'cloudy': 1,
            'rainy': 2,
            'foggy': 3
        }
        df['weather_code'] = df['weather'].map(weather_encoding).fillna(0)  # 默认值为0
    else:
        # 如果没有weather列，创建默认值
        df['weather_code'] = 0
    
    # 如果有temperature列，进行标准化
    if 'temperature' in df.columns:
        df['temp_norm'] = (df['temperature'] - df['temperature'].min()) / (df['temperature'].max() - df['temperature'].min())
    else:
        # 如果没有temperature列，创建默认值
        df['temp_norm'] = 0.5  # 中间值
    
    # 如果没有flow列，创建默认值
    if 'flow' not in df.columns:
        # 使用速度的负相关值作为流量的估计
        df['flow'] = 800 - 10 * df['speed']
        df['flow'] = df['flow'].apply(lambda x: max(100, x))
    
    # 如果没有congestion_index列，创建默认值
    if 'congestion_index' not in df.columns:
        # 基于速度创建简单的拥堵指数（值越高表示越拥堵）
        df['congestion_index'] = df['speed'].apply(lambda x: min(10, max(1, 10 - x/8)))
    
    return df

# 创建时空序列数据
def create_spatio_temporal_sequences(df, timesteps):
    """创建时空序列数据"""
    logger.info('正在优化的时空序列构建...')
    
    # 获取所有节点和时间点
    nodes = df['node_id'].unique()
    num_nodes = len(nodes)
    
    # 按节点和时间排序
    df_sorted = df.sort_values(['timestamp', 'node_id'])
    
    # 创建节点ID到索引的映射
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # 获取时间序列长度
    timestamps = df_sorted['timestamp'].unique()
    num_timestamps = len(timestamps)
    
    # 创建速度数据的宽表 [时间点, 节点]，使用pivot_table提高效率
    speed_pivot = df_sorted.pivot_table(index='timestamp', columns='node_id', values='speed', fill_value=0)
    
    # 确保所有节点都在列中
    for node in nodes:
        if node not in speed_pivot.columns:
            speed_pivot[node] = 0
    
    # 按照节点索引顺序排列列
    speed_pivot = speed_pivot[[idx_to_node[i] for i in range(num_nodes)]]
    
    # 将速度数据转换为numpy数组 [时间点, 节点]
    speed_matrix = speed_pivot.values
    
    # 创建外部特征矩阵 [时间点, 节点, 特征维度]
    # 先构建一个每个时间点的特征矩阵
    external_features_list = []
    feature_columns = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
                       'is_weekend', 'is_holiday', 'weather_code', 'temp_norm', 
                       'flow', 'congestion_index']
    
    # 为每个时间点创建所有节点的外部特征
    # 添加进度提示
    total_timestamps = num_timestamps
    for t in range(num_timestamps):
        # 每100个时间点打印一次进度
        if t % 100 == 0:
            progress = (t / total_timestamps) * 100
            logger.info(f'正在处理时间点 {t}/{total_timestamps} ({progress:.1f}%)')
            
        current_time = timestamps[t]
        time_df = df_sorted[df_sorted['timestamp'] == current_time]
        
        # 创建当前时间点的特征矩阵
        time_features = np.zeros((num_nodes, len(feature_columns)))
        
        # 填充特征
        for node_idx in range(num_nodes):
            node = idx_to_node[node_idx]
            node_row = time_df[time_df['node_id'] == node]
            if not node_row.empty:
                for feat_idx, feat_col in enumerate(feature_columns):
                    time_features[node_idx, feat_idx] = node_row[feat_col].values[0]
        
        # 将当前时间点的特征矩阵添加到列表中
        external_features_list.append(time_features)
    
    # 进度完成提示
    logger.info('所有时间点特征处理完成，开始转换数据格式')
    
    # 转换为numpy数组 [时间点, 节点, 特征维度]
    external_matrix = np.array(external_features_list)
    
    # 确保数据矩阵不为空
    if speed_matrix.shape[0] == 0:
        logger.error('速度矩阵为空，无法构建时空序列')
        raise ValueError('速度矩阵为空，无法构建时空序列')
    
    if external_matrix.shape[0] == 0:
        logger.error('外部特征矩阵为空，无法构建时空序列')
        raise ValueError('外部特征矩阵为空，无法构建时空序列')
    
    # 重新计算有效样本数，确保不会越界
    max_valid_index = min(speed_matrix.shape[0], external_matrix.shape[0]) - timesteps - 1
    num_samples = max(0, max_valid_index)
    
    if num_samples <= 0:
        logger.error(f'有效样本数不足，需要至少 {timesteps+1} 个时间点，实际只有 {max(speed_matrix.shape[0], external_matrix.shape[0])} 个时间点')
        raise ValueError(f'有效样本数不足，需要至少 {timesteps+1} 个时间点')
    
    # 初始化数组
    X = np.zeros((num_samples, timesteps, num_nodes))
    y = np.zeros((num_samples, num_nodes))
    external_features = np.zeros((num_samples, num_nodes, len(feature_columns)))
    
    # 填充数据，使用滑动窗口方法
    for i in range(num_samples):
        # 历史序列
        X[i] = speed_matrix[i:i+timesteps]
        # 目标值
        y[i] = speed_matrix[i+timesteps]
        # 外部特征（使用当前时间点的特征）
        external_features[i] = external_matrix[i+timesteps]
    
    logger.info(f'时空序列构建完成，样本数: {num_samples}')
    
    return X, y, external_features

# 数据标准化
def normalize_data(X, y, external_features):
    """数据标准化"""
    # 创建标准化器
    speed_scaler = MinMaxScaler(feature_range=(0, 1))
    external_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 对速度数据进行标准化
    num_samples, num_timesteps, num_nodes = X.shape
    
    # 重塑X以适应标准化器
    X_reshaped = X.reshape(-1, 1)  # [样本数*时间步*节点数, 1]
    X_scaled = speed_scaler.fit_transform(X_reshaped)
    X_normalized = X_scaled.reshape(num_samples, num_timesteps, num_nodes)  # 恢复原始形状
    
    # 对y进行标准化
    y_reshaped = y.reshape(-1, 1)
    y_scaled = speed_scaler.transform(y_reshaped)
    y_normalized = y_scaled.reshape(num_samples, num_nodes)
    
    # 对外接特征进行标准化
    ext_shape = external_features.shape
    external_reshaped = external_features.reshape(-1, ext_shape[-1])
    external_scaled = external_scaler.fit_transform(external_reshaped)
    external_normalized = external_scaled.reshape(ext_shape)
    
    # 保存标准化器
    scalers = {
        'speed_scaler': speed_scaler,
        'external_scaler': external_scaler
    }
    
    return X_normalized, y_normalized, external_normalized, scalers

# 保存预处理后的数据
def save_preprocessed_data(X, y, external_features, adj_matrix, scalers, dataset_name):
    """保存预处理后的数据"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # 创建保存目录
    processed_dir = os.path.join(data_dir, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    # 保存数据
    np.save(os.path.join(processed_dir, f'{dataset_name}_X.npy'), X)
    np.save(os.path.join(processed_dir, f'{dataset_name}_y.npy'), y)
    np.save(os.path.join(processed_dir, f'{dataset_name}_external.npy'), external_features)
    np.save(os.path.join(processed_dir, f'{dataset_name}_adj_matrix.npy'), adj_matrix)
    
    # 保存标准化器
    import pickle
    with open(os.path.join(processed_dir, f'{dataset_name}_scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    
    logger.info(f'预处理后的数据已保存至: {processed_dir}')
    logger.info(f'训练样本数: {X.shape[0]}')
    logger.info(f'时间步长: {X.shape[1]}')
    logger.info(f'节点数: {X.shape[2]}')