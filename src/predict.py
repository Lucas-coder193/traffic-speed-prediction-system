#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测模块
负责加载训练好的模型并进行交通速度预测
"""

import os
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime, timedelta
from .fuzzy_inference import FuzzyInferenceSystem
from .model_structure import GCNLayer, SpatioTemporalGCNGRU, create_model

# 配置日志
logger = logging.getLogger('TrafficPrediction')

# 加载模型
def load_trained_model(model_name='gcn_gru_fuzzy', dataset_name='SZBZ'):
    """加载训练好的模型"""
    logger.info(f'加载训练好的模型: {model_name}_{dataset_name}')
    
    # 确定模型路径
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_info_path = os.path.join(models_dir, f'{model_name}_{dataset_name}_info.pkl')
    
    # 检查模型信息文件是否存在
    if not os.path.exists(model_info_path):
        # 列出可用的模型信息文件，帮助用户选择正确的数据集/模型
        try:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            available = [f for f in os.listdir(models_dir) if f.endswith('_info.pkl')]
        except Exception:
            available = []
        logger.error(f'模型信息文件不存在: {model_info_path}')
        if available:
            logger.error(f'可用的模型信息文件: {available}')
        raise FileNotFoundError(f'模型信息文件不存在: {model_info_path}')
    
    # 加载模型信息
    try:
        import pickle
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # 优先尝试加载最佳权重 *_best.keras，否则回退到最终模型
        best_model_path = os.path.join(models_dir, f'{model_name}_{dataset_name}_best.keras')
        candidate_model_path = best_model_path if os.path.exists(best_model_path) else model_info['model_path']
        if not os.path.exists(candidate_model_path):
            logger.error(f'模型文件不存在: {candidate_model_path}')
            raise FileNotFoundError(f'模型文件不存在: {candidate_model_path}')
        
        # 首选直接反序列化完整模型（我们已修复自定义层以兼容KerasTensor）
        logger.info(f'尝试直接加载完整模型: {candidate_model_path}（compile=False）')
        model = tf.keras.models.load_model(
            candidate_model_path,
            custom_objects={'GCNLayer': GCNLayer, 'SpatioTemporalGCNGRU': SpatioTemporalGCNGRU},
            compile=False
        )
        logger.info(f'模型已成功加载: {candidate_model_path}')
        
        # 加载其他信息
        scalers = model_info['scalers']
        adj_matrix = model_info['adj_matrix']
        params = model_info['params']
        
        # 加载模糊推理系统
        fuzzy_path = os.path.join(models_dir, f'fuzzy_system_{dataset_name}.pkl')
        if os.path.exists(fuzzy_path):
            fuzzy_system = FuzzyInferenceSystem.load_model(fuzzy_path)
            logger.info(f'模糊推理系统已加载: {fuzzy_path}')
        else:
            fuzzy_system = FuzzyInferenceSystem()
            logger.warning(f'模糊推理系统文件不存在，使用默认系统: {fuzzy_path}')
        
        return model, scalers, adj_matrix, params, fuzzy_system
        
    except Exception as e:
        logger.error(f'加载模型时出错: {e}')
        raise

# ============== 辅助：生成模拟预测所需的输入 ==============
def generate_simulation_history(params):
    """生成模拟的历史速度数据，形状为 (timesteps, num_nodes)"""
    timesteps = int(params.get('timesteps', 12))
    num_nodes = int(params.get('num_nodes', 20))
    # 基础速度 40-70，加入小扰动
    base = np.random.uniform(40, 70, size=(num_nodes,))
    noise = np.random.normal(0, 3, size=(timesteps, num_nodes))
    history = base.reshape(1, -1) + noise
    # 保证合理范围
    history = np.clip(history, 10, 120)
    # 扩展到 timesteps 行
    if history.shape[0] == 1:
        history = np.repeat(history, timesteps, axis=0)
    return history.astype(np.float32)

def generate_simulation_external(params):
    """生成模拟的外部特征，形状为 (num_nodes, 10)"""
    num_nodes = int(params.get('num_nodes', 20))
    ext = np.random.rand(num_nodes, 10)
    return ext.astype(np.float32)

# 准备预测数据
def prepare_prediction_data(historical_data, external_features, adj_matrix, scalers, params):
    """准备预测数据"""
    logger.info('准备预测数据...')
    
    # 获取参数
    timesteps = params['timesteps']
    num_nodes = params['num_nodes']
    
    # 确保历史数据的时间步长正确
    if len(historical_data) < timesteps:
        logger.error(f'历史数据不足，需要{timesteps}个时间步，但只提供了{len(historical_data)}个')
        raise ValueError(f'历史数据不足，需要{timesteps}个时间步，但只提供了{len(historical_data)}个')
    
    # 获取最新的timesteps个时间步的数据
    recent_data = historical_data[-timesteps:]
    
    # 确保数据形状正确
    if recent_data.shape != (timesteps, num_nodes):
        logger.error(f'历史数据形状不正确，期望: ({timesteps}, {num_nodes})，实际: {recent_data.shape}')
        raise ValueError(f'历史数据形状不正确')
    
    # 对数据进行标准化
    speed_scaler = scalers['speed_scaler']
    recent_data_flattened = recent_data.reshape(-1, 1)
    recent_data_scaled = speed_scaler.transform(recent_data_flattened)
    recent_data_scaled = recent_data_scaled.reshape(timesteps, num_nodes)
    
    # 扩展维度以适应模型输入
    X_pred = np.expand_dims(recent_data_scaled, axis=0)  # [1, timesteps, num_nodes]
    
    # 处理外部特征
    if external_features is not None:
        # 确保外部特征形状正确
        if external_features.shape != (num_nodes, 10):  # 假设外部特征维度为10
            logger.error(f'外部特征形状不正确，期望: ({num_nodes}, 10)，实际: {external_features.shape}')
            raise ValueError(f'外部特征形状不正确')
        
        # 对外部特征进行标准化
        external_scaler = scalers['external_scaler']
        external_features_flattened = external_features.reshape(-1, 10)
        external_features_scaled = external_scaler.transform(external_features_flattened)
        external_features_scaled = external_features_scaled.reshape(num_nodes, 10)
        
        # 扩展维度
        ext_pred = np.expand_dims(external_features_scaled, axis=0)  # [1, num_nodes, 10]
    else:
        # 真实场景禁止使用随机外部特征
        logger.error('未提供外部特征 external_features，且不允许使用模拟数据')
        raise ValueError('缺少外部特征 external_features')
    
    # 处理邻接矩阵
    adj_matrix_pred = np.expand_dims(adj_matrix, axis=0)  # [1, num_nodes, num_nodes]
    
    return X_pred, adj_matrix_pred, ext_pred

# 进行预测
def predict_traffic_speed(args):
    """预测交通速度"""
    logger.info('=== 开始交通速度预测 ===')
    
    try:
        # 1. 加载训练好的模型
        model, scalers, adj_matrix, params, fuzzy_system = load_trained_model(args.model, args.dataset)
        
        # 2. 准备预测数据（强制真实数据）
        use_real_data = hasattr(args, 'config_file') and args.config_file is not None
        
        if use_real_data:
            logger.info('使用真实数据集进行预测')
            # 导入数据加载模块
            from .data_preprocessing import load_real_dataset, load_dataset
            
            # 加载配置文件
            from data_config import DataConfig
            data_config = DataConfig()
            config = data_config.load_config(args.config_file)
            
            # 加载真实数据集
            df = load_dataset(args.dataset, config_file=args.config_file)
            
            # 从数据集中提取最新的历史数据
            # 获取最新的历史时间步数据
            timesteps = params['timesteps']
            num_nodes = params['num_nodes']
            
            # 按时间戳排序
            df_sorted = df.sort_values('timestamp')
            
            # 为每个节点提取最新的timesteps个速度数据
            historical_data = []
            for node_id in df_sorted['node_id'].unique()[:num_nodes]:  # 取前num_nodes个节点
                node_data = df_sorted[df_sorted['node_id'] == node_id].tail(timesteps)
                if len(node_data) < timesteps:
                    logger.warning(f'节点 {node_id} 的历史数据不足 {timesteps} 个时间步，使用模拟数据填充')
                    # 使用模拟数据填充
                    node_speeds = generate_simulation_history({'num_nodes': 1, 'timesteps': timesteps}).reshape(-1)
                else:
                    node_speeds = node_data['speed'].values
                historical_data.append(node_speeds)
            
            # 转置数据，使其形状为 [timesteps, num_nodes]
            historical_data = np.array(historical_data).T
            
            # 提取外部特征
            # 这里简化处理，使用最新的外部特征
            latest_external = df_sorted.tail(num_nodes)
            
            # 构建外部特征数组
            external_features = []
            for _, row in latest_external.iterrows():
                # 提取需要的特征
                ext = []
                # 时间特征
                if 'hour' in row:
                    ext.append(row['hour'] / 23.0)  # 归一化到 [0,1]
                else:
                    ext.append(0.5)  # 默认值
                
                if 'day_of_week' in row:
                    ext.append(row['day_of_week'] / 6.0)  # 归一化到 [0,1]
                else:
                    ext.append(0.5)  # 默认值
                
                # 周末和节假日特征
                if 'is_weekend' in row:
                    ext.append(row['is_weekend'])
                else:
                    ext.append(0)
                
                if 'is_holiday' in row:
                    ext.append(row['is_holiday'])
                else:
                    ext.append(0)
                
                # 天气特征
                if 'weather_code' in row:
                    ext.append(row['weather_code'] / 3.0)  # 归一化到 [0,1]
                elif 'weather' in row and 'weather_mapping' in config.get('real_data_config', {}):
                    weather_mapping = config['real_data_config']['weather_mapping']
                    weather_code = weather_mapping.get(str(row['weather']).lower(), 0) / 3.0
                    ext.append(weather_code)
                else:
                    ext.append(0)  # 默认晴天
                
                # 温度特征
                if 'temperature' in row:
                    # 假设温度范围是 0-40 度，归一化到 [0,1]
                    temp_norm = min(max((row['temperature'] - 0) / 40.0, 0), 1)
                    ext.append(temp_norm)
                else:
                    ext.append(0.5)  # 默认适中温度
                
                # 填充到固定长度（假设需要10个特征）
                while len(ext) < 10:
                    ext.append(0.5)  # 使用中间值填充
                
                external_features.append(ext[:10])  # 确保长度为10
            
            external_features = np.array(external_features)
        else:
            # 禁止使用模拟数据
            logger.error('未提供 --config_file，无法加载真实数据；预测阶段禁用模拟数据')
            raise ValueError('请提供 --config_file 指向真实数据配置文件')
        
        # 3. 准备预测数据
        X_pred, adj_matrix_pred, ext_pred = prepare_prediction_data(
            historical_data, external_features, adj_matrix, scalers, params
        )
        
        # 4. 进行预测
        logger.info('进行预测...')
        predictions_scaled = model.predict([X_pred, adj_matrix_pred, ext_pred])
        
        # 5. 反归一化预测结果
        speed_scaler = scalers['speed_scaler']
        predictions = speed_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(predictions_scaled.shape)
        
        # 6. 记录预测结果
        record_predictions(predictions, params)
        
        # 7. 显示预测结果
        display_predictions(predictions, params)
        
        logger.info('=== 交通速度预测完成 ===')
        
        return predictions
        
    except Exception as e:
        logger.error(f'预测过程中出错: {e}')
        raise



# 多步预测
def multi_step_prediction(model, scalers, adj_matrix, params, historical_data, external_features, steps=6):
    """进行多步预测"""
    logger.info(f'进行{steps}步预测...')
    
    # 获取参数
    timesteps = params['timesteps']
    num_nodes = params['num_nodes']
    
    # 确保历史数据足够
    if len(historical_data) < timesteps:
        logger.error(f'历史数据不足，需要{timesteps}个时间步')
        raise ValueError(f'历史数据不足')
    
    # 初始化预测结果列表
    all_predictions = []
    
    # 用于预测的历史数据
    current_history = historical_data.copy()
    
    # 对每个预测步进行预测
    for step in range(steps):
        logger.info(f'预测第{step+1}/{steps}步...')
        
        # 准备预测数据
        X_pred, adj_matrix_pred, ext_pred = prepare_prediction_data(
            current_history, external_features, adj_matrix, scalers, params
        )
        
        # 进行预测
        pred_scaled = model.predict([X_pred, adj_matrix_pred, ext_pred])
        
        # 反归一化预测结果
        speed_scaler = scalers['speed_scaler']
        pred = speed_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
        
        # 保存预测结果
        all_predictions.append(pred[0])
        
        # 更新历史数据，用于下一步预测
        current_history = np.vstack([current_history[1:], pred[0]])
        
        # 保持外部特征不变，或在上层按步提供
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    
    return all_predictions

# 记录预测结果
def record_predictions(predictions, params):
    """记录预测结果"""
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 保存预测结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pred_path = os.path.join(results_dir, f'predictions_{timestamp}.npy')
    np.save(pred_path, predictions)
    
    # 保存预测信息
    import pickle
    pred_info = {
        'timestamp': timestamp,
        'predictions': predictions,
        'params': params
    }
    
    info_path = os.path.join(results_dir, f'predictions_info_{timestamp}.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(pred_info, f)
    
    logger.info(f'预测结果已保存至: {pred_path}')
    logger.info(f'预测信息已保存至: {info_path}')

# 显示预测结果
def display_predictions(predictions, params):
    """显示预测结果"""
    logger.info('预测结果摘要:')
    
    # 计算统计信息
    avg_speed = np.mean(predictions)
    min_speed = np.min(predictions)
    max_speed = np.max(predictions)
    
    logger.info(f'平均预测速度: {avg_speed:.2f} km/h')
    logger.info(f'最低预测速度: {min_speed:.2f} km/h')
    logger.info(f'最高预测速度: {max_speed:.2f} km/h')
    
    # 显示前5个节点的预测结果
    logger.info('前5个节点的预测速度:')
    for i in range(min(5, params['num_nodes'])):
        logger.info(f'节点 {i+1}: {predictions[0, i]:.2f} km/h')
    
    # 生成可视化图表（如果需要）
    try:
        visualize_predictions(predictions, params)
    except Exception as e:
        logger.warning(f'生成可视化图表时出错: {e}')

# 可视化预测结果
def visualize_predictions(predictions, params):
    """可视化预测结果"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 1. 预测速度分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions.flatten(), bins=20, kde=True)
    plt.title('预测速度分布')
    plt.xlabel('速度 (km/h)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    hist_path = os.path.join(results_dir, f"prediction_histogram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(hist_path)
    plt.close()
    logger.info(f'预测速度分布图已保存至: {hist_path}')
    
    # 2. 部分节点的预测速度条形图
    # 只显示前10个节点
    num_nodes_to_show = min(10, params['num_nodes'])
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_nodes_to_show), predictions[0, :num_nodes_to_show])
    plt.title('各节点预测速度')
    plt.xlabel('节点ID')
    plt.ylabel('速度 (km/h)')
    plt.grid(True, alpha=0.3, axis='y')
    bar_path = os.path.join(results_dir, f"prediction_bars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(bar_path)
    plt.close()
    logger.info(f'节点预测速度条形图已保存至: {bar_path}')
    
    # 3. 热力图（如果节点数量适中）
    if params['num_nodes'] <= 100:
        # 重塑预测结果以适应热力图（假设是网格状分布）
        grid_size = int(np.ceil(np.sqrt(params['num_nodes'])))
        heatmap_data = np.zeros((grid_size, grid_size))
        heatmap_data.flat[:predictions.shape[1]] = predictions[0]
        
        plt.figure(figsize=(10, 10))
        sns.heatmap(heatmap_data, annot=False, cmap='RdYlGn_r', cbar_kws={'label': '速度 (km/h)'})
        plt.title('预测速度热力图')
        plt.axis('off')
        heat_path = os.path.join(results_dir, f"prediction_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(heat_path)
        plt.close()
        logger.info(f'预测速度热力图已保存至: {heat_path}')

if __name__ == '__main__':
    # 测试预测功能
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_gru_fuzzy')
    parser.add_argument('--dataset', type=str, default='Los-loop')
    
    args = parser.parse_args()
    predict_traffic_speed(args)