# -*- coding: utf-8 -*-
"""
简单的预测测试脚本
用于测试训练好的模型是否能正常进行预测
"""

import os
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/test_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestPrediction')

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in os.sys.path:
    os.sys.path.append(project_root)

# 导入必要的模块
from src.model_structure import GCNLayer, SpatioTemporalGCNGRU, create_model
from src.fuzzy_inference import FuzzyInferenceSystem

class MockArgs:
    """模拟命令行参数"""
    def __init__(self, model='gcn_gru_fuzzy', dataset='Los-loop'):
        self.model = model
        self.dataset = dataset

# 加载训练好的模型
def load_trained_model(model_name='gcn_gru_fuzzy', dataset_name='Los-loop'):
    """加载训练好的模型"""
    logger.info(f'加载训练好的模型: {model_name}_{dataset_name}')
    
    # 确定模型路径
    models_dir = os.path.join(project_root, 'models')
    model_info_path = os.path.join(models_dir, f'{model_name}_{dataset_name}_info.pkl')
    
    # 检查模型信息文件是否存在
    if not os.path.exists(model_info_path):
        # 列出可用的模型信息文件
        try:
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
        
        # 加载模型
        logger.info(f'尝试直接加载完整模型: {candidate_model_path}')
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
        
        return model, scalers, adj_matrix, params
        
    except Exception as e:
        logger.error(f'加载模型时出错: {e}')
        raise

# 生成模拟预测所需的输入
def generate_simulation_data(params):
    """生成模拟的历史速度数据和外部特征"""
    timesteps = int(params.get('timesteps', 12))
    num_nodes = int(params.get('num_nodes', 20))
    
    # 生成模拟的历史速度数据
    # 基础速度 40-70，加入小扰动
    base = np.random.uniform(40, 70, size=(num_nodes,))
    noise = np.random.normal(0, 3, size=(timesteps, num_nodes))
    historical_data = base.reshape(1, -1) + noise
    historical_data = np.clip(historical_data, 10, 120)  # 保证合理范围
    historical_data = np.repeat(historical_data, timesteps, axis=0)  # 扩展到 timesteps 行
    historical_data = historical_data.astype(np.float32)
    
    # 生成模拟的外部特征
    external_features = np.random.rand(num_nodes, 10).astype(np.float32)
    
    return historical_data, external_features

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
        raise ValueError(f'历史数据不足')
    
    # 获取最新的timesteps个时间步的数据
    recent_data = historical_data[-timesteps:]
    
    # 对数据进行标准化
    speed_scaler = scalers['speed_scaler']
    recent_data_flattened = recent_data.reshape(-1, 1)
    recent_data_scaled = speed_scaler.transform(recent_data_flattened)
    recent_data_scaled = recent_data_scaled.reshape(timesteps, num_nodes)
    
    # 扩展维度以适应模型输入
    X_pred = np.expand_dims(recent_data_scaled, axis=0)  # [1, timesteps, num_nodes]
    
    # 处理外部特征
    if external_features is not None:
        # 对外部特征进行标准化
        external_scaler = scalers['external_scaler']
        external_features_flattened = external_features.reshape(-1, 10)
        external_features_scaled = external_scaler.transform(external_features_flattened)
        external_features_scaled = external_features_scaled.reshape(num_nodes, 10)
        
        # 扩展维度
        ext_pred = np.expand_dims(external_features_scaled, axis=0)  # [1, num_nodes, 10]
    else:
        logger.error('未提供外部特征')
        raise ValueError('缺少外部特征')
    
    # 处理邻接矩阵
    adj_matrix_pred = np.expand_dims(adj_matrix, axis=0)  # [1, num_nodes, num_nodes]
    
    return X_pred, adj_matrix_pred, ext_pred

# 测试预测功能
def test_prediction(model_name='gcn_gru_fuzzy', dataset_name='Los-loop'):
    """测试模型预测功能"""
    logger.info(f'=== 开始测试预测功能: {model_name}_{dataset_name} ===')
    
    try:
        # 1. 加载训练好的模型
        model, scalers, adj_matrix, params = load_trained_model(model_name, dataset_name)
        
        # 2. 生成模拟数据
        logger.info('生成模拟测试数据...')
        historical_data, external_features = generate_simulation_data(params)
        logger.info(f'模拟数据生成完成: 历史数据形状={historical_data.shape}, 外部特征形状={external_features.shape}')
        
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
        
        # 6. 显示预测结果
        logger.info('预测结果摘要:')
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
        
        # 7. 保存预测结果
        results_dir = os.path.join(project_root, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pred_path = os.path.join(results_dir, f'test_predictions_{timestamp}.npy')
        np.save(pred_path, predictions)
        logger.info(f'测试预测结果已保存至: {pred_path}')
        
        logger.info('=== 预测功能测试完成 ===')
        return True
        
    except Exception as e:
        logger.error(f'预测功能测试失败: {e}')
        return False

if __name__ == '__main__':
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_gru_fuzzy')
    parser.add_argument('--dataset', type=str, default='Los-loop')
    
    args = parser.parse_args()
    success = test_prediction(args.model, args.dataset)
    
    # 根据测试结果设置退出码
    os.sys.exit(0 if success else 1)