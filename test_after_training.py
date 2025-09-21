#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练后测试脚本：在模型训练完成后，使用测试集评估模型性能
"""

import sys
import os
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/test_after_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestAfterTraining')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def load_test_data(dataset_name):
    """加载测试数据"""
    logger.info(f'加载{dataset_name}测试数据...')
    data_dir = os.path.join(project_root, 'data', 'processed')
    
    # 加载数据
    X_path = os.path.join(data_dir, f'{dataset_name}_X.npy')
    y_path = os.path.join(data_dir, f'{dataset_name}_y.npy')
    ext_path = os.path.join(data_dir, f'{dataset_name}_external.npy')
    adj_path = os.path.join(data_dir, f'{dataset_name}_adj_matrix.npy')
    scalers_path = os.path.join(data_dir, f'{dataset_name}_scalers.pkl')
    
    # 检查文件是否存在
    for file_path in [X_path, y_path, ext_path, adj_path, scalers_path]:
        if not os.path.exists(file_path):
            logger.error(f'未找到文件: {file_path}')
            return None
    
    # 加载数据
    import pickle
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
        
    X = np.load(X_path)
    y = np.load(y_path)
    external_features = np.load(ext_path)
    adj_matrix = np.load(adj_path)
    
    logger.info(f'测试数据加载完成: X形状={X.shape}, y形状={y.shape}, 外部特征形状={external_features.shape}')
    return X, y, external_features, adj_matrix, scalers

def load_trained_model(model_name, dataset_name):
    """加载训练好的模型"""
    logger.info(f'加载模型: {model_name}_{dataset_name}')
    models_dir = os.path.join(project_root, 'models')
    
    # 优先加载最佳模型
    model_path = os.path.join(models_dir, f'{model_name}_{dataset_name}_best.keras')
    if not os.path.exists(model_path):
        # 尝试加载普通模型
        model_path = os.path.join(models_dir, f'{model_name}_{dataset_name}.keras')
        if not os.path.exists(model_path):
            logger.error(f'未找到模型文件')
            return None
    
    # 加载模型信息
    info_path = os.path.join(models_dir, f'{model_name}_{dataset_name}_info.pkl')
    if not os.path.exists(info_path):
        logger.error(f'未找到模型信息文件')
        return None
    
    import pickle
    with open(info_path, 'rb') as f:
        model_info = pickle.load(f)
    
    # 导入自定义层
    from src.model_structure import GCNLayer, SpatioTemporalGCNGRU
    
    # 加载模型
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'GCNLayer': GCNLayer, 'SpatioTemporalGCNGRU': SpatioTemporalGCNGRU}
        )
        logger.info(f'模型加载成功: {model_path}')
        return model, model_info['scalers'], model_info['adj_matrix'], model_info['params']
    except Exception as e:
        logger.error(f'模型加载失败: {e}')
        return None

def evaluate_model(model, X_test, y_test, ext_test, adj_matrix, scalers):
    """评估模型性能"""
    logger.info('评估模型性能...')
    
    # 扩展邻接矩阵以匹配批次
    adj_matrix_expanded = np.expand_dims(adj_matrix, axis=0)
    adj_matrix_expanded = np.repeat(adj_matrix_expanded, X_test.shape[0], axis=0)
    
    # 创建数据生成器
    class TestDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, X, y, ext_features, adj_matrix, batch_size=32):
            self.X = X
            self.y = y
            self.ext_features = ext_features
            self.adj_matrix = adj_matrix
            self.batch_size = batch_size
            self.indices = np.arange(len(X))
        
        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))
        
        def __getitem__(self, idx):
            batch_indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            ext_batch = self.ext_features[batch_indices]
            adj_batch = self.adj_matrix[batch_indices]
            return (X_batch, adj_batch, ext_batch), y_batch
    
    # 创建测试数据生成器
    test_generator = TestDataGenerator(X_test, y_test, ext_test, adj_matrix_expanded)
    
    # 评估模型
    loss, rmse, mae = model.evaluate(test_generator, verbose=1)
    
    # 反归一化评估指标
    # 假设原始数据的范围是0-120 km/h
    original_rmse = rmse * 120
    original_mae = mae * 120
    
    logger.info(f'测试集性能:')
    logger.info(f'Loss: {loss:.4f}')
    logger.info(f'RMSE (标准化): {rmse:.4f}')
    logger.info(f'RMSE (原始值，km/h): {original_rmse:.4f}')
    logger.info(f'MAE (标准化): {mae:.4f}')
    logger.info(f'MAE (原始值，km/h): {original_mae:.4f}')
    
    # 获取预测值
    y_pred = model.predict(test_generator)
    
    # 反归一化预测值和真实值
    speed_scaler = scalers.get('speed_scaler')
    if speed_scaler is not None:
        y_test_original = speed_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        y_pred_original = speed_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    else:
        y_test_original = y_test
        y_pred_original = y_pred
    
    # 计算准确率
    threshold = 0.1  # 10%的误差范围
    accurate_predictions = np.abs((y_pred_original - y_test_original) / (y_test_original + 1e-8)) <= threshold
    accuracy = np.mean(accurate_predictions)
    
    logger.info(f'准确率 (±{threshold*100}%误差范围): {accuracy*100:.2f}%')
    
    return {
        'loss': loss,
        'rmse': rmse,
        'mae': mae,
        'original_rmse': original_rmse,
        'original_mae': original_mae,
        'accuracy': accuracy,
        'y_test': y_test_original,
        'y_pred': y_pred_original
    }

def plot_results(results, model_name, dataset_name):
    """绘制评估结果"""
    logger.info('绘制评估结果...')
    
    # 创建结果目录
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 绘制预测对比图
    plt.figure(figsize=(12, 6))
    # 随机选择5个节点绘制
    num_nodes = min(5, results['y_test'].shape[1])
    sample_nodes = np.random.choice(results['y_test'].shape[1], num_nodes, replace=False)
    
    for i, node in enumerate(sample_nodes):
        plt.subplot(num_nodes, 1, i+1)
        plt.plot(results['y_test'][:50, node], label='真实值')
        plt.plot(results['y_pred'][:50, node], label='预测值')
        plt.title(f'节点 {node} 速度预测对比')
        plt.ylabel('速度 (km/h)')
        plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'prediction_comparison_{model_name}_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plot_path)
    logger.info(f'预测对比图已保存至: {plot_path}')
    
    # 绘制误差分布图
    plt.figure(figsize=(10, 6))
    errors = np.abs(results['y_pred'] - results['y_test'])
    plt.hist(errors.flatten(), bins=50, alpha=0.7)
    plt.title('预测误差分布')
    plt.xlabel('误差 (km/h)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    error_plot_path = os.path.join(results_dir, f'error_distribution_{model_name}_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(error_plot_path)
    logger.info(f'误差分布图已保存至: {error_plot_path}')
    
    # 保存结果数据
    import pickle
    results_path = os.path.join(results_dir, f'test_results_{model_name}_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f'测试结果已保存至: {results_path}')

def main():
    """主函数"""
    logger.info('=== 开始训练后测试 ===')
    
    # 配置参数
    model_name = 'gcn_gru_fuzzy'
    dataset_name = 'Los-loop'
    
    # 加载训练好的模型
    model_data = load_trained_model(model_name, dataset_name)
    if model_data is None:
        logger.error('模型加载失败，无法进行测试')
        return 1
    model, scalers, adj_matrix, params = model_data
    
    # 加载测试数据
    test_data = load_test_data(dataset_name)
    if test_data is None:
        logger.error('测试数据加载失败，无法进行测试')
        return 1
    X, y, external_features, adj_matrix, scalers = test_data
    
    # 划分测试集（使用最后20%的数据作为测试集）
    test_size = 0.2
    test_samples = int(len(X) * test_size)
    X_test = X[-test_samples:]
    y_test = y[-test_samples:]
    ext_test = external_features[-test_samples:]
    
    # 评估模型
    results = evaluate_model(model, X_test, y_test, ext_test, adj_matrix, scalers)
    
    # 绘制结果
    plot_results(results, model_name, dataset_name)
    
    logger.info('=== 训练后测试完成 ===')
    return 0

if __name__ == '__main__':
    sys.exit(main())