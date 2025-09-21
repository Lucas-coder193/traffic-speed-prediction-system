#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估模块
负责评估模型性能并生成评估报告
"""

import os
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 配置日志
logger = logging.getLogger('TrafficPrediction')

# 加载数据
def load_evaluation_data(dataset_name='SZBZ'):
    """加载评估数据"""
    logger.info(f'加载{dataset_name}数据集用于评估...')
    
    # 确定数据路径
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    
    # 加载测试数据
    X_test_path = os.path.join(data_dir, f'{dataset_name}_X_test.npy')
    y_test_path = os.path.join(data_dir, f'{dataset_name}_y_test.npy')
    adj_matrix_path = os.path.join(data_dir, f'{dataset_name}_adj_matrix.npy')
    external_test_path = os.path.join(data_dir, f'{dataset_name}_external_test.npy')
    scalers_path = os.path.join(data_dir, f'{dataset_name}_scalers.pkl')
    
    # 检查文件是否存在
    for path in [X_test_path, y_test_path, adj_matrix_path, external_test_path, scalers_path]:
        if not os.path.exists(path):
            logger.error(f'文件不存在: {path}')
            raise FileNotFoundError(f'文件不存在: {path}')
    
    # 加载数据
    try:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        adj_matrix = np.load(adj_matrix_path)
        external_test = np.load(external_test_path)
        
        # 加载标准化器
        import pickle
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        logger.info(f'评估数据加载成功，测试集大小: {X_test.shape[0]}')
        return X_test, y_test, adj_matrix, external_test, scalers
        
    except Exception as e:
        logger.error(f'加载评估数据时出错: {e}')
        raise

# 评估模型性能
def evaluate_model(model, X_test, y_test, adj_matrix, external_test, scalers, params=None):
    """评估模型性能"""
    logger.info('评估模型性能...')
    
    try:
        # 扩展邻接矩阵维度以适应模型输入
        batch_size = X_test.shape[0]
        adj_matrix_expanded = np.repeat(adj_matrix[np.newaxis, :, :], batch_size, axis=0)
        
        # 模型预测
        logger.info('进行模型预测...')
        y_pred_scaled = model.predict([X_test, adj_matrix_expanded, external_test])
        
        # 反归一化预测结果和真实值
        speed_scaler = scalers['speed_scaler']
        y_pred = speed_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
        y_true = speed_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        
        # 计算评估指标
        metrics = calculate_metrics(y_true, y_pred)
        
        # 生成评估报告
        report = generate_evaluation_report(metrics, y_true, y_pred, params)
        
        # 可视化评估结果
        visualize_evaluation(y_true, y_pred, metrics)
        
        logger.info('模型评估完成')
        return report, metrics, y_true, y_pred
        
    except Exception as e:
        logger.error(f'评估模型时出错: {e}')
        raise

# 计算评估指标
def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    logger.info('计算评估指标...')
    
    # 扁平化数据以便计算指标
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    # 计算均方误差 (MSE)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    
    # 计算根均方误差 (RMSE)
    rmse = np.sqrt(mse)
    
    # 计算平均绝对误差 (MAE)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    
    # 计算平均绝对百分比误差 (MAPE)
    # 避免除以零的情况
    mask = y_true_flat != 0
    mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    
    # 计算R方 (R²)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # 计算准确率 (根据论文中的定义，这里简单定义为预测值与真实值之差在某个范围内的比例)
    accuracy_5 = np.mean(np.abs(y_true_flat - y_pred_flat) <= 5) * 100  # 误差在5 km/h以内
    accuracy_10 = np.mean(np.abs(y_true_flat - y_pred_flat) <= 10) * 100  # 误差在10 km/h以内
    
    # 组织指标
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Accuracy_5': accuracy_5,
        'Accuracy_10': accuracy_10
    }
    
    # 打印指标
    logger.info('评估指标:')
    for key, value in metrics.items():
        if key in ['MSE', 'RMSE', 'MAE']:
            logger.info(f'{key}: {value:.4f}')
        else:
            logger.info(f'{key}: {value:.4f}%' if key in ['MAPE', 'Accuracy_5', 'Accuracy_10'] else f'{key}: {value:.4f}')
    
    return metrics

# 生成评估报告
def generate_evaluation_report(metrics, y_true, y_pred, params=None):
    """生成评估报告"""
    logger.info('生成评估报告...')
    
    # 创建报告目录
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # 创建报告内容
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_content = []
    report_content.append(f'交通速度预测模型评估报告')
    report_content.append(f'生成时间: {timestamp}')
    report_content.append(f'{"-"*50}')
    
    # 添加参数信息
    if params is not None:
        report_content.append('模型参数:')
        for key, value in params.items():
            report_content.append(f'  {key}: {value}')
        report_content.append(f'{"-"*50}')
    
    # 添加评估指标
    report_content.append('评估指标:')
    for key, value in metrics.items():
        if key in ['MSE', 'RMSE', 'MAE']:
            report_content.append(f'  {key}: {value:.4f}')
        else:
            report_content.append(f'  {key}: {value:.4f}%' if key in ['MAPE', 'Accuracy_5', 'Accuracy_10'] else f'  {key}: {value:.4f}')
    report_content.append(f'{"-"*50}')
    
    # 添加数据统计
    report_content.append('数据统计:')
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    report_content.append(f'  测试样本数: {len(y_true_flat)}')
    report_content.append(f'  真实速度范围: {np.min(y_true_flat):.2f} - {np.max(y_true_flat):.2f} km/h')
    report_content.append(f'  预测速度范围: {np.min(y_pred_flat):.2f} - {np.max(y_pred_flat):.2f} km/h')
    report_content.append(f'{"-"*50}')
    
    # 添加结论
    if metrics['RMSE'] < 5 and metrics['Accuracy_10'] > 90:
        report_content.append('结论: 模型性能优秀')
    elif metrics['RMSE'] < 10 and metrics['Accuracy_10'] > 80:
        report_content.append('结论: 模型性能良好')
    elif metrics['RMSE'] < 15 and metrics['Accuracy_10'] > 70:
        report_content.append('结论: 模型性能一般')
    else:
        report_content.append('结论: 模型性能较差，建议进一步优化')
    
    # 保存报告
    report_path = os.path.join(reports_dir, f'evaluation_report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    logger.info(f'评估报告已保存至: {report_path}')
    
    # 返回报告内容
    return '\n'.join(report_content)

# 可视化评估结果
def visualize_evaluation(y_true, y_pred, metrics):
    """可视化评估结果"""
    logger.info('可视化评估结果...')
    
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 扁平化数据
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    # 1. 真实值 vs 预测值散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=10)
    
    # 添加参考线
    min_val = min(np.min(y_true_flat), np.min(y_pred_flat)) - 5
    max_val = max(np.max(y_true_flat), np.max(y_pred_flat)) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # 设置图表属性
    plt.title('真实值 vs 预测值')
    plt.xlabel('真实速度 (km/h)')
    plt.ylabel('预测速度 (km/h)')
    plt.grid(True, alpha=0.3)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # 添加R²和RMSE信息
    plt.text(0.05, 0.95, f"R² = {metrics['R2']:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.90, f"RMSE = {metrics['RMSE']:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    # 保存图表
    scatter_path = os.path.join(results_dir, f"true_vs_pred_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(scatter_path)
    plt.close()
    logger.info(f'真实值vs预测值散点图已保存至: {scatter_path}')
    
    # 2. 误差分布直方图
    plt.figure(figsize=(10, 6))
    errors = y_pred_flat - y_true_flat
    sns.histplot(errors, bins=50, kde=True)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.title('预测误差分布')
    plt.xlabel('误差 (km/h)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    # 添加MAE信息
    plt.text(0.95, 0.95, f"MAE = {metrics['MAE']:.4f}", transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', horizontalalignment='right')
    
    # 保存图表
    error_hist_path = os.path.join(results_dir, f"error_histogram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(error_hist_path)
    plt.close()
    logger.info(f'预测误差分布图已保存至: {error_hist_path}')
    
    # 3. 部分节点的真实值与预测值对比图
    # 选择几个节点进行展示
    num_nodes = y_true.shape[1] if len(y_true.shape) > 1 else 1
    nodes_to_show = min(5, num_nodes)
    
    if len(y_true.shape) > 1 and y_true.shape[0] > 10:
        # 只显示前50个时间步
        time_steps = min(50, y_true.shape[0])
        
        plt.figure(figsize=(12, 8))
        for i in range(nodes_to_show):
            plt.subplot(nodes_to_show, 1, i+1)
            plt.plot(range(time_steps), y_true[:time_steps, i], 'b-', label='真实值')
            plt.plot(range(time_steps), y_pred[:time_steps, i], 'r--', label='预测值')
            plt.title(f'节点 {i+1} 的速度预测')
            plt.ylabel('速度 (km/h)')
            if i == nodes_to_show - 1:
                plt.xlabel('时间步')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        time_series_path = os.path.join(results_dir, f'time_series_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png')
        plt.savefig(time_series_path)
        plt.close()
        logger.info(f'时间序列对比图已保存至: {time_series_path}')
    
    # 4. 指标雷达图
    # 选择合适的指标进行雷达图展示
    radar_metrics = {
        'RMSE': 1 - min(metrics['RMSE'] / 20, 1),  # 归一化，假设RMSE上限为20
        'MAE': 1 - min(metrics['MAE'] / 15, 1),  # 归一化，假设MAE上限为15
        'MAPE': 1 - min(metrics['MAPE'] / 30, 1),  # 归一化，假设MAPE上限为30%
        'R2': max(0, metrics['R2']),  # 确保R2为正值
        'Accuracy_10': metrics['Accuracy_10'] / 100  # 归一化到0-1
    }
    
    categories = list(radar_metrics.keys())
    values = list(radar_metrics.values())
    
    # 闭合雷达图
    values = values + values[:1]
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles = angles + angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # 设置标签
    plt.xticks(angles[:-1], categories)
    ax.set_yticklabels([])
    plt.title('模型性能雷达图')
    
    # 添加数值标签
    for angle, value, metric in zip(angles[:-1], values[:-1], categories):
        if metric == 'RMSE':
            label = f'{metrics[metric]:.2f}'
        elif metric == 'MAE':
            label = f'{metrics[metric]:.2f}'
        elif metric == 'MAPE':
            label = f'{metrics[metric]:.2f}%'
        elif metric == 'R2':
            label = f'{metrics[metric]:.3f}'
        elif metric == 'Accuracy_10':
            label = f'{metrics[metric]:.1f}%'
        
        ax.annotate(label, xy=(angle, value), xytext=(angle, value + 0.1),
                    ha='center', va='center', fontsize=10)
    
    # 保存图表
    radar_path = os.path.join(results_dir, f'performance_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png')
    plt.savefig(radar_path)
    plt.close()
    logger.info(f'模型性能雷达图已保存至: {radar_path}')

# 运行评估
def run_evaluation(args):
    """运行评估"""
    logger.info('=== 开始模型评估 ===')
    
    try:
        # 1. 加载训练好的模型
        from .predict import load_trained_model
        model, scalers, adj_matrix, params, _ = load_trained_model(args.model, args.dataset)
        
        # 2. 加载评估数据
        X_test, y_test, adj_matrix_test, external_test, data_scalers = load_evaluation_data(args.dataset)
        
        # 如果加载的模型中的scalers与数据中的scalers不同，使用数据中的scalers
        if scalers != data_scalers:
            logger.warning('模型中的scalers与数据中的scalers不同，使用数据中的scalers')
            scalers = data_scalers
        
        # 3. 评估模型
        report, metrics, y_true, y_pred = evaluate_model(
            model, X_test, y_test, adj_matrix_test, external_test, scalers, params
        )
        
        # 4. 打印评估报告摘要
        logger.info('\n评估报告摘要:')
        logger.info(report.split('\n')[0])  # 只打印标题
        logger.info(report.split('\n')[2])  # 只打印分隔线
        for line in report.split('\n'):
            if line.strip().startswith(('MSE:', 'RMSE:', 'MAE:', 'MAPE:', 'R2:', 'Accuracy_')):
                logger.info(line)
        
        logger.info('=== 模型评估完成 ===')
        
        return report, metrics
        
    except Exception as e:
        logger.error(f'评估过程中出错: {e}')
        raise

# 消融实验
def ablation_experiment(args):
    """进行消融实验"""
    logger.info('=== 开始消融实验 ===')
    
    # 这里应该实现消融实验的逻辑
    # 例如，分别去掉天气、节假日、流量、时间段等外部属性，比较性能差异
    
    # 目前简化实现
    logger.info('消融实验功能尚未完全实现')
    
    # 创建实验结果目录
    experiments_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments')
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    
    # 记录实验信息
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_info = {
        'timestamp': timestamp,
        'args': vars(args),
        'status': 'not_implemented'
    }
    
    exp_path = os.path.join(experiments_dir, f'ablation_experiment_{timestamp}.pkl')
    import pickle
    with open(exp_path, 'wb') as f:
        pickle.dump(exp_info, f)
    
    logger.info(f'消融实验信息已保存至: {exp_path}')
    logger.info('=== 消融实验完成 ===')

# 对比实验
def comparison_experiment(args):
    """进行对比实验"""
    logger.info('=== 开始对比实验 ===')
    
    # 这里应该实现对比实验的逻辑
    # 例如，与HA、ARIMA、SVR、T-GCN、AST-GCN等模型对比
    
    # 目前简化实现
    logger.info('对比实验功能尚未完全实现')
    
    # 创建实验结果目录
    experiments_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments')
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    
    # 记录实验信息
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_info = {
        'timestamp': timestamp,
        'args': vars(args),
        'status': 'not_implemented'
    }
    
    exp_path = os.path.join(experiments_dir, f'comparison_experiment_{timestamp}.pkl')
    import pickle
    with open(exp_path, 'wb') as f:
        pickle.dump(exp_info, f)
    
    logger.info(f'对比实验信息已保存至: {exp_path}')
    logger.info('=== 对比实验完成 ===')

if __name__ == '__main__':
    # 测试评估功能
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_gru_fuzzy')
    parser.add_argument('--dataset', type=str, default='SZBZ')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'ablation', 'comparison'])
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        run_evaluation(args)
    elif args.mode == 'ablation':
        ablation_experiment(args)
    elif args.mode == 'comparison':
        comparison_experiment(args)