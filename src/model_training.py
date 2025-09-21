#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练模块
负责模型的训练、验证和保存
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from src.model_structure import create_model
from src.fuzzy_inference import FuzzyInferenceSystem
# 导入训练进度可视化工具
from src.training_visualizer import TrainingVisualizer
import pickle

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 加载预处理数据
def load_preprocessed_data(dataset_name):
    """加载预处理后的数据"""
    logger.info(f'加载数据集: {dataset_name}')
    
    # 确定数据目录
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # 对于Los-loop数据集，使用processed目录
    if dataset_name.startswith('Los-loop'):
        data_dir = os.path.join(base_dir, 'data', 'processed')
        
        # 加载训练数据
        X_file = os.path.join(data_dir, f'{dataset_name}_X.npy')
        if not os.path.exists(X_file):
            raise FileNotFoundError(f'未找到输入数据文件: {X_file}')
        X = np.load(X_file)
        
        # 加载标签数据
        y_file = os.path.join(data_dir, f'{dataset_name}_y.npy')
        if not os.path.exists(y_file):
            raise FileNotFoundError(f'未找到标签数据文件: {y_file}')
        y = np.load(y_file)
        
        # 加载外部特征
        external_file = os.path.join(data_dir, f'{dataset_name}_external.npy')
        if not os.path.exists(external_file):
            raise FileNotFoundError(f'未找到外部特征文件: {external_file}')
        external_features = np.load(external_file)
        
        # 加载邻接矩阵
        adj_file = os.path.join(data_dir, f'{dataset_name}_adj_matrix.npy')
        if not os.path.exists(adj_file):
            raise FileNotFoundError(f'未找到邻接矩阵文件: {adj_file}')
        adj_matrix = np.load(adj_file)
        
        # 加载标准化器
        scalers_file = os.path.join(data_dir, f'{dataset_name}_scalers.pkl')
        if os.path.exists(scalers_file):
            import pickle
            with open(scalers_file, 'rb') as f:
                scalers = pickle.load(f)
            logger.info('已加载预训练的标准化器')
        else:
            # 如果没有预训练的标准化器，则创建新的
            logger.warning('未找到预训练的标准化器，创建新的标准化器')
            scalers = {}
            
            # 标准化交通数据
            speed_scaler = MinMaxScaler(feature_range=(0, 1))
            # 重塑数据以适应MinMaxScaler
            X_flat = X.reshape(-1, X.shape[-1])
            speed_scaler.fit(X_flat)
            scalers['speed_scaler'] = speed_scaler
            
            # 标准化外部特征
            if external_features.size > 0:
                ext_scaler = MinMaxScaler(feature_range=(0, 1))
                ext_flat = external_features.reshape(-1, external_features.shape[-1])
                ext_scaler.fit(ext_flat)
                scalers['external_scaler'] = ext_scaler
    else:
        # 原始的加载逻辑，用于其他数据集
        data_dir = os.path.join(base_dir, 'data')
        
        # 加载交通流量数据
        traffic_file = os.path.join(data_dir, f'traffic_data_{dataset_name}.npy')
        if not os.path.exists(traffic_file):
            raise FileNotFoundError(f'未找到交通数据文件: {traffic_file}')
        traffic_data = np.load(traffic_file)
        
        # 加载外部特征
        external_file = os.path.join(data_dir, f'external_features_{dataset_name}.npy')
        if not os.path.exists(external_file):
            raise FileNotFoundError(f'未找到外部特征文件: {external_file}')
        external_features = np.load(external_file)
        
        # 加载邻接矩阵
        adj_file = os.path.join(data_dir, f'adjacency_matrix_{dataset_name}.npy')
        if not os.path.exists(adj_file):
            raise FileNotFoundError(f'未找到邻接矩阵文件: {adj_file}')
        adj_matrix = np.load(adj_file)
        
        # 创建标签（预测未来一个时间步）
        X = traffic_data[:-1, :]
        y = traffic_data[1:, :]
        
        # 标准化数据
        scalers = {}
        
        # 标准化交通数据
        speed_scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = speed_scaler.fit_transform(X)
        y_scaled = speed_scaler.transform(y)
        scalers['speed_scaler'] = speed_scaler
        
        # 标准化外部特征
        ext_scaler = MinMaxScaler(feature_range=(0, 1))
        external_scaled = ext_scaler.fit_transform(external_features)
        scalers['external_scaler'] = ext_scaler
        
        # 重塑为LSTM需要的格式
        X = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        y = y_scaled
        external_features = external_scaled
    
    logger.info(f'数据集加载完成: X形状={X.shape}, y形状={y.shape}, 外部特征形状={external_features.shape}, 邻接矩阵形状={adj_matrix.shape}')
    return X, y, external_features, adj_matrix, scalers

# 划分数据集
def split_data(X, y, external_features):
    """划分训练集、验证集和测试集"""
    logger.info('划分数据集...')
    
    # 首先划分训练集和测试集
    X_train_val, X_test, y_train_val, y_test, ext_train_val, ext_test = train_test_split(
        X, y, external_features,
        test_size=0.2,
        random_state=42
    )
    
    # 然后从训练集中划分验证集
    X_train, X_val, y_train, y_val, ext_train, ext_val = train_test_split(
        X_train_val, y_train_val, ext_train_val,
        test_size=0.25,  # 这将使训练集占60%，验证集占20%，测试集占20%
        random_state=42
    )
    
    logger.info(f'训练集大小: {X_train.shape[0]}')
    logger.info(f'验证集大小: {X_val.shape[0]}')
    logger.info(f'测试集大小: {X_test.shape[0]}')
    
    return X_train, X_val, X_test, y_train, y_val, y_test, ext_train, ext_val, ext_test

# 创建TensorFlow数据集
def create_tf_dataset(X, y, adj_matrix, external_features, batch_size=32, shuffle=True):
    """创建TensorFlow数据集"""
    # 扩展邻接矩阵以匹配批次大小
    adj_matrix_expanded = np.expand_dims(adj_matrix, axis=0)
    adj_matrix_expanded = np.repeat(adj_matrix_expanded, X.shape[0], axis=0)
    
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'traffic_input': X,
            'adjacency_matrix': adj_matrix_expanded,
            'external_features': external_features
        },
        y
    ))
    
    # 打乱数据
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # 批次处理
    dataset = dataset.batch(batch_size)
    
    # 预取数据以提高性能
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# 自定义回调以显示训练进度
def create_training_callbacks(args, log_dir=None):
    """创建训练回调"""
    callbacks = []
    
    # 创建模型保存目录
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    try:
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f'模型目录已创建: {models_dir}')
    except Exception as e:
        logger.error(f'创建模型目录失败: {e}')
    
    # 模型检查点回调
    checkpoint_path = os.path.join(models_dir, f'{args.model}_{args.dataset}_best.keras')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # 学习率调度器
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(lr_scheduler)
    
    # 早停策略
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # 临时禁用TensorBoard回调，解决路径解析问题
    logger.info('临时禁用TensorBoard回调，使用TrainingVisualizer代替')
    
    # 添加自定义进度打印回调，包含反规划功能和MAE/RMSE评估
    class ProgressPrintCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.previous_loss = float('inf')
            self.best_loss = float('inf')
            self.loss_history = []
            # 初始化评估指标跟踪
            self.best_mae = float('inf')
            self.best_rmse = float('inf')
        
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                current_loss = logs['loss']
                val_loss = logs.get('val_loss', 0)
                self.loss_history.append(current_loss)
                
                # 获取MAE和RMSE值
                current_mae = logs.get('mean_absolute_error', 0)
                current_rmse = logs.get('root_mean_squared_error', 0)
                val_mae = logs.get('val_mean_absolute_error', 0)
                val_rmse = logs.get('val_root_mean_squared_error', 0)
                
                # 计算损失变化率（反规划）
                loss_change = (self.previous_loss - current_loss) / self.previous_loss if self.previous_loss > 0 else 0
                
                # 更新最佳指标
                is_best = False
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    is_best = True
                if current_mae < self.best_mae:
                    self.best_mae = current_mae
                    is_best = True
                if current_rmse < self.best_rmse:
                    self.best_rmse = current_rmse
                    is_best = True
                
                # 计算平滑损失（最后5轮的平均值）
                if len(self.loss_history) >= 5:
                    smoothed_loss = sum(self.loss_history[-5:]) / 5
                else:
                    smoothed_loss = current_loss
                
                # 判断是否为有效训练（MAE和RMSE都在合理范围内且呈下降趋势）
                is_valid_training = True
                # 这里设置一个简单的判断标准：MAE和RMSE不是NaN且在合理范围内
                if np.isnan(current_mae) or np.isnan(current_rmse) or current_mae > 1.0 or current_rmse > 1.0:
                    is_valid_training = False
                
                # 打印详细的训练信息 - 使用纯文本标记代替表情符号
                logger.info(f"轮次 {epoch+1}/{self.params['epochs']}: 损失={current_loss:.6f}, " +
                           f"验证损失={val_loss:.6f}, 损失变化率={loss_change:.2%}, " +
                           f"MAE={current_mae:.6f}, 验证MAE={val_mae:.6f}, " +
                           f"RMSE={current_rmse:.6f}, 验证RMSE={val_rmse:.6f}, " +
                           f"平滑损失={smoothed_loss:.6f}{' [最佳]' if is_best else ''}{' [有效]' if is_valid_training else ' [无效]'}")
                
                # 更新前一轮损失
                self.previous_loss = current_loss
    
    # 将自定义回调添加到callbacks列表
    progress_callback = ProgressPrintCallback()
    callbacks.append(progress_callback)
    
    return callbacks

# 训练模型
def train_model(args):
    """训练模型主函数"""
    logger.info(f'开始训练模型: {args.model}, 数据集: {args.dataset}')
    
    # 创建可视化器实例
    visualizer = TrainingVisualizer(
        model_name=args.model,
        dataset_name=args.dataset
    )
    
    # 加载数据
    X, y, external_features, adj_matrix, scalers = load_preprocessed_data(args.dataset)
    
    # 划分数据
    X_train, X_val, X_test, y_train, y_val, y_test, ext_train, ext_val, ext_test = split_data(
        X, y, external_features
    )
    
    # 邻接矩阵处理 - 不再重复扩展，以减少内存使用
    # 创建一个批次大小的邻接矩阵，在训练过程中重复使用
    adj_matrix_batch = np.expand_dims(adj_matrix, axis=0)
    
    # 创建模型
    num_nodes = X.shape[2]
    logger.info(f'创建模型，节点数: {num_nodes}, 隐藏层大小: {args.hidden_size}')
    model = create_model(
        num_nodes=num_nodes,
        timesteps=args.timesteps,
        hidden_size=args.hidden_size
    )
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    )
    
    # 初始化模糊推理系统
    logger.info('初始化模糊推理系统...')
    fuzzy_system = FuzzyInferenceSystem()
    
    # 准备训练回调
    callbacks = create_training_callbacks(args)
    
    # 创建进度条回调 - 用于集成到TrainingVisualizer
    class VisualizationCallback(tf.keras.callbacks.Callback):
        def __init__(self, visualizer):
            super().__init__()
            self.visualizer = visualizer
            
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                self.visualizer.update_epoch(epoch, logs)
    
    # 添加可视化回调
    visualization_callback = VisualizationCallback(visualizer)
    callbacks.append(visualization_callback)
    
    # 开始训练
    visualizer.start_training(args.epochs)
    
    # 训练模型
    logger.info(f'开始模型训练，共{args.epochs}轮')
    
    # 使用数据生成器处理邻接矩阵，减少内存使用
    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, X, y, ext_features, adj_matrix, batch_size):
            self.X = X
            self.y = y
            self.ext_features = ext_features
            self.adj_matrix = adj_matrix  # 形状: (1, num_nodes, num_nodes)
            self.batch_size = batch_size
            self.indices = np.arange(len(X))
            
            # 使用元组格式的output_signature，让TensorFlow按顺序匹配输入层
            self.output_signature = (
                (tf.TensorSpec(shape=(None,) + X.shape[1:], dtype=tf.float32),
                 tf.TensorSpec(shape=(None,) + adj_matrix.shape[1:], dtype=tf.float32),
                 tf.TensorSpec(shape=(None,) + ext_features.shape[1:], dtype=tf.float32)),
                tf.TensorSpec(shape=(None,) + y.shape[1:], dtype=tf.float32)
            )
        
        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))
        
        def __getitem__(self, idx):
            batch_indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            ext_batch = self.ext_features[batch_indices]
            
            # 为当前批次重复邻接矩阵
            batch_size = len(batch_indices)
            adj_matrix_batch = np.repeat(self.adj_matrix, batch_size, axis=0)
            
            # 返回元组格式的数据，与output_signature匹配
            return (X_batch, adj_matrix_batch, ext_batch), y_batch
    
    # 创建训练和验证数据生成器
    train_generator = DataGenerator(X_train, y_train, ext_train, adj_matrix_batch, args.batch_size)
    val_generator = DataGenerator(X_val, y_val, ext_val, adj_matrix_batch, args.batch_size)
    
    # 使用生成器进行训练
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0  # 禁用默认输出，使用自定义可视化
    )
    
    # 结束训练
    visualizer.end_training()
    
    # 保存模型
    logger.info('保存训练好的模型...')
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    if not os.path.exists(models_dir):
        try:
            os.makedirs(models_dir, exist_ok=True)
        except Exception as e:
            logger.error(f'创建模型目录失败: {e}')
    
    # 保存完整模型
    model_path = os.path.join(models_dir, f'{args.model}_{args.dataset}.keras')
    try:
        model.save(model_path)
        logger.info(f'模型已保存至: {model_path}')
    except Exception as e:
        logger.error(f'保存模型失败: {e}')
    
    # 保存模型参数
    try:
        import pickle
        model_info = {
            'model_path': model_path,
            'scalers': scalers,
            'adj_matrix': adj_matrix,
            'params': {
                'num_nodes': adj_matrix.shape[0],
                'timesteps': args.timesteps,
                'hidden_size': args.hidden_size,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'dataset': args.dataset
            }
        }
        
        info_path = os.path.join(models_dir, f'{args.model}_{args.dataset}_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        logger.info(f'模型信息已保存至: {info_path}')
    except Exception as e:
        logger.error(f'保存模型信息失败: {e}')
    
    # 保存模糊推理系统
    try:
        fuzzy_path = os.path.join(models_dir, f'fuzzy_system_{args.dataset}.pkl')
        fuzzy_system.save_model(fuzzy_path)
        logger.info(f'模糊推理系统已保存至: {fuzzy_path}')
    except Exception as e:
        logger.error(f'保存模糊推理系统失败: {e}')
        
    # 在测试集上评估模型
    results = evaluate_on_test(model, X_test, y_test, ext_test, adj_matrix, scalers)
        
    # 设置测试结果并生成可视化
    if results:
        # 只提取数值型的测试结果用于显示
        numeric_results = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        visualizer.set_test_results(numeric_results)
        
        # 绘制训练进度
        visualizer.plot_training_progress()
        
        # 获取预测值和真实值用于可视化
        y_test_original = results['y_test']
        y_pred_original = results['y_pred']
        
        # 绘制预测对比
        visualizer.plot_prediction_comparison(y_test_original, y_pred_original)
        
        # 绘制误差分布
        visualizer.plot_error_distribution(y_test_original, y_pred_original)

# 在测试集上评估模型
def evaluate_on_test(model, X_test, y_test, ext_test, adj_matrix, scalers):
    """在测试集上评估模型性能"""
    logger.info('在测试集上评估模型性能...')
    
    try:
        # 使用数据生成器处理邻接矩阵，减少内存使用
        class TestDataGenerator(tf.keras.utils.Sequence):
            def __init__(self, X, y, ext_features, adj_matrix, batch_size=32):
                self.X = X
                self.y = y
                self.ext_features = ext_features
                self.adj_matrix = np.expand_dims(adj_matrix, axis=0)  # 形状: (1, num_nodes, num_nodes)
                self.batch_size = batch_size
                self.indices = np.arange(len(X))
                
                # 使用元组格式的output_signature，让TensorFlow按顺序匹配输入层
                self.output_signature = (
                    (tf.TensorSpec(shape=(None,) + X.shape[1:], dtype=tf.float32),
                     tf.TensorSpec(shape=(None,) + self.adj_matrix.shape[1:], dtype=tf.float32),
                     tf.TensorSpec(shape=(None,) + ext_features.shape[1:], dtype=tf.float32)),
                    tf.TensorSpec(shape=(None,) + y.shape[1:], dtype=tf.float32)
                )
            
            def __len__(self):
                return int(np.ceil(len(self.X) / self.batch_size))
            
            def __getitem__(self, idx):
                batch_indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
                X_batch = self.X[batch_indices]
                y_batch = self.y[batch_indices]
                ext_batch = self.ext_features[batch_indices]
                
                # 为当前批次重复邻接矩阵
                batch_size = len(batch_indices)
                adj_matrix_batch = np.repeat(self.adj_matrix, batch_size, axis=0)
                
                # 返回元组格式的数据，与output_signature匹配
                return (X_batch, adj_matrix_batch, ext_batch), y_batch
        
        # 创建测试数据生成器
        test_generator = TestDataGenerator(X_test, y_test, ext_test, adj_matrix, batch_size=32)
        
        # 评估模型
        loss, rmse, mae = model.evaluate(
            test_generator,
            verbose=1
        )
        
        # 获取速度标准化器
        speed_scaler = scalers['speed_scaler']
        
        # 反归一化评估指标
        # 注意：RMSE和MAE需要使用原始数据的标准差进行反归一化
        # 这里简化处理，假设原始数据的范围是0-120 km/h
        original_rmse = rmse * 120  # 假设标准化范围是[0,1]
        original_mae = mae * 120
        
        logger.info(f'测试集性能:')
        logger.info(f'Loss: {loss:.4f}')
        logger.info(f'RMSE (标准化): {rmse:.4f}')
        logger.info(f'RMSE (原始值，km/h): {original_rmse:.4f}')
        logger.info(f'MAE (标准化): {mae:.4f}')
        logger.info(f'MAE (原始值，km/h): {original_mae:.4f}')
        
        # 计算准确率（假设预测速度在真实值的±10%范围内为准确）
        # 使用测试数据生成器进行预测
        y_pred = model.predict(test_generator)
        
        # 反归一化预测值和真实值
        y_test_original = speed_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        y_pred_original = speed_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
        
        # 计算准确率
        threshold = 0.1  # 10%的误差范围
        accurate_predictions = np.abs((y_pred_original - y_test_original) / (y_test_original + 1e-8)) <= threshold
        accuracy = np.mean(accurate_predictions)
        
        logger.info(f'准确率 (±{threshold*100}%误差范围): {accuracy*100:.2f}%')
        
        # 保存评估结果
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        results = {
            'loss': loss,
            'rmse': rmse,
            'mae': mae,
            'original_rmse': original_rmse,
            'original_mae': original_mae,
            'accuracy': accuracy,
            'y_test': y_test_original,
            'y_pred': y_pred_original
        }
        
        import pickle
        results_path = os.path.join(results_dir, f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f'测试结果已保存至: {results_path}')
        
        return results
        
    except Exception as e:
        logger.error(f'测试集评估失败: {e}')
        return None

# 超参数调优
def hyperparameter_tuning(dataset_name):
    """超参数调优"""
    logger.info(f'开始超参数调优: {dataset_name}')
    
    try:
        # 加载数据
        X, y, external_features, adj_matrix, scalers = load_preprocessed_data(dataset_name)
        
        # 划分数据
        X_train, X_val, X_test, y_train, y_val, y_test, ext_train, ext_val, ext_test = split_data(
            X, y, external_features
        )
        
        # 扩展邻接矩阵
        adj_matrix_train = np.expand_dims(adj_matrix, axis=0)
        adj_matrix_train = np.repeat(adj_matrix_train, X_train.shape[0], axis=0)
        
        adj_matrix_val = np.expand_dims(adj_matrix, axis=0)
        adj_matrix_val = np.repeat(adj_matrix_val, X_val.shape[0], axis=0)
        
        # 定义超参数搜索空间
        param_grid = {
            'batch_size': [16, 32, 64],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'hidden_size': [32, 64, 128],
            'timesteps': [6, 12, 24]
        }
        
        # 记录最佳参数和性能
        best_params = None
        best_val_loss = float('inf')
        
        # 网格搜索
        with tqdm(total=len(param_grid['batch_size']) * len(param_grid['learning_rate']) * len(param_grid['hidden_size']), desc='超参数搜索') as pbar:
            for batch_size in param_grid['batch_size']:
                for lr in param_grid['learning_rate']:
                    for hidden_size in param_grid['hidden_size']:
                        # 为每个参数组合创建和训练模型
                        logger.info(f'测试参数组合: batch_size={batch_size}, lr={lr}, hidden_size={hidden_size}')
                        
                        # 创建模型
                        num_nodes = X.shape[2]
                        model = create_model(
                            num_nodes=num_nodes,
                            timesteps=param_grid['timesteps'][1],  # 使用中间值作为默认值
                            hidden_size=hidden_size
                        )
                        
                        # 编译模型
                        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                        model.compile(
                            optimizer=optimizer,
                            loss=tf.keras.losses.MeanSquaredError(),
                            metrics=[tf.keras.metrics.RootMeanSquaredError()]
                        )
                        
                        # 训练模型
                        history = model.fit(
                            x=[X_train, adj_matrix_train, ext_train],
                            y=y_train,
                            validation_data=([X_val, adj_matrix_val, ext_val], y_val),
                            epochs=20,  # 为了节省时间，使用较少的轮数
                            batch_size=batch_size,
                            verbose=0
                        )
                        
                        # 获取验证损失
                        val_loss = min(history.history['val_loss'])
                        
                        # 更新最佳参数
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = {
                                'batch_size': batch_size,
                                'learning_rate': lr,
                                'hidden_size': hidden_size,
                                'timesteps': param_grid['timesteps'][1]
                            }
                            
                        logger.info(f'参数组合性能: val_loss={val_loss:.4f}, best_val_loss={best_val_loss:.4f}')
                        pbar.update(1)
        
        logger.info(f'超参数调优完成！最佳参数: {best_params}')
        
        # 保存最佳参数
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, f'best_params_{dataset_name}.pkl'), 'wb') as f:
            pickle.dump(best_params, f)
        
    except Exception as e:
        logger.error(f'超参数调优失败: {e}')
        best_params = None
    
    return best_params

if __name__ == '__main__':
    # 测试训练功能
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Los-loop')
    parser.add_argument('--model', type=str, default='gcn_gru_fuzzy')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=12)
    
    args = parser.parse_args()
    train_model(args)