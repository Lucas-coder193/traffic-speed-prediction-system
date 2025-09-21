#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型结构模块
实现GCN+GRU+模糊推理融合的深度学习模型
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
try:
    from keras import ops as K
except Exception:
    K = tf
import numpy as np
import logging
import sys
import os
from src.fuzzy_inference import FuzzyInferenceSystem

# 配置日志
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

@tf.keras.utils.register_keras_serializable()
class GCNLayer(layers.Layer):
    """图卷积网络层"""
    def __init__(self, units, activation=None, dropout_rate=0.0, l2_reg=0.0, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
    def build(self, input_shape):
        # 输入形状: [batch_size, num_nodes, features]
        input_dim = input_shape[-1]
        
        # 创建权重矩阵
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            regularizer=regularizers.l2(self.l2_reg),
            trainable=True,
            name='kernel'
        )
        
        # 创建偏置
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        
        # 创建dropout层
        self.dropout = layers.Dropout(self.dropout_rate)
        self.built = True
        
    def call(self, inputs, training=False, adj_matrix=None):
        try:
            # 确保training是布尔值
            is_training = bool(training) if not isinstance(training, str) else training.lower() == 'true'
            
            # 确保inputs是有效的张量
            if inputs is None:
                raise ValueError("GCNLayer received None as input")
            
            # 输入在Keras函数式图中可能是KerasTensor，保持原样以兼容反序列化
            
            # 应用dropout
            if is_training:
                inputs = self.dropout(inputs)
            
            # 计算 GCN 前向传播: A * X * W + b
            
            # 检查adj_matrix是否为None
            if adj_matrix is None:
                # 如果adj_matrix为None，记录警告并使用单位矩阵作为默认值
                logger.warning('GCNLayer.call() called without adj_matrix parameter, using identity matrix')
                
                # 获取input的动态形状
                input_shape = tf.shape(inputs)
                batch_size = input_shape[0]
                num_nodes = input_shape[1]
                
                # 创建单位矩阵作为默认邻接矩阵
                adj_matrix = tf.eye(num_nodes, batch_shape=[batch_size])
            else:
                # 邻接矩阵同样可能是KerasTensor/NumPy，直接使用即可
                pass
            
            # 直接使用TensorFlow操作，不使用Lambda层包装
            # 矩阵乘法: X * W
            x = tf.matmul(inputs, self.kernel)
            
            # 应用邻接矩阵: A * X
            x = tf.einsum('bij,bjk->bik', adj_matrix, x)
            
            # 添加偏置
            x = tf.add(x, self.bias)
            
            # 应用激活函数
            if self.activation is not None:
                activation_fn = tf.keras.activations.get(self.activation)
                x = activation_fn(x)
            
            return x
        except Exception as e:
            logger.error(f"Error in GCNLayer.call(): {str(e)}")
            logger.error(f"Inputs shape: {inputs.shape}")
            logger.error(f"Training mode: {is_training}")
            raise
        
    def compute_output_shape(self, input_shape):
        # 输入形状: [(batch_size, num_nodes, features), (batch_size, num_nodes, num_nodes)]
        # 输出形状: (batch_size, num_nodes, self.units)
        if isinstance(input_shape, list) and len(input_shape) > 0:
            return (input_shape[0][0], input_shape[0][1], self.units)
        elif hasattr(input_shape, '__len__') and len(input_shape) >= 2:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (None, None, self.units)
        
    def get_config(self):
        """获取层配置，用于模型保存和加载"""
        config = super(GCNLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        """从配置创建层"""
        return cls(**config)
        
    # 修改__call__方法，兼容Keras的调用约定
    def __call__(self, *args, **kwargs):
        # 使用父类的 __call__ 以确保自动构建权重（build）与追踪计算图
        return super().__call__(*args, **kwargs)

@tf.keras.utils.register_keras_serializable()
class SpatioTemporalGCNGRU(tf.keras.Model):
    """属性增强时空预测模型：GCN+GRU+模糊推理融合"""
    def __init__(self, num_nodes, timesteps, hidden_size=64, gcn_units=32, dropout_rate=0.3, l2_reg=0.001, **kwargs):
        super(SpatioTemporalGCNGRU, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.timesteps = timesteps
        self.hidden_size = hidden_size
        self.gcn_units = gcn_units
        
        # 初始化模糊推理系统
        self.fuzzy_system = FuzzyInferenceSystem()
        
        # 图卷积层
        self.gcn_layer1 = GCNLayer(
            units=gcn_units,
            activation=tf.nn.relu,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )
        self.gcn_layer2 = GCNLayer(
            units=gcn_units,
            activation=tf.nn.relu,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )
        
        # GRU层用于时间特征提取
        self.gru_layer1 = layers.GRU(
            units=hidden_size,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg)
        )
        self.gru_layer2 = layers.GRU(
            units=hidden_size,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg)
        )
        
        # 外部特征处理层
        self.external_dense1 = layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l2(l2_reg)
        )
        self.external_dense2 = layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l2(l2_reg)
        )
        
        # 融合层
        self.fusion_dense1 = layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l2(l2_reg)
        )
        self.fusion_dense2 = layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l2(l2_reg)
        )
        
        # 输出层
        self.output_layer = layers.Dense(
            units=num_nodes,
            activation=None
        )
        
        # Dropout层
        self.dropout = layers.Dropout(dropout_rate)
        
        # BatchNormalization层
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
        self.batch_norm3 = layers.BatchNormalization()
    
    def call(self, inputs, training=False):
        try:
            logger.info(f"SpatioTemporalGCNGRU.call() called with inputs type: {type(inputs)}")
            
            # 确保training是布尔值
            if isinstance(training, str):
                is_training = training.lower() == 'true'
            else:
                is_training = bool(training)
            logger.info(f"Training mode: {is_training}")
            
            # 解包输入 - 更健壮的方式
            speed_sequences = None
            adj_matrix = None
            external_features = None
            
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                logger.info(f"Inputs is list/tuple with length: {len(inputs)}")
                if len(inputs) >= 1:
                    speed_sequences = inputs[0]
                if len(inputs) >= 2:
                    adj_matrix = inputs[1]
                if len(inputs) >= 3:
                    external_features = inputs[2]
            elif isinstance(inputs, dict):
                logger.info("Inputs is dictionary")
                speed_sequences = inputs.get('traffic_input')
                adj_matrix = inputs.get('adjacency_matrix')
                external_features = inputs.get('external_features')
            else:
                # 如果是单个输入，假设是speed_sequences
                logger.warning("Inputs is not list/tuple/dict, assuming it's speed_sequences")
                speed_sequences = inputs
            
            # 确保所有必要输入都存在
            if speed_sequences is None:
                raise ValueError("speed_sequences input is required")
            
            # 直接使用符号张量，避免强制转换
            logger.info(f"Speed sequences shape: {getattr(speed_sequences, 'shape', None)}")
            input_dyn_shape = tf.shape(speed_sequences)
            batch_dyn = input_dyn_shape[0]
            
            # 检查adj_matrix是否为None，创建默认矩阵
            if adj_matrix is None:
                logger.warning("adj_matrix is None, creating default identity matrix")
                # 获取batch_size和num_nodes
                try:
                    input_shape = tf.shape(speed_sequences)
                    batch_size = input_shape[0]
                    num_nodes = input_shape[2]  # 假设形状为 [batch_size, timesteps, num_nodes]
                except Exception as e:
                    logger.error(f"Error getting input shape: {str(e)}")
                    batch_size = 1
                    num_nodes = self.num_nodes
                # 创建单位矩阵
                adj_matrix = tf.eye(num_nodes, batch_shape=[batch_size])
            else:
                # 保持符号/数值张量原样使用
                logger.info(f"Adjacency matrix shape: {adj_matrix.shape}")
            
            # 检查external_features是否为None，创建默认特征
            if external_features is None:
                logger.warning("external_features is None, creating default zeros")
                try:
                    input_shape = tf.shape(speed_sequences)
                    batch_size = input_shape[0]
                except Exception as e:
                    logger.error(f"Error getting input shape: {str(e)}")
                    batch_size = 1
                # 创建零矩阵
                external_features = tf.zeros((batch_size, self.num_nodes, 10))
            else:
                # 保持符号/数值张量原样使用
                logger.info(f"External features shape: {external_features.shape}")
            
            # 调整输入形状以适应GCN（按时间步展开 -> 逐时间步做GCN -> 还原时间维）
            # 输入 speed_sequences: [batch, timesteps, num_nodes]
            if len(speed_sequences.shape) == 3:
                # 添加特征维: [batch, timesteps, num_nodes, 1]
                speed_reshaped = tf.expand_dims(speed_sequences, axis=-1)
            else:
                speed_reshaped = speed_sequences
            
            # 构造按时间步的批次: [batch*timesteps, num_nodes, 1]
            # 使用已知的 self.timesteps，避免动态张量相乘
            timesteps_dyn = self.timesteps
            # 使用-1自动推断 batch*timesteps
            gcn_input = tf.reshape(speed_reshaped, (-1, self.num_nodes, 1))
            
            # 邻接矩阵展开到每个时间步: [batch*timesteps, num_nodes, num_nodes]
            if adj_matrix is None:
                adj_matrix = tf.eye(self.num_nodes, batch_shape=[batch_dyn])
            adj_exp = tf.expand_dims(adj_matrix, axis=1)  # [batch, 1, n, n]
            adj_tiled4 = tf.tile(adj_exp, (1, timesteps_dyn, 1, 1))  # [batch, timesteps, n, n]
            adj_tiled = tf.reshape(adj_tiled4, (-1, self.num_nodes, self.num_nodes))
            
            # 应用GCN层（逐时间步）
            try:
                x = self.gcn_layer1(gcn_input, adj_matrix=adj_tiled, training=is_training)
                x = self.batch_norm1(x, training=is_training)
                x = self.gcn_layer2(x, adj_matrix=adj_tiled, training=is_training)
                x = self.batch_norm2(x, training=is_training)
            except Exception as e:
                logger.error(f"Error in GCN layers: {str(e)}")
                raise
            
            # 还原时间维: [batch, timesteps, num_nodes*gcn_units]
            node_features = x.shape[-1]
            node_features = int(node_features) if node_features is not None else self.gcn_units
            # 先还原为 [batch, timesteps, num_nodes, features]
            x = tf.reshape(x, (-1, self.timesteps, self.num_nodes, node_features))
            # 再合并节点与特征维 -> [batch, timesteps, num_nodes*features]
            x = tf.reshape(x, (-1, self.timesteps, self.num_nodes * node_features))
            
            # 应用GRU提取时间特征
            try:
                gru_output = self.gru_layer1(x, training=is_training)
                gru_output = self.gru_layer2(gru_output, training=is_training)
            except Exception as e:
                logger.error(f"Error in GRU layers: {str(e)}")
                raise
            
            # 处理外部特征
            try:
                # 计算平均值
                external_mean = tf.reduce_mean(external_features, axis=1)  # [batch_size, features]
                
                # 处理外部特征
                external_processed = self.external_dense1(external_mean)
                external_processed = self.dropout(external_processed, training=is_training)
                external_processed = self.external_dense2(external_processed)
                external_processed = self.batch_norm3(external_processed, training=is_training)
            except Exception as e:
                logger.error(f"Error in external features processing: {str(e)}")
                raise
            
            # 融合时空特征和外部特征
            try:
                fusion_input = tf.keras.backend.concatenate([gru_output, external_processed], axis=-1)
                
                # 应用融合层
                fusion_output = self.fusion_dense1(fusion_input)
                fusion_output = self.dropout(fusion_output, training=is_training)
                fusion_output = self.fusion_dense2(fusion_output)
                
                # 输出预测结果
                predictions = self.output_layer(fusion_output)
            except Exception as e:
                logger.error(f"Error in fusion and output layers: {str(e)}")
                raise
            
            logger.info(f"Predictions shape: {predictions.shape}")
            return predictions
        except Exception as e:
            logger.error(f"Error in SpatioTemporalGCNGRU.call(): {str(e)}")
            logger.error(f"Inputs type: {type(inputs)}")
            logger.error(f"Training type: {type(training)}")
            logger.error(f"Training value: {training}")
            # 添加更多调试信息
            if hasattr(inputs, 'shape'):
                logger.error(f"Inputs shape: {inputs.shape}")
            # 重新抛出异常以便上层捕获
            raise
    
    def build_graph(self):
        """构建模型图并显示摘要"""
        # 创建所有输入层
        speed_sequences = tf.keras.Input(shape=(self.timesteps, self.num_nodes))
        adj_matrix = tf.keras.Input(shape=(self.num_nodes, self.num_nodes))
        external_features = tf.keras.Input(shape=(self.num_nodes, 10))  # 假设外部特征维度为10
        
        # 构建模型
        model = tf.keras.Model(
            inputs=[speed_sequences, adj_matrix, external_features],
            outputs=self.call([speed_sequences, adj_matrix, external_features])
        )
        
        return model
    
    def get_config(self):
        """获取模型配置，用于模型保存和加载"""
        config = {
            'num_nodes': self.num_nodes,
            'timesteps': self.timesteps,
            'hidden_size': self.hidden_size,
            'gcn_units': self.gcn_units,
        }
        return config
    
    @classmethod
    def from_config(cls, config):
        """从配置创建模型"""
        return cls(**config)

# 创建模型函数
def create_model(num_nodes, timesteps, hidden_size=64, gcn_units=32, dropout_rate=0.3, l2_reg=0.001):
    """创建交通速度预测模型"""
    logger.info(f'创建模型: 节点数={num_nodes}, 时间步={timesteps}, 隐藏层大小={hidden_size}')
    
    # 创建模型实例
    model = SpatioTemporalGCNGRU(
        num_nodes=num_nodes,
        timesteps=timesteps,
        hidden_size=hidden_size,
        gcn_units=gcn_units,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg
    )
    
    # 构建完整模型
    full_model = model.build_graph()
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    
    full_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    # 打印模型摘要
    logger.info('模型结构摘要:')
    full_model.summary(print_fn=logger.info)
    
    return full_model

# 自定义训练循环（用于处理复杂输入）
def custom_train_loop(model, train_dataset, val_dataset, epochs, batch_size, callbacks=None):
    """自定义训练循环"""
    # 从tf.keras.callbacks中提取需要的回调
    history = {}
    
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch+1}/{epochs}')
        
        # 训练循环
        train_loss = 0.0
        train_rmse = 0.0
        train_mae = 0.0
        train_steps = 0
        
        # 遍历训练数据集
        for batch_data in train_dataset:
            # 获取输入和标签
            inputs, labels = batch_data
            speed_sequences = inputs['traffic_input']
            adj_matrix = inputs['adjacency_matrix']
            external_features = inputs['external_features']
            
            # 执行训练步骤
            with tf.GradientTape() as tape:
                # 前向传播
                predictions = model([speed_sequences, adj_matrix, external_features], training=True)
                loss_value = tf.keras.losses.mean_squared_error(labels, predictions)
                loss_value = tf.reduce_mean(loss_value)
                
                # 计算其他指标
                rmse_value = tf.keras.metrics.mean_squared_error(labels, predictions)**0.5
                mae_value = tf.keras.metrics.mean_absolute_error(labels, predictions)
                
                rmse_value = tf.reduce_mean(rmse_value)
                mae_value = tf.reduce_mean(mae_value)
            
            # 反向传播和优化
            grads = tape.gradient(loss_value, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # 更新累计指标
            train_loss += loss_value.numpy()
            train_rmse += rmse_value.numpy()
            train_mae += mae_value.numpy()
            train_steps += 1
        
        # 验证循环
        val_loss = 0.0
        val_rmse = 0.0
        val_mae = 0.0
        val_steps = 0
        
        for batch_data in val_dataset:
            # 获取输入和标签
            inputs, labels = batch_data
            speed_sequences = inputs['traffic_input']
            adj_matrix = inputs['adjacency_matrix']
            external_features = inputs['external_features']
            
            # 执行验证步骤
            predictions = model([speed_sequences, adj_matrix, external_features], training=False)
            loss_value = tf.keras.losses.mean_squared_error(labels, predictions)
            loss_value = tf.reduce_mean(loss_value)
            
            # 计算其他指标
            rmse_value = tf.keras.metrics.mean_squared_error(labels, predictions)**0.5
            mae_value = tf.keras.metrics.mean_absolute_error(labels, predictions)
            
            rmse_value = tf.reduce_mean(rmse_value)
            mae_value = tf.reduce_mean(mae_value)
            
            # 更新累计指标
            val_loss += loss_value.numpy()
            val_rmse += rmse_value.numpy()
            val_mae += mae_value.numpy()
            val_steps += 1
        
        # 记录历史
        epoch_history = {
            'loss': train_loss / train_steps if train_steps > 0 else 0,
            'val_loss': val_loss / val_steps if val_steps > 0 else 0,
            'root_mean_squared_error': train_rmse / train_steps if train_steps > 0 else 0,
            'val_root_mean_squared_error': val_rmse / val_steps if val_steps > 0 else 0,
            'mean_absolute_error': train_mae / train_steps if train_steps > 0 else 0,
            'val_mean_absolute_error': val_mae / val_steps if val_steps > 0 else 0
        }
        
        # 更新历史记录
        for key, value in epoch_history.items():
            if key not in history:
                history[key] = []
            history[key].append(value)
        
        # 打印 epoch 结果
        logger.info(f"Train Loss: {epoch_history['loss']:.4f}, Train RMSE: {epoch_history['root_mean_squared_error']:.4f}")
        logger.info(f"Val Loss: {epoch_history['val_loss']:.4f}, Val RMSE: {epoch_history['val_root_mean_squared_error']:.4f}")
    
    return history

# 测试模型
def test_model():
    """测试模型结构"""
    # 参数设置
    num_nodes = 50
    timesteps = 12
    
    # 创建随机邻接矩阵
    adj_matrix = np.random.rand(num_nodes, num_nodes)
    adj_matrix = adj_matrix / np.sum(adj_matrix, axis=1, keepdims=True)  # 归一化
    
    # 创建模型
    model = create_model(num_nodes, timesteps)
    
    # 创建测试数据
    batch_size = 32
    speed_sequences = np.random.rand(batch_size, timesteps, num_nodes)
    external_features = np.random.rand(batch_size, num_nodes, 10)
    
    # 测试前向传播
    predictions = model.predict([speed_sequences, np.expand_dims(adj_matrix, axis=0), external_features])
    print(f'预测结果形状: {predictions.shape}')

if __name__ == '__main__':
    test_model()