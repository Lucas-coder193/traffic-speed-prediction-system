#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练进度可视化工具
提供实时训练进度显示和结果可视化功能
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from tqdm import tqdm
import json
import logging
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

class TrainingVisualizer:
    """训练进度可视化类"""
    def __init__(self, log_dir=None, model_name=None, dataset_name=None):
        """初始化可视化工具"""
        self.log_dir = log_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        self.model_name = model_name or 'unknown_model'
        self.dataset_name = dataset_name or 'unknown_dataset'
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 训练历史
        self.history = {
            'loss': [],
            'val_loss': [],
            'root_mean_squared_error': [],
            'val_root_mean_squared_error': [],
            'mean_absolute_error': [],
            'val_mean_absolute_error': [],
            'loss_change_rate': []  # 添加损失变化率
        }
        
        # 进度条
        self.progress_bar = None
        
        # 可视化配置
        self.fig = None
        self.axes = None
        
        # 测试结果
        self.test_results = {}
        
        # 时间记录
        self.start_time = None
        self.end_time = None
        
        # 上一轮损失，用于计算变化率
        self.previous_loss = float('inf')
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(f'{self.model_name}_{self.dataset_name}_visualizer')
        logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not logger.handlers:
            # 文件处理器
            log_file = os.path.join(self.log_dir, f"visualizer_{self.model_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def start_training(self, epochs):
        """开始训练，初始化进度条和计时器"""
        self.start_time = time.time()
        self.progress_bar = tqdm(total=epochs, desc=f'训练 {self.model_name} 在 {self.dataset_name} 数据集上')
        self.logger.info(f'开始训练模型: {self.model_name}，数据集: {self.dataset_name}，总轮次: {epochs}')
    
    def update_epoch(self, epoch, epoch_history):
        """更新一轮训练的进度和指标"""
        # 计算损失变化率
        current_loss = epoch_history.get('loss', 0)
        if self.previous_loss != float('inf') and self.previous_loss > 0:
            loss_change_rate = (self.previous_loss - current_loss) / self.previous_loss
        else:
            loss_change_rate = 0
        
        # 添加损失变化率到epoch_history
        epoch_history['loss_change_rate'] = loss_change_rate
        
        # 更新历史记录
        for key, value in epoch_history.items():
            if key in self.history:
                self.history[key].append(value)
        
        # 更新上一轮损失
        self.previous_loss = current_loss
        
        # 更新进度条
        if self.progress_bar:
            # 提取本轮关键指标
            loss = epoch_history.get('loss', 0)
            val_loss = epoch_history.get('val_loss', 0)
            rmse = epoch_history.get('root_mean_squared_error', 0)
            val_rmse = epoch_history.get('val_root_mean_squared_error', 0)
            
            # 更新进度条描述
            self.progress_bar.set_postfix(
                loss=f'{loss:.6f}', 
                val_loss=f'{val_loss:.6f}',
                loss_change=f'{loss_change_rate:.2%}',
                rmse=f'{rmse:.4f}',
                val_rmse=f'{val_rmse:.4f}'
            )
            self.progress_bar.update(1)
        
        # 记录日志
        self.logger.info(f"Epoch {epoch+1} 完成: 训练损失={epoch_history.get('loss', 0):.6f}, 验证损失={epoch_history.get('val_loss', 0):.6f}, 损失变化率={loss_change_rate:.2%}")
    
    def end_training(self):
        """结束训练，保存历史记录"""
        self.end_time = time.time()
        training_time = self.end_time - self.start_time
        
        if self.progress_bar:
            self.progress_bar.close()
        
        # 记录训练总时间
        hours, remainder = divmod(int(training_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.logger.info(f'训练完成！总耗时: {hours}小时 {minutes}分钟 {seconds}秒')
        
        # 保存历史记录
        self._save_history()
    
    def _save_history(self):
        """保存训练历史到文件"""
        history_file = os.path.join(self.log_dir, f"training_history_{self.model_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            self.logger.info(f'训练历史已保存到: {history_file}')
        except Exception as e:
            self.logger.error(f'保存训练历史失败: {str(e)}')
    
    def set_test_results(self, results):
        """设置测试结果"""
        self.test_results = results
        self.logger.info(f'测试结果: {results}')
    
    def plot_training_progress(self, save_fig=True):
        """绘制训练进度图表，包含损失变化率"""
        try:
            # 创建图表 - 扩展为3行2列以显示更多信息
            self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 15))
            
            # 绘制损失曲线
            self.axes[0, 0].plot(self.history['loss'], label='训练损失')
            self.axes[0, 0].plot(self.history['val_loss'], label='验证损失')
            self.axes[0, 0].set_title('损失随训练轮次变化')
            self.axes[0, 0].set_xlabel('轮次')
            self.axes[0, 0].set_ylabel('损失值')
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True)
            
            # 绘制损失变化率曲线
            self.axes[0, 1].plot(self.history['loss_change_rate'], 'g-', label='训练损失变化率')
            self.axes[0, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)  # 添加零线参考
            self.axes[0, 1].set_title('损失变化率随训练轮次变化')
            self.axes[0, 1].set_xlabel('轮次')
            self.axes[0, 1].set_ylabel('变化率 (%)')
            # 设置y轴为百分比格式
            self.axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True)
            
            # 绘制RMSE曲线
            self.axes[1, 0].plot(self.history['root_mean_squared_error'], label='训练RMSE')
            self.axes[1, 0].plot(self.history['val_root_mean_squared_error'], label='验证RMSE')
            self.axes[1, 0].set_title('RMSE随训练轮次变化')
            self.axes[1, 0].set_xlabel('轮次')
            self.axes[1, 0].set_ylabel('RMSE值')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True)
            
            # 绘制MAE曲线
            self.axes[1, 1].plot(self.history['mean_absolute_error'], label='训练MAE')
            self.axes[1, 1].plot(self.history['val_mean_absolute_error'], label='验证MAE')
            self.axes[1, 1].set_title('MAE随训练轮次变化')
            self.axes[1, 1].set_xlabel('轮次')
            self.axes[1, 1].set_ylabel('MAE值')
            self.axes[1, 1].legend()
            self.axes[1, 1].grid(True)
            
            # 添加模型和数据集信息
            self.axes[2, 0].axis('off')
            info_text = f"""模型: {self.model_name}\n数据集: {self.dataset_name}\n"""
            
            # 添加最终指标
            if self.history['loss']:
                info_text += f"最终训练损失: {self.history['loss'][-1]:.6f}\n"
                info_text += f"最终验证损失: {self.history['val_loss'][-1]:.6f}\n"
                info_text += f"最终训练RMSE: {self.history['root_mean_squared_error'][-1]:.4f}\n"
                info_text += f"最终验证RMSE: {self.history['val_root_mean_squared_error'][-1]:.4f}\n"
                info_text += f"最终训练MAE: {self.history['mean_absolute_error'][-1]:.4f}\n"
                info_text += f"最终验证MAE: {self.history['val_mean_absolute_error'][-1]:.4f}\n"
            
            # 添加损失趋势分析
            if len(self.history['loss']) > 10:
                # 计算最近5轮的平均损失变化
                recent_changes = self.history['loss_change_rate'][-5:]
                avg_recent_change = sum(recent_changes) / len(recent_changes)
                
                # 判断学习趋势
                if avg_recent_change > 0.02:  # 大于2%的平均下降
                    trend = "学习进展良好"
                elif avg_recent_change > 0:  # 仍在下降，但速度较慢
                    trend = "学习放缓，可能需要调整"
                else:
                    trend = "损失趋于稳定或上升"
                
                info_text += f"\n学习趋势: {trend}\n"
                info_text += f"最近5轮平均损失变化: {avg_recent_change:.2%}\n"
            
            # 添加测试结果
            if self.test_results:
                info_text += f"\n测试结果:\n"
                for key, value in self.test_results.items():
                    info_text += f"{key}: {value:.4f}\n"
            
            # 添加训练时间
            if self.start_time and self.end_time:
                training_time = self.end_time - self.start_time
                hours, remainder = divmod(int(training_time), 3600)
                minutes, seconds = divmod(remainder, 60)
                info_text += f"\n总训练时间: {hours}小时 {minutes}分钟 {seconds}秒"
            
            self.axes[2, 0].text(0.1, 0.1, info_text, fontsize=12, verticalalignment='top')
            
            # 隐藏最后一个子图
            self.axes[2, 1].axis('off')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_fig:
                fig_file = os.path.join(self.log_dir, f"training_progress_{self.model_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(fig_file, dpi=300, bbox_inches='tight')
                self.logger.info(f'训练进度图表已保存到: {fig_file}')
            
            # 显示图表
            plt.show()
            
        except Exception as e:
            self.logger.error(f'绘制训练进度图表失败: {str(e)}')
    
    def plot_prediction_comparison(self, true_values, predictions, sample_indices=None, save_fig=True):
        """绘制预测结果与真实值的对比图表"""
        try:
            # 如果未指定样本索引，随机选择5个样本
            if sample_indices is None:
                num_samples = min(5, len(true_values))
                sample_indices = np.random.choice(len(true_values), num_samples, replace=False)
            
            # 创建图表
            fig, axes = plt.subplots(len(sample_indices), 1, figsize=(15, 4 * len(sample_indices)))
            if len(sample_indices) == 1:
                axes = [axes]  # 确保axes是列表
            
            # 为每个样本绘制对比图
            for i, idx in enumerate(sample_indices):
                true_sample = true_values[idx]
                pred_sample = predictions[idx]
                
                axes[i].plot(true_sample, label='真实值', marker='o')
                axes[i].plot(pred_sample, label='预测值', marker='x')
                axes[i].set_title(f'样本 {idx} 的预测与真实值对比')
                axes[i].set_xlabel('节点')
                axes[i].set_ylabel('速度值')
                axes[i].legend()
                axes[i].grid(True)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_fig:
                fig_file = os.path.join(self.log_dir, f"prediction_comparison_{self.model_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(fig_file, dpi=300, bbox_inches='tight')
                self.logger.info(f'预测对比图表已保存到: {fig_file}')
            
            # 显示图表
            plt.show()
            
        except Exception as e:
            self.logger.error(f'绘制预测对比图表失败: {str(e)}')
    
    def plot_error_distribution(self, true_values, predictions, save_fig=True):
        """绘制预测误差分布图"""
        try:
            # 计算误差
            errors = np.abs(true_values - predictions)
            mean_errors = np.mean(errors, axis=0)  # 每个节点的平均误差
            
            # 创建图表
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 绘制每个节点的平均误差
            axes[0].bar(range(len(mean_errors)), mean_errors)
            axes[0].set_title('各节点平均绝对误差')
            axes[0].set_xlabel('节点')
            axes[0].set_ylabel('平均绝对误差')
            axes[0].grid(True, axis='y')
            
            # 绘制误差分布直方图
            axes[1].hist(errors.flatten(), bins=50, alpha=0.7)
            axes[1].set_title('预测误差分布')
            axes[1].set_xlabel('绝对误差')
            axes[1].set_ylabel('频数')
            axes[1].grid(True, axis='y')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_fig:
                fig_file = os.path.join(self.log_dir, f"error_distribution_{self.model_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(fig_file, dpi=300, bbox_inches='tight')
                self.logger.info(f'误差分布图已保存到: {fig_file}')
            
            # 显示图表
            plt.show()
            
        except Exception as e:
            self.logger.error(f'绘制误差分布图失败: {str(e)}')

# 创建一个简单的示例，展示如何使用这个可视化工具
def example_usage():
    """示例：如何使用训练可视化工具"""
    # 创建可视化器实例
    visualizer = TrainingVisualizer(
        log_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'),
        model_name='example_model',
        dataset_name='example_dataset'
    )
    
    # 模拟训练过程
    epochs = 10
    visualizer.start_training(epochs)
    
    for epoch in range(epochs):
        # 模拟每轮的训练结果
        epoch_history = {
            'loss': 1.0 * (1 - epoch/epochs),
            'val_loss': 1.2 * (1 - epoch/epochs * 0.8),
            'root_mean_squared_error': 1.0 * (1 - epoch/epochs),
            'val_root_mean_squared_error': 1.2 * (1 - epoch/epochs * 0.8),
            'mean_absolute_error': 0.8 * (1 - epoch/epochs),
            'val_mean_absolute_error': 1.0 * (1 - epoch/epochs * 0.8)
        }
        
        # 更新进度
        visualizer.update_epoch(epoch, epoch_history)
        
        # 模拟训练耗时
        time.sleep(0.5)
    
    # 结束训练
    visualizer.end_training()
    
    # 设置测试结果
    visualizer.set_test_results({
        'test_loss': 0.3,
        'test_rmse': 0.55,
        'test_mae': 0.42
    })
    
    # 绘制训练进度
    visualizer.plot_training_progress()
    
    # 模拟预测结果和真实值
    true_values = np.random.rand(100, 50) * 100  # 100个样本，每个样本50个节点
    predictions = true_values + np.random.normal(0, 5, size=true_values.shape)  # 添加噪声作为预测
    
    # 绘制预测对比
    visualizer.plot_prediction_comparison(true_values, predictions)
    
    # 绘制误差分布
    visualizer.plot_error_distribution(true_values, predictions)

if __name__ == '__main__':
    example_usage()