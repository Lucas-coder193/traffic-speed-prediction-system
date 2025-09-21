#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交通车辆速度预测应用系统主程序
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import tensorflow as tf

# 配置日志
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/traffic_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('TrafficPrediction')

# 启用GPU及显存按需增长
def setup_tf_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"已检测到GPU并启用显存按需增长: {len(gpus)} 个GPU")
        else:
            print('未检测到GPU，将使用CPU运行')
    except Exception as e:
        print(f'GPU配置失败，回退CPU: {e}')

# 确保日志目录存在
def ensure_directories():
    directories = ['data', 'logs', 'models', 'results', 'src']
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f'创建目录: {dir_name}')

# 主函数
def main(args):
    global logger
    ensure_directories()
    logger = setup_logging()
    setup_tf_gpu()
    
    logger.info('=== 交通车辆速度预测应用系统 ===')
    logger.info(f'当前模式: {args.mode}')
    
    # 根据模式执行不同的任务
    if args.mode == 'preprocess':
        logger.info('开始数据预处理...')
        # 导入数据预处理模块
        from src.data_preprocessing import preprocess_data
        preprocess_data(args.dataset, args.config_file)
    elif args.mode == 'train':
        logger.info('开始模型训练...')
        # 导入模型训练模块
        from src.model_training import train_model
        train_model(args)
    elif args.mode == 'predict':
        logger.info('开始预测...')
        # 导入预测模块
        from src.predict import predict_traffic_speed
        predict_traffic_speed(args)
    elif args.mode == 'evaluate':
        logger.info('开始模型评估...')
        # 导入评估模块
        from src.evaluate import run_evaluation
        run_evaluation(args)
    elif args.mode == 'web':
        logger.info('启动Web服务...')
        # 导入Web服务模块
        from src.web import run_web_server
        run_web_server(args)
    else:
        logger.error(f'未知模式: {args.mode}')
        sys.exit(1)

if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='交通车辆速度预测应用系统')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'predict', 'evaluate', 'web'],
                        default='train', help='运行模式')
    parser.add_argument('--dataset', type=str, default='SZBZ', choices=['SZBZ', 'Los-loop', 'real'],
                        help='数据集选择')
    parser.add_argument('--model', type=str, default='gcn_gru_fuzzy',
                        help='模型类型')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='GRU隐藏层大小')
    parser.add_argument('--timesteps', type=int, default=12,
                        help='历史时间步')
    parser.add_argument('--port', type=int, default=5000,
                        help='Web服务端口')
    parser.add_argument('--config_file', type=str, default=None, help='配置文件路径，用于指定真实数据集的加载参数')
    
    args = parser.parse_args()
    main(args)