# -*- coding: utf-8 -*-
"""
数据配置模块
负责加载和解析数据配置文件
"""

import os
import json
import logging

# 配置日志
logger = logging.getLogger('TrafficPrediction')

class DataConfig:
    """
    数据配置类，用于管理数据集的加载参数
    """
    
    def __init__(self):
        """初始化数据配置类"""
        # 默认配置
        self.default_config = {
            'real_data_config': {
                'enabled': False,
                'data_file': '',
                'adj_matrix_file': '',
                'has_header': True,
                'delimiter': ',',
                'field_mapping': {},
                'time_format': '%Y-%m-%d %H:%M:%S',
                'weather_mapping': {
                    'sunny': 0,
                    'cloudy': 1,
                    'rainy': 2,
                    'foggy': 3,
                    '晴': 0,
                    '多云': 1,
                    '雨': 2,
                    '雾': 3
                }
            }
        }
    
    def load_config(self, config_file=None):
        """
        加载配置文件
        
        Args:
            config_file: 配置文件路径，如果为None则返回默认配置
            
        Returns:
            dict: 加载的配置
        """
        # 从默认配置开始
        config = self.default_config.copy()
        
        if config_file:
            # 检查配置文件是否存在
            if not os.path.exists(config_file):
                logger.error(f'配置文件不存在: {config_file}')
                return config
            
            try:
                # 加载配置文件
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    
                # 合并用户配置到默认配置
                self._merge_config(config, user_config)
                logger.info(f'成功加载配置文件: {config_file}')
            except json.JSONDecodeError as e:
                logger.error(f'解析配置文件时出错: {e}')
            except Exception as e:
                logger.error(f'加载配置文件时出错: {e}')
        
        return config
    
    def _merge_config(self, base_config, user_config):
        """
        合并配置
        
        Args:
            base_config: 基础配置
            user_config: 用户配置
        """
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                self._merge_config(base_config[key], value)
            else:
                # 直接替换值
                base_config[key] = value
    
    def save_config(self, config, file_path):
        """
        保存配置到文件
        
        Args:
            config: 要保存的配置
            file_path: 保存路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 保存配置文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
            logger.info(f'配置已保存至: {file_path}')
        except Exception as e:
            logger.error(f'保存配置时出错: {e}')
            raise

def create_example_config(output_path='example_config.json'):
    """
    创建示例配置文件
    
    Args:
        output_path: 输出路径
    """
    config = DataConfig()
    example_config = {
        'real_data_config': {
            'enabled': True,
            'data_file': 'data/traffic_data.csv',
            'adj_matrix_file': 'data/adj_matrix.npy',
            'has_header': True,
            'delimiter': ',',
            'field_mapping': {
                'timestamp': 'time',
                'node_id': 'location_id',
                'speed': 'vehicle_speed'
            },
            'time_format': '%Y-%m-%d %H:%M:%S',
            'weather_mapping': {
                'sunny': 0,
                'cloudy': 1,
                'rainy': 2,
                'foggy': 3
            }
        }
    }
    
    config.save_config(example_config, output_path)
    return example_config

if __name__ == '__main__':
    # 创建示例配置文件
    create_example_config()