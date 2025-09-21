#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据配置模块
用于配置真实数据集的加载和处理
"""

import os
import logging
from typing import Dict, Any, Optional

# 配置日志
logger = logging.getLogger('TrafficPrediction')

class DataConfig:
    """数据集配置类"""
    
    def __init__(self):
        # 默认配置
        self.config = {
            'real_data_config': {
                'enabled': False,  # 是否启用真实数据集
                'data_file': '',   # 真实数据集文件路径
                'adj_matrix_file': '',  # 真实邻接矩阵文件路径
                'has_header': True,  # 数据文件是否有表头
                'delimiter': ',',    # 数据文件分隔符
                
                # 字段映射：将用户数据集的字段名映射到系统期望的字段名
                'field_mapping': {
                    'timestamp': 'timestamp',  # 时间戳
                    'node_id': 'node_id',      # 节点ID
                    'speed': 'speed',          # 速度
                    'flow': 'flow',            # 流量
                    'congestion_index': 'congestion_index',  # 拥堵指数
                    'hour': 'hour',            # 小时
                    'day_of_week': 'day_of_week',  # 星期几
                    'is_weekend': 'is_weekend',    # 是否周末
                    'is_holiday': 'is_holiday',    # 是否节假日
                    'weather': 'weather',      # 天气状况
                    'temperature': 'temperature'  # 温度
                },
                
                # 时间格式（如果timestamp字段不是datetime类型）
                'time_format': '%Y-%m-%d %H:%M:%S',
                
                # 天气编码映射
                'weather_mapping': {
                    'sunny': 0,
                    'cloudy': 1,
                    'rainy': 2,
                    'foggy': 3
                }
            }
        }
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_file and os.path.exists(config_file):
            try:
                import json
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 合并用户配置和默认配置
                self.config.update(user_config)
                logger.info(f'已加载配置文件: {config_file}')
            except Exception as e:
                logger.error(f'加载配置文件时出错: {e}')
                logger.info('使用默认配置')
        return self.config
    
    def save_config(self, config_file: str) -> None:
        """保存配置到文件"""
        try:
            import json
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            logger.info(f'配置已保存至: {config_file}')
        except Exception as e:
            logger.error(f'保存配置文件时出错: {e}')

# 创建示例配置JSON文件
def create_example_config():
    """创建示例配置JSON文件"""
    example_config = {
        "real_data_config": {
            "enabled": True,
            "data_file": "data/real_traffic_data.csv",
            "adj_matrix_file": "data/real_adj_matrix.npy",
            "has_header": True,
            "delimiter": ",",
            "field_mapping": {
                "timestamp": "timestamp",
                "node_id": "node_id",
                "speed": "speed",
                "flow": "flow",
                "congestion_index": "congestion",
                "hour": "hour",
                "day_of_week": "day",
                "is_weekend": "weekend",
                "is_holiday": "holiday",
                "weather": "weather",
                "temperature": "temp"
            },
            "time_format": "%Y-%m-%d %H:%M:%S",
            "weather_mapping": {
                "晴": 0,
                "多云": 1,
                "雨": 2,
                "雾": 3
            }
        }
    }
    
    # 保存示例配置文件
    try:
        import json
        with open('data_config_example.json', 'w', encoding='utf-8') as f:
            json.dump(example_config, f, indent=4, ensure_ascii=False)
        print("示例配置文件已创建: data_config_example.json")
    except Exception as e:
        print(f"创建示例配置文件时出错: {e}")

if __name__ == '__main__':
    # 创建示例配置文件
    create_example_config()