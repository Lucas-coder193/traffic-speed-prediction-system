#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模糊推理机制模块
处理外部属性对交通速度的影响
"""

import numpy as np
import logging
import os

# 配置日志
logger = logging.getLogger('TrafficPrediction')

class FuzzyInferenceSystem:
    """模糊推理系统类"""
    
    def __init__(self):
        """初始化模糊推理系统"""
        logger.info('初始化模糊推理系统...')
        
        # 定义模糊集和隶属度函数参数
        self.initialize_fuzzy_sets()
        
        # 定义模糊规则库
        self.initialize_rules()
        
    def initialize_fuzzy_sets(self):
        """初始化模糊集"""
        # 天气模糊集参数
        self.weather_sets = {
            'sunny': {'type': 'gaussian', 'mean': 0, 'sigma': 0.5},  # 晴天
            'cloudy': {'type': 'gaussian', 'mean': 1, 'sigma': 0.5},  # 多云
            'rainy': {'type': 'gaussian', 'mean': 2, 'sigma': 0.5},  # 雨天
            'foggy': {'type': 'gaussian', 'mean': 3, 'sigma': 0.5}   # 雾天
        }
        
        # 时间段模糊集参数
        self.time_sets = {
            'early_morning': {'type': 'trapezoidal', 'a': 0, 'b': 0, 'c': 5, 'd': 6},  # 凌晨
            'morning_rush': {'type': 'trapezoidal', 'a': 6, 'b': 7, 'c': 9, 'd': 10},  # 早高峰
            'noon': {'type': 'trapezoidal', 'a': 10, 'b': 11, 'c': 14, 'd': 15},  # 中午
            'afternoon': {'type': 'trapezoidal', 'a': 15, 'b': 17, 'c': 18, 'd': 19},  # 下午
            'evening_rush': {'type': 'trapezoidal', 'a': 19, 'b': 20, 'c': 22, 'd': 23},  # 晚高峰
            'night': {'type': 'trapezoidal', 'a': 23, 'b': 23, 'c': 24, 'd': 24}  # 夜晚
        }
        
        # 节假日模糊集参数
        self.holiday_sets = {
            'weekday': {'type': 'triangular', 'a': 0, 'b': 0, 'c': 0.5},  # 工作日
            'weekend': {'type': 'triangular', 'a': 0.5, 'b': 1, 'c': 1}   # 周末/节假日
        }
        
        # 交通流量模糊集参数
        self.flow_sets = {
            'low': {'type': 'trapezoidal', 'a': 0, 'b': 0, 'c': 300, 'd': 500},  # 低流量
            'medium': {'type': 'trapezoidal', 'a': 400, 'b': 600, 'c': 800, 'd': 1000},  # 中流量
            'high': {'type': 'trapezoidal', 'a': 900, 'b': 1100, 'c': 1300, 'd': 1500},  # 高流量
            'very_high': {'type': 'trapezoidal', 'a': 1400, 'b': 1600, 'c': 2000, 'd': 2000}  # 极高流量
        }
        
        # 拥堵指数模糊集参数
        self.congestion_sets = {
            'light': {'type': 'trapezoidal', 'a': 0, 'b': 0, 'c': 2, 'd': 3},  # 轻度拥堵
            'moderate': {'type': 'trapezoidal', 'a': 2.5, 'b': 3.5, 'c': 6.5, 'd': 7.5},  # 中度拥堵
            'heavy': {'type': 'trapezoidal', 'a': 7, 'b': 8, 'c': 10, 'd': 10}  # 严重拥堵
        }
        
        # 速度影响模糊集参数（输出）
        self.speed_impact_sets = {
            'positive_large': {'type': 'triangular', 'a': 0.6, 'b': 0.8, 'c': 1.0},
            'positive_small': {'type': 'triangular', 'a': 0.3, 'b': 0.5, 'c': 0.7},
            'neutral': {'type': 'triangular', 'a': -0.2, 'b': 0.0, 'c': 0.2},
            'negative_small': {'type': 'triangular', 'a': -0.7, 'b': -0.5, 'c': -0.3},
            'negative_large': {'type': 'triangular', 'a': -1.0, 'b': -0.8, 'c': -0.6}
        }
    
    def initialize_rules(self):
        """初始化模糊规则库"""
        # 模糊规则库，格式：(天气, 时间段, 节假日, 流量, 拥堵指数) -> 速度影响
        self.rules = [
            # 晴天规则
            ('sunny', 'early_morning', 'weekday', 'low', 'light', 'positive_large'),
            ('sunny', 'morning_rush', 'weekday', 'high', 'heavy', 'negative_large'),
            ('sunny', 'noon', 'weekday', 'medium', 'moderate', 'positive_small'),
            ('sunny', 'afternoon', 'weekday', 'medium', 'moderate', 'neutral'),
            ('sunny', 'evening_rush', 'weekday', 'high', 'heavy', 'negative_large'),
            ('sunny', 'night', 'weekday', 'low', 'light', 'positive_large'),
            # 晴天周末规则
            ('sunny', 'morning_rush', 'weekend', 'medium', 'moderate', 'positive_small'),
            ('sunny', 'evening_rush', 'weekend', 'high', 'moderate', 'negative_small'),
            
            # 多云规则
            ('cloudy', 'early_morning', 'weekday', 'low', 'light', 'positive_large'),
            ('cloudy', 'morning_rush', 'weekday', 'high', 'heavy', 'negative_large'),
            ('cloudy', 'noon', 'weekday', 'medium', 'moderate', 'positive_small'),
            ('cloudy', 'afternoon', 'weekday', 'medium', 'moderate', 'neutral'),
            ('cloudy', 'evening_rush', 'weekday', 'high', 'heavy', 'negative_large'),
            ('cloudy', 'night', 'weekday', 'low', 'light', 'positive_large'),
            
            # 雨天规则
            ('rainy', 'early_morning', 'weekday', 'low', 'light', 'positive_small'),
            ('rainy', 'morning_rush', 'weekday', 'high', 'heavy', 'negative_large'),
            ('rainy', 'noon', 'weekday', 'medium', 'moderate', 'neutral'),
            ('rainy', 'afternoon', 'weekday', 'medium', 'heavy', 'negative_small'),
            ('rainy', 'evening_rush', 'weekday', 'high', 'heavy', 'negative_large'),
            ('rainy', 'night', 'weekday', 'low', 'light', 'positive_small'),
            
            # 雾天规则
            ('foggy', 'early_morning', 'weekday', 'low', 'light', 'neutral'),
            ('foggy', 'morning_rush', 'weekday', 'high', 'heavy', 'negative_large'),
            ('foggy', 'noon', 'weekday', 'medium', 'moderate', 'negative_small'),
            ('foggy', 'afternoon', 'weekday', 'medium', 'heavy', 'negative_small'),
            ('foggy', 'evening_rush', 'weekday', 'high', 'heavy', 'negative_large'),
            ('foggy', 'night', 'weekday', 'low', 'light', 'neutral'),
        ]
    
    def gaussian_membership(self, x, mean, sigma):
        """高斯隶属度函数"""
        return np.exp(-0.5 * np.square((x - mean) / sigma))
    
    def triangular_membership(self, x, a, b, c):
        """三角形隶属度函数"""
        return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))
    
    def trapezoidal_membership(self, x, a, b, c, d):
        """梯形隶属度函数"""
        return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)))
    
    def get_membership(self, x, fuzzy_set):
        """获取变量在模糊集中的隶属度"""
        if fuzzy_set['type'] == 'gaussian':
            return self.gaussian_membership(x, fuzzy_set['mean'], fuzzy_set['sigma'])
        elif fuzzy_set['type'] == 'triangular':
            return self.triangular_membership(x, fuzzy_set['a'], fuzzy_set['b'], fuzzy_set['c'])
        elif fuzzy_set['type'] == 'trapezoidal':
            return self.trapezoidal_membership(x, fuzzy_set['a'], fuzzy_set['b'], fuzzy_set['c'], fuzzy_set['d'])
        else:
            raise ValueError(f"未知的隶属度函数类型: {fuzzy_set['type']}")
    
    def fuzzify(self, external_features):
        """将外部特征模糊化"""
        # 外部特征格式: [hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, is_weekend, is_holiday, weather_code, temp_norm, flow, congestion_index]
        
        # 计算实际小时值（从sin和cos值还原）
        hour_sin, hour_cos = external_features[0], external_features[1]
        hour = (np.arctan2(hour_sin, hour_cos) * 12 / np.pi) % 24
        
        # 获取其他特征
        is_weekend = external_features[4]
        is_holiday = external_features[5]
        weather_code = external_features[6]
        flow = external_features[8] * 2000  # 反归一化流量值
        congestion_index = external_features[9] * 10  # 反归一化拥堵指数
        
        # 判断是否为节假日
        holiday_factor = max(is_weekend, is_holiday)
        
        # 计算各个模糊集的隶属度
        memberships = {
            'weather': {},
            'time': {},
            'holiday': {},
            'flow': {},
            'congestion': {}
        }
        
        # 天气隶属度
        for weather, fuzzy_set in self.weather_sets.items():
            memberships['weather'][weather] = self.get_membership(weather_code, fuzzy_set)
        
        # 时间段隶属度
        for time, fuzzy_set in self.time_sets.items():
            memberships['time'][time] = self.get_membership(hour, fuzzy_set)
        
        # 节假日隶属度
        for holiday, fuzzy_set in self.holiday_sets.items():
            memberships['holiday'][holiday] = self.get_membership(holiday_factor, fuzzy_set)
        
        # 流量隶属度
        for flow_level, fuzzy_set in self.flow_sets.items():
            memberships['flow'][flow_level] = self.get_membership(flow, fuzzy_set)
        
        # 拥堵指数隶属度
        for congestion_level, fuzzy_set in self.congestion_sets.items():
            memberships['congestion'][congestion_level] = self.get_membership(congestion_index, fuzzy_set)
        
        return memberships
    
    def inference(self, memberships):
        """模糊推理"""
        rule_firings = []
        
        for rule in self.rules:
            weather, time, holiday, flow, congestion, speed_impact = rule
            
            # 计算规则触发强度（取最小值）
            firing_strength = min(
                memberships['weather'][weather],
                memberships['time'][time],
                memberships['holiday'][holiday],
                memberships['flow'][flow],
                memberships['congestion'][congestion]
            )
            
            if firing_strength > 0:
                rule_firings.append((speed_impact, firing_strength))
        
        return rule_firings
    
    def defuzzify(self, rule_firings):
        """解模糊化"""
        if not rule_firings:
            return 0.0  # 默认无影响
        
        # 重心法解模糊化
        numerator = 0.0
        denominator = 0.0
        
        # 采样点
        sample_points = np.linspace(-1.0, 1.0, 100)
        
        for speed_impact, firing_strength in rule_firings:
            fuzzy_set = self.speed_impact_sets[speed_impact]
            
            for x in sample_points:
                membership = self.get_membership(x, fuzzy_set)
                # 取触发强度和隶属度的最小值
                weight = min(firing_strength, membership)
                numerator += x * weight
                denominator += weight
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def process_external_features(self, external_features):
        """处理外部特征并返回影响值"""
        # 外部特征格式: [batch_size, num_nodes, feature_dim]
        batch_size, num_nodes, _ = external_features.shape
        
        # 初始化影响值数组
        impacts = np.zeros((batch_size, num_nodes))
        
        # 对每个样本的每个节点进行模糊推理
        for i in range(batch_size):
            for j in range(num_nodes):
                # 获取单个节点的外部特征
                node_features = external_features[i, j]
                
                # 模糊化
                memberships = self.fuzzify(node_features)
                
                # 推理
                rule_firings = self.inference(memberships)
                
                # 解模糊化
                impact = self.defuzzify(rule_firings)
                
                impacts[i, j] = impact
        
        return impacts
    
    def save_model(self, filepath):
        """保存模糊推理系统模型"""
        # 由于模糊系统主要是规则和参数，这里简化为保存规则
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'rules': self.rules,
                'weather_sets': self.weather_sets,
                'time_sets': self.time_sets,
                'holiday_sets': self.holiday_sets,
                'flow_sets': self.flow_sets,
                'congestion_sets': self.congestion_sets,
                'speed_impact_sets': self.speed_impact_sets
            }, f)
        logger.info(f'模糊推理系统已保存至: {filepath}')
    
    @staticmethod
    def load_model(filepath):
        """加载模糊推理系统模型"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        fis = FuzzyInferenceSystem()
        fis.rules = data['rules']
        fis.weather_sets = data['weather_sets']
        fis.time_sets = data['time_sets']
        fis.holiday_sets = data['holiday_sets']
        fis.flow_sets = data['flow_sets']
        fis.congestion_sets = data['congestion_sets']
        fis.speed_impact_sets = data['speed_impact_sets']
        
        logger.info(f'模糊推理系统已从: {filepath} 加载')
        return fis

# 测试模糊推理系统
def test_fuzzy_system():
    """测试模糊推理系统"""
    # 初始化模糊推理系统
    fis = FuzzyInferenceSystem()
    
    # 创建测试外部特征
    # 格式: [hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, is_weekend, is_holiday, weather_code, temp_norm, flow, congestion_index]
    # 测试案例1: 晴天工作日早高峰
    test_feature1 = np.array([
        np.sin(2 * np.pi * 8 / 24),  # 早上8点的sin值
        np.cos(2 * np.pi * 8 / 24),  # 早上8点的cos值
        np.sin(2 * np.pi * 1 / 7),   # 周一的sin值
        np.cos(2 * np.pi * 1 / 7),   # 周一的cos值
        0,  # 非周末
        0,  # 非节假日
        0,  # 晴天
        0.5,  # 温度适中
        0.8,  # 流量高
        0.9   # 拥堵严重
    ])
    
    # 测试案例2: 雨天周末晚上
    test_feature2 = np.array([
        np.sin(2 * np.pi * 20 / 24),  # 晚上8点的sin值
        np.cos(2 * np.pi * 20 / 24),  # 晚上8点的cos值
        np.sin(2 * np.pi * 6 / 7),    # 周日的sin值
        np.cos(2 * np.pi * 6 / 7),    # 周日的cos值
        1,  # 周末
        0,  # 非节假日
        2,  # 雨天
        0.3,  # 温度较低
        0.6,  # 流量中等
        0.7   # 中度拥堵
    ])
    
    # 构建批量测试数据
    test_features = np.array([[test_feature1, test_feature2]])
    
    # 处理外部特征
    impacts = fis.process_external_features(test_features)
    
    print("测试结果:")
    print(f"案例1影响值: {impacts[0, 0]}")
    print(f"案例2影响值: {impacts[0, 1]}")
    
    # 保存模型
    fis.save_model('fuzzy_inference_model.pkl')

if __name__ == '__main__':
    test_fuzzy_system()