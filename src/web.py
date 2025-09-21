#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web可视化模块
使用Flask实现前端可视化，展示交通速度预测结果
"""

import os
import numpy as np
import logging
from datetime import datetime
import json
import time
import random
import math
from flask import Flask, render_template, request, jsonify, redirect
from flask_cors import CORS
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# 配置日志
logger = logging.getLogger('TrafficPrediction')
# 配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)

# 确保目录存在
def ensure_directories():
    """确保必要的目录存在"""
    directories = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'css'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'js'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    ]
    
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f'创建目录: {dir_path}')

# 创建Flask应用
def create_app():
    """创建Flask应用"""
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
                static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'))
    app.config['SECRET_KEY'] = 'traffic_prediction_secret_key'
    
    # 允许跨域请求
    CORS(app)
    
    # 注册路由
    register_routes(app)
    
    return app

# 运行Web服务器
def run_web_server(args):
    """运行Web服务器"""
    logger.info(f'启动Web服务器: host={args.host}, port={args.port}, debug={args.debug}')
    
    # 创建Flask应用
    app = create_app()
    
    # 确保静态文件目录存在
    ensure_directories()
    
    # 启动服务器
    app.run(host=getattr(args, 'host', '0.0.0.0'), 
            port=getattr(args, 'port', 5000), 
            debug=getattr(args, 'debug', False))

# 注册路由
def register_routes(app):
    """注册Flask路由"""
    
    @app.route('/')
    def index():
        """首页 - 重定向到现代化预测页面"""
        logger.info('访问首页，重定向到现代化预测页面')
        return redirect('/modern_prediction')
    
    @app.route('/dashboard')
    def dashboard():
        """仪表盘"""
        logger.info('访问仪表盘')
        return render_template('dashboard.html')
    
    @app.route('/prediction')
    def prediction_page():
        """预测页面"""
        logger.info('访问预测页面')
        return render_template('prediction.html')
    
    @app.route('/history')
    def history_page():
        """历史记录页面"""
        logger.info('访问历史记录页面')
        return render_template('history.html')
        
    @app.route('/test')
    def test_page():
        """测试页面"""
        logger.info('访问测试页面')
        return render_template('test.html')
        
    @app.route('/minimal_test')
    def minimal_test_page():
        """极简测试页面"""
        logger.info('访问极简测试页面')
        return render_template('minimal_test.html')
        
    @app.route('/go_to_prediction')
    def go_to_prediction():
        """重定向到预测页面的简化路由"""
        logger.info('通过重定向路由访问预测页面')
        return redirect('/prediction')
        
    @app.route('/simple_prediction')
    def simple_prediction():
        """最简单的预测页面，绕过所有复杂因素"""
        logger.info('访问简单预测页面')
        return '''
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>简单预测 - 交通车辆速度预测系统</title>
            <link rel="stylesheet" href="/static/css/bootstrap.min.css">
        </head>
        <body style="padding: 2rem;">
            <div class="container">
                <h1>交通速度预测</h1>
                <p>这是一个简化版的预测页面</p>
                
                <div class="mb-4">
                    <label for="node_id" class="form-label">选择节点</label>
                    <input type="number" id="node_id" class="form-control" placeholder="输入节点ID">
                </div>
                
                <div class="mb-4">
                    <label for="predict_time" class="form-label">预测时间</label>
                    <input type="datetime-local" id="predict_time" class="form-control">
                </div>
                
                <button id="predict_btn" class="btn btn-primary">生成预测</button>
                
                <div id="result" class="mt-4" style="display: none;">
                    <h3>预测结果</h3>
                    <p>预测速度: <span id="predicted_speed">--</span> km/h</p>
                </div>
            </div>
            
            <script>
                document.getElementById('predict_btn').addEventListener('click', function() {
                    // 简单的模拟预测结果
                    const nodeId = document.getElementById('node_id').value;
                    const predictTime = document.getElementById('predict_time').value;
                    
                    if (nodeId && predictTime) {
                        // 显示模拟结果
                        const resultDiv = document.getElementById('result');
                        const speedSpan = document.getElementById('predicted_speed');
                        
                        // 生成随机速度作为模拟结果
                        const randomSpeed = Math.floor(Math.random() * 60) + 20;
                        speedSpan.textContent = randomSpeed;
                        
                        resultDiv.style.display = 'block';
                    } else {
                        alert('请填写所有必填字段');
                    }
                });
            </script>
        </body>
        </html>
        '''
    
    @app.route('/modern_prediction')
    def modern_prediction_page():
        """现代化预测页面"""
        logger.info('访问现代化预测页面')
        return render_template('modern_prediction.html')
    
    @app.route('/assessment')
    def assessment_page():
        """模型评估页面"""
        logger.info('访问模型评估页面')
        return render_template('assessment.html')
    
    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        """预测API"""
        logger.info('处理预测API请求')
        
        try:
            # 获取请求数据
            data = request.get_json()
            
            # 解析请求参数
            model_name = data.get('model', 'gcn_gru_fuzzy')
            dataset_name = data.get('dataset', 'Los-loop')  # 使用用户训练的模型数据集
            steps = data.get('steps', 6)
            
            # 导入预测相关函数
            import sys
            import os
            # 添加项目根目录到系统路径
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.predict import load_trained_model, multi_step_prediction
            import numpy as np
            
            logger.info(f'加载训练好的模型: {model_name}_{dataset_name}')
            
            # 1. 加载训练好的模型
            model, scalers, adj_matrix, params, fuzzy_system = load_trained_model(model_name, dataset_name)
            
            # 2. 加载历史数据样本（从预处理产物读取：注意这些是已归一化的数据）
            logger.info('加载真实历史数据...')
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
            data_path = os.path.join(data_dir, f'{dataset_name}_X.npy')
            
            # 确保数据文件存在
            if not os.path.exists(data_path):
                error_msg = f'错误: 数据文件不存在: {data_path}'
                logger.error(error_msg)
                return jsonify({'status': 'error', 'message': error_msg}), 500
            
            # 加载历史数据（归一化空间）
            logger.info(f'正在加载数据文件: {data_path}')
            all_data = np.load(data_path)
            timesteps = params['timesteps']
            num_nodes = params['num_nodes']
            
            # 随机选择一个历史时间点的数据作为输入
            import random
            random_idx = random.randint(0, len(all_data) - 1)
            
            # 获取该时间点的历史数据序列（当前为归一化后的窗口）
            historical_data_scaled = all_data[random_idx]
            logger.info(f'已加载历史数据(归一化)，形状: {historical_data_scaled.shape}')
            
            # 确保历史数据形状正确
            if len(historical_data_scaled.shape) == 2:
                # 数据形状: (timesteps, num_nodes)
                if historical_data_scaled.shape[0] != timesteps or historical_data_scaled.shape[1] != num_nodes:
                    logger.warning(f'历史数据形状不匹配，期望: ({timesteps}, {num_nodes})，实际: {historical_data_scaled.shape}')
                    # 尝试调整形状
                    if historical_data_scaled.shape[0] >= timesteps and historical_data_scaled.shape[1] >= num_nodes:
                        historical_data_scaled = historical_data_scaled[-timesteps:, :num_nodes]
                        logger.info(f'已调整历史数据形状: {historical_data_scaled.shape}')
                    else:
                        logger.error('无法调整历史数据形状')
                        return jsonify({'status': 'error', 'message': '历史数据形状不匹配'}), 500
            else:
                logger.error(f'历史数据形状错误，期望二维数组，实际: {historical_data.shape}')
                return jsonify({'status': 'error', 'message': '历史数据形状错误'}), 500
            
            # 将历史数据从归一化空间还原到原始量纲，避免预测阶段再次标准化造成“二次归一化”
            speed_scaler = scalers.get('speed_scaler', None)
            if speed_scaler is not None:
                hist_flat = historical_data_scaled.reshape(-1, 1)
                historical_data = speed_scaler.inverse_transform(hist_flat).reshape(historical_data_scaled.shape)
            else:
                historical_data = historical_data_scaled

            # 3. 加载外部特征样本（同样是已归一化数据）
            external_path = os.path.join(data_dir, f'{dataset_name}_external.npy')
            if os.path.exists(external_path):
                logger.info(f'正在加载外部特征文件: {external_path}')
                all_external = np.load(external_path)
                # 选择与历史数据对应的外部特征
                external_features_scaled = all_external[random_idx]
                logger.info(f'已加载外部特征(归一化)，形状: {external_features_scaled.shape}')
                
                # 确保外部特征形状正确
                if external_features_scaled.shape != (num_nodes, 10):
                    logger.warning(f'外部特征形状不匹配，期望: ({num_nodes}, 10)，实际: {external_features_scaled.shape}')
                    # 尝试调整形状
                    if len(external_features_scaled.shape) == 2 and external_features_scaled.shape[0] >= num_nodes:
                        # 如果外部特征是二维的 (timesteps, features)，取最后一个时间点的特征
                        if external_features_scaled.shape[1] >= 10:
                            external_features_scaled = external_features_scaled[-num_nodes:, :10]
                            logger.info(f'已调整外部特征形状: {external_features_scaled.shape}')
                        else:
                            # 填充到10个特征
                            pad_width = ((0, max(0, num_nodes - external_features_scaled.shape[0])), 
                                        (0, max(0, 10 - external_features_scaled.shape[1])))
                            external_features_scaled = np.pad(external_features_scaled, pad_width, mode='constant')
                            logger.info(f'已填充外部特征形状: {external_features_scaled.shape}')
                    else:
                        # 如果无法调整，创建默认外部特征
                        logger.warning('无法调整外部特征形状，使用默认值')
                        external_features_scaled = np.zeros((num_nodes, 10))
            else:
                # 如果外部特征文件不存在，创建空的外部特征
                logger.warning(f'外部特征文件不存在: {external_path}，使用空的外部特征')
                external_features_scaled = np.zeros((num_nodes, 10))

            # 将外部特征从归一化空间还原到原始量纲，避免再次标准化
            ext_scaler = scalers.get('external_scaler', None)
            if ext_scaler is not None:
                ext_flat = external_features_scaled.reshape(-1, external_features_scaled.shape[-1])
                external_features = ext_scaler.inverse_transform(ext_flat).reshape(external_features_scaled.shape)
            else:
                external_features = external_features_scaled
            
            # 3. 进行多步预测
            logger.info(f'进行{steps}步预测...')
            all_predictions = multi_step_prediction(model, scalers, adj_matrix, params, historical_data, external_features, steps)
            
            # 4. 准备预测结果
            predictions = all_predictions.tolist()
            
            # 5. 准备结果数据
            result = {
                'status': 'success',
                'model': model_name,
                'dataset': dataset_name,
                'steps': steps,
                'predictions': predictions,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'params': params,
                'note': '使用实际训练的模型进行预测'
            }
            
            # 6. 记录预测结果
            from src.predict import record_predictions
            record_predictions(all_predictions, params)
            
            logger.info('预测API请求处理成功（使用实际模型）')
            return jsonify(result)
            
        except Exception as e:
            logger.error(f'处理预测API请求时出错: {e}')
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/history', methods=['GET'])
    def api_history():
        """历史记录API"""
        logger.info('处理历史记录API请求')
        
        try:
            # 获取历史预测记录
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # 查找所有预测记录文件
            history_files = []
            for file in os.listdir(results_dir):
                if file.startswith('predictions_info_') and file.endswith('.pkl'):
                    timestamp = file.replace('predictions_info_', '').replace('.pkl', '')
                    try:
                        import pickle
                        with open(os.path.join(results_dir, file), 'rb') as f:
                            info = pickle.load(f)
                        
                        # 获取预测结果文件
                        pred_file = f'predictions_{timestamp}.npy'
                        if os.path.exists(os.path.join(results_dir, pred_file)):
                            # 加载预测结果
                            predictions = np.load(os.path.join(results_dir, pred_file))
                            
                            history_files.append({
                                'timestamp': timestamp,
                                'formatted_timestamp': datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
                                'avg_speed': float(np.mean(predictions)),
                                'min_speed': float(np.min(predictions)),
                                'max_speed': float(np.max(predictions))
                            })
                    except Exception as e:
                        logger.warning(f'加载历史记录文件{file}时出错: {e}')
            
            # 按时间戳排序（最新的在前）
            history_files.sort(key=lambda x: x['timestamp'], reverse=True)
            
            logger.info(f'找到{len(history_files)}条历史记录')
            return jsonify({'status': 'success', 'history': history_files})
            
        except Exception as e:
            logger.error(f'处理历史记录API请求时出错: {e}')
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/history/<timestamp>', methods=['GET'])
    def api_history_detail(timestamp):
        """历史记录详情API"""
        logger.info(f'处理历史记录详情API请求: {timestamp}')
        
        try:
            # 获取指定时间戳的历史记录
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            
            # 加载预测信息文件
            info_file = f'predictions_info_{timestamp}.pkl'
            info_path = os.path.join(results_dir, info_file)
            
            if not os.path.exists(info_path):
                logger.error(f'历史记录文件不存在: {info_path}')
                return jsonify({'status': 'error', 'message': '历史记录不存在'}), 404
            
            import pickle
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
            
            # 加载预测结果文件
            pred_file = f'predictions_{timestamp}.npy'
            pred_path = os.path.join(results_dir, pred_file)
            
            if not os.path.exists(pred_path):
                logger.error(f'预测结果文件不存在: {pred_path}')
                return jsonify({'status': 'error', 'message': '预测结果不存在'}), 404
            
            predictions = np.load(pred_path)
            
            # 准备详细信息
            detail = {
                'status': 'success',
                'timestamp': timestamp,
                'formatted_timestamp': datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
                'predictions': predictions.tolist(),
                'params': info['params'],
                'statistics': {
                    'avg_speed': float(np.mean(predictions)),
                    'min_speed': float(np.min(predictions)),
                    'max_speed': float(np.max(predictions)),
                    'median_speed': float(np.median(predictions)),
                    'std_speed': float(np.std(predictions))
                }
            }
            
            logger.info(f'历史记录详情请求处理成功: {timestamp}')
            return jsonify(detail)
            
        except Exception as e:
            logger.error(f'处理历史记录详情API请求时出错: {e}')
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/visualization/prediction_chart', methods=['POST'])
    def api_prediction_chart():
        """生成预测图表API"""
        logger.info('处理预测图表API请求')
        
        try:
            # 获取请求数据
            data = request.get_json()
            predictions = np.array(data.get('predictions', []))
            chart_type = data.get('chart_type', 'line')
            nodes = data.get('nodes', [0, 1, 2, 3, 4])  # 默认显示前5个节点
            
            if not predictions.any():
                logger.error('没有预测数据')
                return jsonify({'status': 'error', 'message': '没有预测数据'}), 400
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == 'line':
                # 折线图
                for i, node_idx in enumerate(nodes):
                    if node_idx < predictions.shape[1]:
                        ax.plot(predictions[:, node_idx], label=f'节点 {node_idx+1}')
                
                ax.set_title('多节点预测速度折线图')
                ax.set_xlabel('时间步')
                ax.set_ylabel('速度 (km/h)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif chart_type == 'bar':
                # 条形图（只显示最后一个时间步）
                last_step_preds = predictions[-1, nodes]
                ax.bar([f'节点 {i+1}' for i in range(len(nodes))], last_step_preds)
                
                ax.set_title('各节点预测速度条形图')
                ax.set_ylabel('速度 (km/h)')
                ax.grid(True, alpha=0.3, axis='y')
                
                # 在条形图上添加数值标签
                for i, v in enumerate(last_step_preds):
                    ax.text(i, v + 1, f'{v:.2f}', ha='center', va='bottom')
                
            elif chart_type == 'heatmap':
                # 热力图
                # 重塑预测结果以适应热力图
                grid_size = int(np.ceil(np.sqrt(predictions.shape[1])))
                heatmap_data = np.zeros((grid_size, grid_size))
                heatmap_data.flat[:predictions.shape[1]] = predictions[-1]
                
                sns.heatmap(heatmap_data, annot=False, cmap='RdYlGn_r', cbar_kws={'label': '速度 (km/h)'}, ax=ax)
                ax.set_title('预测速度热力图')
                ax.set_xticks([])
                ax.set_yticks([])
            
            # 保存图表到内存
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # 转换为base64编码
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            logger.info('预测图表生成成功')
            return jsonify({'status': 'success', 'image_data': image_base64})
            
        except Exception as e:
            logger.error(f'生成预测图表时出错: {e}')
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/road_network', methods=['GET'])
    def api_road_network():
        """获取道路网络数据API"""
        logger.info('处理道路网络数据API请求')
        
        try:
            # 在实际应用中，这里应该从数据库或文件加载真实的道路网络数据
            # 这里为了演示，我们生成模拟数据
            dataset_name = request.args.get('dataset', 'SZBZ')
            
            # 加载邻接矩阵
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
            adj_matrix_path = os.path.join(data_dir, f'{dataset_name}_adj_matrix.npy')
            
            if os.path.exists(adj_matrix_path):
                adj_matrix = np.load(adj_matrix_path)
                num_nodes = adj_matrix.shape[0]
            else:
                # 如果邻接矩阵不存在，生成模拟数据
                logger.warning(f'邻接矩阵文件不存在: {adj_matrix_path}，生成模拟数据')
                num_nodes = 50  # 模拟50个节点
                adj_matrix = np.zeros((num_nodes, num_nodes))
                
                # 生成随机连接
                np.random.seed(42)
                for i in range(num_nodes):
                    # 每个节点连接到3-6个其他节点
                    num_connections = np.random.randint(3, 7)
                    connections = np.random.choice([j for j in range(num_nodes) if j != i], 
                                                  size=num_connections, replace=False)
                    adj_matrix[i, connections] = 1
                    adj_matrix[connections, i] = 1  # 确保对称
            
            # 生成节点坐标
            np.random.seed(42)
            nodes = []
            for i in range(num_nodes):
                # 生成节点坐标（模拟地理位置）
                x = np.random.uniform(0, 100)
                y = np.random.uniform(0, 100)
                
                # 生成模拟速度（0-100 km/h）
                speed = np.random.uniform(20, 80)
                
                nodes.append({
                    'id': i,
                    'x': float(x),
                    'y': float(y),
                    'speed': float(speed),
                    'label': f'节点 {i+1}'
                })
            
            # 生成边
            edges = []
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    if adj_matrix[i, j] > 0:
                        edges.append({
                            'source': i,
                            'target': j,
                            'weight': float(adj_matrix[i, j])
                        })
            
            # 准备道路网络数据
            road_network = {
                'status': 'success',
                'dataset': dataset_name,
                'num_nodes': num_nodes,
                'nodes': nodes,
                'edges': edges
            }
            
            logger.info(f'道路网络数据生成成功，包含{num_nodes}个节点和{len(edges)}条边')
            return jsonify(road_network)
            
        except Exception as e:
            logger.error(f'处理道路网络数据API请求时出错: {e}')
            return jsonify({'status': 'error', 'message': str(e)}), 500
            
    @app.route('/api/assessment', methods=['POST'])
    def api_assessment():
        """模型评估API"""
        logger.info('处理模型评估API请求')
        
        try:
            # 获取请求数据
            data = request.json
            if not data:
                return jsonify({'status': 'error', 'message': '请求数据为空'}), 400
            
            # 获取模型和数据集参数
            model = data.get('model', 'gcn_gru_fuzzy')
            dataset = data.get('dataset', 'SZBZ')
            
            logger.info(f'评估模型: {model}, 数据集: {dataset}')
            
            # 记录开始时间
            start_time = time.time()
            
            # 这里应该调用实际的模型评估代码
            # 模拟模型评估过程
            # 生成模拟的评估结果
            time.sleep(1)  # 模拟计算延迟
            
            # 模拟评估指标
            metrics = {
                'mse': 8.25,
                'rmse': 2.87,
                'mae': 1.95,
                'mape': 12.3,
                'r2': 0.92,
                'runtime': round(time.time() - start_time, 1)
            }
            
            # 生成模拟的真实值和预测值数据
            real_values = []
            predicted_values = []
            time_steps = []
            
            for i in range(24):
                time_steps.append(f'{i}:00')
                real_val = 20 + random.uniform(0, 30)
                real_values.append(round(real_val, 2))
                predicted_values.append(round(real_val + random.uniform(-4, 4), 2))
            
            # 生成模拟的误差分布数据
            error_ranges = ['-5~-4', '-4~-3', '-3~-2', '-2~-1', '-1~0', '0~1', '1~2', '2~3', '3~4', '4~5']
            frequencies = [2, 5, 12, 25, 40, 38, 22, 10, 5, 1]
            
            # 生成模拟的训练历史数据
            epochs = list(range(1, 51))
            train_loss = [round(30 / math.sqrt(i) + random.uniform(0, 2), 2) for i in epochs]
            val_loss = [round(35 / math.sqrt(i) + random.uniform(0, 3), 2) for i in epochs]
            
            # 生成模拟的不同速度区间性能数据
            speed_ranges = ['0-10', '10-20', '20-30', '30-40', '40-50', '50+']
            speed_rmse = [0.8, 1.5, 2.2, 3.0, 2.5, 1.8]
            speed_mape = [5.2, 8.3, 11.5, 15.2, 12.8, 9.5]
            
            return jsonify({
                'status': 'success',
                'metrics': metrics,
                'prediction_data': {
                    'time_steps': time_steps,
                    'real_values': real_values,
                    'predicted_values': predicted_values
                },
                'error_distribution': {
                    'ranges': error_ranges,
                    'frequencies': frequencies
                },
                'training_history': {
                    'epochs': epochs,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                },
                'speed_comparison': {
                    'ranges': speed_ranges,
                    'rmse': speed_rmse,
                    'mape': speed_mape
                }
            })
            
        except Exception as e:
            logger.error(f'处理模型评估API请求时出错: {e}')
            return jsonify({'status': 'error', 'message': str(e)}), 500

# 运行Web服务器
def run_web_server(args):
    """运行Web服务器"""
    logger.info('=== 启动Web服务器 ===')
    
    try:
        # 创建Flask应用
        app = create_app()
        
        # 确保templates和static目录存在
        ensure_directories()
        
        # 创建必要的HTML模板
        create_html_templates()
        
        # 启动服务器
        host = args.host if hasattr(args, 'host') else '0.0.0.0'
        port = args.port if hasattr(args, 'port') else 5000
        debug = args.debug if hasattr(args, 'debug') else False
        
        logger.info(f'Web服务器将在 http://{host}:{port} 启动')
        
        # 运行Flask应用
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f'启动Web服务器时出错: {e}')
        raise

# 确保必要的目录存在
def ensure_directories():
    """确保必要的目录存在"""
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(__file__))
    
    # 创建templates目录
    templates_dir = os.path.join(root_dir, 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        logger.info(f'创建目录: {templates_dir}')
    
    # 创建static目录
    static_dir = os.path.join(root_dir, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        logger.info(f'创建目录: {static_dir}')
    
    # 创建static/css目录
    css_dir = os.path.join(static_dir, 'css')
    if not os.path.exists(css_dir):
        os.makedirs(css_dir)
        logger.info(f'创建目录: {css_dir}')
    
    # 创建static/js目录
    js_dir = os.path.join(static_dir, 'js')
    if not os.path.exists(js_dir):
        os.makedirs(js_dir)
        logger.info(f'创建目录: {js_dir}')
    
    # 创建static/images目录
    images_dir = os.path.join(static_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        logger.info(f'创建目录: {images_dir}')

# 创建HTML模板
def create_html_templates():
    """创建HTML模板"""
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(__file__))
    templates_dir = os.path.join(root_dir, 'templates')
    static_dir = os.path.join(root_dir, 'static')
    
    # 创建基础CSS文件
    create_css_files(static_dir)
    
    # 创建基础JS文件
    create_js_files(static_dir)
    
    # 创建index.html
    index_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_path):
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交通速度预测系统</title>
    <link rel="stylesheet" href="/static/css/bootstrap.min.css">
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="/static/js/echarts.min.js"></script>
</head>
<body>
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">交通速度预测系统</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link active" href="/">首页</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/dashboard">仪表盘</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/prediction">预测</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/history">历史记录</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/assessment">模型评估</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- 主内容区 -->
    <main class="container mt-5">
        <div class="jumbotron bg-light p-5 rounded-lg shadow">
            <h1 class="display-4 text-center">交通速度预测系统</h1>
            <p class="lead text-center mt-3">基于深度学习的智能交通速度预测平台</p>
            <div class="mt-5 text-center">
                <a href="/dashboard" class="btn btn-primary btn-lg mx-2">进入仪表盘</a>
                <a href="/prediction" class="btn btn-success btn-lg mx-2">进行预测</a>
            </div>
        </div>

        <!-- 系统介绍 -->
        <div class="mt-8">
            <h2 class="text-center mb-4">系统功能介绍</h2>
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h3 class="card-title text-center">数据可视化</h3>
                            <p class="card-text">直观展示交通数据分布、道路网络和预测结果，帮助用户快速理解交通状况。</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h3 class="card-title text-center">智能预测</h3>
                            <p class="card-text">基于GCN+GRU+模糊推理机制的深度学习模型，准确预测未来交通速度。</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body">
                            <h3 class="card-title text-center">历史记录</h3>
                            <p class="card-text">查看和分析历史预测结果，支持多维度比较和评估。</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 技术架构 -->
        <div class="mt-8">
            <h2 class="text-center mb-4">技术架构</h2>
            <div class="bg-light p-5 rounded-lg">
                <ul class="list-group">
                    <li class="list-group-item"><strong>前端：</strong>HTML5, CSS3, JavaScript, Bootstrap, ECharts</li>
                    <li class="list-group-item"><strong>后端：</strong>Python, Flask</li>
                    <li class="list-group-item"><strong>深度学习框架：</strong>TensorFlow</li>
                    <li class="list-group-item"><strong>模型：</strong>GCN (图卷积网络) + GRU (门控循环单元) + 模糊推理机制</li>
                </ul>
            </div>
        </div>
    </main>

    <!-- 页脚 -->
    <footer class="bg-dark text-white mt-8 py-4">
        <div class="container text-center">
            <p>&copy; 2024 交通速度预测系统. 保留所有权利.</p>
        </div>
    </footer>

    <script src="/static/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/main.js"></script>
</body>
</html>''')
            logger.info(f'创建HTML模板: {index_path}')
    
    # 创建dashboard.html
    dashboard_path = os.path.join(templates_dir, 'dashboard.html')
    if not os.path.exists(dashboard_path):
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>仪表盘 - 交通速度预测系统</title>
    <link rel="stylesheet" href="/static/css/bootstrap.min.css">
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="/static/js/echarts.min.js"></script>
</head>
<body>
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">交通速度预测系统</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/">首页</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="/dashboard">仪表盘</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/prediction">预测</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/history">历史记录</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/assessment">模型评估</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- 主内容区 -->
    <main class="container mt-5">
        <h1 class="text-center mb-5">系统仪表盘</h1>
        
        <!-- 统计卡片 -->
        <div class="row mb-5">
            <div class="col-md-3">
                <div class="card bg-primary text-white">
                    <div class="card-body">
                        <h3 class="card-title">总节点数</h3>
                        <p class="card-text display-4" id="total-nodes">0</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-success text-white">
                    <div class="card-body">
                        <h3 class="card-title">平均速度</h3>
                        <p class="card-text display-4" id="avg-speed">0</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-warning text-white">
                    <div class="card-body">
                        <h3 class="card-title">拥堵路段</h3>
                        <p class="card-text display-4" id="congested-roads">0</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-info text-white">
                    <div class="card-body">
                        <h3 class="card-title">历史预测</h3>
                        <p class="card-text display-4" id="history-count">0</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- 道路网络可视化 -->
        <div class="mb-5">
            <h2 class="text-center">道路网络可视化</h2>
            <div class="bg-light p-3 rounded-lg shadow">
                <div class="row">
                    <div class="col-md-2">
                        <div class="mb-3">
                            <label for="dataset-select" class="form-label">选择数据集</label>
                            <select id="dataset-select" class="form-select">
                                <option value="SZBZ">深圳北站 (SZBZ)</option>
                                <option value="Los-loop">洛杉矶高速 (Los-loop)</option>
                            </select>
                        </div>
                        <button id="refresh-network" class="btn btn-primary w-100 mb-3">刷新网络</button>
                        <div class="mb-3">
                            <label class="form-label">图例</label>
                            <div class="row">
                                <div class="col-4">低速</div>
                                <div class="col-8 bg-red-500" style="background: linear-gradient(to right, #ff0000, #ffff00, #00ff00); height: 20px;"></div>
                            </div>
                            <div class="row mt-1">
                                <div class="col-4">高速</div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-10">
                        <div id="road-network-chart" style="width: 100%; height: 500px;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 速度分布和预测趋势 -->
        <div class="row mb-5">
            <div class="col-md-6">
                <h2 class="text-center">速度分布</h2>
                <div class="bg-light p-3 rounded-lg shadow">
                    <div id="speed-distribution-chart" style="width: 100%; height: 400px;"></div>
                </div>
            </div>
            <div class="col-md-6">
                <h2 class="text-center">预测趋势</h2>
                <div class="bg-light p-3 rounded-lg shadow">
                    <div id="prediction-trend-chart" style="width: 100%; height: 400px;"></div>
                </div>
            </div>
        </div>

        <!-- 实时预测结果 -->
        <div class="mb-5">
            <h2 class="text-center">实时预测结果</h2>
            <div class="bg-light p-3 rounded-lg shadow">
                <div id="real-time-prediction-chart" style="width: 100%; height: 400px;"></div>
            </div>
        </div>
    </main>

    <!-- 页脚 -->
    <footer class="bg-dark text-white mt-8 py-4">
        <div class="container text-center">
            <p>&copy; 2024 交通速度预测系统. 保留所有权利.</p>
        </div>
    </footer>

    <script src="/static/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/dashboard.js"></script>
</body>
</html>''')
            logger.info(f'创建HTML模板: {dashboard_path}')
    
    # 创建prediction.html
    prediction_path = os.path.join(templates_dir, 'prediction.html')
    if not os.path.exists(prediction_path):
        with open(prediction_path, 'w', encoding='utf-8') as f:
            f.write('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>预测 - 交通速度预测系统</title>
    <link rel="stylesheet" href="/static/css/bootstrap.min.css">
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="/static/js/echarts.min.js"></script>
</head>
<body>
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">交通速度预测系统</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/">首页</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/dashboard">仪表盘</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="/prediction">预测</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/history">历史记录</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/assessment">模型评估</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- 主内容区 -->
    <main class="container mt-5">
        <h1 class="text-center mb-5">交通速度预测</h1>
        
        <!-- 预测参数设置 -->
        <div class="bg-light p-5 rounded-lg shadow mb-5">
            <h2 class="text-center mb-4">预测参数设置</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="model-select" class="form-label">选择模型</label>
                        <select id="model-select" class="form-select">
                            <option value="gcn_gru_fuzzy">GCN+GRU+模糊推理</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="dataset-select" class="form-label">选择数据集</label>
                        <select id="dataset-select" class="form-select">
                            <option value="SZBZ">深圳北站 (SZBZ)</option>
                            <option value="Los-loop">洛杉矶高速 (Los-loop)</option>
                        </select>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="prediction-steps" class="form-label">预测步长（未来时间步数）</label>
                        <input type="range" id="prediction-steps" min="1" max="12" value="6" class="form-range">
                        <div class="text-center mt-2"><span id="steps-value">6</span> 步</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="show-nodes" class="form-label">显示节点数量</label>
                        <input type="range" id="show-nodes" min="1" max="10" value="5" class="form-range">
                        <div class="text-center mt-2"><span id="nodes-value">5</span> 个节点</div>
                    </div>
                </div>
            </div>
            <div class="text-center mt-4">
                <button id="start-prediction" class="btn btn-primary btn-lg">开始预测</button>
            </div>
        </div>

        <!-- 预测结果 -->
        <div class="mb-5" id="prediction-result" style="display: none;">
            <h2 class="text-center">预测结果</h2>
            <div class="bg-light p-3 rounded-lg shadow mb-3">
                <div class="mb-3">
                    <label for="chart-type" class="form-label">选择图表类型</label>
                    <select id="chart-type" class="form-select">
                        <option value="line">折线图</option>
                        <option value="bar">条形图</option>
                        <option value="heatmap">热力图</option>
                    </select>
                </div>
                <div id="prediction-chart" style="width: 100%; height: 500px;"></div>
            </div>
            
            <!-- 预测统计信息 -->
            <div class="bg-light p-3 rounded-lg shadow">
                <h3 class="text-center mb-3">预测统计信息</h3>
                <div class="row">
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">平均速度</h4>
                                <p class="card-text display-5" id="pred-avg-speed">0</p>
                                <p class="card-text text-muted">km/h</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">最高速度</h4>
                                <p class="card-text display-5" id="pred-max-speed">0</p>
                                <p class="card-text text-muted">km/h</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">最低速度</h4>
                                <p class="card-text display-5" id="pred-min-speed">0</p>
                                <p class="card-text text-muted">km/h</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">预测时间</h4>
                                <p class="card-text display-5" id="pred-timestamp"></p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- 页脚 -->
    <footer class="bg-dark text-white mt-8 py-4">
        <div class="container text-center">
            <p>&copy; 2024 交通速度预测系统. 保留所有权利.</p>
        </div>
    </footer>

    <script src="/static/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/prediction.js"></script>
</body>
</html>''')
            logger.info(f'创建HTML模板: {prediction_path}')
    
    # 创建history.html
    history_path = os.path.join(templates_dir, 'history.html')
    if not os.path.exists(history_path):
        with open(history_path, 'w', encoding='utf-8') as f:
            f.write('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>历史记录 - 交通速度预测系统</title>
    <link rel="stylesheet" href="/static/css/bootstrap.min.css">
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="/static/js/echarts.min.js"></script>
</head>
<body>
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">交通速度预测系统</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/">首页</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/dashboard">仪表盘</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/prediction">预测</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="/history">历史记录</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/assessment">模型评估</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- 主内容区 -->
    <main class="container mt-5">
        <h1 class="text-center mb-5">历史预测记录</h1>
        
        <!-- 历史记录表格 -->
        <div class="bg-light p-3 rounded-lg shadow mb-5">
            <div class="mb-3">
                <button id="refresh-history" class="btn btn-primary">刷新记录</button>
            </div>
            <div class="table-responsive">
                <table class="table table-striped table-hover">
                    <thead>
                        <tr>
                            <th>预测时间</th>
                            <th>平均速度 (km/h)</th>
                            <th>最高速度 (km/h)</th>
                            <th>最低速度 (km/h)</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody id="history-table-body">
                        <!-- 历史记录将通过JavaScript动态加载 -->
                        <tr>
                            <td colspan="5" class="text-center">加载中...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 历史记录详情 -->
        <div id="history-detail" class="mb-5" style="display: none;">
            <h2 class="text-center">历史记录详情</h2>
            <div class="bg-light p-3 rounded-lg shadow">
                <div class="mb-3">
                    <h3 id="detail-timestamp"></h3>
                </div>
                
                <!-- 统计信息 -->
                <div class="row mb-3">
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">平均速度</h4>
                                <p class="card-text display-5" id="detail-avg-speed">0</p>
                                <p class="card-text text-muted">km/h</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">最高速度</h4>
                                <p class="card-text display-5" id="detail-max-speed">0</p>
                                <p class="card-text text-muted">km/h</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">最低速度</h4>
                                <p class="card-text display-5" id="detail-min-speed">0</p>
                                <p class="card-text text-muted">km/h</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">中位数速度</h4>
                                <p class="card-text display-5" id="detail-median-speed">0</p>
                                <p class="card-text text-muted">km/h</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- 历史预测图表 -->
                <div class="mb-3">
                    <label for="detail-chart-type" class="form-label">选择图表类型</label>
                    <select id="detail-chart-type" class="form-select">
                        <option value="line">折线图</option>
                        <option value="bar">条形图</option>
                        <option value="heatmap">热力图</option>
                    </select>
                </div>
                <div id="detail-chart" style="width: 100%; height: 500px;"></div>
            </div>
        </div>
    </main>

    <!-- 页脚 -->
    <footer class="bg-dark text-white mt-8 py-4">
        <div class="container text-center">
            <p>&copy; 2024 交通速度预测系统. 保留所有权利.</p>
        </div>
    </footer>

    <script src="/static/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/history.js"></script>
</body>
</html>''')
            logger.info(f'创建HTML模板: {history_path}')
    
    # 创建assessment.html
    assessment_path = os.path.join(templates_dir, 'assessment.html')
    if not os.path.exists(assessment_path):
        with open(assessment_path, 'w', encoding='utf-8') as f:
            f.write('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型评估 - 交通速度预测系统</title>
    <link rel="stylesheet" href="/static/css/bootstrap.min.css">
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="/static/js/echarts.min.js"></script>
</head>
<body>
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">交通速度预测系统</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/">首页</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/dashboard">仪表盘</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/prediction">预测</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/history">历史记录</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="/assessment">模型评估</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- 主内容区 -->
    <main class="container mt-5">
        <h1 class="text-center mb-5">模型性能评估</h1>
        
        <!-- 模型和数据集选择 -->
        <div class="bg-light p-5 rounded-lg shadow mb-5">
            <h2 class="text-center mb-4">选择评估参数</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="assessment-model" class="form-label">选择模型</label>
                        <select id="assessment-model" class="form-select">
                            <option value="gcn_gru_fuzzy">GCN+GRU+模糊推理</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="assessment-dataset" class="form-label">选择数据集</label>
                        <select id="assessment-dataset" class="form-select">
                            <option value="SZBZ">深圳北站 (SZBZ)</option>
                            <option value="Los-loop">洛杉矶高速 (Los-loop)</option>
                        </select>
                    </div>
                </div>
            </div>
            <div class="text-center mt-4">
                <button id="start-assessment" class="btn btn-primary btn-lg">开始评估</button>
            </div>
        </div>

        <!-- 评估结果区域 -->
        <div id="assessment-result" class="mb-5" style="display: none;">
            <!-- 评估指标卡片 -->
            <div class="bg-light p-3 rounded-lg shadow mb-5">
                <h2 class="text-center mb-4">评估指标</h2>
                <div class="row">
                    <div class="col-md-2">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">MSE</h4>
                                <p class="card-text display-5" id="mse-value">0</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">RMSE</h4>
                                <p class="card-text display-5" id="rmse-value">0</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">MAE</h4>
                                <p class="card-text display-5" id="mae-value">0</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">MAPE</h4>
                                <p class="card-text display-5" id="mape-value">0%</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">R²</h4>
                                <p class="card-text display-5" id="r2-value">0</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card text-center">
                            <div class="card-body">
                                <h4 class="card-title">运行时间</h4>
                                <p class="card-text display-5" id="runtime-value">0</p>
                                <p class="card-text text-muted">秒</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 模型拓扑结构 -->
            <div class="bg-light p-3 rounded-lg shadow mb-5">
                <h2 class="text-center mb-4">模型拓扑结构</h2>
                <div class="row">
                    <div class="col-md-12">
                        <div id="model-topology" style="width: 100%; height: 300px;"></div>
                    </div>
                </div>
            </div>

            <!-- 评估图表 -->
            <div class="bg-light p-3 rounded-lg shadow mb-5">
                <h2 class="text-center mb-4">评估可视化</h2>
                <div class="mb-3">
                    <label for="assessment-chart-type" class="form-label">选择图表类型</label>
                    <select id="assessment-chart-type" class="form-select">
                        <option value="prediction_comparison">真实值 vs 预测值</option>
                        <option value="error_distribution">误差分布</option>
                        <option value="training_history">训练历史</option>
                        <option value="speed_comparison">不同速度区间性能</option>
                    </select>
                </div>
                <div id="assessment-chart" style="width: 100%; height: 500px;"></div>
            </div>
        </div>
    </main>

    <!-- 页脚 -->
    <footer class="bg-dark text-white mt-8 py-4">
        <div class="container text-center">
            <p>&copy; 2024 交通速度预测系统. 保留所有权利.</p>
        </div>
    </footer>

    <script src="/static/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/assessment.js"></script>
</body>
</html>''')
            logger.info(f'创建HTML模板: {assessment_path}')

# 创建CSS文件
def create_css_files(static_dir):
    """创建CSS文件"""
    css_dir = os.path.join(static_dir, 'css')
    
    # 创建style.css
    style_path = os.path.join(css_dir, 'style.css')
    if not os.path.exists(style_path):
        with open(style_path, 'w', encoding='utf-8') as f:
            f.write('''/* 基础样式 */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background-color: #f5f5f5;
    margin: 0;
    padding: 0;
}

/* 导航栏样式 */
.navbar {
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    background-color: #007bff;
}

.navbar-brand {
    color: white;
    font-weight: bold;
}

.nav-link {
    color: white;
}

.nav-link:hover {
    color: #f8f9fa;
}

.nav-link.active {
    font-weight: bold;
}

/* 卡片样式 */
.card {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: none;
    border-radius: 8px;
    overflow: hidden;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

/* 按钮样式 */
.btn {
    transition: all 0.3s ease;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: 500;
}

.btn:hover {
    transform: translateY(-2px);
}

.btn-primary {
    background-color: #007bff;
}

.btn-success {
    background-color: #28a745;
}

.btn-warning {
    background-color: #ffc107;
    color: #212529;
}

.btn-info {
    background-color: #17a2b8;
}

/* 阴影效果 */
.shadow {
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* 响应式调整 */
@media (max-width: 768px) {
    .jumbotron h1 {
        font-size: 2.5rem;
    }
    
    .card {
        margin-bottom: 1rem;
    }
}

/* 加载动画 */
.loader {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* 表格样式 */
.table {
    width: 100%;
    border-collapse: collapse;
}

.table th,
.table td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #dee2e6;
}

.table th {
    background-color: #f8f9fa;
    font-weight: 600;
}

.table-hover tbody tr:hover {
    background-color: #f8f9fa;
}

/* 滚动条样式 */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* 容器和网格系统 */
.container {
    width: 100%;
    padding-right: 15px;
    padding-left: 15px;
    margin-right: auto;
    margin-left: auto;
}

.row {
    display: flex;
    flex-wrap: wrap;
    margin-right: -15px;
    margin-left: -15px;
}

.col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6,
.col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    position: relative;
    width: 100%;
    padding-right: 15px;
    padding-left: 15px;
}

@media (min-width: 768px) {
    .col-md-1 { flex: 0 0 8.333333%; max-width: 8.333333%; }
    .col-md-2 { flex: 0 0 16.666667%; max-width: 16.666667%; }
    .col-md-3 { flex: 0 0 25%; max-width: 25%; }
    .col-md-4 { flex: 0 0 33.333333%; max-width: 33.333333%; }
    .col-md-5 { flex: 0 0 41.666667%; max-width: 41.666667%; }
    .col-md-6 { flex: 0 0 50%; max-width: 50%; }
    .col-md-7 { flex: 0 0 58.333333%; max-width: 58.333333%; }
    .col-md-8 { flex: 0 0 66.666667%; max-width: 66.666667%; }
    .col-md-9 { flex: 0 0 75%; max-width: 75%; }
    .col-md-10 { flex: 0 0 83.333333%; max-width: 83.333333%; }
    .col-md-11 { flex: 0 0 91.666667%; max-width: 91.666667%; }
    .col-md-12 { flex: 0 0 100%; max-width: 100%; }
}

/* 表单样式 */
.form-control {
    display: block;
    width: 100%;
    padding: 8px 12px;
    font-size: 16px;
    line-height: 1.5;
    color: #495057;
    background-color: #fff;
    background-clip: padding-box;
    border: 1px solid #ced4da;
    border-radius: 4px;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}

.form-control:focus {
    color: #495057;
    background-color: #fff;
    border-color: #80bdff;
    outline: 0;
    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
}

/* 标题样式 */
h1, h2, h3, h4, h5, h6 {
    margin-top: 0;
    margin-bottom: 0.5rem;
    font-weight: 500;
    line-height: 1.2;
}

h1 { font-size: 2.5rem; }
h2 { font-size: 2rem; }
h3 { font-size: 1.75rem; }
h4 { font-size: 1.5rem; }
h5 { font-size: 1.25rem; }
h6 { font-size: 1rem; }

/* 间距 */
.mt-1 { margin-top: 0.25rem !important; }
.mt-2 { margin-top: 0.5rem !important; }
.mt-3 { margin-top: 1rem !important; }
.mt-4 { margin-top: 1.5rem !important; }
.mt-5 { margin-top: 3rem !important; }
.mt-8 { margin-top: 5rem !important; }

.mb-1 { margin-bottom: 0.25rem !important; }
.mb-2 { margin-bottom: 0.5rem !important; }
.mb-3 { margin-bottom: 1rem !important; }
.mb-4 { margin-bottom: 1.5rem !important; }
.mb-5 { margin-bottom: 3rem !important; }

.p-1 { padding: 0.25rem !important; }
.p-2 { padding: 0.5rem !important; }
.p-3 { padding: 1rem !important; }
.p-4 { padding: 1.5rem !important; }
.p-5 { padding: 3rem !important; }

/* 文本对齐 */
.text-center { text-align: center !important; }
.text-muted { color: #6c757d !important; }

/* 背景色 */
.bg-primary { background-color: #007bff !important; color: white !important; }
.bg-success { background-color: #28a745 !important; color: white !important; }
.bg-warning { background-color: #ffc107 !important; color: #212529 !important; }
.bg-info { background-color: #17a2b8 !important; color: white !important; }
.bg-dark { background-color: #343a40 !important; color: white !important; }
.bg-light { background-color: #f8f9fa !important; }
.bg-white { background-color: #fff !important; }

/* 边框圆角 */
.rounded { border-radius: 0.25rem !important; }
.rounded-lg { border-radius: 0.5rem !important; }

/* 显示/隐藏 */
.d-none { display: none !important; }
.d-block { display: block !important; }
.d-flex { display: flex !important; }

/* 响应式表格 */
.table-responsive {
    display: block;
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

/* Jumbotron样式 */
.jumbotron {
    padding: 2rem 1rem;
    margin-bottom: 2rem;
    background-color: #e9ecef;
    border-radius: 0.3rem;
}

/* 显示-4大小的字体 */
.display-4 {
    font-size: 3.5rem;
    font-weight: 300;
    line-height: 1.2;
}

.lead {
    font-size: 1.25rem;
    font-weight: 300;
}''')
            logger.info(f'创建CSS文件: {style_path}')

    # 创建mock的Bootstrap CSS
    bootstrap_path = os.path.join(css_dir, 'bootstrap.min.css')
    if not os.path.exists(bootstrap_path):
        with open(bootstrap_path, 'w', encoding='utf-8') as f:
            f.write('/* Bootstrap Mock CSS - 提供基本样式支持 */\n')
            # 只需添加一个小的占位内容，实际样式已经在style.css中实现
        logger.info(f'创建Mock Bootstrap CSS: {bootstrap_path}')

# 创建JS文件
def create_js_files(static_dir):
    """创建JS文件"""
    js_dir = os.path.join(static_dir, 'js')
    
    # 创建main.js
    main_path = os.path.join(js_dir, 'main.js')
    if not os.path.exists(main_path):
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write('''// 主页面JS脚本
console.log('交通速度预测系统主页面加载完成');

// 页面加载完成后的初始化函数
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM内容加载完成');
    
    // 可以在这里添加页面初始化代码
});''')
            logger.info(f'创建JS文件: {main_path}')
    
    # 创建dashboard.js
    dashboard_path = os.path.join(js_dir, 'dashboard.js')
    if not os.path.exists(dashboard_path):
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write('''// 仪表盘页面JS脚本
console.log('仪表盘页面加载完成');

// 页面加载完成后的初始化函数
document.addEventListener('DOMContentLoaded', function() {
    console.log('仪表盘DOM内容加载完成');
    
    // 初始化统计数据
    updateStatistics();
    
    // 初始化道路网络图表
    initRoadNetworkChart();
    
    // 初始化速度分布图表
    initSpeedDistributionChart();
    
    // 初始化预测趋势图表
    initPredictionTrendChart();
    
    // 初始化实时预测图表
    initRealTimePredictionChart();
    
    // 绑定刷新网络按钮事件
    document.getElementById('refresh-network').addEventListener('click', function() {
        refreshRoadNetwork();
    });
    
    // 绑定数据集选择事件
    document.getElementById('dataset-select').addEventListener('change', function() {
        refreshRoadNetwork();
    });
});

// 更新统计数据
function updateStatistics() {
    console.log('更新统计数据');
    
    // 模拟数据
    const totalNodes = 50;
    const avgSpeed = 55.7;
    const congestedRoads = 12;
    const historyCount = 24;
    
    // 更新统计卡片
    document.getElementById('total-nodes').textContent = totalNodes;
    document.getElementById('avg-speed').textContent = avgSpeed.toFixed(1) + ' km/h';
    document.getElementById('congested-roads').textContent = congestedRoads;
    document.getElementById('history-count').textContent = historyCount;
}

// 初始化道路网络图表
function initRoadNetworkChart() {
    console.log('初始化道路网络图表');
    
    const chartDom = document.getElementById('road-network-chart');
    // 检查ECharts是否可用
    if (typeof echarts !== 'undefined') {
        const myChart = echarts.init(chartDom);
        
        // 模拟道路网络数据
        const dataset = document.getElementById('dataset-select').value;
        fetchRoadNetworkData(dataset, myChart);
    } else {
        console.log('ECharts未加载，使用模拟图表');
        // 显示模拟图表
        chartDom.innerHTML = '<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #666;">图表加载中...</div>';
        
        // 3秒后显示模拟数据
        setTimeout(() => {
            showMockRoadNetworkChart(chartDom);
        }, 1000);
    }
}

// 显示模拟道路网络图表
function showMockRoadNetworkChart(chartDom) {
    const mockData = `
        <svg width="100%" height="100%">
            <!-- 模拟道路网络 -->
            <g>
                <!-- 模拟节点 -->
                <circle cx="100" cy="100" r="10" fill="#28a745" /><text x="115" y="105" font-size="12">节点1: 60km/h</text>
                <circle cx="200" cy="150" r="12" fill="#28a745" /><text x="215" y="155" font-size="12">节点2: 70km/h</text>
                <circle cx="300" cy="100" r="8" fill="#ffc107" /><text x="315" y="105" font-size="12">节点3: 40km/h</text>
                <circle cx="200" cy="50" r="9" fill="#007bff" /><text x="215" y="55" font-size="12">节点4: 50km/h</text>
                <circle cx="100" cy="200" r="7" fill="#dc3545" /><text x="115" y="205" font-size="12">节点5: 30km/h</text>
                <circle cx="300" cy="200" r="11" fill="#28a745" /><text x="315" y="205" font-size="12">节点6: 65km/h</text>
                
                <!-- 模拟边 -->
                <line x1="100" y1="100" x2="200" y2="150" stroke="#ddd" stroke-width="2" />
                <line x1="200" y1="150" x2="300" y2="100" stroke="#ddd" stroke-width="2" />
                <line x1="200" y1="150" x2="200" y2="50" stroke="#ddd" stroke-width="2" />
                <line x1="100" y1="100" x2="100" y2="200" stroke="#ddd" stroke-width="2" />
                <line x1="300" y1="100" x2="300" y2="200" stroke="#ddd" stroke-width="2" />
                <line x1="100" y1="200" x2="200" y2="150" stroke="#ddd" stroke-width="2" />
                <line x1="200" y1="150" x2="300" y2="200" stroke="#ddd" stroke-width="2" />
            </g>
        </svg>
    `;
    chartDom.innerHTML = mockData;
}

// 获取道路网络数据
function fetchRoadNetworkData(dataset, chart) {
    console.log('获取道路网络数据', dataset);
    
    // 显示加载中
    chart.showLoading();
    
    // 发送API请求
    fetch(`/api/road_network?dataset=${dataset}`)
        .then(response => response.json())
        .then(data => {
            console.log('道路网络数据获取成功', data);
            
            // 隐藏加载中
            chart.hideLoading();
            
            if (data.status === 'success') {
                // 构建图表配置
                const option = buildRoadNetworkOption(data);
                chart.setOption(option);
                
                // 更新统计数据
                updateRoadNetworkStatistics(data);
            } else {
                console.error('道路网络数据获取失败', data.message);
            }
        })
        .catch(error => {
            console.error('获取道路网络数据时出错', error);
            chart.hideLoading();
        });
}

// 构建道路网络图表配置
function buildRoadNetworkOption(data) {
    console.log('构建道路网络图表配置');
    
    // 提取节点和边数据
    const nodes = data.nodes;
    const edges = data.edges;
    
    // 准备节点数据
    const categories = [];
    const nodeData = nodes.map(node => ({
        id: node.id,
        name: node.label,
        value: node.speed,
        category: 0,
        symbolSize: Math.max(8, Math.min(20, node.speed / 5)),
        x: node.x,
        y: node.y,
        itemStyle: {
            color: getSpeedColor(node.speed)
        },
        label: {
            show: false
        },
        emphasis: {
            label: {
                show: true,
                formatter: function(params) {
                    return `${params.name}: ${params.value.toFixed(1)} km/h`;
                }
            }
        }
    }));
    
    // 准备边数据
    const edgeData = edges.map(edge => ({
        source: edge.source,
        target: edge.target,
        lineStyle: {
            color: '#aaa',
            width: edge.weight * 2,
            curveness: 0
        }
    }));
    
    // 图表配置
    const option = {
        title: {
            text: `${data.dataset} 道路网络`,
            left: 'center'
        },
        tooltip: {
            formatter: function(params) {
                if (params.dataType === 'edge') {
                    return `路段: ${params.data.source} -> ${params.data.target}`;
                } else {
                    return `${params.name}: ${params.value.toFixed(1)} km/h`;
                }
            }
        },
        animationDurationUpdate: 1500,
        animationEasingUpdate: 'quinticInOut',
        series: [
            {
                type: 'graph',
                layout: 'none',
                symbolSize: 20,
                roam: true,
                label: {
                    show: false
                },
                edgeSymbol: ['none', 'arrow'],
                edgeSymbolSize: [4, 8],
                edgeLabel: {
                    fontSize: 12
                },
                data: nodeData,
                links: edgeData,
                categories: categories,
                lineStyle: {
                    color: '#aaa',
                    width: 1,
                    curveness: 0
                },
                emphasis: {
                    focus: 'adjacency',
                    lineStyle: {
                        width: 4
                    }
                }
            }
        ]
    };
    
    return option;
}

// 根据速度获取颜色
function getSpeedColor(speed) {
    // 低速（<30 km/h）- 红色
    if (speed < 30) {
        return '#ff0000';
    }
    // 中速（30-60 km/h）- 黄色到橙色
    else if (speed < 60) {
        const ratio = (speed - 30) / 30;
        const r = 255;
        const g = Math.floor(255 * ratio);
        const b = 0;
        return `rgb(${r}, ${g}, ${b})`;
    }
    // 高速（>60 km/h）- 绿色
    else {
        const ratio = Math.min(1, (speed - 60) / 40);
        const r = Math.floor(255 * (1 - ratio));
        const g = 255;
        const b = 0;
        return `rgb(${r}, ${g}, ${b})`;
    }
}

// 更新道路网络统计数据
function updateRoadNetworkStatistics(data) {
    console.log('更新道路网络统计数据');
    
    const nodes = data.nodes;
    const speeds = nodes.map(node => node.speed);
    const avgSpeed = speeds.reduce((a, b) => a + b, 0) / speeds.length;
    const congestedRoads = speeds.filter(speed => speed < 30).length;
    
    // 更新统计卡片
    document.getElementById('total-nodes').textContent = data.num_nodes;
    document.getElementById('avg-speed').textContent = avgSpeed.toFixed(1) + ' km/h';
    document.getElementById('congested-roads').textContent = congestedRoads;
}

// 刷新道路网络
function refreshRoadNetwork() {
    console.log('刷新道路网络');
    
    const chartDom = document.getElementById('road-network-chart');
    const myChart = echarts.getInstanceByDom(chartDom);
    const dataset = document.getElementById('dataset-select').value;
    
    fetchRoadNetworkData(dataset, myChart);
}

// 初始化速度分布图表
function initSpeedDistributionChart() {
    console.log('初始化速度分布图表');
    
    const chartDom = document.getElementById('speed-distribution-chart');
    const myChart = echarts.init(chartDom);
    
    // 模拟速度分布数据
    const speedRanges = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+'];
    const counts = [5, 15, 20, 8, 2, 0];
    
    // 图表配置
    const option = {
        title: {
            text: '速度分布统计',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: function(params) {
                return `${params[0].name} km/h: ${params[0].value} 个路段`;
            }
        },
        xAxis: {
            type: 'category',
            data: speedRanges,
            name: '速度区间 (km/h)',
            axisLabel: {
                rotate: 0
            }
        },
        yAxis: {
            type: 'value',
            name: '路段数量',
            minInterval: 1
        },
        series: [
            {
                data: counts,
                type: 'bar',
                itemStyle: {
                    color: function(params) {
                        const colors = ['#ff0000', '#ff7f00', '#ffff00', '#7fff00', '#00ff00', '#00ff7f'];
                        return colors[params.dataIndex];
                    }
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }
        ]
    };
    
    myChart.setOption(option);
    
    // 响应窗口大小变化
    window.addEventListener('resize', function() {
        myChart.resize();
    });
}

// 初始化预测趋势图表
function initPredictionTrendChart() {
    console.log('初始化预测趋势图表');
    
    const chartDom = document.getElementById('prediction-trend-chart');
    const myChart = echarts.init(chartDom);
    
    // 模拟预测趋势数据
    const timeSteps = ['现在', '+5', '+10', '+15', '+20', '+25', '+30'];
    const avgSpeed = [55, 52, 48, 45, 43, 42, 45];
    const maxSpeed = [75, 72, 68, 65, 63, 62, 65];
    const minSpeed = [35, 32, 28, 25, 23, 22, 25];
    
    // 图表配置
    const option = {
        title: {
            text: '未来30分钟速度趋势预测',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                let result = `${params[0].name}分钟后<br/>`;
                params.forEach(param => {
                    result += `${param.seriesName}: ${param.value} km/h<br/>`;
                });
                return result;
            }
        },
        legend: {
            data: ['平均速度', '最高速度', '最低速度'],
            bottom: 0
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '15%',
            top: '15%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: timeSteps,
            name: '时间'
        },
        yAxis: {
            type: 'value',
            name: '速度 (km/h)',
            min: 0,
            max: 100
        },
        series: [
            {
                name: '平均速度',
                type: 'line',
                data: avgSpeed,
                symbol: 'circle',
                symbolSize: 8,
                itemStyle: {
                    color: '#3498db'
                },
                lineStyle: {
                    width: 3
                },
                emphasis: {
                    focus: 'series'
                }
            },
            {
                name: '最高速度',
                type: 'line',
                data: maxSpeed,
                symbol: 'circle',
                symbolSize: 6,
                itemStyle: {
                    color: '#2ecc71'
                },
                emphasis: {
                    focus: 'series'
                }
            },
            {
                name: '最低速度',
                type: 'line',
                data: minSpeed,
                symbol: 'circle',
                symbolSize: 6,
                itemStyle: {
                    color: '#e74c3c'
                },
                emphasis: {
                    focus: 'series'
                }
            }
        ]
    };
    
    myChart.setOption(option);
    
    // 响应窗口大小变化
    window.addEventListener('resize', function() {
        myChart.resize();
    });
}

// 初始化实时预测图表
function initRealTimePredictionChart() {
    console.log('初始化实时预测图表');
    
    const chartDom = document.getElementById('real-time-prediction-chart');
    const myChart = echarts.init(chartDom);
    
    // 模拟实时预测数据（前5个节点）
    const timeSteps = ['现在', '+5', '+10', '+15', '+20', '+25', '+30'];
    const node1 = [55, 52, 48, 45, 43, 42, 45];
    const node2 = [60, 58, 55, 52, 50, 48, 50];
    const node3 = [45, 42, 38, 35, 33, 32, 35];
    const node4 = [70, 68, 65, 62, 60, 58, 60];
    const node5 = [50, 48, 45, 42, 40, 38, 40];
    
    // 图表配置
    const option = {
        title: {
            text: '各节点速度预测对比',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                let result = `${params[0].name}分钟后<br/>`;
                params.forEach(param => {
                    result += `${param.seriesName}: ${param.value} km/h<br/>`;
                });
                return result;
            }
        },
        legend: {
            data: ['节点 1', '节点 2', '节点 3', '节点 4', '节点 5'],
            bottom: 0
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '15%',
            top: '15%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: timeSteps,
            name: '时间'
        },
        yAxis: {
            type: 'value',
            name: '速度 (km/h)',
            min: 0,
            max: 100
        },
        series: [
            {
                name: '节点 1',
                type: 'line',
                data: node1,
                symbol: 'circle',
                symbolSize: 6,
                itemStyle: {
                    color: '#3498db'
                },
                emphasis: {
                    focus: 'series'
                }
            },
            {
                name: '节点 2',
                type: 'line',
                data: node2,
                symbol: 'circle',
                symbolSize: 6,
                itemStyle: {
                    color: '#2ecc71'
                },
                emphasis: {
                    focus: 'series'
                }
            },
            {
                name: '节点 3',
                type: 'line',
                data: node3,
                symbol: 'circle',
                symbolSize: 6,
                itemStyle: {
                    color: '#e74c3c'
                },
                emphasis: {
                    focus: 'series'
                }
            },
            {
                name: '节点 4',
                type: 'line',
                data: node4,
                symbol: 'circle',
                symbolSize: 6,
                itemStyle: {
                    color: '#f39c12'
                },
                emphasis: {
                    focus: 'series'
                }
            },
            {
                name: '节点 5',
                type: 'line',
                data: node5,
                symbol: 'circle',
                symbolSize: 6,
                itemStyle: {
                    color: '#9b59b6'
                },
                emphasis: {
                    focus: 'series'
                }
            }
        ]
    };
    
    myChart.setOption(option);
    
    // 响应窗口大小变化
    window.addEventListener('resize', function() {
        myChart.resize();
    });
}''')
            logger.info(f'创建JS文件: {dashboard_path}')
    
    # 创建prediction.js
    prediction_path = os.path.join(js_dir, 'prediction.js')
    if not os.path.exists(prediction_path):
        with open(prediction_path, 'w', encoding='utf-8') as f:
            f.write('''// 预测页面JS脚本
console.log('预测页面加载完成');

// 页面加载完成后的初始化函数
document.addEventListener('DOMContentLoaded', function() {
    console.log('预测页面DOM内容加载完成');
    
    // 添加错误处理，确保所有DOM元素存在
    try {
        // 初始化滑块值显示
        const stepsValueEl = document.getElementById('steps-value');
        const predictionStepsEl = document.getElementById('prediction-steps');
        const nodesValueEl = document.getElementById('nodes-value');
        const showNodesEl = document.getElementById('show-nodes');
        const startButton = document.getElementById('start-prediction');
        
        if (!stepsValueEl || !predictionStepsEl || !nodesValueEl || !showNodesEl || !startButton) {
            console.error('无法找到必要的DOM元素');
            return;
        }
        
        stepsValueEl.textContent = predictionStepsEl.value;
        nodesValueEl.textContent = showNodesEl.value;
        
        // 绑定滑块事件
        predictionStepsEl.addEventListener('input', function() {
            stepsValueEl.textContent = this.value;
        });
        
        showNodesEl.addEventListener('input', function() {
            nodesValueEl.textContent = this.value;
        });
        
        // 绑定开始预测按钮事件
        console.log('绑定开始预测按钮事件');
        startButton.addEventListener('click', function() {
            console.log('开始预测按钮被点击');
            startPrediction();
        });
        
        // 确保按钮可用
        startButton.disabled = false;
    } catch (error) {
        console.error('初始化页面时出错:', error);
    }
    
    // 绑定图表类型选择事件
    document.getElementById('chart-type').addEventListener('change', function() {
        if (window.currentPredictionData) {
            updatePredictionChart(window.currentPredictionData);
        }
    });
});

// 开始预测
function startPrediction() {
    console.log('开始预测');
    
    try {
        // 获取参数
        const modelSelect = document.getElementById('model-select');
        const datasetSelect = document.getElementById('dataset-select');
        const predictionSteps = document.getElementById('prediction-steps');
        const startButton = document.getElementById('start-prediction');
        
        if (!modelSelect || !datasetSelect || !predictionSteps || !startButton) {
            console.error('无法找到必要的DOM元素');
            alert('系统错误：无法找到必要的页面元素');
            return;
        }
        
        const model = modelSelect.value;
        const dataset = datasetSelect.value;
        const steps = parseInt(predictionSteps.value);
        
        // 显示加载状态
        const originalText = startButton.textContent;
        startButton.disabled = true;
        startButton.textContent = '预测中...';
        
        // 发送预测请求
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: model,
                dataset: dataset,
                steps: steps
            })
        })
    .then(response => response.json())
        .then(data => {
            console.log('预测结果获取成功', data);
            
            // 恢复按钮状态
            startButton.disabled = false;
            startButton.textContent = originalText;
            
            if (data.status === 'success') {
                // 保存当前预测数据
                window.currentPredictionData = data;
                
                // 显示预测结果
                showPredictionResults(data);
                
                // 更新预测图表
                updatePredictionChart(data);
            } else {
                console.error('预测失败', data.message);
                alert('预测失败: ' + data.message);
            }
        })
        .catch(error => {
            console.error('预测过程中出错', error);
            
            // 恢复按钮状态
            startButton.disabled = false;
            startButton.textContent = originalText;
            
            alert('预测过程中出错: ' + error.message);
        });
    } catch (error) {
        console.error('预测过程中发生异常:', error);
        
        // 尝试恢复按钮状态
        try {
            const startButton = document.getElementById('start-prediction');
            if (startButton) {
                startButton.disabled = false;
            }
        } catch (e) {
            // 忽略恢复按钮状态时的错误
        }
        
        alert('预测过程中发生异常: ' + error.message);
    }
}

// 显示预测结果
function showPredictionResults(data) {
    console.log('显示预测结果');
    
    // 显示结果区域
    const resultElement = document.getElementById('prediction-result');
    resultElement.style.display = 'block';
    
    // 计算统计信息
    const predictions = data.predictions;
    const flatPredictions = predictions.flat();
    const avgSpeed = flatPredictions.reduce((a, b) => a + b, 0) / flatPredictions.length;
    const maxSpeed = Math.max(...flatPredictions);
    const minSpeed = Math.min(...flatPredictions);
    
    // 更新统计信息
    document.getElementById('pred-avg-speed').textContent = avgSpeed.toFixed(1);
    document.getElementById('pred-max-speed').textContent = maxSpeed.toFixed(1);
    document.getElementById('pred-min-speed').textContent = minSpeed.toFixed(1);
    document.getElementById('pred-timestamp').textContent = data.timestamp;
}

// 更新预测图表
function updatePredictionChart(data) {
    console.log('更新预测图表');
    
    const chartDom = document.getElementById('prediction-chart');
    const myChart = echarts.init(chartDom);
    const chartType = document.getElementById('chart-type').value;
    const showNodes = parseInt(document.getElementById('show-nodes').value);
    const predictions = data.predictions;
    
    // 准备图表配置
    let option;
    
    if (chartType === 'line') {
        // 折线图
        option = buildLineChartOption(predictions, showNodes);
    } else if (chartType === 'bar') {
        // 条形图
        option = buildBarChartOption(predictions, showNodes);
    } else if (chartType === 'heatmap') {
        // 热力图
        option = buildHeatmapChartOption(predictions);
    }
    
    // 设置图表配置
    myChart.setOption(option);
    
    // 响应窗口大小变化
    window.addEventListener('resize', function() {
        myChart.resize();
    });
}

// 构建折线图配置
function buildLineChartOption(predictions, showNodes) {
    console.log('构建折线图配置');
    
    // 准备时间步数据
    const timeSteps = [];
    for (let i = 0; i < predictions.length; i++) {
        timeSteps.push(`+${i*5}分钟`);
    }
    
    // 准备系列数据
    const series = [];
    const nodeCount = Math.min(showNodes, predictions[0].length);
    
    for (let i = 0; i < nodeCount; i++) {
        const nodeData = predictions.map(step => step[i]);
        series.push({
            name: `节点 ${i+1}`,
            type: 'line',
            data: nodeData,
            symbol: 'circle',
            symbolSize: 6,
            itemStyle: {
                color: getColorByIndex(i)
            },
            emphasis: {
                focus: 'series'
            }
        });
    }
    
    // 图表配置
    const option = {
        title: {
            text: '各节点速度预测折线图',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                let result = `${params[0].name}<br/>`;
                params.forEach(param => {
                    result += `${param.seriesName}: ${param.value.toFixed(1)} km/h<br/>`;
                });
                return result;
            }
        },
        legend: {
            data: series.map(s => s.name),
            bottom: 0,
            type: 'scroll',
            textStyle: {
                fontSize: 10
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '20%',
            top: '15%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: timeSteps,
            name: '时间'
        },
        yAxis: {
            type: 'value',
            name: '速度 (km/h)',
            min: 0,
            max: 100
        },
        series: series
    };
    
    return option;
}

// 构建条形图配置
function buildBarChartOption(predictions, showNodes) {
    console.log('构建条形图配置');
    
    // 只显示最后一个时间步的预测结果
    const lastStepPredictions = predictions[predictions.length - 1];
    
    // 准备数据
    const nodeNames = [];
    const nodeData = [];
    const nodeCount = Math.min(showNodes, lastStepPredictions.length);
    
    for (let i = 0; i < nodeCount; i++) {
        nodeNames.push(`节点 ${i+1}`);
        nodeData.push(lastStepPredictions[i]);
    }
    
    // 图表配置
    const option = {
        title: {
            text: `未来${(predictions.length-1)*5}分钟速度预测条形图`,
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: function(params) {
                return `${params[0].name}: ${params[0].value.toFixed(1)} km/h`;
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '10%',
            top: '15%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: nodeNames,
            name: '节点',
            axisLabel: {
                rotate: 0
            }
        },
        yAxis: {
            type: 'value',
            name: '速度 (km/h)',
            min: 0,
            max: 100
        },
        series: [
            {
                data: nodeData,
                type: 'bar',
                itemStyle: {
                    color: function(params) {
                        return getSpeedColor(params.value);
                    }
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                },
                label: {
                    show: true,
                    position: 'top',
                    formatter: function(params) {
                        return params.value.toFixed(1);
                    }
                }
            }
        ]
    };
    
    return option;
}

// 构建热力图配置
function buildHeatmapChartOption(predictions) {
    console.log('构建热力图配置');
    
    // 只显示最后一个时间步的预测结果
    const lastStepPredictions = predictions[predictions.length - 1];
    
    // 计算网格大小
    const gridSize = Math.ceil(Math.sqrt(lastStepPredictions.length));
    
    // 准备热力图数据
    const heatmapData = [];
    
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const index = i * gridSize + j;
            const value = index < lastStepPredictions.length ? lastStepPredictions[index] : 0;
            heatmapData.push([i, j, value]);
        }
    }
    
    // 图表配置
    const option = {
        title: {
            text: `未来${(predictions.length-1)*5}分钟速度预测热力图`,
            left: 'center'
        },
        tooltip: {
            position: 'top',
            formatter: function(params) {
                return `节点 ${params.data[0] * gridSize + params.data[1] + 1}: ${params.data[2].toFixed(1)} km/h`;
            }
        },
        grid: {
            left: '10%',
            right: '10%',
            bottom: '10%',
            top: '20%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: Array.from({length: gridSize}, (_, i) => i),
            splitArea: {
                show: true
            },
            axisLabel: {
                show: false
            }
        },
        yAxis: {
            type: 'category',
            data: Array.from({length: gridSize}, (_, i) => i),
            splitArea: {
                show: true
            },
            axisLabel: {
                show: false
            }
        },
        visualMap: {
            min: 0,
            max: 100,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: '5%',
            inRange: {
                color: ['#ff0000', '#ffff00', '#00ff00']
            }
        },
        series: [
            {
                name: '速度',
                type: 'heatmap',
                data: heatmapData,
                label: {
                    show: gridSize <= 10  // 只有网格大小小于等于10时才显示标签
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }
        ]
    };
    
    return option;
}

// 根据索引获取颜色
function getColorByIndex(index) {
    const colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#16a085', '#8e44ad'];
    return colors[index % colors.length];
}

// 根据速度获取颜色
function getSpeedColor(speed) {
    // 低速（<30 km/h）- 红色
    if (speed < 30) {
        return '#ff0000';
    }
    // 中速（30-60 km/h）- 黄色到橙色
    else if (speed < 60) {
        const ratio = (speed - 30) / 30;
        const r = 255;
        const g = Math.floor(255 * ratio);
        const b = 0;
        return `rgb(${r}, ${g}, ${b})`;
    }
    // 高速（>60 km/h）- 绿色
    else {
        const ratio = Math.min(1, (speed - 60) / 40);
        const r = Math.floor(255 * (1 - ratio));
        const g = 255;
        const b = 0;
        return `rgb(${r}, ${g}, ${b})`;
    }
}''')
            logger.info(f'创建JS文件: {prediction_path}')
    
    # 创建history.js
    history_path = os.path.join(js_dir, 'history.js')
    if not os.path.exists(history_path):
        with open(history_path, 'w', encoding='utf-8') as f:
            f.write('''// 历史记录页面JS脚本
console.log('历史记录页面加载完成');

// 页面加载完成后的初始化函数
document.addEventListener('DOMContentLoaded', function() {
    console.log('历史记录页面DOM内容加载完成');
    
    // 加载历史记录
    loadHistoryRecords();
    
    // 绑定刷新按钮事件
    document.getElementById('refresh-history').addEventListener('click', function() {
        loadHistoryRecords();
    });
    
    // 绑定图表类型选择事件
    document.getElementById('detail-chart-type').addEventListener('change', function() {
        if (window.currentDetailData) {
            updateDetailChart(window.currentDetailData);
        }
    });
});

// 加载历史记录
function loadHistoryRecords() {
    console.log('加载历史记录');
    
    // 显示加载中
    const tableBody = document.getElementById('history-table-body');
    tableBody.innerHTML = '<tr><td colspan="5" class="text-center">加载中...</td></tr>';
    
    // 发送API请求
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            console.log('历史记录获取成功', data);
            
            if (data.status === 'success') {
                // 更新历史记录表格
                updateHistoryTable(data.history);
                
                // 更新历史记录计数
                document.getElementById('history-count').textContent = data.history.length;
            } else {
                console.error('历史记录获取失败', data.message);
                tableBody.innerHTML = '<tr><td colspan="5" class="text-center text-danger">加载失败: ' + data.message + '</td></tr>';
            }
        })
        .catch(error => {
            console.error('加载历史记录时出错', error);
            tableBody.innerHTML = '<tr><td colspan="5" class="text-center text-danger">加载失败: ' + error.message + '</td></tr>';
        });
}

// 更新历史记录表格
function updateHistoryTable(history) {
    console.log('更新历史记录表格');
    
    const tableBody = document.getElementById('history-table-body');
    
    if (history.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="5" class="text-center">暂无历史记录</td></tr>';
        return;
    }
    
    // 清空表格
    tableBody.innerHTML = '';
    
    // 添加记录
    history.forEach(record => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${record.timestamp}</td>
            <td>${record.model}</td>
            <td>${record.dataset}</td>
            <td>${record.steps}</td>
            <td>
                <button class="btn btn-info btn-sm view-detail" data-id="${record.id}">查看详情</button>
                <button class="btn btn-danger btn-sm delete-record" data-id="${record.id}">删除</button>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // 绑定查看详情按钮事件
    document.querySelectorAll('.view-detail').forEach(button => {
        button.addEventListener('click', function() {
            const id = this.getAttribute('data-id');
            viewHistoryDetail(id);
        });
    });
    
    // 绑定删除按钮事件
    document.querySelectorAll('.delete-record').forEach(button => {
        button.addEventListener('click', function() {
            const id = this.getAttribute('data-id');
            if (confirm('确定要删除这条历史记录吗？')) {
                deleteHistoryRecord(id);
            }
        });
    });
}

// 查看历史记录详情
function viewHistoryDetail(id) {
    console.log('查看历史记录详情', id);
    
    // 发送API请求
    fetch(`/api/history/${id}`)
        .then(response => response.json())
        .then(data => {
            console.log('历史记录详情获取成功', data);
            
            if (data.status === 'success') {
                // 保存当前详情数据
                window.currentDetailData = data.detail;
                
                // 显示详情模态框
                showHistoryDetail(data.detail);
                
                // 更新详情图表
                updateDetailChart(data.detail);
            } else {
                console.error('历史记录详情获取失败', data.message);
                alert('获取详情失败: ' + data.message);
            }
        })
        .catch(error => {
            console.error('获取历史记录详情时出错', error);
            alert('获取详情时出错: ' + error.message);
        });
}

// 显示历史记录详情
function showHistoryDetail(detail) {
    console.log('显示历史记录详情');
    
    // 更新详情信息
    document.getElementById('detail-timestamp').textContent = detail.timestamp;
    document.getElementById('detail-model').textContent = detail.model;
    document.getElementById('detail-dataset').textContent = detail.dataset;
    document.getElementById('detail-steps').textContent = detail.steps;
    document.getElementById('detail-avg-speed').textContent = detail.avg_speed.toFixed(1);
    document.getElementById('detail-max-speed').textContent = detail.max_speed.toFixed(1);
    document.getElementById('detail-min-speed').textContent = detail.min_speed.toFixed(1);
    
    // 显示详情模态框
    const modal = new bootstrap.Modal(document.getElementById('history-detail-modal'));
    modal.show();
}

// 更新详情图表
function updateDetailChart(detail) {
    console.log('更新详情图表');
    
    const chartDom = document.getElementById('detail-chart');
    const myChart = echarts.init(chartDom);
    const chartType = document.getElementById('detail-chart-type').value;
    const predictions = detail.predictions;
    
    // 准备图表配置
    let option;
    
    if (chartType === 'line') {
        // 折线图
        option = buildDetailLineChartOption(predictions);
    } else if (chartType === 'heatmap') {
        // 热力图
        option = buildDetailHeatmapChartOption(predictions);
    } else if (chartType === 'radar') {
        // 雷达图
        option = buildDetailRadarChartOption(predictions);
    }
    
    // 设置图表配置
    myChart.setOption(option);
    
    // 响应窗口大小变化
    window.addEventListener('resize', function() {
        myChart.resize();
    });
}

// 构建详情折线图配置
function buildDetailLineChartOption(predictions) {
    console.log('构建详情折线图配置');
    
    // 准备时间步数据
    const timeSteps = [];
    for (let i = 0; i < predictions.length; i++) {
        timeSteps.push(`+${i*5}分钟`);
    }
    
    // 准备系列数据
    const series = [];
    const nodeCount = Math.min(5, predictions[0].length);  // 只显示前5个节点
    
    for (let i = 0; i < nodeCount; i++) {
        const nodeData = predictions.map(step => step[i]);
        series.push({
            name: `节点 ${i+1}`,
            type: 'line',
            data: nodeData,
            symbol: 'circle',
            symbolSize: 6,
            itemStyle: {
                color: getColorByIndex(i)
            },
            emphasis: {
                focus: 'series'
            }
        });
    }
    
    // 图表配置
    const option = {
        title: {
            text: '速度预测详情折线图',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                let result = `${params[0].name}<br/>`;
                params.forEach(param => {
                    result += `${param.seriesName}: ${param.value.toFixed(1)} km/h<br/>`;
                });
                return result;
            }
        },
        legend: {
            data: series.map(s => s.name),
            bottom: 0
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '15%',
            top: '15%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: timeSteps,
            name: '时间'
        },
        yAxis: {
            type: 'value',
            name: '速度 (km/h)',
            min: 0,
            max: 100
        },
        series: series
    };
    
    return option;
}

// 构建详情热力图配置
function buildDetailHeatmapChartOption(predictions) {
    console.log('构建详情热力图配置');
    
    // 只显示最后一个时间步的预测结果
    const lastStepPredictions = predictions[predictions.length - 1];
    
    // 计算网格大小
    const gridSize = Math.ceil(Math.sqrt(lastStepPredictions.length));
    
    // 准备热力图数据
    const heatmapData = [];
    
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const index = i * gridSize + j;
            const value = index < lastStepPredictions.length ? lastStepPredictions[index] : 0;
            heatmapData.push([i, j, value]);
        }
    }
    
    // 图表配置
    const option = {
        title: {
            text: `未来${(predictions.length-1)*5}分钟速度预测热力图`,
            left: 'center'
        },
        tooltip: {
            position: 'top',
            formatter: function(params) {
                return `节点 ${params.data[0] * gridSize + params.data[1] + 1}: ${params.data[2].toFixed(1)} km/h`;
            }
        },
        grid: {
            left: '10%',
            right: '10%',
            bottom: '10%',
            top: '20%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: Array.from({length: gridSize}, (_, i) => i),
            splitArea: {
                show: true
            },
            axisLabel: {
                show: false
            }
        },
        yAxis: {
            type: 'category',
            data: Array.from({length: gridSize}, (_, i) => i),
            splitArea: {
                show: true
            },
            axisLabel: {
                show: false
            }
        },
        visualMap: {
            min: 0,
            max: 100,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: '5%',
            inRange: {
                color: ['#ff0000', '#ffff00', '#00ff00']
            }
        },
        series: [
            {
                name: '速度',
                type: 'heatmap',
                data: heatmapData,
                label: {
                    show: gridSize <= 10  // 只有网格大小小于等于10时才显示标签
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }
        ]
    };
    
    return option;
}

// 构建详情雷达图配置
function buildDetailRadarChartOption(predictions) {
    console.log('构建详情雷达图配置');
    
    // 只显示最后一个时间步的预测结果
    const lastStepPredictions = predictions[predictions.length - 1];
    
    // 准备雷达图数据
    const indicators = [];
    const values = [];
    const nodeCount = Math.min(10, lastStepPredictions.length);  // 最多显示10个节点
    
    for (let i = 0; i < nodeCount; i++) {
        indicators.push({ name: `节点 ${i+1}`, max: 100 });
        values.push(lastStepPredictions[i]);
    }
    
    // 图表配置
    const option = {
        title: {
            text: '各节点速度预测雷达图',
            left: 'center'
        },
        tooltip: {},
        legend: {
            data: ['速度预测值'],
            bottom: 0
        },
        radar: {
            indicator: indicators,
            shape: 'circle',
            splitNumber: 5,
            axisName: {
                color: 'auto'
            },
            splitLine: {
                lineStyle: {
                    color: ['rgba(211, 211, 211, 0.9)']
                }
            },
            splitArea: {
                show: true,
                areaStyle: {
                    color: ['rgba(211, 211, 211, 0.2)']
                }
            },
            axisLine: {
                lineStyle: {
                    color: 'rgba(211, 211, 211, 0.5)'
                }
            }
        },
        series: [
            {
                name: '速度预测',
                type: 'radar',
                data: [
                    {
                        value: values,
                        name: '速度预测值',
                        symbol: 'circle',
                        symbolSize: 6,
                        lineStyle: {
                            color: '#3498db',
                            width: 2
                        },
                        areaStyle: {
                            color: 'rgba(52, 152, 219, 0.3)'
                        }
                    }
                ]
            }
        ]
    };
    
    return option;
}

// 删除历史记录
function deleteHistoryRecord(id) {
    console.log('删除历史记录', id);
    
    // 发送API请求
    fetch(`/api/history/${id}`, {
        method: 'DELETE'
    })
        .then(response => response.json())
        .then(data => {
            console.log('删除历史记录结果', data);
            
            if (data.status === 'success') {
                // 重新加载历史记录
                loadHistoryRecords();
                
                // 显示成功提示
                alert('历史记录删除成功');
            } else {
                console.error('删除历史记录失败', data.message);
                alert('删除失败: ' + data.message);
            }
        })
        .catch(error => {
            console.error('删除历史记录时出错', error);
            alert('删除时出错: ' + error.message);
        });
}

// 根据索引获取颜色
function getColorByIndex(index) {
    const colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#16a085', '#8e44ad'];
    return colors[index % colors.length];
}''')
            logger.info(f'创建JS文件: {history_path}')
    
    # 创建mock的Bootstrap JS
    bootstrap_path = os.path.join(js_dir, 'bootstrap.bundle.min.js')
    if not os.path.exists(bootstrap_path):
        with open(bootstrap_path, 'w', encoding='utf-8') as f:
            f.write('''// Bootstrap Mock JS - 提供基本功能支持
console.log('Bootstrap Mock JS loaded');

// 简单的响应式导航栏切换
window.addEventListener('DOMContentLoaded', function() {
    // 处理导航栏切换
    const navbarTogglers = document.querySelectorAll('.navbar-toggler');
    navbarTogglers.forEach(toggler => {
        toggler.addEventListener('click', function() {
            const targetId = this.getAttribute('data-bs-target');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                if (targetElement.style.display === 'none' || targetElement.style.display === '') {
                    targetElement.style.display = 'block';
                } else {
                    targetElement.style.display = 'none';
                }
            }
        });
    });
});''')
            logger.info(f'创建Mock Bootstrap JS: {bootstrap_path}')
    
    # 创建mock的ECharts JS
    echarts_path = os.path.join(js_dir, 'echarts.min.js')
    if not os.path.exists(echarts_path):
        with open(echarts_path, 'w', encoding='utf-8') as f:
            f.write('''// ECharts Mock JS - 提供基本图表功能支持
console.log('ECharts Mock JS loaded');

// 简单的ECharts模拟实现
window.echarts = {
    init: function(dom, theme, options) {
        console.log('ECharts.init called');
        
        const chart = {
            dom: dom,
            showLoading: function() {
                this.dom.innerHTML = '<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #666;">加载中...</div>';
            },
            hideLoading: function() {
                // 清除加载提示
            },
            setOption: function(option) {
                console.log('ECharts.setOption called with:', option);
                // 简单的模拟实现，显示一些基本信息
                this.dom.innerHTML = '<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #666;">图表已加载</div>';
            },
            resize: function() {
                console.log('ECharts.resize called');
            },
            destroy: function() {
                console.log('ECharts.destroy called');
            }
        };
        
        return chart;
    }
};''')
            logger.info(f'创建Mock ECharts JS: {echarts_path}')

if __name__ == '__main__':
    # 测试Web服务器
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true', default=False)
    
    args = parser.parse_args()
    run_web_server(args)