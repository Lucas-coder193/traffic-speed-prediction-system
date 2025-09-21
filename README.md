# traffic-speed-prediction-system

## 项目简介

This project develops an intelligent traffic speed prediction system using a hybrid GCN-GRU-Fuzzy model to capture spatial-temporal traffic patterns. Featuring a Flask-based interface, it enables dataset selection, parameter configuration, and real-time result visualization, supporting efficient traffic management.

这是一个基于深度学习的交通车辆速度预测应用系统，结合了图卷积网络（GCN）、门控循环单元（GRU）和模糊推理机制，能够预测未来某一时段交通路段上的车辆速度。系统不仅考虑了交通数据的时序性和空间性，还融入了天气、节假日、时间段等外部因素，提高了预测的准确性和可解释性。

## 系统架构

系统包含以下核心模块：

1. **数据预处理模块**：负责数据加载、清洗、特征工程和标准化等工作
2. **模糊推理机制**：处理外部属性（天气、节假日、时间段等）对交通速度的影响
3. **深度学习模型**：基于GCN+GRU的时空预测模型，融合模糊推理结果
4. **模型训练与评估**：负责模型的训练、调优和性能评估
5. **Web可视化界面**：提供用户友好的界面展示预测结果和历史记录

## 安装指南

### 前提条件

- Python 3.7+（推荐3.10+）
- pip包管理器

### 安装步骤

1. 克隆或下载项目代码到本地

2. 安装项目依赖：

```bash
pip install -r requirements.txt
```

## 使用说明

### 命令行模式

系统支持以下几种运行模式，可以通过命令行参数指定：

1. **数据预处理模式**

```bash
python main.py preprocess
```

该模式会处理原始数据，生成训练所需的特征数据。

2. **模型训练模式**

```bash
python main.py train
```

该模式会使用预处理后的数据训练模型，并保存训练好的模型。

3. **预测模式**

```bash
python main.py predict
```

该模式会使用训练好的模型进行交通速度预测。

4. **评估模式**

```bash
python main.py evaluate
```

该模式会评估模型的性能，并生成评估报告。

5. **Web服务模式**

```bash
python main.py web
```

该模式会启动Web服务器，用户可以通过浏览器访问系统界面。

### Web界面使用

启动Web服务后，可以通过浏览器访问 `http://localhost:5000` 进入系统首页。

系统界面包含以下几个主要部分：

- **首页**：系统介绍和功能概览
- **仪表盘**：实时监控和分析交通状况
- **预测**：输入参数进行交通速度预测
- **历史记录**：查看和管理过去的预测结果

## 数据说明

### 模拟数据集
系统内置了两种模拟数据集：

1. **深圳北站出租车数据集（SZBZ）**
2. **美国洛杉矶高速公路环路检测器数据集（Los-loop）**

如果没有提供实际数据，系统会自动生成模拟数据用于演示。

### 使用真实数据集

#### 1. 准备真实数据集

您需要准备一个CSV格式的交通数据集，包含以下字段：

| 字段名 | 类型 | 描述 |
|-------|------|------|
| timestamp | datetime | 时间戳，例如：2023-01-01 00:00:00 |
| node_id | string/int | 道路节点ID |
| speed | float | 车辆速度 |
| flow | float | 交通流量（可选） |
| congestion_index | float | 拥堵指数（可选） |
| hour | int | 小时（可选，如果不提供会自动从timestamp计算） |
| day_of_week | int | 星期几（可选，如果不提供会自动从timestamp计算） |
| is_weekend | int | 是否周末（可选，如果不提供会自动计算） |
| is_holiday | int | 是否节假日（可选） |
| weather | string | 天气状况（可选） |
| temperature | float | 温度（可选） |

#### 2. 准备邻接矩阵

您需要准备一个表示道路网络拓扑结构的邻接矩阵，可以保存为NumPy的.npy格式文件。

邻接矩阵是一个[num_nodes × num_nodes]的矩阵，其中num_nodes是道路节点的数量。矩阵中的值表示节点之间的连接关系，通常为0或1，或者表示连接强度的权重。

#### 3. 配置系统

使用生成的示例配置文件作为模板，创建您自己的配置文件：

```bash
python data_config.py
```

这将生成一个名为`data_config_example.json`的示例配置文件。复制此文件并重命名为`data_config.json`，然后根据您的数据集修改配置：

```json
{
  "real_data_config": {
    "enabled": true,
    "data_file": "data/real_traffic_data.csv",
    "adj_matrix_file": "data/real_adj_matrix.npy",
    "has_header": true,
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
```

- `enabled`: 设置为true表示启用真实数据集
- `data_file`: 真实数据集文件路径
- `adj_matrix_file`: 邻接矩阵文件路径
- `has_header`: 数据文件是否有表头
- `delimiter`: 数据文件的分隔符
- `field_mapping`: 字段映射，将您的数据集中的字段名映射到系统期望的字段名
- `time_format`: 时间戳格式
- `weather_mapping`: 天气状况映射，将文本表示的天气映射为数字编码

#### 4. 使用真实数据集运行系统

使用以下命令运行系统，指定使用真实数据集并提供配置文件路径：

```bash
# 数据预处理
python main.py --mode preprocess --dataset real --config_file data_config.json

# 模型训练
python main.py --mode train --dataset real --model GCN_GRU --epochs 100 --batch_size 32 --learning_rate 0.001 --config_file data_config.json

# 模型预测
python main.py --mode predict --dataset real --model GCN_GRU --config_file data_config.json

# 模型评估
python main.py --mode evaluate --dataset real --model GCN_GRU --config_file data_config.json
```

## 模型说明

系统采用的深度学习模型主要包含以下几个部分：

1. **图卷积网络（GCN）**：捕获道路网络的空间相关性
2. **门控循环单元（GRU）**：捕获交通数据的时间依赖性
3. **模糊推理机制**：处理外部属性对交通速度的影响
4. **融合层**：将时空特征与外部属性影响值融合，输出预测结果

## 超参数设置

模型的主要超参数包括：

- 学习率：0.001
- 优化器：Adam
- 损失函数：均方误差（MSE）
- 批量大小：32
- 时间步长：12
- GRU隐藏单元数：64

## 注意事项

1. 首次运行时，系统会自动创建必要的目录和生成模拟数据
2. 如需使用实际数据，请将数据放置在 `data/raw` 目录下
3. 训练好的模型将保存在 `models` 目录下
4. 预测结果和评估报告将保存在 `results` 目录下

## 系统要求

- 推荐使用具有GPU支持的环境以加速模型训练
- 浏览器支持：Chrome 90+、Firefox 88+、Safari 14+、Edge 90+
- 分辨率：1024x768及以上分辨率

## License

[MIT License](LICENSE)
