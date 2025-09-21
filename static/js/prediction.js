// 预测页面JS脚本
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
}