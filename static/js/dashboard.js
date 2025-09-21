// 仪表盘页面JS脚本
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
}