// 历史记录页面JS脚本
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
}