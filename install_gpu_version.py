#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorFlow GPU版本安装脚本
该脚本将帮助您安装TensorFlow的GPU版本及其依赖
"""

import os
import sys
import subprocess
import time
import platform

# 配置常量
REQUIREMENTS_FILE = 'requirements-gpu.txt'

# 打印彩色文本函数
def print_color(text, color='white'):
    colors = {
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37'
    }
    color_code = colors.get(color.lower(), '37')
    print(f'\033[{color_code}m{text}\033[0m')

def check_system():
    """检查当前系统环境"""
    print_color("=== 系统环境检查 ===", 'cyan')
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    
    # 检查是否已安装TensorFlow
    try:
        import tensorflow as tf
        print(f"当前安装的TensorFlow版本: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"可用GPU数量: {len(gpus)}")
        if len(gpus) > 0:
            print_color("检测到GPU，当前TensorFlow可能已经配置了GPU支持！", 'green')
        else:
            print_color("未检测到可用GPU，或当前TensorFlow是CPU版本。", 'yellow')
        return True
    except ImportError:
        print_color("未安装TensorFlow。", 'yellow')
        return False

def check_nvidia_driver():
    """检查NVIDIA驱动是否安装"""
    print_color("\n=== NVIDIA驱动检查 ===", 'cyan')
    try:
        if platform.system() == 'Windows':
            # 在Windows上使用nvidia-smi命令检查
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            print(result.stdout)
            return True
        else:
            # 在Linux/Mac上也尝试使用nvidia-smi命令
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            print(result.stdout)
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print_color("未找到NVIDIA驱动或NVIDIA显卡。", 'red')
        return False

def install_tensorflow_gpu():
    """安装TensorFlow GPU版本"""
    print_color("\n=== 开始安装TensorFlow GPU版本 ===", 'cyan')
    
    try:
        # 首先卸载当前的TensorFlow（如果有）
        print("正在卸载当前的TensorFlow...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow', 'tensorflow-intel'],
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 直接安装TensorFlow GPU版本（不通过requirements文件避免编码问题）
        print("正在安装TensorFlow GPU版本...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow[and-cuda]==2.16.1'],
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        # 安装其他依赖
        print("正在安装其他依赖...")
        dependencies = [
            'pandas==2.2.2',
            'numpy==1.26.4',
            'matplotlib==3.8.4',
            'seaborn==0.13.2',
            'scikit-learn==1.4.2',
            'networkx==3.3',
            'flask==3.0.3',
            'fuzzywuzzy==0.18.0',
            'python-Levenshtein==0.23.0'
        ]
        for dep in dependencies:
            print(f"安装 {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep],
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        print_color("TensorFlow GPU版本安装成功！", 'green')
        return True
    except subprocess.CalledProcessError as e:
        print_color(f"安装失败: {e}", 'red')
        print("请查看错误信息并尝试手动安装。")
        return False

def verify_installation():
    """验证TensorFlow GPU安装是否成功"""
    print_color("\n=== 验证TensorFlow GPU安装 ===", 'cyan')
    try:
        import tensorflow as tf
        print(f"TensorFlow版本: {tf.__version__}")
        
        # 检查GPU设备
        gpus = tf.config.list_physical_devices('GPU')
        print(f"可用GPU数量: {len(gpus)}")
        for gpu in gpus:
            print(f"GPU设备: {gpu}")
        
        # 检查CUDA和cuDNN
        print(f"CUDA可用: {tf.test.is_built_with_cuda()}")
        
        if len(gpus) > 0:
            # 简单测试GPU是否可用
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
                b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
                c = tf.matmul(a, b)
                print(f"在GPU上计算结果: {c}")
                print_color("TensorFlow GPU安装成功并可用！", 'green')
        else:
            print_color("未检测到可用GPU，TensorFlow可能仍在使用CPU。", 'yellow')
            print_color("可能的原因:", 'yellow')
            print_color("1. 没有NVIDIA GPU", 'yellow')
            print_color("2. NVIDIA驱动未正确安装", 'yellow')
            print_color("3. CUDA和cuDNN版本与TensorFlow不兼容", 'yellow')
            print_color("4. 需要重启电脑以加载驱动", 'yellow')
    except Exception as e:
        print_color(f"验证失败: {e}", 'red')

def main():
    """主函数"""
    print_color("====================================", 'magenta')
    print_color("     TensorFlow GPU版本安装助手     ", 'magenta')
    print_color("====================================", 'magenta')
    
    # 提示用户需要的前置条件
    print_color("\n重要提示：", 'yellow')
    print("1. 安装TensorFlow GPU版本前，需要先安装:")
    print("   - NVIDIA显卡驱动（建议最新版本）")
    print("   - CUDA Toolkit 12.3")
    print("   - cuDNN 8.9")
    print("2. 本脚本将自动卸载当前的TensorFlow并安装GPU版本")
    
    # 等待用户确认
    user_input = input("\n是否继续安装？(y/n): ")
    if user_input.lower() != 'y':
        print_color("安装已取消。", 'yellow')
        return 0
    
    # 检查系统环境
    check_system()
    
    # 检查NVIDIA驱动
    if not check_nvidia_driver():
        user_input = input("\n未检测到NVIDIA驱动，是否继续安装？(y/n): ")
        if user_input.lower() != 'y':
            print_color("安装已取消。", 'yellow')
            return 0
    
    # 安装TensorFlow GPU版本
    if install_tensorflow_gpu():
        # 验证安装
        verify_installation()
    
    print_color("\n=== 安装脚本执行完成 ===", 'cyan')
    print("请参考以下建议:")
    print("1. 如果已安装CUDA和cuDNN但仍无法使用GPU，请检查版本兼容性")
    print("2. 尝试重启电脑以确保驱动和库正确加载")
    print("3. 如果遇到CUDA错误，请尝试安装与您CUDA版本兼容的TensorFlow版本")
    return 0

if __name__ == '__main__':
    sys.exit(main())