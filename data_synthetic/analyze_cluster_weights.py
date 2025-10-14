#!/usr/bin/env python3
"""
分析cluster权重文件，了解每个cluster的权重分布
"""

import pickle
import numpy as np
import pandas as pd

def analyze_cluster_weights(weights_file):
    """分析cluster权重文件"""
    print(f"正在分析权重文件: {weights_file}")
    
    # 加载权重文件
    with open(weights_file, 'rb') as f:
        data = pickle.load(f)
    
    print("\n=== 权重文件结构 ===")
    print(f"数据类型: {type(data)}")
    print(f"主要键: {list(data.keys())}")
    
    # 分析cluster权重
    if 'cluster_weights' in data:
        cluster_weights = data['cluster_weights']
        print(f"\n=== Cluster权重信息 ===")
        print(f"Cluster数量: {len(cluster_weights)}")
        
        # 显示每个cluster的权重
        total_weight = sum(cluster_weights.values())
        print(f"总权重: {total_weight}")
        
        print("\n各cluster权重详情:")
        for cluster_id, weight in sorted(cluster_weights.items()):
            percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
            print(f"  {cluster_id}: {weight:.6f} ({percentage:.2f}%)")
    
    # 分析cluster信息
    if 'cluster_info' in data:
        cluster_info = data['cluster_info']
        print(f"\n=== Cluster信息 ===")
        print(f"Cluster信息数量: {len(cluster_info)}")
        
        total_samples = 0
        print("\n各cluster样本数量:")
        for cluster_idx, info in sorted(cluster_info.items()):
            size = info.get('size', 0)
            total_samples += size
            print(f"  Cluster {cluster_idx:2d}: {size:5d} 样本 (文件: {info.get('file', 'N/A')})")
        
        print(f"\n总样本数: {total_samples}")
    
    return data

if __name__ == "__main__":
    weights_file = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/mixing_results_all/cluster_mixing_weights.pkl"
    data = analyze_cluster_weights(weights_file)