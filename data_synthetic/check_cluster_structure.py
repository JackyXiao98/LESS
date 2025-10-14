#!/usr/bin/env python3
"""
检查cluster文件结构，了解ID存储格式
"""

import pickle
import numpy as np

def check_cluster_structure(cluster_file):
    """检查cluster文件结构"""
    print(f"正在检查cluster文件: {cluster_file}")
    
    with open(cluster_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\n=== 文件结构 ===")
    print(f"数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"字典键: {list(data.keys())}")
        
        for key, value in data.items():
            print(f"\n{key}:")
            print(f"  类型: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"  形状: {value.shape}")
            elif hasattr(value, '__len__'):
                print(f"  长度: {len(value)}")
            
            # 显示前几个元素
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                print(f"  前5个元素: {value[:5]}")
    
    elif isinstance(data, (list, np.ndarray)):
        print(f"数组长度: {len(data)}")
        print(f"前5个元素: {data[:5]}")
    
    return data

if __name__ == "__main__":
    # 检查几个不同的cluster文件
    cluster_files = [
        "/Users/bytedance/Desktop/Github/LESS/data_synthetic/data_process/cluster_embeddings/cluster_03.pkl",  # 权重最高的cluster
        "/Users/bytedance/Desktop/Github/LESS/data_synthetic/data_process/cluster_embeddings/cluster_15.pkl",  # 权重第二高的cluster
        "/Users/bytedance/Desktop/Github/LESS/data_synthetic/data_process/cluster_embeddings/cluster_00.pkl"   # 权重为0的cluster
    ]
    
    for cluster_file in cluster_files:
        data = check_cluster_structure(cluster_file)
        print("\n" + "="*50 + "\n")