#!/usr/bin/env python3
"""
根据cluster权重进行加权采样，生成5000条样本的新CSV文件
"""

import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path

def load_cluster_weights(weights_file):
    """加载cluster权重"""
    with open(weights_file, 'rb') as f:
        data = pickle.load(f)
    return data['cluster_weights'], data['cluster_info']

def calculate_sample_counts(cluster_weights, total_samples=5000):
    """根据权重计算每个cluster应该采样的数量"""
    sample_counts = {}
    
    # 过滤掉权重为0或负数的cluster
    valid_weights = {k: max(0, v) for k, v in cluster_weights.items() if v > 0}
    
    if not valid_weights:
        raise ValueError("没有有效的cluster权重")
    
    # 归一化权重
    total_weight = sum(valid_weights.values())
    normalized_weights = {k: v / total_weight for k, v in valid_weights.items()}
    
    # 计算每个cluster的采样数量
    allocated_samples = 0
    for cluster_id, weight in normalized_weights.items():
        count = int(weight * total_samples)
        sample_counts[cluster_id] = count
        allocated_samples += count
    
    # 处理由于四舍五入导致的差异
    remaining = total_samples - allocated_samples
    if remaining > 0:
        # 将剩余样本分配给权重最高的cluster
        sorted_clusters = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)
        for i in range(remaining):
            cluster_id = sorted_clusters[i % len(sorted_clusters)][0]
            sample_counts[cluster_id] += 1
    
    return sample_counts, normalized_weights

def load_cluster_indices(cluster_file):
    """加载cluster文件中的indices"""
    with open(cluster_file, 'rb') as f:
        data = pickle.load(f)
    return data['indices']

def sample_from_clusters(cluster_dir, sample_counts, seed=42):
    """从每个cluster中采样指定数量的样本"""
    np.random.seed(seed)
    sampled_indices = []
    
    print("正在从各cluster采样...")
    for cluster_id, count in sample_counts.items():
        if count == 0:
            continue
            
        # 构建cluster文件路径
        cluster_num = int(cluster_id.split('_')[1])
        cluster_file = os.path.join(cluster_dir, f"cluster_{cluster_num:02d}.pkl")
        
        if not os.path.exists(cluster_file):
            print(f"警告: cluster文件不存在: {cluster_file}")
            continue
        
        # 加载cluster的indices
        indices = load_cluster_indices(cluster_file)
        
        if len(indices) < count:
            print(f"警告: {cluster_id} 只有 {len(indices)} 个样本，但需要 {count} 个")
            sampled = indices  # 取所有可用的样本
        else:
            # 随机采样
            sampled = np.random.choice(indices, size=count, replace=False)
        
        sampled_indices.extend(sampled)
        print(f"  {cluster_id}: 采样 {len(sampled)} 个样本 (权重要求: {count})")
    
    return np.array(sampled_indices)

def extract_data_by_indices(csv_file, indices):
    """根据indices从原始CSV文件中提取数据"""
    print(f"正在从 {csv_file} 提取数据...")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 确保indices在有效范围内
    valid_indices = indices[indices < len(df)]
    if len(valid_indices) < len(indices):
        print(f"警告: {len(indices) - len(valid_indices)} 个索引超出范围")
    
    # 提取对应的行
    sampled_data = df.iloc[valid_indices].copy()
    
    # 添加原始索引列
    sampled_data['original_index'] = valid_indices
    
    return sampled_data

def main():
    # 文件路径
    weights_file = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/mixing_results_all/cluster_mixing_weights.pkl"
    cluster_dir = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/data_process/cluster_embeddings"
    original_csv = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/data_process/raw/yelp_huggingface_gpt2_secpe_10_600.csv"
    output_csv = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/weighted_sampled_5000.csv"
    
    # 1. 加载权重
    print("=== 步骤1: 加载cluster权重 ===")
    cluster_weights, cluster_info = load_cluster_weights(weights_file)
    
    # 2. 计算采样数量
    print("\n=== 步骤2: 计算采样数量 ===")
    sample_counts, normalized_weights = calculate_sample_counts(cluster_weights, total_samples=5000)
    
    print("各cluster采样数量:")
    total_planned = 0
    for cluster_id, count in sorted(sample_counts.items()):
        weight = normalized_weights[cluster_id]
        total_planned += count
        print(f"  {cluster_id}: {count:4d} 样本 (权重: {weight:.4f})")
    print(f"总计划采样: {total_planned}")
    
    # 3. 从clusters采样
    print("\n=== 步骤3: 从clusters采样 ===")
    sampled_indices = sample_from_clusters(cluster_dir, sample_counts)
    print(f"实际采样总数: {len(sampled_indices)}")
    
    # 4. 提取原始数据
    print("\n=== 步骤4: 提取原始数据 ===")
    sampled_data = extract_data_by_indices(original_csv, sampled_indices)
    
    # 5. 保存结果
    print("\n=== 步骤5: 保存结果 ===")
    sampled_data.to_csv(output_csv, index=False)
    print(f"结果已保存到: {output_csv}")
    print(f"最终样本数: {len(sampled_data)}")
    
    # 6. 显示统计信息
    print("\n=== 统计信息 ===")
    print(f"原始数据行数: {pd.read_csv(original_csv).shape[0]}")
    print(f"采样数据行数: {len(sampled_data)}")
    print(f"采样比例: {len(sampled_data) / pd.read_csv(original_csv).shape[0] * 100:.2f}%")
    
    # 显示数据预览
    print("\n=== 数据预览 ===")
    print(sampled_data.head())
    
    return sampled_data

if __name__ == "__main__":
    sampled_data = main()