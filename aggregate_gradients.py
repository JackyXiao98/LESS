#!/usr/bin/env python3
"""
聚合脚本：将所有GPU的梯度和ID文件合并成最终的单一文件

该脚本用于将分布式训练中各个GPU生成的梯度文件和对应的ID文件合并成最终的统一文件。
支持自定义路径模板和输出目录。

使用方法:
    python aggregate_gradients.py --base_path /path/to/grads --experiment_name llama2-7b-p0.05-lora-seed3 --checkpoint_name dolly-ckpt212-adam --num_gpus 8 --dim 8192
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm


def aggregate_gradients_and_ids(
    base_path: str,
    experiment_name: str,
    checkpoint_name: str,
    num_gpus: int = 8,
    dim: int = 8192,
    output_dir: str = None
) -> None:
    """
    聚合所有GPU的梯度和ID文件
    
    Args:
        base_path: 基础路径，例如 "/mnt/bn/pilab0/yt/github/grads"
        experiment_name: 实验名称，例如 "llama2-7b-p0.05-lora-seed3"
        checkpoint_name: 检查点名称，例如 "dolly-ckpt212-adam"
        num_gpus: GPU数量，默认8
        dim: 梯度维度，默认8192
        output_dir: 输出目录，如果为None则自动生成
    """
    
    # 构建路径模板
    if output_dir is None:
        output_dir = os.path.join(base_path, experiment_name, checkpoint_name, f"dim{dim}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始聚合 {num_gpus} 个GPU的梯度和ID文件...")
    print(f"输出目录: {output_dir}")
    
    # 初始化存储列表
    all_gradients = []
    all_ids = []
    
    # 循环处理每个GPU的结果
    for rank in tqdm(range(num_gpus), desc="处理GPU"):
        # 构建当前rank的路径
        rank_dir = os.path.join(base_path, experiment_name, checkpoint_name, f"rank_{rank}", f"dim{dim}")
        
        # 检查路径是否存在
        if not os.path.exists(rank_dir):
            print(f"警告: 路径不存在 {rank_dir}")
            continue
        
        # 加载梯度文件
        grad_file = os.path.join(rank_dir, "all_orig.pt")
        if os.path.exists(grad_file):
            try:
                gradients = torch.load(grad_file, map_location='cpu')
                all_gradients.append(gradients)
                print(f"Rank {rank}: 加载梯度文件 {grad_file}, 形状: {gradients.shape}")
            except Exception as e:
                print(f"错误: 无法加载梯度文件 {grad_file}: {e}")
                continue
        else:
            print(f"警告: 梯度文件不存在 {grad_file}")
            continue
        
        # 加载ID文件
        ids_file = os.path.join(rank_dir, "all_ids.pkl")
        if os.path.exists(ids_file):
            try:
                with open(ids_file, 'rb') as f:
                    ids = pickle.load(f)
                all_ids.extend(ids)
                print(f"Rank {rank}: 加载ID文件 {ids_file}, 数量: {len(ids)}")
            except Exception as e:
                print(f"错误: 无法加载ID文件 {ids_file}: {e}")
                # 如果ID文件加载失败，移除对应的梯度
                if all_gradients:
                    all_gradients.pop()
                continue
        else:
            print(f"警告: ID文件不存在 {ids_file}")
            # 如果没有ID文件，为对应的梯度生成默认ID
            if len(all_gradients) > len(all_ids):
                num_samples = gradients.shape[0]
                default_ids = [f"rank_{rank}_sample_{i}" for i in range(num_samples)]
                all_ids.extend(default_ids)
                print(f"Rank {rank}: 生成默认ID，数量: {len(default_ids)}")
    
    # 检查是否有数据需要合并
    if not all_gradients:
        print("错误: 没有找到任何梯度文件进行合并")
        return
    
    # 合并所有梯度
    print("正在合并梯度张量...")
    try:
        final_gradients = torch.cat(all_gradients, dim=0)
        print(f"合并后的梯度形状: {final_gradients.shape}")
    except Exception as e:
        print(f"错误: 无法合并梯度张量: {e}")
        return
    
    # 验证梯度和ID数量是否匹配
    if len(all_ids) != final_gradients.shape[0]:
        print(f"警告: ID数量 ({len(all_ids)}) 与梯度样本数量 ({final_gradients.shape[0]}) 不匹配")
        # 调整ID列表长度
        if len(all_ids) > final_gradients.shape[0]:
            all_ids = all_ids[:final_gradients.shape[0]]
            print(f"截断ID列表到 {len(all_ids)} 个")
        else:
            # 补充缺失的ID
            missing_count = final_gradients.shape[0] - len(all_ids)
            for i in range(missing_count):
                all_ids.append(f"missing_id_{i}")
            print(f"补充 {missing_count} 个缺失的ID")
    
    # 保存最终的梯度文件
    final_grad_file = os.path.join(output_dir, "all_orig.pt")
    try:
        torch.save(final_gradients, final_grad_file)
        print(f"成功保存最终梯度文件: {final_grad_file}")
        print(f"最终梯度形状: {final_gradients.shape}")
    except Exception as e:
        print(f"错误: 无法保存梯度文件 {final_grad_file}: {e}")
        return
    
    # 保存最终的ID文件
    final_ids_file = os.path.join(output_dir, "all_ids.pkl")
    try:
        with open(final_ids_file, 'wb') as f:
            pickle.dump(all_ids, f)
        print(f"成功保存最终ID文件: {final_ids_file}")
        print(f"最终ID数量: {len(all_ids)}")
    except Exception as e:
        print(f"错误: 无法保存ID文件 {final_ids_file}: {e}")
        return
    
    print("聚合完成!")
    print(f"最终文件:")
    print(f"  梯度文件: {final_grad_file}")
    print(f"  ID文件: {final_ids_file}")
    print(f"  样本总数: {final_gradients.shape[0]}")
    print(f"  梯度维度: {final_gradients.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description='聚合分布式训练的梯度和ID文件')
    
    parser.add_argument('--base_path', type=str, required=True,
                        help='基础路径，例如 /mnt/bn/pilab0/yt/github/grads')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='实验名称，例如 llama2-7b-p0.05-lora-seed3')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                        help='检查点名称，例如 dolly-ckpt212-adam')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='GPU数量，默认8')
    parser.add_argument('--dim', type=int, default=8192,
                        help='梯度投影维度，默认8192')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，如果不指定则自动生成')
    
    args = parser.parse_args()
    
    # 验证基础路径是否存在
    if not os.path.exists(args.base_path):
        print(f"错误: 基础路径不存在: {args.base_path}")
        return
    
    # 执行聚合
    aggregate_gradients_and_ids(
        base_path=args.base_path,
        experiment_name=args.experiment_name,
        checkpoint_name=args.checkpoint_name,
        num_gpus=args.num_gpus,
        dim=args.dim,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()