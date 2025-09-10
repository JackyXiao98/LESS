#!/usr/bin/env python3
"""
验证脚本：检查梯度与ID映射的正确性

该脚本用于验证修改后的代码是否正确地保存了梯度与数据样本ID的映射关系。
包括检查文件完整性、数量一致性和顺序正确性。

使用方法:
    python validate_gradient_id_mapping.py --output_dir /path/to/output --check_aggregated
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def validate_single_rank_files(rank_dir: str, rank: int) -> Tuple[bool, Dict]:
    """
    验证单个rank目录下的文件
    
    Args:
        rank_dir: rank目录路径
        rank: rank编号
        
    Returns:
        (is_valid, info_dict): 验证结果和信息字典
    """
    info = {
        'rank': rank,
        'grad_files': [],
        'id_files': [],
        'total_samples': 0,
        'errors': []
    }
    
    if not os.path.exists(rank_dir):
        info['errors'].append(f"目录不存在: {rank_dir}")
        return False, info
    
    # 获取所有梯度文件和ID文件
    files = os.listdir(rank_dir)
    grad_files = [f for f in files if f.startswith('grads-') and f.endswith('.pt')]
    id_files = [f for f in files if f.startswith('ids-') and f.endswith('.pkl')]
    
    # 排序文件
    grad_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
    id_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
    
    info['grad_files'] = grad_files
    info['id_files'] = id_files
    
    # 检查文件数量是否匹配
    if len(grad_files) != len(id_files):
        info['errors'].append(f"梯度文件数量 ({len(grad_files)}) 与ID文件数量 ({len(id_files)}) 不匹配")
        return False, info
    
    # 检查每对文件
    total_samples = 0
    for grad_file, id_file in zip(grad_files, id_files):
        # 检查文件名编号是否匹配
        grad_num = int(grad_file.split('-')[1].split('.')[0])
        id_num = int(id_file.split('-')[1].split('.')[0])
        
        if grad_num != id_num:
            info['errors'].append(f"文件编号不匹配: {grad_file} vs {id_file}")
            return False, info
        
        # 加载并检查内容
        try:
            grad_path = os.path.join(rank_dir, grad_file)
            id_path = os.path.join(rank_dir, id_file)
            
            gradients = torch.load(grad_path, map_location='cpu')
            with open(id_path, 'rb') as f:
                ids = pickle.load(f)
            
            # 检查样本数量是否匹配
            if gradients.shape[0] != len(ids):
                info['errors'].append(f"文件 {grad_file}: 梯度样本数 ({gradients.shape[0]}) 与ID数量 ({len(ids)}) 不匹配")
                return False, info
            
            total_samples += gradients.shape[0]
            
        except Exception as e:
            info['errors'].append(f"加载文件失败 {grad_file}/{id_file}: {e}")
            return False, info
    
    info['total_samples'] = total_samples
    
    # 检查合并文件
    all_orig_path = os.path.join(rank_dir, 'all_orig.pt')
    all_ids_path = os.path.join(rank_dir, 'all_ids.pkl')
    
    if os.path.exists(all_orig_path) and os.path.exists(all_ids_path):
        try:
            all_gradients = torch.load(all_orig_path, map_location='cpu')
            with open(all_ids_path, 'rb') as f:
                all_ids = pickle.load(f)
            
            # 检查合并文件的样本数量
            if all_gradients.shape[0] != len(all_ids):
                info['errors'].append(f"合并文件: 梯度样本数 ({all_gradients.shape[0]}) 与ID数量 ({len(all_ids)}) 不匹配")
                return False, info
            
            # 检查合并文件的总样本数是否正确
            if all_gradients.shape[0] != total_samples:
                info['errors'].append(f"合并文件样本数 ({all_gradients.shape[0]}) 与分片文件总数 ({total_samples}) 不匹配")
                return False, info
            
            info['merged_samples'] = all_gradients.shape[0]
            
        except Exception as e:
            info['errors'].append(f"加载合并文件失败: {e}")
            return False, info
    
    return len(info['errors']) == 0, info


def validate_aggregated_files(output_dir: str) -> Tuple[bool, Dict]:
    """
    验证聚合后的最终文件
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        (is_valid, info_dict): 验证结果和信息字典
    """
    info = {
        'final_grad_file': None,
        'final_id_file': None,
        'total_samples': 0,
        'grad_shape': None,
        'errors': []
    }
    
    final_grad_path = os.path.join(output_dir, 'all_orig.pt')
    final_id_path = os.path.join(output_dir, 'all_ids.pkl')
    
    info['final_grad_file'] = final_grad_path
    info['final_id_file'] = final_id_path
    
    # 检查文件是否存在
    if not os.path.exists(final_grad_path):
        info['errors'].append(f"最终梯度文件不存在: {final_grad_path}")
        return False, info
    
    if not os.path.exists(final_id_path):
        info['errors'].append(f"最终ID文件不存在: {final_id_path}")
        return False, info
    
    # 加载并验证文件
    try:
        final_gradients = torch.load(final_grad_path, map_location='cpu')
        with open(final_id_path, 'rb') as f:
            final_ids = pickle.load(f)
        
        info['grad_shape'] = final_gradients.shape
        info['total_samples'] = len(final_ids)
        
        # 检查样本数量是否匹配
        if final_gradients.shape[0] != len(final_ids):
            info['errors'].append(f"最终文件: 梯度样本数 ({final_gradients.shape[0]}) 与ID数量 ({len(final_ids)}) 不匹配")
            return False, info
        
        # 检查ID的唯一性（可选）
        unique_ids = set(final_ids)
        if len(unique_ids) != len(final_ids):
            duplicate_count = len(final_ids) - len(unique_ids)
            info['errors'].append(f"发现 {duplicate_count} 个重复的ID")
            # 这可能不是错误，取决于数据集的特性
        
    except Exception as e:
        info['errors'].append(f"加载最终文件失败: {e}")
        return False, info
    
    return len(info['errors']) == 0, info


def main():
    parser = argparse.ArgumentParser(description='验证梯度与ID映射的正确性')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径')
    parser.add_argument('--check_ranks', action='store_true',
                        help='检查各个rank的文件')
    parser.add_argument('--check_aggregated', action='store_true',
                        help='检查聚合后的最终文件')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='GPU数量，默认8')
    parser.add_argument('--dim', type=int, default=8192,
                        help='梯度维度，默认8192')
    
    args = parser.parse_args()
    
    if not args.check_ranks and not args.check_aggregated:
        print("请指定 --check_ranks 或 --check_aggregated 或两者都指定")
        return
    
    print("开始验证梯度与ID映射...")
    print(f"输出目录: {args.output_dir}")
    
    all_valid = True
    
    # 检查各个rank的文件
    if args.check_ranks:
        print(f"\n=== 检查各个rank的文件 ===")
        
        for rank in range(args.num_gpus):
            rank_dir = os.path.join(args.output_dir, f"rank_{rank}", f"dim{args.dim}")
            print(f"\n检查 Rank {rank}: {rank_dir}")
            
            is_valid, info = validate_single_rank_files(rank_dir, rank)
            
            if is_valid:
                print(f"✓ Rank {rank} 验证通过")
                print(f"  梯度文件数: {len(info['grad_files'])}")
                print(f"  ID文件数: {len(info['id_files'])}")
                print(f"  总样本数: {info['total_samples']}")
                if 'merged_samples' in info:
                    print(f"  合并文件样本数: {info['merged_samples']}")
            else:
                print(f"✗ Rank {rank} 验证失败")
                for error in info['errors']:
                    print(f"  错误: {error}")
                all_valid = False
    
    # 检查聚合后的文件
    if args.check_aggregated:
        print(f"\n=== 检查聚合后的最终文件 ===")
        
        final_output_dir = os.path.join(args.output_dir, f"dim{args.dim}")
        print(f"检查最终文件: {final_output_dir}")
        
        is_valid, info = validate_aggregated_files(final_output_dir)
        
        if is_valid:
            print("✓ 最终文件验证通过")
            print(f"  梯度文件: {info['final_grad_file']}")
            print(f"  ID文件: {info['final_id_file']}")
            print(f"  总样本数: {info['total_samples']}")
            print(f"  梯度形状: {info['grad_shape']}")
        else:
            print("✗ 最终文件验证失败")
            for error in info['errors']:
                print(f"  错误: {error}")
            all_valid = False
    
    # 总结
    print(f"\n=== 验证总结 ===")
    if all_valid:
        print("✓ 所有验证都通过！梯度与ID映射正确。")
    else:
        print("✗ 发现验证错误，请检查上述错误信息。")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())