#!/usr/bin/env python3
"""
专门验证dolly数据集ID提取的简化脚本

该脚本专门用于验证dolly数据集中ID的提取逻辑，
确保从"dolly_125"这样的ID中正确提取数字部分。

使用方法:
    python validate_dolly_id.py --data_file ../data/train/processed/dolly/dolly_data.jsonl --max_samples 100
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from less.data_selection.get_training_dataset import get_training_dataset


def setup_distributed():
    """设置分布式环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # 初始化分布式进程组
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        # 设置CUDA设备
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        return rank, world_size, local_rank, device
    else:
        return 0, 1, 0, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def extract_dolly_id_number(dolly_id: str) -> int:
    """
    从dolly ID中提取数字部分
    
    Args:
        dolly_id: dolly ID字符串，例如 "dolly_125"
        
    Returns:
        提取的数字，例如 125
    """
    if isinstance(dolly_id, str) and dolly_id.startswith("dolly_"):
        match = re.search(r'dolly_(\d+)', dolly_id)
        if match:
            return int(match.group(1))
    return -1


def load_and_analyze_dolly_data(data_file: str, max_samples: int = None) -> Dict:
    """
    加载并分析dolly数据
    
    Args:
        data_file: 数据文件路径
        max_samples: 最大样本数
        
    Returns:
        分析结果字典
    """
    print(f"加载dolly数据: {data_file}")
    
    original_data = []
    id_analysis = {
        'total_samples': 0,
        'has_id_field': 0,
        'dolly_id_format': 0,
        'other_id_format': 0,
        'no_id_field': 0,
        'id_examples': [],
        'id_numbers': []
    }
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    sample = json.loads(line.strip())
                    original_data.append(sample)
                    
                    id_analysis['total_samples'] += 1
                    
                    if 'id' in sample:
                        id_analysis['has_id_field'] += 1
                        sample_id = sample['id']
                        
                        # 保存前10个ID作为示例
                        if len(id_analysis['id_examples']) < 10:
                            id_analysis['id_examples'].append(sample_id)
                        
                        # 分析ID格式
                        if isinstance(sample_id, str) and sample_id.startswith("dolly_"):
                            id_analysis['dolly_id_format'] += 1
                            id_number = extract_dolly_id_number(sample_id)
                            if id_number >= 0:
                                id_analysis['id_numbers'].append(id_number)
                        else:
                            id_analysis['other_id_format'] += 1
                    else:
                        id_analysis['no_id_field'] += 1
                    
                    if max_samples and len(original_data) >= max_samples:
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num+1}行JSON解析失败: {e}")
                    continue
    
    print(f"数据分析完成:")
    print(f"  总样本数: {id_analysis['total_samples']}")
    print(f"  有ID字段: {id_analysis['has_id_field']}")
    print(f"  dolly格式ID: {id_analysis['dolly_id_format']}")
    print(f"  其他格式ID: {id_analysis['other_id_format']}")
    print(f"  无ID字段: {id_analysis['no_id_field']}")
    print(f"  ID示例: {id_analysis['id_examples'][:5]}")
    
    if id_analysis['id_numbers']:
        print(f"  ID数字范围: {min(id_analysis['id_numbers'])} - {max(id_analysis['id_numbers'])}")
    
    return original_data, id_analysis


def validate_distributed_sampling(
    data_file: str,
    rank: int,
    world_size: int,
    device: torch.device,
    max_samples: int = None
) -> Dict:
    """
    验证分布式采样的ID映射
    
    Args:
        data_file: 数据文件路径
        rank: 当前rank
        world_size: 总进程数
        device: 设备
        max_samples: 最大样本数
        
    Returns:
        验证结果
    """
    print(f"[Rank {rank}] 开始验证分布式采样...")
    
    # 加载原始数据
    original_data, id_analysis = load_and_analyze_dolly_data(data_file, max_samples)
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 获取处理后的数据集
    dataset = get_training_dataset([data_file], tokenizer, max_seq_length=512)
    print(f"[Rank {rank}] 处理后数据集大小: {len(dataset)}")
    
    # 创建DataLoader
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=data_collator
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=data_collator
        )
    
    print(f"[Rank {rank}] DataLoader大小: {len(dataloader)}")
    
    # 验证采样映射
    validation_results = {
        'rank': rank,
        'world_size': world_size,
        'original_data_size': len(original_data),
        'dataset_size': len(dataset),
        'dataloader_size': len(dataloader),
        'processed_batches': 0,
        'correct_id_mappings': 0,
        'incorrect_id_mappings': 0,
        'missing_original_id': 0,
        'mapping_examples': [],
        'errors': []
    }
    
    # 测试前20个样本的映射
    test_samples = min(20, len(dataloader))
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= test_samples:
            break
            
        try:
            # 模拟collect_grad_reps.py中的索引获取逻辑
            if hasattr(dataloader.sampler, 'dataset'):
                # DistributedSampler情况
                actual_indices = list(dataloader.sampler)
                dataset_idx = actual_indices[batch_idx] if batch_idx < len(actual_indices) else batch_idx
            else:
                # 常规sampler情况
                dataset_idx = batch_idx
            
            validation_results['processed_batches'] += 1
            
            # 获取原始数据的ID
            original_id = None
            if dataset_idx < len(original_data) and 'id' in original_data[dataset_idx]:
                original_id = original_data[dataset_idx]['id']
            
            # 模拟collect_grad_reps.py中的ID提取逻辑
            extracted_id = None
            if dataset_idx < len(dataset):
                dataset_sample = dataset[dataset_idx]
                if hasattr(dataset_sample, 'get') and 'id' in dataset_sample:
                    extracted_id = dataset_sample['id']
                elif isinstance(dataset_sample, dict) and 'id' in dataset_sample:
                    extracted_id = dataset_sample['id']
                else:
                    extracted_id = f"sample_{dataset_idx}"
            
            # 验证映射正确性
            mapping_info = {
                'batch_idx': batch_idx,
                'dataset_idx': dataset_idx,
                'original_id': original_id,
                'extracted_id': extracted_id,
                'is_correct': False
            }
            
            if original_id is None:
                validation_results['missing_original_id'] += 1
                mapping_info['status'] = 'missing_original_id'
            elif original_id == extracted_id:
                validation_results['correct_id_mappings'] += 1
                mapping_info['is_correct'] = True
                mapping_info['status'] = 'correct'
            else:
                validation_results['incorrect_id_mappings'] += 1
                mapping_info['status'] = 'incorrect'
            
            # 如果是dolly格式，提取数字进行额外验证
            if original_id and original_id.startswith("dolly_"):
                original_number = extract_dolly_id_number(original_id)
                mapping_info['original_number'] = original_number
                mapping_info['expected_dataset_idx'] = original_number
                mapping_info['actual_dataset_idx'] = dataset_idx
                mapping_info['index_matches'] = (original_number == dataset_idx)
            
            validation_results['mapping_examples'].append(mapping_info)
            
            if batch_idx % 5 == 0:
                print(f"[Rank {rank}] 处理样本 {batch_idx}: dataset_idx={dataset_idx}, "
                      f"original_id={original_id}, extracted_id={extracted_id}")
                
        except Exception as e:
            error_info = {
                'batch_idx': batch_idx,
                'error': str(e)
            }
            validation_results['errors'].append(error_info)
            print(f"[Rank {rank}] 处理批次 {batch_idx} 时出错: {e}")
    
    # 计算统计信息
    total_processed = validation_results['processed_batches']
    if total_processed > 0:
        correct_rate = validation_results['correct_id_mappings'] / total_processed
        validation_results['correct_rate'] = correct_rate
        print(f"[Rank {rank}] 验证完成: {validation_results['correct_id_mappings']}/{total_processed} "
              f"正确 ({correct_rate:.2%})")
    
    return validation_results


def print_detailed_results(all_results: List[Dict]):
    """
    打印详细的验证结果
    
    Args:
        all_results: 所有rank的验证结果
    """
    print("\n" + "="*80)
    print("详细验证结果")
    print("="*80)
    
    # 汇总统计
    total_processed = sum(r['processed_batches'] for r in all_results)
    total_correct = sum(r['correct_id_mappings'] for r in all_results)
    total_incorrect = sum(r['incorrect_id_mappings'] for r in all_results)
    total_missing = sum(r['missing_original_id'] for r in all_results)
    
    print(f"总体统计:")
    print(f"  参与验证的GPU数: {len(all_results)}")
    print(f"  总处理样本数: {total_processed}")
    print(f"  正确映射: {total_correct}")
    print(f"  错误映射: {total_incorrect}")
    print(f"  缺失原始ID: {total_missing}")
    
    if total_processed > 0:
        overall_rate = total_correct / total_processed
        print(f"  总体正确率: {overall_rate:.2%}")
        print(f"  验证结果: {'✓ 通过' if total_incorrect == 0 else '✗ 失败'}")
    
    # 显示每个GPU的详细信息
    print(f"\n各GPU详细信息:")
    for result in all_results:
        rank = result['rank']
        processed = result['processed_batches']
        correct = result['correct_id_mappings']
        rate = result.get('correct_rate', 0)
        
        print(f"  GPU {rank}:")
        print(f"    处理样本: {processed}")
        print(f"    正确映射: {correct}")
        print(f"    正确率: {rate:.2%}")
        print(f"    数据集大小: {result['dataset_size']}")
        print(f"    DataLoader大小: {result['dataloader_size']}")
    
    # 显示映射示例
    print(f"\n映射示例 (前几个样本):")
    for result in all_results[:2]:  # 只显示前2个GPU
        rank = result['rank']
        print(f"  GPU {rank}:")
        for example in result['mapping_examples'][:5]:
            status_symbol = "✓" if example['is_correct'] else "✗"
            print(f"    {status_symbol} 批次{example['batch_idx']}: "
                  f"dataset_idx={example['dataset_idx']}, "
                  f"original_id={example['original_id']}, "
                  f"extracted_id={example['extracted_id']}")
            
            # 如果有dolly数字信息，也显示
            if 'original_number' in example:
                index_match = "✓" if example['index_matches'] else "✗"
                print(f"      {index_match} dolly数字={example['original_number']}, "
                      f"期望索引={example['expected_dataset_idx']}, "
                      f"实际索引={example['actual_dataset_idx']}")


def main():
    parser = argparse.ArgumentParser(description='验证dolly数据集的ID提取逻辑')
    
    parser.add_argument('--data_file', type=str, required=True,
                        help='dolly数据文件路径')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='最大测试样本数，默认100')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 设置分布式环境
    rank, world_size, local_rank, device = setup_distributed()
    
    try:
        print(f"开始验证dolly数据集ID提取")
        print(f"数据文件: {args.data_file}")
        print(f"最大样本数: {args.max_samples}")
        print(f"GPU配置: Rank {rank}/{world_size}")
        
        # 验证数据文件
        if not os.path.exists(args.data_file):
            print(f"错误: 数据文件不存在: {args.data_file}")
            return 1
        
        # 执行验证
        validation_results = validate_distributed_sampling(
            args.data_file, rank, world_size, device, args.max_samples
        )
        
        # 收集结果
        if world_size > 1:
            all_results = [None] * world_size
            dist.all_gather_object(all_results, validation_results)
        else:
            all_results = [validation_results]
        
        # 打印结果（只在rank 0）
        if rank == 0:
            print_detailed_results(all_results)
            
            # 保存结果
            if args.output_file:
                import json
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"\n结果已保存到: {args.output_file}")
        
        # 同步
        if world_size > 1:
            dist.barrier()
        
        # 判断是否成功
        total_incorrect = sum(r['incorrect_id_mappings'] for r in all_results)
        return 0 if total_incorrect == 0 else 1
        
    except Exception as e:
        print(f"[Rank {rank}] 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    exit(main())