#!/usr/bin/env python3
"""
为 allenai/tulu-3-sft-mixture 数据集添加 chat_template_kwargs 列
并使用并行处理保存为 parquet 格式

使用方法:
    python add_column.py [--output-dir OUTPUT_DIR] [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS]

参数说明:
    --output-dir: 输出目录路径 (默认: /mnt/hdfs/selection/yingtai_sft/tulu_3_raw_add_column)
    --batch-size: 每个批次的样本数 (默认: 1000)
    --num-workers: 并行处理的进程数 (默认: min(cpu_count(), 8))
"""

import os
import json
import gc
import time
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from datasets import load_dataset, Dataset, Features, Value, Sequence
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义要添加的列内容
CHAT_TEMPLATE_KWARGS = {
    "custom_instructions": "",
    "enable_thinking": False,
    "python_tools": [],
    "xml_tools": []
}

def add_chat_template_kwargs(example):
    """为单个样本添加 chat_template_kwargs 列"""
    example['chat_template_kwargs'] = CHAT_TEMPLATE_KWARGS
    return example

def process_batch(batch_data, batch_idx):
    """
    处理一个批次的数据
    """
    logger.info(f"处理批次 {batch_idx}，包含 {len(batch_data)} 个样本")
    
    # 为批次中的每个样本添加列
    processed_batch = []
    for example in batch_data:
        processed_example = add_chat_template_kwargs(example)
        processed_batch.append(processed_example)
    
    return processed_batch

def save_batch_to_parquet(batch_data, output_dir, batch_idx):
    """
    将批次数据保存为 parquet 文件
    """
    try:
        # 验证数据结构
        if len(batch_data) == 0:
            logger.warning(f"批次 {batch_idx} 为空，跳过")
            return None
            
        if not isinstance(batch_data[0], dict):
            logger.error(f"批次 {batch_idx}: 数据格式错误，期望字典但得到 {type(batch_data[0])}")
            return None
        
        # 定义明确的 Features schema
        features = Features({
            'id': Value('string'),
            'messages': [  # messages 是一个列表，每个元素是一个字典
                {
                    'content': Value('string'),
                    'role': Value('string')
                }
            ],
            'source': Value('string'),
            'chat_template_kwargs': {
                'custom_instructions': Value('string'),
                'enable_thinking': Value('bool'),
                'python_tools': Sequence(Value('string')),
                'xml_tools': Sequence(Value('string'))
            }
        })
        
        # 创建 Dataset 对象，明确指定 features
        batch_dataset = Dataset.from_list(batch_data, features=features)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为 parquet 文件
        output_file = os.path.join(output_dir, f"batch_{batch_idx:06d}.parquet")
        batch_dataset.to_parquet(output_file)
        
        logger.info(f"批次 {batch_idx} 已保存到 {output_file} ({len(batch_data)} 个样本)")
        return output_file
    except Exception as e:
        logger.error(f"保存批次 {batch_idx} 时出错: {e}")
        return None

def process_and_save_batch(args):
    """
    处理并保存单个批次的数据（用于并行处理）
    """
    batch_data, output_dir, batch_idx = args
    try:
        # 处理批次
        processed_batch = process_batch(batch_data, batch_idx)
        
        # 保存批次
        output_file = save_batch_to_parquet(processed_batch, output_dir, batch_idx)
        return output_file
    except Exception as e:
        logger.error(f"处理批次 {batch_idx} 时出错: {e}")
        return None

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="为 tulu-3-sft-mixture 数据集添加 chat_template_kwargs 列")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/mnt/hdfs/selection/yingtai_sft/tulu_3_raw_add_column",
        help="输出目录路径"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=30000,
        help="每个批次的样本数"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=min(cpu_count(), 16),
        help="并行处理的进程数"
    )
    return parser.parse_args()

def main():
    """
    主函数：加载数据集，添加列，并行处理并保存
    """
    # 解析命令行参数
    args = parse_args()
    
    # 配置参数
    output_dir = args.output_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_batches_in_memory = num_workers * 2  # 内存中最多保持的批次数
    
    logger.info(f"开始处理数据集，输出目录: {output_dir}")
    logger.info(f"批次大小: {batch_size}, 工作进程数: {num_workers}")
    
    try:
        # 加载数据集
        logger.info("正在加载 allenai/tulu-3-sft-mixture 数据集...")
        dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
        
        # 收集批次数据
        batch_data = []
        batch_idx = 0
        processed_files = []
        batches_to_process = []
        
        logger.info("开始处理数据...")
        
        # 创建进程池
        start_time = time.time()
        total_samples = 0
        
        with Pool(processes=num_workers) as pool:
            for example in tqdm(dataset, desc="处理样本"):
                batch_data.append(example)
                total_samples += 1
                
                # 当批次达到指定大小时，添加到待处理队列
                if len(batch_data) >= batch_size:
                    batches_to_process.append((batch_data.copy(), output_dir, batch_idx))
                    batch_data = []
                    batch_idx += 1
                    
                    # 当积累了足够的批次时，开始并行处理
                    if len(batches_to_process) >= max_batches_in_memory:
                        logger.info(f"开始并行处理 {len(batches_to_process)} 个批次...")
                        batch_start_time = time.time()
                        
                        try:
                            results = pool.map(process_and_save_batch, batches_to_process)
                            
                            # 收集结果
                            successful_batches = 0
                            for result in results:
                                if result:
                                    processed_files.append(result)
                                    successful_batches += 1
                            
                            batch_time = time.time() - batch_start_time
                            logger.info(f"批次处理完成: {successful_batches}/{len(batches_to_process)} 成功, 耗时 {batch_time:.2f}s")
                            
                        except Exception as e:
                            logger.error(f"批次处理失败: {e}")
                        
                        # 清空待处理队列并强制垃圾回收
                        batches_to_process = []
                        gc.collect()
            
            # 处理最后一个不完整的批次
            if batch_data:
                batches_to_process.append((batch_data, output_dir, batch_idx))
                batch_idx += 1
            
            # 处理剩余的批次
            if batches_to_process:
                logger.info(f"处理剩余的 {len(batches_to_process)} 个批次...")
                try:
                    results = pool.map(process_and_save_batch, batches_to_process)
                    
                    # 收集结果
                    for result in results:
                        if result:
                            processed_files.append(result)
                except Exception as e:
                    logger.error(f"最终批次处理失败: {e}")
        
        total_time = time.time() - start_time
        
        logger.info(f"数据处理完成！共生成 {len(processed_files)} 个 parquet 文件")
        logger.info(f"文件保存在: {output_dir}")
        
        # 打印详细统计信息
        logger.info(f"总处理时间: {total_time:.2f}s")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"总批次数: {batch_idx}")
        logger.info(f"平均处理速度: {total_samples/total_time:.2f} 样本/秒")
        logger.info(f"成功处理的批次: {len(processed_files)}/{batch_idx}")
        
        logger.info("处理完成的文件列表:")
        for file_path in processed_files[:10]:  # 只显示前10个文件
            logger.info(f"  - {file_path}")
        if len(processed_files) > 10:
            logger.info(f"  ... 还有 {len(processed_files) - 10} 个文件")
            
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise

def verify_output(output_dir="/mnt/hdfs/selection/yingtai_sft/tulu_3_raw_add_column"):
    """
    验证输出文件的正确性
    """
    if not os.path.exists(output_dir):
        logger.error(f"输出目录不存在: {output_dir}")
        return False
    
    parquet_files = [f for f in os.listdir(output_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        logger.error("未找到任何 parquet 文件")
        return False
    
    logger.info(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    # 检查第一个文件的结构
    first_file = os.path.join(output_dir, parquet_files[0])
    try:
        sample_dataset = Dataset.from_parquet(first_file)
        logger.info(f"样本文件列名: {sample_dataset.column_names}")
        logger.info(f"数据集 features: {sample_dataset.features}")
        
        if "chat_template_kwargs" in sample_dataset.column_names:
            sample_row = sample_dataset[0]
            logger.info(f"chat_template_kwargs 样本内容: {sample_row['chat_template_kwargs']}")
            
            # 验证 schema 类型
            features = sample_dataset.features
            if 'messages' in features:
                logger.info(f"messages 字段类型: {type(features['messages'])}")
            if 'chat_template_kwargs' in features and 'python_tools' in features['chat_template_kwargs']:
                logger.info(f"python_tools 字段类型: {type(features['chat_template_kwargs']['python_tools'])}")
                logger.info(f"xml_tools 字段类型: {type(features['chat_template_kwargs']['xml_tools'])}")
            
            logger.info("验证成功：chat_template_kwargs 列已正确添加")
            return True
        else:
            logger.error("验证失败：未找到 chat_template_kwargs 列")
            return False
            
    except Exception as e:
        logger.error(f"验证文件时出错: {e}")
        return False

if __name__ == "__main__":
    args = parse_args()
    main()
    
    logger.info("开始验证输出文件...")
    if verify_output(args.output_dir):
        logger.info("所有任务完成！")
    else:
        logger.error("验证失败，请检查输出文件")