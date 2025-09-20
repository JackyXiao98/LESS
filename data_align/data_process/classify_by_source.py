#!/usr/bin/env python3
"""
根据 source 字段将 parquet 数据分类并存储
从 /mnt/hdfs/selection/yingtai_sft/tulu_3_raw_add_column 读取数据
按 source 分类后存储到 /mnt/hdfs/selection/yingtai_sft/tulu_3_by_source

使用方法:
    python classify_by_source.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR] [--num-workers NUM_WORKERS]

参数说明:
    --input-dir: 输入目录路径 (默认: /mnt/hdfs/selection/yingtai_sft/tulu_3_raw_add_column)
    --output-dir: 输出目录路径 (默认: /mnt/hdfs/selection/yingtai_sft/tulu_3_by_source)
    --num-workers: 并行处理的进程数 (默认: min(cpu_count(), 8))
"""

import os
import gc
import time
import argparse
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
from datasets import Dataset, Features, Value, Sequence, concatenate_datasets
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_source_distribution(input_dir):
    """
    分析输入目录中所有parquet文件的source字段分布
    """
    logger.info("开始分析source字段分布...")
    
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return None
    
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    if not parquet_files:
        logger.error(f"在 {input_dir} 中未找到parquet文件")
        return None
    
    logger.info(f"找到 {len(parquet_files)} 个parquet文件")
    
    source_counter = Counter()
    total_samples = 0
    
    for file_name in tqdm(parquet_files, desc="分析文件"):
        file_path = os.path.join(input_dir, file_name)
        try:
            # 读取parquet文件
            dataset = Dataset.from_parquet(file_path)
            
            # 统计source字段
            for source in dataset['source']:
                source_counter[source] += 1
                total_samples += 1
                
        except Exception as e:
            logger.error(f"读取文件 {file_name} 时出错: {e}")
            continue
    
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"发现 {len(source_counter)} 个不同的source:")
    
    for source, count in source_counter.most_common():
        percentage = (count / total_samples) * 100
        logger.info(f"  {source}: {count} 样本 ({percentage:.2f}%)")
    
    return source_counter

def process_file_by_source(args):
    """
    处理单个parquet文件，按source分类数据
    """
    file_path, output_dir, file_idx = args
    
    try:
        logger.info(f"处理文件 {file_idx}: {os.path.basename(file_path)}")
        
        # 读取parquet文件
        dataset = Dataset.from_parquet(file_path)
        
        # 按source分组数据
        source_data = defaultdict(list)
        
        for i, example in enumerate(dataset):
            source = example['source']
            source_data[source].append(example)
        
        # 为每个source创建输出文件
        output_files = {}
        for source, examples in source_data.items():
            if not examples:
                continue
                
            # 创建source对应的输出目录
            source_dir = os.path.join(output_dir, source.replace('/', '_').replace(' ', '_'))
            os.makedirs(source_dir, exist_ok=True)
            
            # 定义Features schema (与原始数据保持一致)
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
            
            # 创建Dataset并保存
            source_dataset = Dataset.from_list(examples, features=features)
            output_file = os.path.join(source_dir, f"batch_{file_idx:06d}.parquet")
            source_dataset.to_parquet(output_file)
            
            output_files[source] = output_file
            logger.info(f"  {source}: {len(examples)} 样本 -> {output_file}")
        
        return output_files
        
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {e}")
        return {}

def merge_source_files(output_dir):
    """
    合并每个source目录下的所有parquet文件
    """
    logger.info("开始合并各source的parquet文件...")
    
    source_dirs = [d for d in os.listdir(output_dir) 
                   if os.path.isdir(os.path.join(output_dir, d))]
    
    for source_dir_name in tqdm(source_dirs, desc="合并source文件"):
        source_dir_path = os.path.join(output_dir, source_dir_name)
        parquet_files = [f for f in os.listdir(source_dir_path) 
                        if f.endswith('.parquet')]
        
        if not parquet_files:
            continue
            
        logger.info(f"合并 {source_dir_name}: {len(parquet_files)} 个文件")
        
        try:
            # 读取所有parquet文件并合并
            datasets = []
            for file_name in parquet_files:
                file_path = os.path.join(source_dir_path, file_name)
                dataset = Dataset.from_parquet(file_path)
                datasets.append(dataset)
            
            # 合并所有数据集
            if datasets:
                merged_dataset = concatenate_datasets(datasets)
                
                # 保存合并后的文件
                merged_file = os.path.join(output_dir, f"{source_dir_name}.parquet")
                merged_dataset.to_parquet(merged_file)
                
                logger.info(f"  合并完成: {len(merged_dataset)} 样本 -> {merged_file}")
                
                # 删除临时目录
                import shutil
                shutil.rmtree(source_dir_path)
                
        except Exception as e:
            logger.error(f"合并 {source_dir_name} 时出错: {e}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="根据source字段分类parquet数据")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="/mnt/hdfs/selection/yingtai_sft/tulu_3_raw_add_column",
        help="输入目录路径"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/mnt/hdfs/selection/yingtai_sft/tulu_3_by_source",
        help="输出目录路径"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=min(cpu_count(), 8),
        help="并行处理的进程数"
    )
    parser.add_argument(
        "--analyze-only", 
        action="store_true",
        help="仅分析source分布，不进行数据分类"
    )
    return parser.parse_args()

def main():
    """
    主函数：分析数据分布，按source分类并保存
    """
    # 解析命令行参数
    args = parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_workers = args.num_workers
    
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"工作进程数: {num_workers}")
    
    try:
        # 分析source分布
        source_counter = analyze_source_distribution(input_dir)
        if source_counter is None:
            return
        
        if args.analyze_only:
            logger.info("仅分析模式，程序结束")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有parquet文件
        parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
        file_paths = [(os.path.join(input_dir, f), output_dir, i) 
                     for i, f in enumerate(parquet_files)]
        
        logger.info(f"开始并行处理 {len(file_paths)} 个文件...")
        start_time = time.time()
        
        # 并行处理文件
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_file_by_source, file_paths),
                total=len(file_paths),
                desc="处理文件"
            ))
        
        # 统计处理结果
        total_output_files = 0
        for result in results:
            total_output_files += len(result)
        
        processing_time = time.time() - start_time
        logger.info(f"文件处理完成，耗时 {processing_time:.2f}s")
        logger.info(f"生成了 {total_output_files} 个临时文件")
        
        # 合并每个source的文件
        merge_source_files(output_dir)
        
        total_time = time.time() - start_time
        logger.info(f"所有任务完成！总耗时 {total_time:.2f}s")
        
        # 验证输出
        verify_output(output_dir, source_counter)
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise

def verify_output(output_dir, expected_sources):
    """
    验证输出文件的正确性
    """
    logger.info("开始验证输出文件...")
    
    if not os.path.exists(output_dir):
        logger.error(f"输出目录不存在: {output_dir}")
        return False
    
    parquet_files = [f for f in os.listdir(output_dir) if f.endswith('.parquet')]
    
    logger.info(f"找到 {len(parquet_files)} 个输出文件")
    logger.info(f"期望的source数量: {len(expected_sources)}")
    
    # 检查每个文件
    total_samples = 0
    for file_name in parquet_files:
        file_path = os.path.join(output_dir, file_name)
        try:
            dataset = Dataset.from_parquet(file_path)
            source_name = file_name.replace('.parquet', '')
            
            # 验证所有样本的source都是一致的
            unique_sources = set(dataset['source'])
            if len(unique_sources) != 1:
                logger.warning(f"文件 {file_name} 包含多个source: {unique_sources}")
            
            sample_count = len(dataset)
            total_samples += sample_count
            logger.info(f"  {source_name}: {sample_count} 样本")
            
        except Exception as e:
            logger.error(f"验证文件 {file_name} 时出错: {e}")
    
    logger.info(f"验证完成，总样本数: {total_samples}")
    return True

if __name__ == "__main__":
    main()