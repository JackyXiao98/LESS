#!/usr/bin/env python3
"""
验证数据集测试脚本
展示每个数据集的样本数据
"""

import sys
import logging
from get_validation import ValidationDatasetBuilder

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_single_dataset(dataset_name: str, num_samples: int = 5):
    """测试单个数据集"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试数据集: {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    
    builder = ValidationDatasetBuilder()
    
    if dataset_name not in builder.processors:
        logger.error(f"不支持的数据集: {dataset_name}")
        return
    
    processor = builder.processors[dataset_name]
    
    try:
        # 加载和处理数据
        data = processor.load_and_process_data()
        logger.info(f"成功加载 {len(data)} 条 {dataset_name} 数据")
        
        # 展示样本
        samples_to_show = min(num_samples, len(data))
        for i in range(samples_to_show):
            sample = data[i]
            logger.info(f"\n--- 样本 {i+1} ---")
            logger.info(f"ID: {sample['id']}")
            logger.info(f"来源: {sample['source']}")
            
            user_content = sample['messages'][0]['content']
            assistant_content = sample['messages'][1]['content']
            
            # 限制显示长度
            max_length = 200
            if len(user_content) > max_length:
                user_display = user_content[:max_length] + "..."
            else:
                user_display = user_content
                
            if len(assistant_content) > max_length:
                assistant_display = assistant_content[:max_length] + "..."
            else:
                assistant_display = assistant_content
            
            logger.info(f"用户: {user_display}")
            logger.info(f"助手: {assistant_display}")
        
        logger.info(f"\n{dataset_name} 数据集测试完成! 总计: {len(data)} 条数据")
        return len(data)
        
    except Exception as e:
        logger.error(f"测试 {dataset_name} 时出错: {str(e)}")
        return 0


def test_all_datasets(num_samples: int = 3):
    """测试所有数据集"""
    logger.info("开始测试所有验证数据集...")
    
    builder = ValidationDatasetBuilder()
    results = {}
    
    for dataset_name in builder.processors.keys():
        try:
            count = test_single_dataset(dataset_name, num_samples)
            results[dataset_name] = count
        except Exception as e:
            logger.error(f"测试 {dataset_name} 失败: {str(e)}")
            results[dataset_name] = 0
    
    # 显示总结
    logger.info(f"\n{'='*60}")
    logger.info("测试总结")
    logger.info(f"{'='*60}")
    
    total_samples = 0
    for dataset_name, count in results.items():
        logger.info(f"{dataset_name:15}: {count:6} 条数据")
        total_samples += count
    
    logger.info(f"{'总计':15}: {total_samples:6} 条数据")
    logger.info("所有数据集测试完成!")
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="验证数据集测试工具")
    parser.add_argument("--dataset", type=str, help="测试指定的数据集")
    parser.add_argument("--samples", type=int, default=3, help="每个数据集展示的样本数")
    parser.add_argument("--all", action="store_true", help="测试所有数据集")
    
    args = parser.parse_args()
    
    if args.dataset:
        test_single_dataset(args.dataset, args.samples)
    elif args.all:
        test_all_datasets(args.samples)
    else:
        # 默认测试几个主要数据集
        datasets_to_test = ["mmlu", "gsm8k", "humaneval"]
        for dataset in datasets_to_test:
            test_single_dataset(dataset, args.samples)


if __name__ == "__main__":
    main()