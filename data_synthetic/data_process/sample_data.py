#!/usr/bin/env python3
"""
数据采样脚本
功能：
1. 从yelp-train.csv采样N=1000个数据
2. 从yelp_huggingface_gpt2_secpe_10_600.csv随机采样5个不重复subset，每个subset包含M=1000个数据
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSampler:
    """数据采样器类"""
    
    def __init__(self, raw_data_dir: str = "raw", output_dir: str = "sampled"):
        """
        初始化数据采样器
        
        Args:
            raw_data_dir: 原始数据目录
            output_dir: 输出数据目录
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置随机种子以确保可重现性
        np.random.seed(42)
    
    def sample_yelp_train(self, n_samples: int = 1000, chunk_size: int = 10000) -> bool:
        """
        从yelp-train.csv采样N个数据
        
        Args:
            n_samples: 采样数量
            chunk_size: 分块读取大小，用于处理大文件
            
        Returns:
            bool: 是否成功
        """
        input_file = self.raw_data_dir / "Yelp-train.csv"
        output_file = self.output_dir / f"yelp_train_sampled_{n_samples}.csv"
        
        if not input_file.exists():
            logger.error(f"输入文件不存在: {input_file}")
            return False
        
        try:
            logger.info(f"开始从 {input_file} 采样 {n_samples} 个数据...")
            
            # 首先获取文件总行数（不包括header）
            total_rows = sum(1 for _ in open(input_file)) - 1
            logger.info(f"文件总行数: {total_rows}")
            
            if total_rows < n_samples:
                logger.warning(f"文件总行数 ({total_rows}) 小于采样数量 ({n_samples})，将采样所有数据")
                n_samples = total_rows
            
            # 生成随机索引
            sample_indices = np.random.choice(total_rows, size=n_samples, replace=False)
            sample_indices = sorted(sample_indices)
            
            # 分块读取并采样
            sampled_data = []
            current_row = 0
            sample_idx = 0
            
            for chunk in pd.read_csv(input_file, chunksize=chunk_size):
                chunk_start = current_row
                chunk_end = current_row + len(chunk)
                
                # 找到在当前chunk中的采样索引
                while sample_idx < len(sample_indices) and sample_indices[sample_idx] < chunk_end:
                    if sample_indices[sample_idx] >= chunk_start:
                        local_idx = sample_indices[sample_idx] - chunk_start
                        sampled_data.append(chunk.iloc[local_idx])
                    sample_idx += 1
                
                current_row = chunk_end
                
                # 如果已经采样完成，退出循环
                if sample_idx >= len(sample_indices):
                    break
            
            # 保存采样数据
            if sampled_data:
                sampled_df = pd.DataFrame(sampled_data)
                sampled_df.to_csv(output_file, index=False)
                logger.info(f"成功采样 {len(sampled_df)} 个数据，保存到: {output_file}")
                return True
            else:
                logger.error("采样失败，没有获取到数据")
                return False
                
        except Exception as e:
            logger.error(f"采样yelp-train.csv时发生错误: {str(e)}")
            return False
    
    def sample_yelp_huggingface_subsets(self, t_subsets: int = 5, m_samples: int = 1000, 
                                      chunk_size: int = 10000) -> bool:
        """
        从yelp_huggingface_gpt2_secpe_10_600.csv随机采样t个不重复subset
        确保子集之间没有重复数据
        
        Args:
            t_subsets: subset数量
            m_samples: 每个subset的样本数量
            chunk_size: 分块读取大小
            
        Returns:
            bool: 是否成功
        """
        input_file = self.raw_data_dir / "yelp_huggingface_gpt2_secpe_10_600.csv"
        
        if not input_file.exists():
            logger.error(f"输入文件不存在: {input_file}")
            return False
        
        try:
            logger.info(f"开始从 {input_file} 采样 {t_subsets} 个subset，每个包含 {m_samples} 个数据...")
            
            # 读取整个文件以正确处理包含换行符的CSV
            logger.info("正在读取文件...")
            df = pd.read_csv(input_file)
            total_rows = len(df)
            logger.info(f"文件总行数: {total_rows}")
            
            # 去重处理，基于文本内容
            logger.info("正在去除重复数据...")
            df_unique = df.drop_duplicates(subset=['text'], keep='first')
            unique_rows = len(df_unique)
            logger.info(f"去重后数据量: {unique_rows} (去除了 {total_rows - unique_rows} 个重复项)")
            
            total_needed = t_subsets * m_samples
            if unique_rows < total_needed:
                logger.error(f"去重后数据不足: 需要 {total_needed} 个样本，但只有 {unique_rows} 个")
                return False
            
            # 重置索引
            df_unique = df_unique.reset_index(drop=True)
            
            # 生成不重复的随机索引
            all_indices = np.random.choice(unique_rows, size=total_needed, replace=False)
            
            # 将索引分成t个subset，每个subset准确包含m_samples个样本
            subset_indices = []
            for i in range(t_subsets):
                start_idx = i * m_samples
                end_idx = start_idx + m_samples
                subset_indices.append(all_indices[start_idx:end_idx])
            
            success_count = 0
            
            for i, indices in enumerate(subset_indices):
                # 采样当前subset
                subset_df = df_unique.iloc[indices].copy()
                
                # 保存subset
                output_file = self.output_dir / f"yelp_huggingface_subset_{i+1}_{len(subset_df)}.csv"
                subset_df.to_csv(output_file, index=False)
                logger.info(f"成功保存subset {i+1}: {len(subset_df)} 个数据 -> {output_file}")
                success_count += 1
            
            logger.info(f"成功创建 {success_count}/{t_subsets} 个subset")
            return success_count == t_subsets
            
        except Exception as e:
            logger.error(f"采样yelp_huggingface文件时发生错误: {str(e)}")
            return False
    
    def run_sampling(self, yelp_train_samples: int = 1000, 
                    huggingface_subsets: int = 5, 
                    huggingface_samples_per_subset: int = 1000) -> None:
        """
        运行完整的采样流程
        
        Args:
            yelp_train_samples: yelp-train.csv的采样数量
            huggingface_subsets: huggingface文件的subset数量
            huggingface_samples_per_subset: 每个subset的样本数量
        """
        logger.info("开始数据采样流程...")
        
        # 任务1: 采样yelp-train.csv
        logger.info("=" * 50)
        logger.info("任务1: 采样yelp-train.csv")
        success1 = self.sample_yelp_train(yelp_train_samples)
        
        # 任务2: 采样yelp_huggingface文件
        logger.info("=" * 50)
        logger.info("任务2: 采样yelp_huggingface_gpt2_secpe_10_600.csv")
        success2 = self.sample_yelp_huggingface_subsets(huggingface_subsets, huggingface_samples_per_subset)
        
        # 总结
        logger.info("=" * 50)
        logger.info("采样流程完成!")
        logger.info(f"yelp-train.csv采样: {'成功' if success1 else '失败'}")
        logger.info(f"yelp_huggingface文件采样: {'成功' if success2 else '失败'}")
        
        if success1 and success2:
            logger.info("所有采样任务都已成功完成!")
        else:
            logger.warning("部分采样任务失败，请检查日志信息")


def main():
    """主函数"""
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    raw_dir = script_dir / "raw"
    output_dir = script_dir / "sampled"
    
    # 创建采样器
    sampler = DataSampler(raw_data_dir=str(raw_dir), output_dir=str(output_dir))
    
    # 运行采样
    sampler.run_sampling(
        yelp_train_samples=1000,           # N=1000
        huggingface_subsets=5,             # t=5
        huggingface_samples_per_subset=1000 # M=1000
    )


if __name__ == "__main__":
    main()