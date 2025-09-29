#!/usr/bin/env python3
"""
采样结果验证脚本
用于验证数据采样的正确性
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_sampling_results(sampled_dir: str = "sampled"):
    """
    验证采样结果的正确性
    
    Args:
        sampled_dir: 采样结果目录
    """
    sampled_path = Path(sampled_dir)
    
    if not sampled_path.exists():
        logger.error(f"采样结果目录不存在: {sampled_path}")
        return False
    
    logger.info("开始验证采样结果...")
    
    # 验证yelp_train采样文件
    yelp_train_file = sampled_path / "yelp_train_sampled_1000.csv"
    if yelp_train_file.exists():
        try:
            df = pd.read_csv(yelp_train_file)
            logger.info(f"✓ yelp_train_sampled_1000.csv: {len(df)} 行数据")
            logger.info(f"  列名: {list(df.columns)}")
            
            if len(df) == 1000:
                logger.info("  ✓ 数据行数正确")
            else:
                logger.warning(f"  ⚠ 数据行数不正确，期望1000行，实际{len(df)}行")
        except Exception as e:
            logger.error(f"  ✗ 读取文件失败: {str(e)}")
    else:
        logger.error("✗ yelp_train_sampled_1000.csv 文件不存在")
    
    # 验证huggingface subset文件
    subset_files = list(sampled_path.glob("yelp_huggingface_subset_*_1000.csv"))
    subset_files.sort()
    
    if len(subset_files) == 5:
        logger.info(f"✓ 找到 {len(subset_files)} 个 huggingface subset 文件")
    else:
        logger.warning(f"⚠ huggingface subset 文件数量不正确，期望5个，实际{len(subset_files)}个")
    
    all_text_hashes = set()
    total_samples = 0
    
    for i, file_path in enumerate(subset_files, 1):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✓ {file_path.name}: {len(df)} 行数据")
            logger.info(f"  列名: {list(df.columns)}")
            
            if len(df) == 1000:
                logger.info("  ✓ 数据行数正确")
            else:
                logger.warning(f"  ⚠ 数据行数不正确，期望1000行，实际{len(df)}行")
            
            # 检查数据唯一性（通过文本内容的哈希值）
            current_text_hashes = set(df['text'].apply(lambda x: hash(str(x))))
            overlap = all_text_hashes.intersection(current_text_hashes)
            if overlap:
                logger.warning(f"  ⚠ 发现重复数据: {len(overlap)} 个")
            else:
                logger.info("  ✓ 与其他subset无重复数据")
            
            all_text_hashes.update(current_text_hashes)
            total_samples += len(df)
            
        except Exception as e:
            logger.error(f"  ✗ 读取文件失败: {str(e)}")
    
    # 总结
    logger.info("=" * 50)
    logger.info("验证结果总结:")
    logger.info(f"- yelp_train采样文件: {'存在' if yelp_train_file.exists() else '不存在'}")
    logger.info(f"- huggingface subset文件数量: {len(subset_files)}/5")
    logger.info(f"- 总采样数据量: {total_samples}")
    logger.info(f"- 唯一文本数量: {len(all_text_hashes)}")
    
    if total_samples == 5000 and len(all_text_hashes) == total_samples:
        logger.info("✓ 所有验证通过！采样结果正确。")
        return True
    else:
        logger.warning("⚠ 验证发现问题，请检查采样结果。")
        return False


def show_sample_data(sampled_dir: str = "sampled", n_samples: int = 3):
    """
    显示采样数据的示例
    
    Args:
        sampled_dir: 采样结果目录
        n_samples: 显示的样本数量
    """
    sampled_path = Path(sampled_dir)
    
    logger.info("=" * 50)
    logger.info("采样数据示例:")
    
    # 显示yelp_train数据示例
    yelp_train_file = sampled_path / "yelp_train_sampled_1000.csv"
    if yelp_train_file.exists():
        try:
            df = pd.read_csv(yelp_train_file)
            logger.info(f"\n--- yelp_train_sampled_1000.csv 前{n_samples}行 ---")
            print(df.head(n_samples).to_string())
        except Exception as e:
            logger.error(f"读取yelp_train文件失败: {str(e)}")
    
    # 显示第一个huggingface subset数据示例
    subset_files = list(sampled_path.glob("yelp_huggingface_subset_*_1000.csv"))
    if subset_files:
        try:
            df = pd.read_csv(subset_files[0])
            logger.info(f"\n--- {subset_files[0].name} 前{n_samples}行 ---")
            print(df.head(n_samples).to_string())
        except Exception as e:
            logger.error(f"读取huggingface subset文件失败: {str(e)}")


def main():
    """主函数"""
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    sampled_dir = script_dir / "sampled"
    
    # 验证采样结果
    success = verify_sampling_results(str(sampled_dir))
    
    # 显示数据示例
    show_sample_data(str(sampled_dir))
    
    if success:
        logger.info("\n🎉 验证完成！所有采样结果都正确。")
    else:
        logger.warning("\n⚠️ 验证发现问题，请检查采样过程。")


if __name__ == "__main__":
    main()