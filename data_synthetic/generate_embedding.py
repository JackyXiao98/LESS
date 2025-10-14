#!/usr/bin/env python3
"""
生成sentence embedding的脚本
支持处理不同格式的CSV文件，为每条数据生成embedding并保存
"""

import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
import torch

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32):
        """
        初始化embedding生成器
        
        Args:
            model_name: sentence-transformers模型名称
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {self.device}")
        
    def load_model(self):
        """加载sentence transformer模型"""
        if self.model is None:
            logger.info(f"加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("模型加载完成")
    
    def preprocess_yelp_train_text(self, row: pd.Series) -> str:
        """
        预处理yelp_train数据，将多列合并为纯文本
        
        Args:
            row: pandas Series，包含text, label1, label2列
            
        Returns:
            合并后的纯文本
        """
        text = str(row['text']).strip()
        label1 = str(row['label1']).strip()
        label2 = str(row['label2']).strip()
        
        # 合并文本，用空格分隔
        combined_text = f"{text} {label1} {label2}"
        return combined_text
    
    def preprocess_huggingface_text(self, row: pd.Series) -> str:
        """
        预处理huggingface数据，直接使用text列
        
        Args:
            row: pandas Series，包含text列
            
        Returns:
            处理后的文本
        """
        return str(row['text']).strip()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        生成文本的sentence embeddings
        
        Args:
            texts: 文本列表
            
        Returns:
            embeddings数组，形状为(n_texts, embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"生成{len(texts)}条文本的embeddings")
        
        # 批量生成embeddings
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def process_csv_file(self, csv_path: str, output_dir: str) -> Dict[str, Any]:
        """
        处理单个CSV文件，生成embeddings
        
        Args:
            csv_path: CSV文件路径
            output_dir: 输出目录
            
        Returns:
            处理结果信息
        """
        csv_path = Path(csv_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"处理文件: {csv_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        logger.info(f"读取到{len(df)}条数据")
        
        # 根据文件名判断数据格式
        if 'yelp_train_sampled' in csv_path.name:
            # yelp_train格式：需要合并多列
            texts = [self.preprocess_yelp_train_text(row) for _, row in df.iterrows()]
            file_type = 'yelp_train'
        else:
            # huggingface格式：直接使用text列
            texts = [self.preprocess_huggingface_text(row) for _, row in df.iterrows()]
            file_type = 'huggingface'
        
        # 生成embeddings
        embeddings = self.generate_embeddings(texts)
        
        # 保存embeddings
        output_file = output_dir / f"{csv_path.stem}_embeddings.pkl"
        embedding_data = {
            'embeddings': embeddings,
            'texts': texts,
            'file_type': file_type,
            'original_file': str(csv_path),
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1]
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        logger.info(f"Embeddings已保存到: {output_file}")
        logger.info(f"Embedding维度: {embeddings.shape}")
        
        return {
            'input_file': str(csv_path),
            'output_file': str(output_file),
            'num_texts': len(texts),
            'embedding_shape': embeddings.shape,
            'file_type': file_type
        }
    
    def process_directory(self, input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        处理目录中的所有CSV文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            
        Returns:
            所有文件的处理结果
        """
        input_dir = Path(input_dir)
        results = []
        
        # 查找所有CSV文件
        csv_files = list(input_dir.glob('*.csv'))
        logger.info(f"找到{len(csv_files)}个CSV文件")
        
        for csv_file in sorted(csv_files):
            try:
                result = self.process_csv_file(csv_file, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"处理文件{csv_file}时出错: {e}")
                results.append({
                    'input_file': str(csv_file),
                    'error': str(e)
                })
        
        return results

def main():
    """主函数"""
    # 配置路径
    input_dir = "./data_process/sampled"
    output_dir = "./data_process/embeddings"
    
    # 创建embedding生成器
    generator = EmbeddingGenerator(
        model_name='stsb-roberta-base-v2',
        batch_size=32
    )
    
    # 处理所有CSV文件
    logger.info("开始生成embeddings...")
    results = generator.process_directory(input_dir, output_dir)
    
    # 打印结果摘要
    logger.info("\n" + "="*50)
    logger.info("处理结果摘要:")
    logger.info("="*50)
    
    successful = 0
    failed = 0
    
    for result in results:
        if 'error' in result:
            logger.error(f"❌ {result['input_file']}: {result['error']}")
            failed += 1
        else:
            logger.info(f"✅ {result['input_file']}")
            logger.info(f"   输出: {result['output_file']}")
            logger.info(f"   数据量: {result['num_texts']}")
            logger.info(f"   Embedding形状: {result['embedding_shape']}")
            logger.info(f"   文件类型: {result['file_type']}")
            successful += 1
    
    logger.info(f"\n总计: {successful}个文件成功, {failed}个文件失败")

if __name__ == "__main__":
    main()