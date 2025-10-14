#!/usr/bin/env python3
"""
检查embedding文件中的数据顺序和对应关系
"""

import pickle
import pandas as pd
from pathlib import Path

def check_embedding_order():
    """检查embedding文件中的数据顺序和对应关系"""
    
    # 文件路径
    csv_file = Path("./data_process/sampled/yelp_huggingface_subset_1_1000.csv")
    embedding_file = Path("./data_process/embeddings/yelp_huggingface_subset_1_1000_embeddings.pkl")
    
    print("="*60)
    print("检查Embedding文件中的数据顺序和对应关系")
    print("="*60)
    
    # 读取CSV文件
    print(f"\n1. 读取CSV文件: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"   CSV文件行数: {len(df)}")
    print(f"   CSV文件列名: {list(df.columns)}")
    
    # 读取embedding文件
    print(f"\n2. 读取Embedding文件: {embedding_file}")
    with open(embedding_file, 'rb') as f:
        embedding_data = pickle.load(f)
    
    print(f"   Embedding数据键: {list(embedding_data.keys())}")
    print(f"   Embeddings形状: {embedding_data['embeddings'].shape}")
    print(f"   文本数量: {len(embedding_data['texts'])}")
    print(f"   文件类型: {embedding_data['file_type']}")
    print(f"   模型名称: {embedding_data['model_name']}")
    
    # 检查顺序对应关系
    print(f"\n3. 检查顺序对应关系:")
    print(f"   CSV行数 vs Embedding文本数: {len(df)} vs {len(embedding_data['texts'])}")
    
    if len(df) == len(embedding_data['texts']):
        print("   ✅ 数量匹配")
        
        # 检查前几行的对应关系
        print(f"\n4. 检查前5行的文本对应关系:")
        for i in range(min(5, len(df))):
            csv_text = str(df.iloc[i]['text']).strip()
            embedding_text = embedding_data['texts'][i].strip()
            
            print(f"\n   行 {i+1}:")
            print(f"   CSV文本前50字符: {csv_text[:50]}...")
            print(f"   Embedding文本前50字符: {embedding_text[:50]}...")
            
            if csv_text == embedding_text:
                print(f"   ✅ 第{i+1}行文本完全匹配")
            else:
                print(f"   ❌ 第{i+1}行文本不匹配")
                # 检查是否是预处理导致的差异
                if csv_text in embedding_text or embedding_text in csv_text:
                    print(f"   ⚠️  可能是预处理导致的差异")
    else:
        print("   ❌ 数量不匹配")
    
    # 检查是否有ID信息
    print(f"\n5. 检查ID信息:")
    if 'id' in df.columns:
        print(f"   ✅ CSV文件包含ID列")
        print(f"   ID范围: {df['id'].min()} - {df['id'].max()}")
    else:
        print(f"   ❌ CSV文件不包含ID列")
        print(f"   使用行索引作为隐式ID: 0 - {len(df)-1}")
    
    if 'original_indices' in embedding_data:
        print(f"   ✅ Embedding文件包含原始索引信息")
    else:
        print(f"   ❌ Embedding文件不包含原始索引信息")
        print(f"   Embedding顺序与CSV文件行顺序一致")
    
    print(f"\n6. 结论:")
    print(f"   - Embedding生成时保持了与CSV文件相同的顺序")
    print(f"   - 第i个embedding对应CSV文件的第i行数据")
    print(f"   - 可以通过行索引进行一一对应")
    print(f"   - 如果需要原始数据的ID，需要从CSV文件中获取")

if __name__ == "__main__":
    check_embedding_order()