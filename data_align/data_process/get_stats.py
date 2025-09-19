from collections import Counter
from datasets import load_dataset

def count_source_values():
    """
    加载 allenai/tulu-3-sft-mixture 数据集，并统计 'train' 分割中
    'source' 字段每个值的出现次数。
    """
    print("正在加载数据集 'allenai/tulu-3-sft-mixture'...")
    
    try:
        # 从 Hugging Face Hub 加载数据集的训练分割
        # stream=True 模式可以避免一次性将整个数据集下载到内存，对于大数据集更高效
        dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
        print("数据集加载成功，开始统计 'source' 字段...")
        
        # 使用 collections.Counter 来高效地统计每个 source 值的数量
        source_counts = Counter()
        
        # 迭代数据集并更新计数器
        for example in dataset:
            source_value = example.get('source')
            if source_value:
                source_counts[source_value] += 1
        
        print("\n--- 统计结果 ---")
        
        if not source_counts:
            print("数据集中未找到 'source' 字段或数据集为空。")
            return

        # 按照数据量从高到低排序并打印结果
        total_count = sum(source_counts.values())
        print(f"总计数据量: {total_count}")
        print("各 'source' 的数据量分布如下 (从高到低):")
        
        for source, count in source_counts.most_common():
            percentage = (count / total_count) * 100
            print(f"- {source:<25}: {count:>8} 条\t({percentage:.2f}%)")
            
    except Exception as e:
        print(f"在处理过程中发生错误: {e}")
        print("请检查您的网络连接或数据集名称是否正确。")


import time
from datasets import load_dataset
from transformers import AutoTokenizer

def calculate_average_token_length():
    """
    加载 allenai/tulu-3-sft-mixture 数据集，使用 Qwen2 Tokenizer 
    统计并计算每条数据的平均 token 长度。
    """
    # --- 配置 ---
    # 数据集名称
    DATASET_NAME = "allenai/tulu-3-sft-mixture"
    
    # Tokenizer 模型名称
    TOKENIZER_NAME = "Qwen/Qwen3-8B"
    
    # --- 初始化 ---
    print(f"正在加载 Tokenizer: '{TOKENIZER_NAME}'...")
    try:
        # 加载 Qwen2 tokenizer，需要设置 trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"加载 Tokenizer失败: {e}")
        print("请确保已安装所有必要的依赖库 (transformers, accelerate, tiktoken)。")
        return

    print(f"正在以流式方式加载数据集: '{DATASET_NAME}'...")
    # 使用流式处理，避免内存溢出
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    # --- 统计 ---
    total_tokens = 0
    total_samples = 0
    start_time = time.time()

    print("开始遍历数据集并计算 Token 长度...")
    
    # 遍历数据集的每一条样本
    for example in dataset:
        # 数据集中的 'messages' 字段是一个包含多轮对话的列表
        # 我们需要将所有对话内容合并成一个长字符串进行计费
        if 'messages' in example and isinstance(example['messages'], list):
            # 将所有轮次对话的 'content' 字段用换行符连接起来
            full_text = "\n".join(
                [msg['content'] for msg in example['messages'] if msg and 'content' in msg and isinstance(msg['content'], str)]
            )
            
            # 使用 tokenizer 对文本进行编码，并计算 token 数量
            # tokenizer.encode 比 tokenizer() 更直接，仅返回 token ID 列表
            token_ids = tokenizer.encode(full_text, add_special_tokens=True)
            
            total_tokens += len(token_ids)
            total_samples += 1
            
            # 每处理 1000 条数据就打印一次进度，方便观察
            if total_samples % 1000 == 0:
                elapsed_time = time.time() - start_time
                samples_per_sec = total_samples / elapsed_time
                # 使用 '\r' 实现原地更新，避免刷屏
                print(f"已处理 {total_samples} 条数据... ({samples_per_sec:.2f} 条/秒)", end='\r')
                average_length = total_tokens / total_samples
                print("\n--- 统计结果 ---")
                print(f"总共处理的数据条数: {total_samples:,}")
                print(f"总共统计的 Token 数: {total_tokens:,}")
                print(f"每条数据的平均 Token 长度: {average_length:.2f}")
    
    # 换行以结束进度条的显示
    print("\n数据集遍历完成。")

    # --- 计算并输出结果 ---
    if total_samples > 0:
        average_length = total_tokens / total_samples
        
        print("\n--- 统计结果 ---")
        print(f"总共处理的数据条数: {total_samples:,}")
        print(f"总共统计的 Token 数: {total_tokens:,}")
        print(f"每条数据的平均 Token 长度: {average_length:.2f}")
    else:
        print("数据集中没有找到任何可处理的样本。")


if __name__ == "__main__":
    calculate_average_token_length()