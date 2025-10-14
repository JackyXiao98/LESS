import json
import os
from datasets import load_dataset, Dataset, Features, Value, Sequence
import tiktoken
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
from tqdm import tqdm
import pandas as pd

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

# 使用 cl100k_base 编码器, 它被广泛用于 OpenAI 的模型中, 作为一个通用的Token计算标准
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """使用tiktoken计算文本的token数量"""
    return len(tokenizer.encode(text))

class ParquetDatasetProcessor:
    """Parquet数据集处理基类"""
    
    def __init__(self, dataset_name, parquet_path, token_budget):
        self.dataset_name = dataset_name
        self.parquet_path = parquet_path
        self.token_budget = token_budget
        
    def load_parquet(self):
        """加载parquet文件"""
        try:
            df = pd.read_parquet(self.parquet_path)
            logger.info(f"成功加载 {self.dataset_name} 数据集，共 {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"Error loading parquet file {self.dataset_name}: {e}")
            return None
    
    def validate_sample(self, sample):
        """验证样本数据，子类需要重写此方法"""
        raise NotImplementedError("Subclasses must implement validate_sample method")
    
    def build_messages(self, sample):
        """构建SFT格式的消息，子类需要重写此方法"""
        raise NotImplementedError("Subclasses must implement build_messages method")
    
    def should_filter_sample(self, sample):
        """判断是否应该过滤样本，默认不过滤"""
        return False
    
    def process(self):
        """处理数据集的主要方法"""
        logger.info(f"Processing {self.dataset_name} dataset...")
        
        df = self.load_parquet()
        if df is None:
            return []
        
        # 随机打乱数据顺序，实现随机选取
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        logger.info(f"Shuffled {len(df)} samples for random selection")
        
        processed_count = 0
        skipped_count = 0
        filtered_count = 0
        total_tokens = 0
        processed_data = []
        
        for _, row in df.iterrows():
            try:
                sample = row.to_dict()
                
                # 数据验证
                if not self.validate_sample(sample):
                    skipped_count += 1
                    continue
                
                # 过滤检查
                if self.should_filter_sample(sample):
                    filtered_count += 1
                    continue
                
                # 构建SFT格式的消息
                messages = self.build_messages(sample)
                
                # 计算token数量
                message_text = json.dumps(messages, ensure_ascii=False)
                tokens = len(tokenizer.encode(message_text))
                
                if total_tokens + tokens > self.token_budget:
                    break
                
                total_tokens += tokens
                processed_count += 1
                
                # 添加chat_template_kwargs列
                processed_sample = {
                    "messages": messages,
                    "source": self.dataset_name,
                    "chat_template_kwargs": CHAT_TEMPLATE_KWARGS.copy()
                }
                processed_data.append(processed_sample)
                
            except Exception as e:
                logger.error(f"Error processing sample in {self.dataset_name}: {e}")
                skipped_count += 1
                continue
        
        # 记录处理结果
        if filtered_count > 0:
            logger.info(f"{self.dataset_name} dataset processed: {processed_count} samples, {skipped_count} skipped, {filtered_count} filtered, {total_tokens} tokens")
        else:
            logger.info(f"{self.dataset_name} dataset processed: {processed_count} samples, {skipped_count} skipped, {total_tokens} tokens")
        
        return processed_data


class GeneralParquetProcessor(ParquetDatasetProcessor):
    """通用Parquet数据集处理器"""
    
    def __init__(self, dataset_name, parquet_path, token_budget):
        super().__init__(dataset_name, parquet_path, token_budget)
        # 过滤关键词
        self.filter_keywords = ["math", "calculate", "solve"]
    
    def validate_sample(self, sample):
        """验证通用数据集样本"""
        # 检查是否有messages字段
        if 'messages' in sample and isinstance(sample['messages'], list):
            # 如果已经是messages格式，检查是否有效
            messages = sample['messages']
            if len(messages) >= 2:
                return True
        
        # 检查其他可能的字段组合
        if sample.get('question') and sample.get('response'):
            return True
        if sample.get('instruction') and sample.get('output'):
            return True
        if sample.get('input') and sample.get('output'):
            return True
        
        return False
    
    def should_filter_sample(self, sample):
        """过滤数学相关内容"""
        # 检查不同字段中的数学关键词
        text_to_check = ""
        
        if 'messages' in sample and isinstance(sample['messages'], list):
            for msg in sample['messages']:
                if isinstance(msg, dict) and 'content' in msg:
                    text_to_check += msg['content'] + " "
        elif sample.get('question'):
            text_to_check = sample['question']
        elif sample.get('instruction'):
            text_to_check = sample['instruction']
        elif sample.get('input'):
            text_to_check = sample['input']
        
        text_lower = text_to_check.lower()
        return any(keyword in text_lower for keyword in self.filter_keywords)
    
    def build_messages(self, sample):
        """构建通用数据集的消息格式"""
        # 如果已经是messages格式，直接使用
        if 'messages' in sample and isinstance(sample['messages'], list):
            return sample['messages']
        
        messages = []
        
        # 添加系统提示（如果存在）
        system_prompt = sample.get('system_prompt', '').strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 根据不同的字段组合构建消息
        if sample.get('question') and sample.get('response'):
            messages.extend([
                {"role": "user", "content": sample['question'].strip()},
                {"role": "assistant", "content": sample['response'].strip()}
            ])
        elif sample.get('instruction') and sample.get('output'):
            messages.extend([
                {"role": "user", "content": sample['instruction'].strip()},
                {"role": "assistant", "content": sample['output'].strip()}
            ])
        elif sample.get('input') and sample.get('output'):
            messages.extend([
                {"role": "user", "content": sample['input'].strip()},
                {"role": "assistant", "content": sample['output'].strip()}
            ])
        
        return messages


def save_dataset_to_parquet(dataset_data, dataset_name, output_dir):
    """
    将单个数据集保存为 parquet 文件
    """
    try:
        # 验证数据结构
        if len(dataset_data) == 0:
            logger.warning(f"数据集 {dataset_name} 为空，跳过")
            return None
            
        if not isinstance(dataset_data[0], dict):
            logger.error(f"数据集 {dataset_name}: 数据格式错误，期望字典但得到 {type(dataset_data[0])}")
            return None
        
        # 为每个样本添加id字段
        for i, sample in enumerate(dataset_data):
            if 'id' not in sample:
                sample['id'] = f"{dataset_name}_sample_{i:06d}"
        
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
        dataset = Dataset.from_list(dataset_data, features=features)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为 parquet 文件
        output_file = os.path.join(output_dir, f"{dataset_name}.parquet")
        dataset.to_parquet(output_file)
        
        logger.info(f"数据集 {dataset_name} 已保存到 {output_file} ({len(dataset_data)} 个样本)")
        return output_file
    except Exception as e:
        logger.error(f"保存数据集 {dataset_name} 时出错: {e}")
        return None


def save_datasets_by_source(datasets_dict, output_dir):
    """
    为每个数据集保存单独的parquet文件
    """
    logger.info(f"开始保存数据集，每个数据集一个文件")
    
    successful_files = []
    
    for dataset_name, dataset_data in datasets_dict.items():
        logger.info(f"正在保存 {dataset_name} 数据集...")
        
        output_file = save_dataset_to_parquet(dataset_data, dataset_name, output_dir)
        if output_file:
            successful_files.append(output_file)
    
    logger.info(f"成功保存 {len(successful_files)} 个数据集文件")
    return successful_files


def main():
    """主函数，处理指定路径下的parquet文件并保存为新的数据集"""
    
    # 数据集配置 - 根据图二的token数目配置
    # 注意：由于无法访问原始路径，这里使用占位符路径，实际使用时需要修改
    source_data_path = "/mnt/bn/pilab0/yt/general_n_safety_datasets_250925"
    
    datasets_config = {
        "nemotron_qwen3_2507_no_think_chat": {
            "parquet_file": "nemotron_qwen3_2507_no_think_chat_8k.parquet",
            "token_budget": 6302185  # 根据图二的数据
        },
        "open_r1_qwen3_2507_think_coding": {
            "parquet_file": "open_r1_qwen3_2507_think_coding_8k.parquet", 
            "token_budget": 4005982
        },
        "open_r1_qwen3_2507_think_math": {
            "parquet_file": "open_r1_qwen3_2507_think_math_8k.parquet",
            "token_budget": 10667522
        },
        "tulu3_qwen3_2507_no_think_coding": {
            "parquet_file": "tulu3_qwen3_2507_no_think_coding_8k.parquet",
            "token_budget": 8928727
        },
        "tulu3_qwen3_2507_no_think_instruction": {
            "parquet_file": "tulu3_qwen3_2507_no_think_instruction_8k.parquet",
            "token_budget": 1117800
        },
        "tulu3_qwen3_2507_no_think_knowledge": {
            "parquet_file": "tulu3_qwen3_2507_no_think_knowledge_8k.parquet",
            "token_budget": 4188235
        },
        "tulu3_qwen3_2507_no_think_math": {
            "parquet_file": "tulu3_qwen3_2507_no_think_math_8k.parquet",
            "token_budget": 8928557
        },
        "tulu3_qwen3_2507_no_think_multilingual": {
            "parquet_file": "tulu3_qwen3_2507_no_think_multilingual_8k.parquet",
            "token_budget": 6534542
        },
        "tulu3_qwen3_2507_think_knowledge": {
            "parquet_file": "tulu3_qwen3_2507_think_knowledge_8k.parquet",
            "token_budget": 10667639
        },
        "tulu3_qwen3_2507_think_multilingual": {
            "parquet_file": "tulu3_qwen3_2507_think_multilingual_8k.parquet",
            "token_budget": 10665255
        }
    }
    
    logger.info("开始构建通用数据集...")
    
    # 处理所有数据集
    datasets_dict = {}
    
    for dataset_name, config in datasets_config.items():
        logger.info(f"开始处理 {dataset_name} 数据集...")
        
        try:
            # 构建完整的parquet文件路径
            parquet_path = os.path.join(source_data_path, config["parquet_file"])
            
            # 检查文件是否存在
            if not os.path.exists(parquet_path):
                logger.warning(f"文件不存在: {parquet_path}，跳过处理")
                continue
            
            # 创建处理器并处理数据
            processor = GeneralParquetProcessor(
                dataset_name, 
                parquet_path, 
                config["token_budget"]
            )
            
            processed_data = processor.process()
            
            if processed_data:
                datasets_dict[dataset_name] = processed_data
                logger.info(f"{dataset_name} 数据集处理完成，获得 {len(processed_data)} 个样本")
            else:
                logger.warning(f"{dataset_name} 数据集处理失败或无数据")
                
        except Exception as e:
            logger.error(f"处理 {dataset_name} 数据集时出错: {e}")
            continue
    
    # 为每个数据集保存单独的parquet文件
    if datasets_dict:
        output_dir = "general_datasets_parquet"
        logger.info(f"开始保存数据到 {output_dir} 目录...")
        
        successful_files = save_datasets_by_source(datasets_dict, output_dir)
        
        # 统计总样本数
        total_samples = sum(len(data) for data in datasets_dict.values())
        
        logger.info(f"所有数据集处理完成！")
        logger.info(f"成功保存 {len(successful_files)} 个数据集文件，总计 {total_samples} 个样本")
        logger.info(f"输出目录: {output_dir}")
        
        # 显示每个数据集的详细信息
        for dataset_name, data in datasets_dict.items():
            logger.info(f"  - {dataset_name}.parquet: {len(data)} 个样本")
    else:
        logger.error("没有成功处理任何数据集")


if __name__ == "__main__":
    main()