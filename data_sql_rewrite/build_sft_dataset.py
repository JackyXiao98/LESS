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

# --- 数据集配置 ---
# 配置来源于 "instructions.md" 文件
# 每个数据集的处理器函数在下面定义
DATASET_CONFIG = [
    {
        "name": "spider",
        "huggingface_id": "xlangai/spider",  # 修正为正确的数据集ID
        "split": "train",
        "token_budget": 25_000_000,
        "processor": "process_sql_dataset",
        "output_file": "spider_sft.jsonl",
        "question_field": "question",
        "answer_field": "query"
    },
    {
        "name": "bird",
        "huggingface_id": "birdsql/bird23-train-filtered",  # 修正为正确的数据集ID
        "split": "train",
        "token_budget": 15_000_000,
        "processor": "process_sql_dataset",
        "output_file": "bird_sft.jsonl",
        "question_field": "question",
        "answer_field": "SQL"
    },
    {
        "name": "code",
        "huggingface_id": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "token_budget": 20_000_000,
        "processor": "process_code_dataset",
        "output_file": "code_sft.jsonl"
    },
    {
        "name": "math",
        "huggingface_id": "gsm8k",
        "split": "train",
        "subset": "main",
        "token_budget": 15_000_000,
        "processor": "process_math_dataset",
        "output_file": "math_sft.jsonl"
    },
    {
        "name": "general",
        "huggingface_id": "Open-Orca/OpenOrca",
        "split": "train",
        "token_budget": 25_000_000,
        "processor": "process_general_dataset",
        "output_file": "general_sft.jsonl"
    },
]

def count_tokens(text):
    """使用tiktoken计算文本的token数量"""
    return len(tokenizer.encode(text))

class DatasetProcessor:
    """数据集处理基类"""
    
    def __init__(self, dataset_name, huggingface_id, token_budget, config=None):
        self.dataset_name = dataset_name
        self.huggingface_id = huggingface_id
        self.token_budget = token_budget
        self.config = config
        
    def load_dataset(self):
        """加载数据集"""
        try:
            if self.config:
                return load_dataset(self.huggingface_id, self.config, split="train")
            else:
                return load_dataset(self.huggingface_id, split="train")
        except Exception as e:
            logger.error(f"Error loading dataset {self.dataset_name}: {e}")
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
        
        dataset = self.load_dataset()
        if dataset is None:
            return []
        
        processed_count = 0
        skipped_count = 0
        filtered_count = 0
        total_tokens = 0
        processed_data = []
        
        for sample in dataset:
            try:
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


class SQLDatasetProcessor(DatasetProcessor):
    """SQL数据集处理器"""
    
    def __init__(self, dataset_name, huggingface_id, token_budget, config=None):
        super().__init__(dataset_name, huggingface_id, token_budget, config)
        # 根据数据集确定字段名映射
        if dataset_name == "spider":
            self.question_field = "question"
            self.answer_field = "query"
        elif dataset_name == "bird":
            self.question_field = "question"
            self.answer_field = "SQL"
        elif dataset_name == "gretelai":
            self.question_field = "sql_prompt"
            self.answer_field = "sql"
        elif dataset_name == "ajithnarayanan":
            self.question_field = "question"
            self.answer_field = "sql"
        elif dataset_name == "sql_context":
            self.question_field = "question"
            self.answer_field = "answer"
        else:
            # 默认字段名
            self.question_field = "question"
            self.answer_field = "answer"
        
        logger.info(f"使用字段映射: question='{self.question_field}', answer='{self.answer_field}'")
    
    def validate_sample(self, sample):
        """验证SQL数据集样本"""
        if not sample.get(self.question_field) or not isinstance(sample[self.question_field], str) or not sample[self.question_field].strip():
            return False
        if not sample.get(self.answer_field) or not isinstance(sample[self.answer_field], str) or not sample[self.answer_field].strip():
            return False
        return True
    
    def build_messages(self, sample):
        """构建SQL数据集的消息格式"""
        return [
            {"role": "user", "content": sample[self.question_field].strip()},
            {"role": "assistant", "content": sample[self.answer_field].strip()}
        ]


class CodeDatasetProcessor(DatasetProcessor):
    """代码数据集处理器"""
    
    def validate_sample(self, sample):
        """验证代码数据集样本"""
        if not sample.get('instruction') or not isinstance(sample['instruction'], str) or not sample['instruction'].strip():
            return False
        if not sample.get('output') or not isinstance(sample['output'], str) or not sample['output'].strip():
            return False
        return True
    
    def build_messages(self, sample):
        """构建代码数据集的消息格式"""
        return [
            {"role": "user", "content": sample['instruction'].strip()},
            {"role": "assistant", "content": sample['output'].strip()}
        ]


class MathDatasetProcessor(DatasetProcessor):
    """数学数据集处理器"""
    
    def validate_sample(self, sample):
        """验证数学数据集样本"""
        if not sample.get('question') or not isinstance(sample['question'], str) or not sample['question'].strip():
            return False
        if not sample.get('answer') or not isinstance(sample['answer'], str) or not sample['answer'].strip():
            return False
        return True
    
    def build_messages(self, sample):
        """构建数学数据集的消息格式"""
        return [
            {"role": "user", "content": sample['question'].strip()},
            {"role": "assistant", "content": sample['answer'].strip()}
        ]


class GeneralDatasetProcessor(DatasetProcessor):
    """通用数据集处理器"""
    
    def __init__(self, dataset_name, huggingface_id, token_budget, config=None):
        super().__init__(dataset_name, huggingface_id, token_budget, config)
        # 过滤关键词
        self.filter_keywords = ["math", "calculate", "solve"]
    
    def validate_sample(self, sample):
        """验证通用数据集样本"""
        if not sample.get('question') or not isinstance(sample['question'], str) or not sample['question'].strip():
            return False
        if not sample.get('response') or not isinstance(sample['response'], str) or not sample['response'].strip():
            return False
        return True
    
    def should_filter_sample(self, sample):
        """过滤数学相关内容"""
        question_lower = sample['question'].lower()
        return any(keyword in question_lower for keyword in self.filter_keywords)
    
    def build_messages(self, sample):
        """构建通用数据集的消息格式"""
        messages = []
        
        # 添加系统提示（如果存在）
        system_prompt = sample.get('system_prompt', '').strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend([
            {"role": "user", "content": sample['question'].strip()},
            {"role": "assistant", "content": sample['response'].strip()}
        ])
        
        return messages


# 为了保持向后兼容性，保留原有的函数接口
def process_sql_dataset(dataset_name, huggingface_id, token_budget):
    """处理SQL数据集 - 兼容性函数"""
    processor = SQLDatasetProcessor(dataset_name, huggingface_id, token_budget)
    return processor.process()

def process_code_dataset(dataset_name, huggingface_id, token_budget):
    """处理代码数据集 - 兼容性函数"""
    processor = CodeDatasetProcessor(dataset_name, huggingface_id, token_budget)
    return processor.process()

def process_math_dataset(dataset_name, huggingface_id, token_budget, config=None):
    """处理数学数据集 - 兼容性函数"""
    processor = MathDatasetProcessor(dataset_name, huggingface_id, token_budget, config)
    return processor.process()

def process_general_dataset(dataset_name, huggingface_id, token_budget):
    """处理通用数据集 - 兼容性函数"""
    processor = GeneralDatasetProcessor(dataset_name, huggingface_id, token_budget)
    return processor.process()


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
        
        # 为每个样本添加id字段
        for i, sample in enumerate(batch_data):
            if 'id' not in sample:
                sample['id'] = f"batch_{batch_idx:06d}_sample_{i:06d}"
        
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
        # 保存批次
        output_file = save_batch_to_parquet(batch_data, output_dir, batch_idx)
        return output_file
    except Exception as e:
        logger.error(f"处理批次 {batch_idx} 时出错: {e}")
        return None


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


def save_datasets_parallel(all_datasets, output_dir, batch_size=1000, num_workers=None):
    """
    使用并行处理保存所有数据集为parquet格式（按批次分割）
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    logger.info(f"开始并行保存数据，批次大小: {batch_size}, 工作进程数: {num_workers}")
    
    # 合并所有数据集
    all_data = []
    for dataset_data in all_datasets:
        all_data.extend(dataset_data)
    
    logger.info(f"总共 {len(all_data)} 个样本需要保存")
    
    # 分批处理
    batches = []
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size]
        batches.append((batch, output_dir, i // batch_size))
    
    logger.info(f"分为 {len(batches)} 个批次进行并行处理")
    
    # 并行处理
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_and_save_batch, batches),
            total=len(batches),
            desc="保存批次"
        ))
    
    # 统计结果
    successful_files = [r for r in results if r is not None]
    logger.info(f"成功保存 {len(successful_files)} 个批次文件")
    
    return successful_files


def main():
    """主函数，按顺序处理所有数据集并保存为parquet格式"""
    
    # 数据集配置
    datasets_config = {
        "spider": {
            "huggingface_id": "xlangai/spider",
            "token_budget": 50000,
            "processor": process_sql_dataset
        },
        "bird": {
            "huggingface_id": "birdsql/bird23-train-filtered",
            "token_budget": 50000,
            "processor": process_sql_dataset
        },
        "gretelai": {
            "huggingface_id": "gretelai/synthetic_text_to_sql",
            "token_budget": 50000,
            "processor": process_sql_dataset
        },
        "sql_context": {
            "huggingface_id": "b-mc2/sql-create-context",
            "token_budget": 50000,
            "processor": process_sql_dataset
        },
        "ajithnarayanan": {
            "huggingface_id": "ajithnarayanan/sql",
            "token_budget": 50000,
            "processor": process_sql_dataset
        },
        "code_alpaca": {
            "huggingface_id": "sahil2801/CodeAlpaca-20k",
            "token_budget": 100000,
            "processor": process_code_dataset
        },
        "gsm8k": {
            "huggingface_id": "gsm8k",
            "config": "main",
            "token_budget": 100000,
            "processor": process_math_dataset
        },
        "open_orca": {
            "huggingface_id": "Open-Orca/OpenOrca",
            "token_budget": 200000,
            "processor": process_general_dataset
        }
    }
    
    logger.info("开始构建SFT数据集...")
    
    # 处理所有数据集
    datasets_dict = {}
    
    for dataset_name, config in datasets_config.items():
        logger.info(f"开始处理 {dataset_name} 数据集...")
        
        try:
            # 检查是否有配置参数
            if "config" in config:
                processed_data = config["processor"](
                    dataset_name, 
                    config["huggingface_id"], 
                    config["token_budget"],
                    config["config"]
                )
            else:
                processed_data = config["processor"](
                    dataset_name, 
                    config["huggingface_id"], 
                    config["token_budget"]
                )
            
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
        output_dir = "sft_datasets_parquet"
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
