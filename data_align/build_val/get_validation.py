#!/usr/bin/env python3
"""
验证数据集构建系统
从 Hugging Face Hub 上的多个数据源创建高质量的验证数据集
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 基础输出目录
# BASE_OUTPUT_DIR = "/mnt/hdfs/selection/yingtai_sft/tulu3_validation"
BASE_OUTPUT_DIR = "./tulu3_validation"


class BaseDatasetProcessor(ABC):
    """数据集处理器基类"""
    
    def __init__(self, dataset_name: str, output_dir: str = BASE_OUTPUT_DIR):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.dataset_output_dir = os.path.join(output_dir, dataset_name)
        self.output_file = os.path.join(self.dataset_output_dir, "validation.jsonl")
        
    def ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.dataset_output_dir, exist_ok=True)
        logger.info(f"输出目录已创建: {self.dataset_output_dir}")
    
    def create_message_format(self, user_content: str, assistant_content: str, 
                            sample_id: str) -> Dict[str, Any]:
        """创建标准的消息格式"""
        return {
            "id": sample_id,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "source": self.dataset_name
        }
    
    def save_to_jsonl(self, data: List[Dict[str, Any]]):
        """保存数据到JSONL文件"""
        self.ensure_output_dir()
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"已保存 {len(data)} 条数据到 {self.output_file}")
    
    @abstractmethod
    def load_and_process_data(self) -> List[Dict[str, Any]]:
        """加载和处理数据的抽象方法"""
        pass
    
    def process(self):
        """处理数据集的主方法"""
        logger.info(f"开始处理 {self.dataset_name} 数据集...")
        try:
            processed_data = self.load_and_process_data()
            self.save_to_jsonl(processed_data)
            logger.info(f"{self.dataset_name} 数据集处理完成!")
            return processed_data
        except Exception as e:
            logger.error(f"处理 {self.dataset_name} 数据集时出错: {str(e)}")
            raise


class MMLUProcessor(BaseDatasetProcessor):
    """MMLU数据集处理器"""
    
    def __init__(self):
        super().__init__("mmlu")
    
    def load_and_process_data(self) -> List[Dict[str, Any]]:
        """处理MMLU数据集"""
        dataset = load_dataset("cais/mmlu", "all")
        validation_data = dataset["validation"]
        
        processed_data = []
        for idx, sample in enumerate(tqdm(validation_data, desc="处理MMLU数据")):
            # 构建问题和选项
            question = sample["question"]
            choices = sample["choices"]
            
            # 格式化选项
            options_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            user_content = f"{question}\n\n{options_text}"
            
            # 获取正确答案
            answer_idx = sample["answer"]
            assistant_content = chr(65 + answer_idx)  # 转换为A, B, C, D
            
            sample_id = f"mmlu_{idx}"
            
            processed_item = self.create_message_format(
                user_content, assistant_content, sample_id
            )
            processed_data.append(processed_item)
        
        return processed_data


class GSM8KProcessor(BaseDatasetProcessor):
    """GSM8K数据集处理器"""
    
    def __init__(self):
        super().__init__("gsm8k")
    
    def load_and_process_data(self) -> List[Dict[str, Any]]:
        """处理GSM8K数据集"""
        dataset = load_dataset("gsm8k", "main")
        train_data = dataset["train"]
        
        # 取前1000个样本作为验证集
        validation_samples = train_data.select(range(min(1000, len(train_data))))
        
        processed_data = []
        for idx, sample in enumerate(tqdm(validation_samples, desc="处理GSM8K数据")):
            user_content = sample["question"]
            assistant_content = sample["answer"]
            sample_id = f"gsm8k_{idx}"
            
            processed_item = self.create_message_format(
                user_content, assistant_content, sample_id
            )
            processed_data.append(processed_item)
        
        return processed_data


class HumanEvalProcessor(BaseDatasetProcessor):
    """HumanEval数据集处理器 (使用MBPP作为代理)"""
    
    def __init__(self):
        super().__init__("humaneval")
    
    def load_and_process_data(self) -> List[Dict[str, Any]]:
        """处理MBPP数据集作为HumanEval的代理"""
        dataset = load_dataset("mbpp")
        test_data = dataset["test"]
        
        processed_data = []
        for idx, sample in enumerate(tqdm(test_data, desc="处理HumanEval(MBPP)数据")):
            user_content = sample["text"]
            assistant_content = sample["code"]
            sample_id = f"humaneval_{idx}"
            
            processed_item = self.create_message_format(
                user_content, assistant_content, sample_id
            )
            processed_data.append(processed_item)
        
        return processed_data


class TruthfulQAProcessor(BaseDatasetProcessor):
    """TruthfulQA数据集处理器"""
    
    def __init__(self):
        super().__init__("truthfulqa")
    
    def load_and_process_data(self) -> List[Dict[str, Any]]:
        """处理TruthfulQA数据集"""
        dataset = load_dataset("truthful_qa", "generation")
        validation_data = dataset["validation"]
        
        processed_data = []
        for idx, sample in enumerate(tqdm(validation_data, desc="处理TruthfulQA数据")):
            user_content = sample["question"]
            assistant_content = sample["best_answer"]
            sample_id = f"truthfulqa_{idx}"
            
            processed_item = self.create_message_format(
                user_content, assistant_content, sample_id
            )
            processed_data.append(processed_item)
        
        return processed_data


class DROPProcessor(BaseDatasetProcessor):
    """DROP数据集处理器"""
    
    def __init__(self):
        super().__init__("drop")
    
    def load_and_process_data(self) -> List[Dict[str, Any]]:
        """处理DROP数据集"""
        dataset = load_dataset("drop")
        validation_data = dataset["validation"]
        
        processed_data = []
        for idx, sample in enumerate(tqdm(validation_data, desc="处理DROP数据")):
            user_content = sample["question"]
            
            # 获取第一个答案
            answers_spans = sample["answers_spans"]
            if answers_spans and len(answers_spans["spans"]) > 0:
                assistant_content = answers_spans["spans"][0]
            else:
                # 如果没有spans，尝试使用number或date
                if "number" in sample["answers_spans"] and sample["answers_spans"]["number"]:
                    assistant_content = str(sample["answers_spans"]["number"])
                elif "date" in sample["answers_spans"] and sample["answers_spans"]["date"]:
                    assistant_content = str(sample["answers_spans"]["date"])
                else:
                    continue  # 跳过没有答案的样本
            
            sample_id = f"drop_{idx}"
            
            processed_item = self.create_message_format(
                user_content, assistant_content, sample_id
            )
            processed_data.append(processed_item)
        
        return processed_data


class SafetyProcessor(BaseDatasetProcessor):
    """Safety数据集处理器 (使用Anthropic/hh-rlhf)"""
    
    def __init__(self):
        super().__init__("safety")
    
    def parse_conversation(self, conversation_text: str) -> tuple[str, str]:
        """解析对话文本，提取用户和助手的内容"""
        # 分割对话
        parts = conversation_text.split("\n\nAssistant:")
        if len(parts) != 2:
            raise ValueError("无法解析对话格式")
        
        # 提取用户内容
        human_part = parts[0]
        if not human_part.startswith("\n\nHuman:"):
            raise ValueError("对话格式不正确")
        user_content = human_part.replace("\n\nHuman:", "").strip()
        
        # 提取助手内容
        assistant_content = parts[1].strip()
        
        return user_content, assistant_content
    
    def load_and_process_data(self) -> List[Dict[str, Any]]:
        """处理Safety数据集"""
        dataset = load_dataset("Anthropic/hh-rlhf")
        test_data = dataset["test"]
        
        processed_data = []
        for idx, sample in enumerate(tqdm(test_data, desc="处理Safety数据")):
            try:
                chosen_text = sample["chosen"]
                user_content, assistant_content = self.parse_conversation(chosen_text)
                sample_id = f"safety_{idx}"
                
                processed_item = self.create_message_format(
                    user_content, assistant_content, sample_id
                )
                processed_data.append(processed_item)
            except Exception as e:
                logger.warning(f"跳过样本 {idx}: {str(e)}")
                continue
        
        return processed_data


class HendrycksMathProcessor(BaseDatasetProcessor):
    """Hendrycks' MATH数据集处理器"""
    
    def __init__(self):
        super().__init__("hendrycks_math")
    
    def load_and_process_data(self) -> List[Dict[str, Any]]:
        """处理Hendrycks' MATH数据集"""
        # 所有可用的配置
        configs = ['algebra', 'counting_and_probability', 'geometry', 
                  'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
        
        processed_data = []
        global_idx = 0
        
        for config in configs:
            dataset = load_dataset("EleutherAI/hendrycks_math", config)
            train_data = dataset["train"]
            
            for sample in tqdm(train_data, desc=f"处理Hendrycks MATH数据 ({config})"):
                user_content = sample["problem"]
                assistant_content = sample["solution"]
                sample_id = f"hendrycks_math_{global_idx}"
                
                processed_item = self.create_message_format(
                    user_content, assistant_content, sample_id
                )
                processed_data.append(processed_item)
                global_idx += 1
        
        return processed_data


class ValidationDatasetBuilder:
    """验证数据集构建器主类"""
    
    def __init__(self):
        self.processors = {
            "mmlu": MMLUProcessor(),
            "gsm8k": GSM8KProcessor(),
            "humaneval": HumanEvalProcessor(),
            "truthfulqa": TruthfulQAProcessor(),
            "drop": DROPProcessor(),
            "safety": SafetyProcessor(),
            "hendrycks_math": HendrycksMathProcessor()
        }
    
    def build_all_datasets(self):
        """构建所有数据集"""
        logger.info("开始构建所有验证数据集...")
        results = {}
        
        for name, processor in self.processors.items():
            try:
                data = processor.process()
                results[name] = len(data)
            except Exception as e:
                logger.error(f"构建 {name} 数据集失败: {str(e)}")
                results[name] = 0
        
        logger.info("所有数据集构建完成!")
        logger.info("构建结果:")
        for name, count in results.items():
            logger.info(f"  {name}: {count} 条数据")
        
        return results
    
    def build_single_dataset(self, dataset_name: str):
        """构建单个数据集"""
        if dataset_name not in self.processors:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        processor = self.processors[dataset_name]
        return processor.process()


def test_datasets(num_samples: int = 5):
    """测试函数：展示每个数据集的前几个样本"""
    logger.info(f"开始测试，每个数据集展示前 {num_samples} 个样本...")
    
    builder = ValidationDatasetBuilder()
    
    for dataset_name, processor in builder.processors.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"测试数据集: {dataset_name.upper()}")
        logger.info(f"{'='*50}")
        
        try:
            # 处理数据集
            data = processor.load_and_process_data()
            
            # 展示前几个样本
            samples_to_show = min(num_samples, len(data))
            for i in range(samples_to_show):
                sample = data[i]
                logger.info(f"\n样本 {i+1}:")
                logger.info(f"ID: {sample['id']}")
                logger.info(f"用户: {sample['messages'][0]['content'][:200]}...")
                logger.info(f"助手: {sample['messages'][1]['content'][:200]}...")
                logger.info(f"来源: {sample['source']}")
            
            logger.info(f"\n{dataset_name} 总计: {len(data)} 条数据")
            
        except Exception as e:
            logger.error(f"测试 {dataset_name} 时出错: {str(e)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="验证数据集构建器")
    parser.add_argument("--test", action="store_true", help="运行测试模式")
    parser.add_argument("--build", action="store_true", help="运行构建模式")
    parser.add_argument("--dataset", type=str, help="指定要测试的数据集")
    parser.add_argument("--samples", type=int, default=5, help="测试时显示的样本数量")
    
    args = parser.parse_args()
    
    if args.test:
        test_datasets(args.dataset, args.samples)
    elif args.build:
        # 运行构建模式
        builder = ValidationDatasetBuilder()
        builder.build_all_datasets()
    else:
        # 默认运行测试模式
        test_datasets(args.dataset, args.samples)


if __name__ == "__main__":
    main()