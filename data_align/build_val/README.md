# Tulu3 验证数据集构建系统

这个系统用于从多个Hugging Face Hub数据源创建高质量的验证数据集。

## 功能特性

- 支持7个主要基准数据集：MMLU、GSM8K、HumanEval (使用MBPP)、TruthfulQA、DROP、Safety (使用Anthropic/hh-rlhf)、Hendrycks' MATH
- 统一的JSON Lines输出格式
- 模块化的类架构，易于扩展
- 完整的测试和验证功能

## 输出格式

每个`.jsonl`文件包含JSON对象，格式如下：
```json
{
  "id": "数据集名称_索引",
  "messages": [
    {
      "role": "user", 
      "content": "用户问题内容"
    },
    {
      "role": "assistant",
      "content": "助手回答内容"
    }
  ],
  "source": "数据集名称"
}
```

## 使用方法

### 1. 安装依赖
```bash
pip install datasets tqdm
```

### 2. 测试单个数据集
```bash
# 测试MMLU数据集，显示2个样本
python get_validation.py --test --dataset mmlu --samples 2

# 测试所有数据集，每个显示5个样本
python get_validation.py --test --samples 5
```

### 3. 构建所有验证数据集
```bash
python get_validation.py --build
```

### 4. 使用测试脚本
```bash
# 测试所有数据集
python test_validation.py

# 测试特定数据集
python test_validation.py --dataset gsm8k --samples 3
```

## 支持的数据集

| 数据集 | 来源 | 数据量 | 描述 |
|--------|------|--------|------|
| MMLU | `cais/mmlu` | 1,531 | 多任务语言理解 |
| GSM8K | `gsm8k` | 1,000 | 小学数学问题 |
| HumanEval | `mbpp` | 500 | 代码生成 (使用MBPP作为代理) |
| TruthfulQA | `truthful_qa` | 817 | 真实性问答 |
| DROP | `drop` | 9,535 | 阅读理解 |
| Safety | `Anthropic/hh-rlhf` | 2,609 | 安全性对话 |
| Hendrycks Math | `EleutherAI/hendrycks_math` | 7,500 | 数学问题求解 |

**总计：23,492 条验证数据**

## 输出目录结构

```
tulu3_validation/
├── mmlu/
│   └── validation.jsonl
├── gsm8k/
│   └── validation.jsonl
├── humaneval/
│   └── validation.jsonl
├── truthfulqa/
│   └── validation.jsonl
├── drop/
│   └── validation.jsonl
├── safety/
│   └── validation.jsonl
└── hendrycks_math/
    └── validation.jsonl
```

## 扩展新数据集

要添加新的数据集处理器：

1. 继承`BaseDatasetProcessor`类
2. 实现`load_and_process_data()`方法
3. 在`ValidationDatasetBuilder`中注册新处理器
4. 在测试函数中添加相应的测试逻辑

## 文件说明

- `get_validation.py`: 主要的数据集构建脚本
- `test_validation.py`: 独立的测试脚本
- `README.md`: 本说明文件
- `tulu3_validation/`: 生成的验证数据集目录

## 注意事项

- 确保有足够的磁盘空间（约15GB）
- 网络连接稳定，用于下载Hugging Face数据集
- 某些数据集可能需要较长时间下载和处理