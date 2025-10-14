# build_dataset_general.py 使用说明

## 概述

`build_dataset_general.py` 是一个用于处理通用数据集的脚本，模仿了 `build_sft_dataset.py` 的结构，专门用于处理 parquet 格式的数据文件。

## 功能特点

1. **Parquet文件处理**: 直接读取和处理 parquet 格式的数据文件
2. **Token预算控制**: 根据预设的token数量限制处理数据
3. **多种数据格式支持**: 支持多种消息格式（messages、question/response、instruction/output等）
4. **数据过滤**: 自动过滤数学相关内容
5. **标准化输出**: 输出标准的SFT训练格式

## 数据集配置

脚本根据图二的token数目配置了以下数据集：

| 数据集名称 | Token预算 | 文件名 |
|-----------|----------|--------|
| nemotron_qwen3_2507_no_think_chat | 6,302,185 | nemotron_qwen3_2507_no_think_chat_8k.parquet |
| open_r1_qwen3_2507_think_coding | 4,005,982 | open_r1_qwen3_2507_think_coding_8k.parquet |
| open_r1_qwen3_2507_think_math | 10,667,522 | open_r1_qwen3_2507_think_math_8k.parquet |
| safety_cn_bias | 19,200,145 | safety_cn_bias.parquet |
| safety_rated_seal_safety_tier1 | 6,880,195 | safety_rated_seal_safety_tier1.parquet |
| safety_rated_seal_safety_tier2 | 11,949,778 | safety_rated_seal_safety_tier2.parquet |
| safety_rated_seal_safety_tier3 | 9,969,993 | safety_rated_seal_safety_tier3.parquet |
| tulu3_qwen3_2507_no_think_coding | 8,928,727 | tulu3_qwen3_2507_no_think_coding_8k.parquet |
| tulu3_qwen3_2507_no_think_instruction | 1,117,800 | tulu3_qwen3_2507_no_think_instruction_8k.parquet |
| tulu3_qwen3_2507_no_think_knowledge | 4,188,235 | tulu3_qwen3_2507_no_think_knowledge_8k.parquet |
| tulu3_qwen3_2507_no_think_math | 8,928,557 | tulu3_qwen3_2507_no_think_math_8k.parquet |
| tulu3_qwen3_2507_no_think_multilingual | 6,534,542 | tulu3_qwen3_2507_no_think_multilingual_8k.parquet |
| tulu3_qwen3_2507_think_coding | 10,667,639 | tulu3_qwen3_2507_think_coding_8k.parquet |
| tulu3_qwen3_2507_think_knowledge | 10,667,639 | tulu3_qwen3_2507_think_knowledge_8k.parquet |
| tulu3_qwen3_2507_think_multilingual | 10,665,255 | tulu3_qwen3_2507_think_multilingual_8k.parquet |

## 使用方法

### 1. 修改源数据路径

在运行脚本之前，需要修改 `main()` 函数中的 `source_data_path` 变量：

```python
source_data_path = "/mnt/bn/pilab0/yt/general_n_safety_datasets_250925"
```

将其修改为实际的数据文件路径。

### 2. 运行脚本

```bash
cd /Users/bytedance/Desktop/Github/LESS/data_sql_rewrite
python build_dataset_general.py
```

### 3. 输出结果

脚本会在当前目录下创建 `general_datasets_parquet` 文件夹，包含处理后的数据集文件：

```
general_datasets_parquet/
├── nemotron_qwen3_2507_no_think_chat.parquet
├── open_r1_qwen3_2507_think_coding.parquet
├── open_r1_qwen3_2507_think_math.parquet
├── safety_cn_bias.parquet
├── safety_rated_seal_safety_tier1.parquet
├── safety_rated_seal_safety_tier2.parquet
├── safety_rated_seal_safety_tier3.parquet
├── tulu3_qwen3_2507_no_think_coding.parquet
├── tulu3_qwen3_2507_no_think_instruction.parquet
├── tulu3_qwen3_2507_no_think_knowledge.parquet
├── tulu3_qwen3_2507_no_think_math.parquet
├── tulu3_qwen3_2507_no_think_multilingual.parquet
├── tulu3_qwen3_2507_think_coding.parquet
├── tulu3_qwen3_2507_think_knowledge.parquet
└── tulu3_qwen3_2507_think_multilingual.parquet
```

## 输出格式

每个输出的parquet文件包含以下字段：

- `id`: 样本唯一标识符
- `messages`: 标准的对话格式消息列表
- `source`: 数据来源标识
- `chat_template_kwargs`: 聊天模板参数

## 依赖项

脚本需要以下Python包：

```
datasets
tiktoken
pyarrow
pandas
tqdm
```

## 注意事项

1. 确保源数据路径正确且文件存在
2. 脚本会自动过滤包含数学关键词的样本
3. Token预算控制确保不会超出预设限制
4. 如果源文件不存在，脚本会跳过该数据集并继续处理其他数据集

## 自定义配置

如需修改配置，可以调整 `main()` 函数中的 `datasets_config` 字典：

- `parquet_file`: 源parquet文件名
- `token_budget`: 该数据集的token预算限制

## 错误处理

脚本包含完善的错误处理机制：

- 文件不存在时会跳过并记录警告
- 数据格式错误时会跳过样本并记录错误
- 处理过程中的异常会被捕获并记录