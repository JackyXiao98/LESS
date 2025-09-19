# Add Column Script 使用说明

## 功能描述

`add_column.py` 脚本用于为 `allenai/tulu-3-sft-mixture` 数据集添加 `chat_template_kwargs` 列，并使用并行处理将结果保存为 parquet 格式。

## 添加的列内容

每个样本都会添加一个名为 `chat_template_kwargs` 的列，内容为：

```json
{
    "custom_instructions": "",
    "enable_thinking": false,
    "python_tools": [],
    "xml_tools": []
}
```

## 使用方法

### 基本使用

```bash
python add_column.py
```

### 自定义参数

```bash
python add_column.py \
    --output-dir /path/to/your/output \
    --batch-size 2000 \
    --num-workers 16
```

## 参数说明

- `--output-dir`: 输出目录路径
  - 默认值: `/mnt/hdfs/selection/yingtai_sft/tulu_3_raw_add_column`
  - 脚本会自动创建目录（如果不存在）

- `--batch-size`: 每个批次的样本数
  - 默认值: `1000`
  - 较大的批次可能提高效率，但会占用更多内存

- `--num-workers`: 并行处理的进程数
  - 默认值: `min(cpu_count(), 8)`
  - 建议不要超过 CPU 核心数

## 输出格式

- 输出文件格式: Parquet
- 文件命名: `batch_XXXXXX.parquet` (6位数字编号)
- 每个文件包含一个批次的处理结果

## 性能特性

1. **流式处理**: 使用 streaming=True 避免一次性加载整个数据集到内存
2. **并行处理**: 使用多进程池并行处理多个批次
3. **内存管理**: 定期清理内存，避免内存溢出
4. **错误处理**: 包含完善的错误处理和日志记录

## 依赖要求

```bash
pip install datasets pyarrow tqdm
```

## 示例输出

```
2024-01-01 10:00:00,000 - INFO - 开始处理数据集，输出目录: /mnt/hdfs/selection/yingtai_sft/tulu_3_raw_add_column
2024-01-01 10:00:00,000 - INFO - 批次大小: 1000, 工作进程数: 8
2024-01-01 10:00:01,000 - INFO - 正在加载 allenai/tulu-3-sft-mixture 数据集...
2024-01-01 10:00:05,000 - INFO - 开始处理数据...
2024-01-01 10:01:00,000 - INFO - 开始并行处理 16 个批次...
2024-01-01 10:01:30,000 - INFO - 批次处理完成: 16/16 成功, 耗时 30.00s
...
2024-01-01 10:30:00,000 - INFO - 数据处理完成！共生成 1500 个 parquet 文件
2024-01-01 10:30:00,000 - INFO - 总处理时间: 1800.00s
2024-01-01 10:30:00,000 - INFO - 总样本数: 1500000
2024-01-01 10:30:00,000 - INFO - 平均处理速度: 833.33 样本/秒
```

## 验证功能

脚本运行完成后会自动验证输出文件的正确性，检查：
- 输出目录是否存在
- 是否生成了 parquet 文件
- `chat_template_kwargs` 列是否正确添加
- 列内容是否符合预期

## 注意事项

1. 确保有足够的磁盘空间存储输出文件
2. 网络连接稳定（需要从 Hugging Face Hub 下载数据）
3. 建议在有足够内存的机器上运行
4. 可以通过调整 `batch_size` 和 `num_workers` 来优化性能