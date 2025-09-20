# 按Source分类Parquet数据脚本使用说明

## 概述

`classify_by_source.py` 脚本用于将包含 `chat_template_kwargs` 列的parquet数据按照 `source` 字段进行分类，每个不同的source会生成一个单独的parquet文件。

## 功能特性

- 🔍 **自动分析**: 分析输入数据中所有不同的source及其分布
- 🚀 **并行处理**: 使用多进程并行处理，提高处理速度
- 📊 **数据验证**: 自动验证输出文件的正确性
- 🗂️ **自动合并**: 将同一source的多个批次文件自动合并
- 📝 **详细日志**: 提供详细的处理进度和统计信息

## 使用方法

### 基本用法

```bash
# 使用默认参数
python classify_by_source.py

# 指定输入和输出目录
python classify_by_source.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output

# 指定并行进程数
python classify_by_source.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output \
    --num-workers 16
```

### 仅分析模式

如果你只想分析数据中有哪些source，而不进行实际分类：

```bash
python classify_by_source.py \
    --input-dir /path/to/input \
    --analyze-only
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | `/mnt/hdfs/selection/yingtai_sft/tulu_3_raw_add_column` | 输入目录路径 |
| `--output-dir` | `/mnt/hdfs/selection/yingtai_sft/tulu_3_by_source` | 输出目录路径 |
| `--num-workers` | `min(cpu_count(), 8)` | 并行处理的进程数 |
| `--analyze-only` | `False` | 仅分析source分布，不进行数据分类 |

## 输入数据要求

输入目录中的parquet文件必须包含以下字段：

```python
{
    'id': str,
    'messages': [{'content': str, 'role': str}],
    'source': str,  # 用于分类的字段
    'chat_template_kwargs': {
        'custom_instructions': str,
        'enable_thinking': bool,
        'python_tools': [str],
        'xml_tools': [str]
    }
}
```

## 输出结果

脚本会在输出目录中为每个不同的source创建一个parquet文件：

```
/mnt/hdfs/selection/yingtai_sft/tulu_3_by_source/
├── source_A.parquet
├── source_B.parquet
├── source_C.parquet
└── ...
```

每个文件包含该source的所有数据样本。

## 处理流程

1. **分析阶段**: 扫描所有输入文件，统计source字段的分布
2. **分类阶段**: 并行处理每个输入文件，按source分组数据
3. **合并阶段**: 将同一source的多个临时文件合并为单个文件
4. **验证阶段**: 验证输出文件的正确性和完整性

## 性能优化建议

- **调整进程数**: 根据你的CPU核心数和内存大小调整 `--num-workers` 参数
- **监控内存**: 如果内存不足，可以减少并行进程数
- **存储性能**: 确保输入和输出目录在高性能存储上

## 示例输出

```
2024-01-20 10:00:00 - INFO - 开始分析source字段分布...
2024-01-20 10:00:05 - INFO - 找到 100 个parquet文件
2024-01-20 10:00:30 - INFO - 总样本数: 1000000
2024-01-20 10:00:30 - INFO - 发现 19 个不同的source:
2024-01-20 10:00:30 - INFO -   source_A: 150000 样本 (15.00%)
2024-01-20 10:00:30 - INFO -   source_B: 120000 样本 (12.00%)
...
2024-01-20 10:05:00 - INFO - 开始并行处理 100 个文件...
2024-01-20 10:10:00 - INFO - 文件处理完成，耗时 300.00s
2024-01-20 10:12:00 - INFO - 开始合并各source的parquet文件...
2024-01-20 10:15:00 - INFO - 所有任务完成！总耗时 900.00s
```

## 测试

使用提供的测试脚本验证功能：

```bash
python test_classify.py
```

## 注意事项

- 确保输入目录存在且包含parquet文件
- 输出目录会自动创建
- 处理大量数据时请确保有足够的磁盘空间
- 建议在处理前先使用 `--analyze-only` 模式了解数据分布

## 故障排除

### 常见问题

1. **内存不足**: 减少 `--num-workers` 参数
2. **磁盘空间不足**: 清理临时文件或使用更大的存储
3. **权限问题**: 确保对输入和输出目录有读写权限

### 日志分析

脚本会输出详细的日志信息，包括：
- 处理进度
- 错误信息
- 性能统计
- 验证结果

如果遇到问题，请查看日志中的错误信息进行排查。