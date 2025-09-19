# Dolly数据集ID验证工具

本工具用于验证dolly数据集在分布式训练中的ID映射正确性，确保`get_info.py`中获取的`dataset_id`与`collect_grad_reps.py`中的ID提取逻辑一致。

## 文件结构

```
test/
├── validate_dolly_id.py           # 主验证脚本
├── run_validate_dolly_id.sh       # 运行脚本
├── test_dolly_validator.py        # 单元测试脚本
└── DOLLY_ID_VALIDATION_README.md  # 本说明文档
```

## 核心功能

### 1. validate_dolly_id.py
专门针对dolly数据集的ID验证脚本，主要功能：

- **数据分析**: 分析dolly数据集的ID格式和分布
- **分布式采样验证**: 模拟`get_info.py`的分布式数据加载过程
- **ID映射验证**: 验证`DistributedSampler`获取的索引与原始数据ID的对应关系
- **详细报告**: 生成详细的验证报告和统计信息

### 2. run_validate_dolly_id.sh
便捷的运行脚本，支持：

- 自动检测数据文件存在性
- 灵活配置GPU数量和样本数
- 单GPU和多GPU分布式模式自动切换
- 结果摘要显示

### 3. test_dolly_validator.py
单元测试脚本，用于验证核心功能：

- ID提取函数测试
- 数据加载功能测试
- 边界情况测试

## 使用方法

### 快速开始

```bash
# 进入test目录
cd /Users/bytedance/Desktop/Github/LESS/test

# 使用默认参数（8个GPU，100个样本）
./run_validate_dolly_id.sh

# 自定义参数
./run_validate_dolly_id.sh 4 50  # 4个GPU，50个样本
```

### 详细使用

#### 1. 直接运行验证脚本

```bash
# 单GPU模式
python validate_dolly_id.py \
    --data_file ../data/train/processed/dolly/dolly_data.jsonl \
    --max_samples 100 \
    --output_file results.json

# 多GPU分布式模式
torchrun --nproc_per_node=8 --master_port=29501 \
    validate_dolly_id.py \
    --data_file ../data/train/processed/dolly/dolly_data.jsonl \
    --max_samples 100 \
    --output_file results.json
```

#### 2. 运行单元测试

```bash
python test_dolly_validator.py
```

## 验证逻辑

### ID提取验证

脚本验证以下关键映射关系：

1. **原始数据ID**: 从`dolly_data.jsonl`中读取的`id`字段（如`"dolly_125"`）
2. **数字提取**: 从dolly ID中提取数字部分（如`125`）
3. **采样器索引**: `DistributedSampler`分配给当前GPU的数据索引
4. **数据集索引**: 处理后数据集中的实际索引

### 验证步骤

1. **数据加载**: 加载原始dolly数据，分析ID格式
2. **数据集创建**: 使用`get_training_dataset`创建处理后的数据集
3. **分布式采样**: 创建`DistributedSampler`进行数据分配
4. **索引映射**: 验证采样器索引与原始数据ID的对应关系
5. **结果统计**: 计算正确率和错误统计

## 输出结果

### 控制台输出示例

```
开始验证dolly数据集ID提取
数据文件: ../data/train/processed/dolly/dolly_data.jsonl
最大样本数: 100
GPU配置: Rank 0/8

数据分析完成:
  总样本数: 100
  有ID字段: 100
  dolly格式ID: 100
  其他格式ID: 0
  无ID字段: 0
  ID示例: ['dolly_0', 'dolly_1', 'dolly_2', 'dolly_3', 'dolly_4']
  ID数字范围: 0 - 99

[Rank 0] 验证完成: 12/12 正确 (100.00%)

================================================================================
详细验证结果
================================================================================
总体统计:
  参与验证的GPU数: 8
  总处理样本数: 96
  正确映射: 96
  错误映射: 0
  缺失原始ID: 0
  总体正确率: 100.00%
  验证结果: ✓ 通过
```

### JSON输出文件

```json
[
  {
    "rank": 0,
    "world_size": 8,
    "original_data_size": 100,
    "dataset_size": 100,
    "dataloader_size": 12,
    "processed_batches": 12,
    "correct_id_mappings": 12,
    "incorrect_id_mappings": 0,
    "missing_original_id": 0,
    "correct_rate": 1.0,
    "mapping_examples": [
      {
        "batch_idx": 0,
        "dataset_idx": 0,
        "original_id": "dolly_0",
        "extracted_id": "dolly_0",
        "is_correct": true,
        "status": "correct",
        "original_number": 0,
        "expected_dataset_idx": 0,
        "actual_dataset_idx": 0,
        "index_matches": true
      }
    ]
  }
]
```

## 参数说明

### validate_dolly_id.py 参数

- `--data_file`: dolly数据文件路径（必需）
- `--max_samples`: 最大测试样本数，默认100
- `--output_file`: 输出结果文件路径（可选）

### run_validate_dolly_id.sh 参数

- 第1个参数: GPU数量，默认8
- 第2个参数: 最大样本数，默认100

### 环境变量

- `MASTER_PORT`: 分布式训练主端口，默认29501
- `CUDA_VISIBLE_DEVICES`: 自动设置，基于GPU数量

## 验证重点

### 关键验证点

1. **ID格式一致性**: 确保dolly ID格式为`dolly_数字`
2. **索引映射正确性**: 验证`DistributedSampler`的索引分配
3. **数据完整性**: 确保所有样本都被正确处理
4. **分布式一致性**: 验证多GPU环境下的数据分配

### 对应代码位置

- **get_info.py L217-224**: `DistributedSampler`的使用和数据加载
- **collect_grad_reps.py L290-293**: 采样器索引获取逻辑
- **collect_grad_reps.py L299-302**: 原始数据集ID提取逻辑

## 故障排除

### 常见问题

1. **数据文件不存在**
   ```
   错误: 数据文件不存在: ../data/train/processed/dolly/dolly_data.jsonl
   ```
   解决: 检查数据文件路径是否正确

2. **GPU数量配置错误**
   ```
   错误: GPU数量必须在1-8之间
   ```
   解决: 使用有效的GPU数量（1-8）

3. **分布式初始化失败**
   ```
   RuntimeError: Default process group has not been initialized
   ```
   解决: 确保使用`torchrun`启动多GPU模式

4. **ID映射错误**
   ```
   验证结果: ✗ 失败
   ```
   解决: 检查数据预处理逻辑和采样器配置

### 调试建议

1. **先运行单元测试**: `python test_dolly_validator.py`
2. **使用小样本测试**: 设置`max_samples=10`进行快速验证
3. **单GPU模式调试**: 先用1个GPU验证基本功能
4. **检查日志输出**: 查看详细的映射示例和错误信息

## 扩展性

### 支持其他数据集

要支持其他数据集格式，需要修改：

1. **ID提取逻辑**: 修改`extract_dolly_id_number`函数
2. **数据格式分析**: 调整`load_and_analyze_dolly_data`函数
3. **验证逻辑**: 更新ID对比和验证规则

### 性能优化

- 使用`max_samples`参数限制测试样本数
- 调整批次大小以适应内存限制
- 使用更少的GPU进行快速验证

## 总结

本工具提供了完整的dolly数据集ID验证解决方案，确保分布式训练中数据采样的正确性。通过详细的验证报告和灵活的配置选项，可以有效发现和解决ID映射问题。