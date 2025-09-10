# 梯度与数据样本ID映射实现

本文档详细说明了如何在PyTorch分布式训练中实现梯度与数据样本ID的精确映射和存储。

## 概述

该实现允许在分布式训练过程中：
1. **同步保存**梯度和对应的数据样本ID
2. **精确映射**每个梯度张量行与其对应的数据样本
3. **自动聚合**所有GPU的结果到单一文件
4. **验证正确性**确保映射关系的准确性

## 文件结构

```
LESS/
├── less/data_selection/
│   ├── collect_grad_reps.py     # 修改后的梯度收集脚本
│   └── get_info.py              # 修改后的主训练脚本
├── aggregate_gradients.py       # 新增：聚合脚本
├── validate_gradient_id_mapping.py  # 新增：验证脚本
├── example_usage.py             # 新增：使用示例
└── GRADIENT_ID_MAPPING_README.md    # 本文档
```

## 核心修改

### 1. collect_grad_reps.py 修改

#### 新增功能
- **ID追踪**: 在`collect_grads`函数中添加了`dataset`参数
- **同步保存**: 修改`_save`函数同时保存梯度和ID文件
- **自动合并**: 增强`merge_and_normalize_info`函数处理ID文件

#### 关键修改点

```python
# 1. 函数签名修改
def collect_grads(model, dataloader, loss_computer, num_params, 
                 output_dir, grad_type="adam", projector=None, 
                 proj_dim=8192, device="cuda", dataset=None):  # 新增dataset参数

# 2. ID收集逻辑
sample_ids = []
for batch_idx, batch in enumerate(dataloader):
    if dataset is not None:
        # 提取样本ID的逻辑
        if hasattr(dataloader.sampler, 'dataset'):
            indices = [dataloader.sampler.dataset.indices[i] for i in range(...)]
        else:
            indices = list(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        
        batch_sample_ids = []
        for idx in indices:
            if isinstance(dataset[idx], dict) and 'id' in dataset[idx]:
                batch_sample_ids.append(dataset[idx]['id'])
            else:
                batch_sample_ids.append(f"sample_{idx}")
        
        sample_ids.extend(batch_sample_ids)

# 3. 同步保存
_save(all_grads, output_dir, f"grads-{saved_idx}", sample_ids)
```

### 2. get_info.py 修改

```python
# 在调用collect_grads时传递dataset参数
collect_grads(
    model=model,
    dataloader=dataloader,
    loss_computer=loss_computer,
    num_params=num_params,
    output_dir=output_dir,
    grad_type=args.grad_type,
    projector=projector,
    proj_dim=args.proj_dim,
    device=device,
    dataset=dataset  # 新增参数
)
```

## 新增脚本

### 1. aggregate_gradients.py

**功能**: 将所有GPU的梯度和ID文件聚合成最终的单一文件

**使用方法**:
```bash
python aggregate_gradients.py \
    --base_path /path/to/output \
    --output_dir /path/to/final/output \
    --num_gpus 8 \
    --dim 8192
```

**输出文件**:
- `all_orig.pt`: 聚合后的梯度张量
- `all_ids.pkl`: 对应的样本ID列表

### 2. validate_gradient_id_mapping.py

**功能**: 验证梯度与ID映射的正确性

**使用方法**:
```bash
# 验证各个rank的文件
python validate_gradient_id_mapping.py \
    --output_dir /path/to/output \
    --check_ranks \
    --num_gpus 8

# 验证聚合后的文件
python validate_gradient_id_mapping.py \
    --output_dir /path/to/output \
    --check_aggregated \
    --dim 8192

# 同时验证
python validate_gradient_id_mapping.py \
    --output_dir /path/to/output \
    --check_ranks \
    --check_aggregated \
    --num_gpus 8 \
    --dim 8192
```

### 3. example_usage.py

**功能**: 展示完整的工作流程

**使用方法**:
```bash
python example_usage.py \
    --config_path /path/to/config.json \
    --output_dir /path/to/output \
    --num_gpus 8 \
    --dim 8192
```

## 文件命名规范

### 分片文件（每个GPU）
```
rank_0/dim8192/
├── grads-160.pt      # 前160个样本的梯度
├── ids-160.pkl       # 前160个样本的ID
├── grads-320.pt      # 前320个样本的梯度
├── ids-320.pkl       # 前320个样本的ID
├── ...
├── all_orig.pt       # 该GPU的所有梯度合并
└── all_ids.pkl       # 该GPU的所有ID合并
```

### 最终聚合文件
```
dim8192/
├── all_orig.pt       # 所有GPU的梯度聚合
└── all_ids.pkl       # 所有GPU的ID聚合
```

## 数据格式

### 梯度文件 (.pt)
```python
# 形状: [num_samples, proj_dim]
gradients = torch.load('grads-160.pt')
print(gradients.shape)  # torch.Size([160, 8192])
```

### ID文件 (.pkl)
```python
# 列表格式
import pickle
with open('ids-160.pkl', 'rb') as f:
    ids = pickle.load(f)
print(len(ids))  # 160
print(ids[0])    # 'sample_12345' 或实际的数据ID
```

## 使用流程

### 1. 完整工作流程

```bash
# 步骤1: 运行分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    less/data_selection/get_info.py \
    --config_path config.json \
    --output_dir /path/to/output

# 步骤2: 聚合所有GPU结果
python aggregate_gradients.py \
    --base_path /path/to/output \
    --output_dir /path/to/output/dim8192 \
    --num_gpus 8 \
    --dim 8192

# 步骤3: 验证结果
python validate_gradient_id_mapping.py \
    --output_dir /path/to/output \
    --check_ranks \
    --check_aggregated \
    --num_gpus 8 \
    --dim 8192
```

### 2. 使用示例脚本

```bash
# 一键执行完整流程
python example_usage.py \
    --config_path config.json \
    --output_dir /path/to/output \
    --num_gpus 8 \
    --dim 8192

# 跳过训练，只进行聚合和验证
python example_usage.py \
    --config_path config.json \
    --output_dir /path/to/output \
    --skip_training
```

## 关键特性

### 1. 顺序一致性保证
- 梯度张量的每一行与ID列表的对应位置严格对应
- `gradients[i]` 对应 `ids[i]` 的样本

### 2. 分布式支持
- 自动处理`DistributedSampler`的索引映射
- 支持多GPU并行训练

### 3. 错误处理
- 文件不存在时的优雅处理
- 数据加载失败时的错误报告
- 数量不匹配时的验证警告

### 4. 灵活性
- 支持自定义保存间隔
- 支持不同的梯度投影维度
- 兼容现有的训练流程

## 验证检查项

验证脚本会检查以下项目：

1. **文件完整性**
   - 梯度文件和ID文件数量匹配
   - 文件命名规范正确

2. **数量一致性**
   - 每个梯度文件的样本数与对应ID文件的ID数量匹配
   - 合并文件的总样本数正确

3. **顺序正确性**
   - 文件按训练顺序正确排序
   - 合并后的顺序与原始顺序一致

4. **数据完整性**
   - 所有文件可以正常加载
   - 数据格式正确

## 故障排除

### 常见问题

1. **ID文件缺失**
   - 确保在调用`collect_grads`时传递了`dataset`参数
   - 检查数据集是否包含'id'字段

2. **数量不匹配**
   - 验证DataLoader的batch_size设置
   - 检查DistributedSampler的配置

3. **文件加载失败**
   - 确保有足够的磁盘空间
   - 检查文件权限

4. **内存不足**
   - 调整保存间隔（减少每次保存的样本数）
   - 使用更小的投影维度

### 调试技巧

```python
# 检查数据集结构
print(f"Dataset sample: {dataset[0]}")
print(f"Keys: {dataset[0].keys() if isinstance(dataset[0], dict) else 'Not a dict'}")

# 检查DataLoader
for i, batch in enumerate(dataloader):
    print(f"Batch {i}: {batch.keys()}")
    if i == 0:
        break

# 检查保存的文件
import torch
import pickle

gradients = torch.load('grads-160.pt')
with open('ids-160.pkl', 'rb') as f:
    ids = pickle.load(f)

print(f"Gradients shape: {gradients.shape}")
print(f"IDs count: {len(ids)}")
print(f"First few IDs: {ids[:5]}")
```

## 性能考虑

1. **存储开销**: ID文件相对较小，通常不会显著增加存储需求
2. **计算开销**: ID提取的计算开销很小
3. **I/O开销**: 额外的文件写入操作，但与梯度保存并行进行

## 扩展性

该实现可以轻松扩展以支持：
- 其他类型的元数据保存
- 不同的文件格式（JSON、HDF5等）
- 更复杂的ID提取逻辑
- 自定义验证规则

## 总结

该实现提供了一个完整、健壮的解决方案，用于在PyTorch分布式训练中精确映射和存储梯度与数据样本ID。通过严格的顺序保证和全面的验证机制，确保了数据的完整性和正确性。