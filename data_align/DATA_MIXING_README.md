# Data Mixing Ratio Calculation Tool

这个工具使用 MMD (Maximum Mean Discrepancy) 优化算法来计算训练数据集的最优混合比例，以最好地匹配验证数据集的分布。

## 文件说明

- `calculate_mixing_ratio.py`: 主要的计算脚本
- `run_mixing_calculation.sh`: 运行示例脚本
- `mmd_data_mixing.py`: MMD优化算法的实现

## 使用方法

### 1. 直接使用 Python 脚本

```bash
python calculate_mixing_ratio.py \
    --train_base_path /mnt/hdfs/selection/yingtai_sft/lora_grads \
    --val_base_path /mnt/hdfs/selection/yingtai_sft/lora_val_grads \
    --experiment_name tulu3-Qwen3-8B-p0.05-lora-seed3 \
    --dim dim8192 \
    --output_dir ./mixing_results \
    --rff_dimension 1000 \
    --sigma_sample_size 1000 \
    --ridge_penalty 1e-7 \
    --random_seed 42
```

### 2. 使用 Shell 脚本

```bash
chmod +x run_mixing_calculation.sh
./run_mixing_calculation.sh
```

## 参数说明

### 必需参数

- `--train_base_path`: 训练梯度文件的基础路径
- `--val_base_path`: 验证梯度文件的基础路径  
- `--experiment_name`: 实验名称 (例如: tulu3-Qwen3-8B-p0.05-lora-seed3)

### 可选参数

- `--dim`: 维度文件夹名称 (默认: dim8192)
- `--output_dir`: 结果输出目录 (默认: ./mixing_results)
- `--rff_dimension`: RFF维度 (默认: 1000)
- `--sigma_sample_size`: Sigma计算的采样大小 (默认: 1000)
- `--ridge_penalty`: Ridge惩罚项 (默认: 1e-7)
- `--random_seed`: 随机种子 (默认: 42)

## 文件路径格式

工具会自动查找以下格式的梯度文件：

**训练数据:**
```
{train_base_path}/{experiment_name}/{dataset_name}/{dim}/all_origin.pt
```

**验证数据:**
```
{val_base_path}/{experiment_name}/{dataset_name}/{dim}/all_origin.pt
```

### 示例路径

**训练数据示例:**
```
/mnt/hdfs/selection/yingtai_sft/lora_grads/tulu3-Qwen3-8B-p0.05-lora-seed3/ai2-adapt-dev_coconot_converted-ckpt368-adam/dim8192/all_origin.pt
```

**验证数据示例:**
```
/mnt/hdfs/selection/yingtai_sft/lora_val_grads/tulu3-Qwen3-8B-p0.05-lora-seed3/drop-ckpt368-adam/dim8192/all_origin.pt
```

## 输出结果

### 控制台输出

脚本会在控制台显示：
- 找到的训练和验证数据集数量
- MMD优化过程的日志信息
- 最终的混合比例结果表格

### JSON 结果文件

结果会保存为 JSON 文件，包含：

```json
{
  "experiment_name": "tulu3-Qwen3-8B-p0.05-lora-seed3",
  "timestamp": "2024-01-20T10:30:45.123456",
  "total_weight": 1.0,
  "mixing_ratios": {
    "dataset1": {
      "weight": 0.35,
      "percentage": 35.0,
      "file_path": "/path/to/dataset1/all_origin.pt"
    },
    "dataset2": {
      "weight": 0.25,
      "percentage": 25.0,
      "file_path": "/path/to/dataset2/all_origin.pt"
    }
  },
  "summary": {
    "num_training_datasets": 19,
    "num_validation_datasets": 7,
    "training_datasets": ["dataset1", "dataset2", ...],
    "validation_datasets": ["val_dataset1", "val_dataset2", ...]
  },
  "file_paths": {
    "training": ["/path/to/train1.pt", "/path/to/train2.pt", ...],
    "validation": ["/path/to/val1.pt", "/path/to/val2.pt", ...]
  }
}
```

## MMD 算法参数

### RFF (Random Fourier Features) 参数

- **rff_dimension**: RFF空间的目标维度，更高的维度提供更好的近似但计算成本更高
- **sigma_bandwidth**: 高斯核的带宽参数，如果设置为 None 且 auto_sigma=True，会自动计算
- **auto_sigma**: 是否使用中位数启发式自动计算 sigma
- **sigma_sample_size**: 用于 sigma 计算的采样大小

### 优化参数

- **ridge_penalty**: Ridge 惩罚项，用于数值稳定性
- **random_seed**: 随机种子，确保结果可重现

## 注意事项

1. 确保所有梯度文件都存在且格式正确 (.pt 文件包含 2D tensor)
2. 训练和验证数据的梯度维度必须一致
3. 计算时间取决于数据集大小和 RFF 维度
4. 结果权重之和应该等于 1.0

## 故障排除

### 常见错误

1. **FileNotFoundError**: 检查文件路径是否正确
2. **维度不匹配**: 确保所有梯度文件的维度一致
3. **内存不足**: 减少 rff_dimension 或 sigma_sample_size
4. **QP求解失败**: 尝试增加 ridge_penalty

### 调试建议

- 使用较小的 rff_dimension 进行快速测试
- 检查日志输出中的文件发现信息
- 验证梯度文件是否可以正常加载