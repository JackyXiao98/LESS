# 数据采样脚本使用说明

## 功能概述

`sample_data.py` 脚本实现了两个主要的数据采样功能：

1. **从 yelp-train.csv 采样 N=1000 个数据**
2. **从 yelp_huggingface_gpt2_secpe_10_600.csv 随机采样 5 个不重复的 subset，每个 subset 包含 M=1000 个数据**

## 文件结构

```
data_process/
├── raw/                          # 原始数据文件夹
│   ├── Yelp-train.csv           # 需要采样的第一个文件
│   └── yelp_huggingface_gpt2_secpe_10_600.csv  # 需要采样的第二个文件
├── sampled/                     # 输出文件夹（自动创建）
│   ├── yelp_train_sampled_1000.csv              # yelp-train采样结果
│   ├── yelp_huggingface_subset_1_1000.csv       # 第1个subset
│   ├── yelp_huggingface_subset_2_1000.csv       # 第2个subset
│   ├── yelp_huggingface_subset_3_1000.csv       # 第3个subset
│   ├── yelp_huggingface_subset_4_1000.csv       # 第4个subset
│   └── yelp_huggingface_subset_5_1000.csv       # 第5个subset
├── sample_data.py               # 主脚本
└── README.md                    # 本说明文件
```

## 使用方法

### 1. 准备数据文件

将以下文件放入 `raw/` 文件夹：
- `Yelp-train.csv`
- `yelp_huggingface_gpt2_secpe_10_600.csv`

### 2. 运行脚本

```bash
# 在 data_process 目录下运行
python sample_data.py
```

### 3. 自定义参数

如果需要修改采样参数，可以编辑 `main()` 函数中的参数：

```python
sampler.run_sampling(
    yelp_train_samples=1000,           # 修改 yelp-train.csv 的采样数量
    huggingface_subsets=5,             # 修改 subset 数量
    huggingface_samples_per_subset=1000 # 修改每个 subset 的样本数量
)
```

## 脚本特性

### 内存优化
- 使用分块读取（chunk_size=10000）处理大文件
- 避免将整个文件加载到内存中
- 适合处理GB级别的大型CSV文件

### 随机采样
- 使用 `numpy.random.choice` 进行无重复随机采样
- 设置固定随机种子（42）确保结果可重现
- 对于 huggingface 文件，确保 5 个 subset 之间没有重复数据

### 数据去重
- 自动去除重复数据，确保子集之间没有重复内容
- 使用高效的索引管理避免数据重复
- 保证采样结果的数据质量

### 错误处理
- 检查输入文件是否存在
- 验证文件行数是否足够采样需求
- 详细的日志输出，便于调试和监控进度

### 日志功能
- 实时显示采样进度
- 记录文件行数、采样数量等关键信息
- 错误和警告信息的详细输出

## 输出文件说明

1. **yelp_train_sampled_1000.csv**: 从 Yelp-train.csv 随机采样的 1000 条数据
   - 列名: ['text', 'label1', 'label2']
   - 实际数据行数: 1000 行

2. **yelp_huggingface_subset_X_1000.csv**: 从 yelp_huggingface_gpt2_secpe_10_600.csv 采样的第 X 个 subset，每个包含 1000 条数据
   - 列名: ['text', 'business_category', 'review_stars']
   - 实际数据行数: 每个文件 1000 行
   - 5 个 subset 之间没有重复数据

**注意**: 由于文本字段中包含换行符，使用 `wc -l` 命令统计的行数可能会比实际数据行数多。建议使用 pandas 来验证实际的数据行数。

## 注意事项

1. 确保原始数据文件有足够的行数满足采样需求
2. 脚本会自动创建 `sampled/` 输出目录
3. 如果文件已存在，会被覆盖
4. 采样过程中会显示详细的进度信息

## 依赖库

```bash
pip install pandas numpy
```

## 故障排除

### 文件不存在错误
- 检查 `raw/` 文件夹中是否有正确的文件名
- 确认文件路径和文件名拼写正确

### 内存不足
- 可以减小 `chunk_size` 参数（默认 10000）
- 确保系统有足够的可用内存

### 采样数量不足
- 检查原始文件的行数是否满足采样需求
- 脚本会自动调整采样数量不超过文件总行数