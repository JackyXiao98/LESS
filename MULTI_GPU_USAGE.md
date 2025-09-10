# 多GPU数据并行梯度计算使用说明

## 概述

本文档说明如何使用修改后的代码进行8GPU数据并行梯度计算和存储。

## 主要修改

### 1. get_info.py 修改

#### 新增导入
```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
```

#### 新增命令行参数
- `--world_size`: GPU总数 (默认: 1)
- `--rank`: 当前进程rank (默认: 0) 
- `--local_rank`: 本地GPU ID (默认: 0)
- `--master_addr`: 主节点地址 (默认: "127.0.0.1")
- `--master_port`: 主节点端口 (默认: "29500")

#### 新增函数
- `setup_distributed()`: 初始化分布式训练环境
- `cleanup_distributed()`: 清理分布式训练环境
- `main_worker()`: 主要工作函数，处理单个GPU的任务

#### 主要改进
- 支持分布式数据加载 (DistributedSampler)
- 模型包装为DDP (如果使用多GPU)
- 每个rank的输出保存到独立目录
- 自动处理进程间通信

### 2. 新增多GPU Shell脚本

创建了 `get_train_lora_grads_multi_gpu.sh`，支持：
- 自动检测可用GPU数量
- 设置分布式训练环境变量
- 启动多进程训练
- 结果保存到分rank目录

## 使用方法

### 单GPU使用 (原有方式)
```bash
bash less/scripts/get_info/grad/get_train_lora_grads.sh \
  train_file.jsonl \
  model_path \
  output_path \
  1024 \
  sgd \
  0
```

### 多GPU使用 (推荐: torchrun)
```bash
bash less/scripts/get_info/grad/get_train_lora_grads_torchrun.sh \
  train_file.jsonl \
  model_path \
  output_path \
  1024 \
  sgd \
  8 \
  29500
```

### 多GPU使用 (备选: multiprocessing)
```bash
bash less/scripts/get_info/grad/get_train_lora_grads_multi_gpu.sh \
  train_file.jsonl \
  model_path \
  output_path \
  1024 \
  sgd \
  8 \
  29500
```

参数说明：
1. `train_file.jsonl`: 训练数据文件
2. `model_path`: 模型路径
3. `output_path`: 输出路径
4. `1024`: 梯度投影维度
5. `sgd`: 梯度类型
6. `8`: 使用的GPU数量 (可选，默认8)
7. `29500`: 主节点端口 (可选，默认29500)

## 输出结构

### 单GPU输出
```
output_path/
├── gradients.pt
└── other_files...
```

### 多GPU输出
```
output_path/
├── rank_0/
│   ├── gradients.pt
│   └── other_files...
├── rank_1/
│   ├── gradients.pt
│   └── other_files...
├── ...
└── rank_7/
    ├── gradients.pt
    └── other_files...
```

## 性能优势

1. **数据并行**: 数据集自动分割到各个GPU
2. **内存效率**: 每个GPU只处理部分数据
3. **计算加速**: 8GPU并行计算，理论上8倍加速
4. **存储分布**: 梯度结果分别存储，避免内存瓶颈

## torchrun vs multiprocessing

### torchrun 优势 (推荐)
- **更好的进程管理**: 自动处理进程启动和清理
- **错误处理**: 更好的错误报告和故障恢复
- **环境变量**: 自动设置分布式训练环境变量
- **标准化**: PyTorch官方推荐的分布式启动方式
- **简化配置**: 无需手动设置rank和world_size

### multiprocessing 方式 (备选)
- **兼容性**: 适用于不支持torchrun的环境
- **自定义控制**: 更多的进程控制选项
- **调试友好**: 更容易进行单步调试

## 注意事项

1. **GPU要求**: 确保有足够的GPU资源
2. **内存要求**: 每个GPU需要足够内存加载模型
3. **网络通信**: 多GPU间需要高速互联
4. **结果合并**: 如需合并结果，需要额外处理各rank的输出

## 技术实现

### 核心修改

1. **分布式训练支持**
   - 添加了`torch.distributed`和`torch.multiprocessing`支持
   - 实现了`setup_distributed()`和`cleanup_distributed()`函数
   - 支持NCCL后端进行GPU间通信

2. **命令行参数扩展**
   ```python
   --world_size: GPU数量
   --rank: 当前进程排名
   --local_rank: 本地GPU排名
   --master_addr: 主节点地址
   --master_port: 通信端口
   ```

3. **设备分配优化**
   - **移除了`device_map="auto"`**: 避免模型跨设备分布
   - **显式设备分配**: 每个进程的模型完全在其指定GPU上
   - **设备一致性**: 确保所有张量操作在同一设备上
   ```python
   # 修改前（有问题）
   model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
   
   # 修改后（正确）
   model = AutoModelForCausalLM.from_pretrained(path, device_map=None)
   model = model.to(device)  # device = f"cuda:{rank}"
   ```

4. **数据并行处理**
   - 使用`DistributedSampler`确保数据不重复
   - 每个GPU处理不同的数据子集
   - 使用`DistributedDataParallel`包装模型

## 故障排除

### 常见问题

1. **CUDA设备不可用**
   ```bash
   RuntimeError: CUDA is not available
   ```
   - 检查CUDA安装和GPU驱动
   - 确认PyTorch CUDA版本匹配

2. **端口被占用**
   ```bash
   RuntimeError: Address already in use
   ```
   - 更改master_port参数
   - 检查并终止占用端口的进程

3. **内存不足**
   ```bash
   RuntimeError: CUDA out of memory
   ```
   - 减少batch_size
   - 减少使用的GPU数量
   - 使用gradient checkpointing

4. **进程同步问题**
   - 确保所有GPU可见且可用
   - 检查网络连接（多节点情况）
   - 验证NCCL后端可用性

5. **DataLoader TypeError**
   ```bash
   TypeError: get_dataloader() got an unexpected keyword argument 'sampler'
   ```
   - 这个问题已在最新版本中修复
   - 现在直接使用torch.utils.data.DataLoader创建分布式数据加载器
   - 使用DataCollatorForSeq2Seq进行正确的批处理

6. **设备不匹配错误**
   ```bash
   RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
   ```
   - 这个问题已在最新版本中修复
   - 移除了`device_map="auto"`，改为显式设备分配
   - 每个GPU进程的模型完全在其指定的设备上
   - 确保所有张量操作在同一设备上进行

## 测试验证

运行测试脚本验证功能：
```bash
python3 test_multi_gpu_gradient.py
```

该脚本会验证：
- 代码修改完整性
- 分布式环境设置
- 多GPU功能可用性