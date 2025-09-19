# RBF vs RFF Kernel Comparison Implementation

## 概述

本实现为MMD数据混合优化添加了RBF（径向基函数）kernel与RFF（随机傅里叶特征）方法的对比功能。RBF方法提供精确的kernel计算，而RFF方法提供高效的近似计算。

## 实现的功能

### 1. MMDDataMixerRBF子类

- **位置**: `mmd_data_mixing.py`
- **继承**: 继承自`MMDDataMixer`类
- **功能**: 使用精确的RBF kernel计算MMD，而不是RFF近似

#### 核心方法

1. **`_compute_rbf_kernel_matrix(X, Y)`**
   - 计算RBF kernel矩阵
   - 公式: `k(x, y) = exp(-||x - y||^2 / (2 * σ^2))`
   - 高效的向量化实现

2. **`_compute_mmd_squared_rbf(X, Y)`**
   - 计算两个数据集之间的平方MMD
   - 公式: `MMD²(X, Y) = E[k(x, x')] - 2*E[k(x, y)] + E[k(y, y')]`
   - 使用无偏估计（排除对角线元素）

3. **`solve_qp(train_tensors, val_tensor)`**
   - 使用RBF kernel解决二次规划问题
   - 直接计算数据集间的MMD距离矩阵

### 2. 对比测试功能

- **位置**: `example_usage.py`
- **函数**: `example_rbf_vs_rff_comparison()`

#### 对比流程

1. **RFF方法**: 自动计算sigma参数
2. **RBF方法**: 使用相同的sigma参数
3. **结果对比**: 计算权重差异的各种指标

#### 输出指标

- 每个数据集的权重差异
- L2范数差异
- L1范数差异  
- 最大绝对差异

## 技术特点

### 1. Sigma参数一致性

- RFF方法首先使用中位数启发式自动计算sigma
- RBF方法使用完全相同的sigma值
- 确保公平的对比

### 2. 数学正确性

- **RBF Kernel**: `k(x, y) = exp(-γ * ||x - y||²)`, 其中 `γ = 1/(2σ²)`
- **MMD计算**: 使用无偏估计，排除自相关项
- **QP求解**: 最小化 `Σᵢⱼ wᵢwⱼ MMD²(Dᵢ, Dⱼ) - 2Σᵢ wᵢ MMD²(Dᵢ, V)`

### 3. 性能考虑

- **RBF**: O(n²) 复杂度，精确但计算量大
- **RFF**: O(nd) 复杂度，近似但高效
- **适用场景**: RBF适合小数据集验证，RFF适合大规模应用

## 使用方法

### 基本使用

```python
from mmd_data_mixing import MMDDataMixer, MMDDataMixerRBF

# 运行对比测试
python3 example_usage.py
```

### 单独使用RBF方法

```python
# 使用自动sigma计算
mixer_rbf = MMDDataMixerRBF(auto_sigma=True, sigma_sample_size=100)

# 使用指定sigma
mixer_rbf = MMDDataMixerRBF(sigma_bandwidth=1.5, auto_sigma=False)

# 优化混合权重
results = mixer_rbf.optimize_mixing_weights(train_paths, val_paths)
```

## 验证和测试

### 1. 自动验证脚本

- **`verify_sigma_consistency.py`**: 验证sigma参数一致性
- **`test_rbf_rff_comparison.py`**: 全面的功能测试

### 2. 测试覆盖

- ✅ 类定义和继承
- ✅ RBF方法实现
- ✅ 对比函数实现
- ✅ 数学公式正确性
- ✅ Sigma参数一致性
- ✅ 文件结构完整性

## 理论背景

### RBF vs RFF对比

| 方面 | RBF (精确) | RFF (近似) |
|------|------------|------------|
| 计算复杂度 | O(n²) | O(nd) |
| 内存使用 | O(n²) | O(nd) |
| 精确性 | 完全精确 | 近似，随d增加而改善 |
| 适用场景 | 小数据集、验证 | 大数据集、生产环境 |

### 预期结果

- 小的权重差异（通常 < 0.01）
- RFF近似质量随特征维度增加而提高
- 两种方法应产生相似的数据集排序

## 文件结构

```
data_align/
├── mmd_data_mixing.py              # 主要实现（包含MMDDataMixerRBF）
├── example_usage.py                # 示例和对比测试
├── verify_sigma_consistency.py     # Sigma一致性验证
├── test_rbf_rff_comparison.py      # 功能测试套件
└── RBF_VS_RFF_COMPARISON.md       # 本文档
```

## 依赖要求

- `torch`: 张量计算和RBF kernel实现
- `numpy`: 数值计算
- `cvxpy`: 二次规划求解

## 安装和运行

```bash
# 安装依赖
pip install torch numpy cvxpy

# 运行验证
python3 verify_sigma_consistency.py
python3 test_rbf_rff_comparison.py

# 运行对比测试
python3 example_usage.py
```

## 总结

本实现成功添加了RBF kernel的精确MMD计算方法，并提供了与现有RFF方法的全面对比。通过确保相同的sigma参数和严格的测试验证，用户可以：

1. 验证RFF近似的质量
2. 在小数据集上获得精确结果
3. 理解两种方法的权衡
4. 根据具体需求选择合适的方法

实现遵循了良好的软件工程实践，包括完整的测试覆盖、清晰的文档和模块化设计。