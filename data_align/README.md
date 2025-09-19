# MMD-based Data Mixing Optimization

This module implements optimal data mixing ratio calculation for multiple training datasets using Maximum Mean Discrepancy (MMD) minimization with Random Fourier Features (RFF).

## Overview

The goal is to find optimal mixing weights `w` for N training datasets that minimize the MMD between the weighted average distribution of training sets and the distribution of a target validation set. The implementation uses:

1. **Random Fourier Features (RFF)** to transform high-dimensional gradient embeddings into a lower-dimensional space
2. **MMD objective formulation** as a convex Quadratic Program (QP)
3. **Convex optimization** to solve for optimal mixing weights

## Files

- `mmd_data_mixing.py` - Main implementation with the `MMDDataMixer` class
- `example_usage.py` - Example usage and demonstrations
- `requirements.txt` - Required Python dependencies
- `README.md` - This documentation file

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For better optimization performance, install additional solvers:
```bash
pip install osqp ecos scs
```

## Quick Start

### Basic Usage

```python
from mmd_data_mixing import MMDDataMixer

# Initialize the mixer
mixer = MMDDataMixer(
    rff_dimension=100,
    sigma_bandwidth=3.0,
    ridge_penalty=1e-5,
    random_seed=42
)

# Define your gradient file paths
train_paths = [
    '/path/to/train_dataset1.pt',
    '/path/to/train_dataset2.pt',
    '/path/to/train_dataset3.pt'
]

val_paths = [
    '/path/to/val_dataset1.pt',
    '/path/to/val_dataset2.pt'
]

# Optimize mixing weights
results = mixer.optimize_mixing_weights(train_paths, val_paths)

# Print results
for path, weight in results.items():
    print(f"Dataset: {path} -> Weight: {weight:.4f}")
```

### Command Line Usage

```bash
python mmd_data_mixing.py \
    --train_paths /path/to/train1.pt /path/to/train2.pt /path/to/train3.pt \
    --val_paths /path/to/val1.pt /path/to/val2.pt \
    --rff_dimension 100 \
    --sigma_bandwidth 3.0 \
    --ridge_penalty 1e-5 \
    --output_file results.json
```

## Algorithm Details

### 1. Random Fourier Features (RFF)

The RFF transformation approximates the RBF kernel to make MMD calculation tractable:

```
φ(x) = √(2/D) * cos(Ω^T x + b)
```

Where:
- `Ω` is a random frequency matrix sampled from `N(0, σ^(-2) I_d)`
- `b` is a random phase vector sampled from `Unif[0, 2π]`
- `D` is the target RFF dimension
- `σ` is the Gaussian kernel bandwidth

### 2. MMD Objective

The MMD between weighted training distribution and validation distribution is:

```
MMD²(P_w, Q) = ||μ_w - μ_Q||²
```

Where:
- `μ_w = Σᵢ wᵢ μᵢ` is the weighted average of training means
- `μ_Q` is the validation mean
- `μᵢ` is the mean feature vector of training dataset i

### 3. Quadratic Programming

The optimization problem is formulated as:

```
min_w  w^T A w - 2b^T w
s.t.   1^T w = 1, w ≥ 0
```

Where:
- `A_ij = μᵢ^T μⱼ` (Gram matrix of training means)
- `bᵢ = μᵢ^T μ_Q` (cross-correlation with validation mean)

## Parameters

### MMDDataMixer Parameters

- `rff_dimension` (int, default=100): Target dimension for RFF space
- `sigma_bandwidth` (float, default=3.0): Gaussian kernel bandwidth
- `ridge_penalty` (float, default=1e-5): Ridge penalty for numerical stability
- `random_seed` (int, default=42): Random seed for reproducibility

### Parameter Tuning Guidelines

- **RFF Dimension**: Higher values provide better kernel approximation but increase computation
- **Sigma Bandwidth**: Controls kernel width; smaller values focus on local similarities
- **Ridge Penalty**: Prevents numerical instability; increase if solver fails

## Input Data Format

The script expects gradient embedding files in PyTorch `.pt` format:
- Each file should contain a 2D tensor of shape `(num_examples, gradient_dimension)`
- All files should have the same gradient dimension
- Files are loaded using `torch.load()`

## Output Format

The optimization returns a dictionary mapping each training dataset path to its optimal weight:

```python
{
    '/path/to/train_dataset1.pt': 0.3456,
    '/path/to/train_dataset2.pt': 0.2134,
    '/path/to/train_dataset3.pt': 0.4410
}
```

Weights sum to 1.0 and are non-negative.

## Examples

Run the example script to see demonstrations:

```bash
python example_usage.py
```

This will show:
1. Basic usage with synthetic data
2. Parameter tuning effects
3. Real-world usage patterns

## Error Handling

The implementation includes comprehensive error handling for:
- Missing or invalid input files
- Dimension mismatches between datasets
- Numerical optimization failures
- Invalid parameter values

## Performance Considerations

- **Memory**: RFF transformation reduces memory requirements compared to full kernel methods
- **Computation**: Complexity is O(N²D + ND²) where N is number of datasets and D is RFF dimension
- **Scalability**: Can handle large datasets by processing in batches if needed

## Theoretical Background

This implementation is based on the theory of Maximum Mean Discrepancy and Random Fourier Features:

1. **MMD**: A kernel-based distance measure between probability distributions
2. **RFF**: An efficient approximation method for shift-invariant kernels
3. **Convex Optimization**: Guarantees global optimum for the mixing weight problem

## Troubleshooting

### Common Issues

1. **Solver Failure**: Increase `ridge_penalty` or try different CVXPY solvers
2. **Memory Issues**: Reduce `rff_dimension` or process datasets in smaller batches
3. **Poor Results**: Tune `sigma_bandwidth` based on your data characteristics

### Debug Mode

Enable detailed logging by setting the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this implementation in your research, please cite the relevant papers on MMD and Random Fourier Features.

## License

This implementation is provided under the same license as the LESS project.