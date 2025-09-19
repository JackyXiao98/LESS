#!/usr/bin/env python3
"""
Example Usage Script for MMD-based Data Mixing Optimization

This script demonstrates how to use the MMDDataMixer class to find optimal
mixing weights for training datasets.
"""

from mmd_data_mixing import MMDDataMixer, MMDDataMixerRBF
import torch
import numpy as np
import os

def create_sample_data(num_sample=10000):
    """
    Create sample gradient data for demonstration purposes.
    In practice, these would be loaded from actual gradient files.
    """
    print("Creating sample gradient data for demonstration...")
    
    # Create sample directory
    sample_dir = "sample_gradients"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Set random seed for reproducible sample data
    torch.manual_seed(42)
    
    half_num_sample = num_sample // 2

    # Create sample training datasets (3 datasets)
    train_paths = []
    for i in range(2):
        # Each dataset has different characteristics
        if i == 0:
            # Dataset 1: More focused on certain features
            data = torch.randn([1000, num_sample]) * 1 + torch.tensor([-1.0] * num_sample)
        elif i == 1:
            # Dataset 2: Different distribution
            data = torch.randn([1000, num_sample]) * 1 + torch.tensor([1.0] * num_sample)
        else:
            # Dataset 3: Mixed characteristics
            data = torch.randn([1000, num_sample]) * 1 + torch.tensor([1] * num_sample)
        
        path = os.path.join(sample_dir, f"train_dataset_{i}.pt")
        torch.save(data, path)
        train_paths.append(path)
        print(f"Created training dataset {i+1}: {path} (shape: {data.shape})")
    
    # Create sample validation datasets (2 datasets)
    val_paths = []
    for i in range(2):
        if i == 0:
            # Validation dataset 1: Similar to a mix of training datasets
            data = torch.randn([100, num_sample]) * 1/torch.sqrt(torch.tensor([2.0])) + torch.tensor([0.0] * num_sample)
        else:
            # Validation dataset 2: Another validation set
            data = torch.randn([100, num_sample]) * 1/torch.sqrt(torch.tensor([2.0])) + torch.tensor([0.0] * num_sample)
        
        path = os.path.join(sample_dir, f"val_dataset_{i}.pt")
        torch.save(data, path)
        val_paths.append(path)
        print(f"Created validation dataset {i+1}: {path} (shape: {data.shape})")
    
    return train_paths, val_paths

def example_basic_usage():
    """
    Demonstrate basic usage of the MMDDataMixer.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Create sample data
    train_paths, val_paths = create_sample_data()
    
    # Initialize the mixer with default parameters
    mixer = MMDDataMixer(
        rff_dimension=100,
        sigma_bandwidth=3.0,
        ridge_penalty=1e-5,
        random_seed=42
    )
    
    # Optimize mixing weights
    results = mixer.optimize_mixing_weights(train_paths, val_paths)
    
    # Display results
    print("\nOptimal Mixing Weights:")
    for i, (path, weight) in enumerate(results.items()):
        dataset_name = os.path.basename(path)
        print(f"  {dataset_name}: {weight:.4f}")
    
    # Clean up sample data
    import shutil
    shutil.rmtree("sample_gradients")
    
    return results

def example_auto_sigma():
    """
    Demonstrate automatic sigma selection using median heuristic.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1.5: Automatic Sigma Selection")
    print("="*60)
    
    # Create sample data
    train_paths, val_paths = create_sample_data()
    
    # Initialize the mixer with automatic sigma selection
    mixer = MMDDataMixer(
        rff_dimension=100,
        sigma_bandwidth=None,  # Will be computed automatically
        ridge_penalty=1e-5,
        random_seed=42,
        auto_sigma=True,
        sigma_sample_size=50
    )
    
    print("Using median heuristic to compute sigma automatically...")
    
    # Optimize mixing weights (sigma will be computed during this process)
    results = mixer.optimize_mixing_weights(train_paths, val_paths)
    
    # Display results
    print(f"\nAutomatically computed sigma: {mixer.sigma:.6f}")
    print("\nOptimal Mixing Weights:")
    for i, (path, weight) in enumerate(results.items()):
        dataset_name = os.path.basename(path)
        print(f"  {dataset_name}: {weight:.4f}")
    
    # Clean up sample data
    import shutil
    shutil.rmtree("sample_gradients")
    
    return results

def example_rbf_vs_rff_comparison():
    """
    Compare RBF kernel (exact) vs RFF (approximation) methods using the same sigma.
    """
    print("\n" + "="*60)
    print("EXAMPLE: RBF vs RFF Comparison")
    print("="*60)
    
    # Create sample data
    train_paths, val_paths = create_sample_data()
    
    print("Comparing RBF (exact) vs RFF (approximation) methods...")
    print("Both methods will use the same automatically computed sigma parameter.")
    
    # First, use RFF method to compute sigma automatically
    print("\n1. RFF Method (Random Fourier Features):")
    print("-" * 40)
    
    mixer_rff = MMDDataMixer(
        rff_dimension=1000,
        auto_sigma=True,
        sigma_sample_size=100,
        random_seed=42
    )
    
    # Optimize with RFF method
    results_rff = mixer_rff.optimize_mixing_weights(train_paths, val_paths)
    sigma_computed = mixer_rff.sigma
    
    print(f"Computed sigma: {sigma_computed:.6f}")
    print("Optimal weights (RFF):")
    for path, weight in results_rff.items():
        dataset_name = os.path.basename(path)
        print(f"  {dataset_name}: {weight:.4f}")
    
    # Now use RBF method with the same sigma
    print("\n2. RBF Method (Exact Kernel Computation):")
    print("-" * 40)
    
    mixer_rbf = MMDDataMixerRBF(
        sigma_bandwidth=sigma_computed,  # Use the same sigma
        auto_sigma=False,  # Don't recompute sigma
        random_seed=42
    )
    
    # Optimize with RBF method
    results_rbf = mixer_rbf.optimize_mixing_weights(train_paths, val_paths)
    
    print(f"Using sigma: {mixer_rbf.sigma:.6f}")
    print("Optimal weights (RBF):")
    for path, weight in results_rbf.items():
        dataset_name = os.path.basename(path)
        print(f"  {dataset_name}: {weight:.4f}")
    
    # Compare results
    print("\n3. Comparison Results:")
    print("-" * 40)
    
    print("Weight differences (RBF - RFF):")
    for path in results_rff.keys():
        dataset_name = os.path.basename(path)
        diff = results_rbf[path] - results_rff[path]
        print(f"  {dataset_name}: {diff:+.6f}")
    
    # Calculate weight vector differences
    weights_rff = np.array(list(results_rff.values()))
    weights_rbf = np.array(list(results_rbf.values()))
    
    l2_diff = np.linalg.norm(weights_rbf - weights_rff)
    l1_diff = np.sum(np.abs(weights_rbf - weights_rff))
    max_diff = np.max(np.abs(weights_rbf - weights_rff))
    
    print(f"\nWeight vector differences:")
    print(f"  L2 norm: {l2_diff:.6f}")
    print(f"  L1 norm: {l1_diff:.6f}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    
    # Theoretical note
    print(f"\nNote: RFF is an approximation of RBF kernel.")
    print(f"Small differences are expected due to the stochastic nature of RFF.")
    print(f"RBF provides exact kernel computation, while RFF uses {mixer_rff.D} random features.")
    
    # Clean up sample data
    import shutil
    shutil.rmtree("sample_gradients")
    
    return {
        'rff_results': results_rff,
        'rbf_results': results_rbf,
        'sigma': sigma_computed,
        'l2_diff': l2_diff,
        'l1_diff': l1_diff,
        'max_diff': max_diff
    }

def example_parameter_tuning():
    """
    Demonstrate parameter tuning effects.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Parameter Tuning")
    print("="*60)
    
    # Create sample data
    train_paths, val_paths = create_sample_data()
    
    # Test different parameter combinations
    parameter_sets = [
        {"rff_dimension": 1000, "sigma_bandwidth": 40.0, "name": "Low Sigma"},
        {"rff_dimension": 1000, "sigma_bandwidth": 50.0, "name": "Default Parameters"},
        {"rff_dimension": 1000, "sigma_bandwidth": 60.0, "name": "High Sigma"},
    ]
    
    results_comparison = {}
    
    for params in parameter_sets:
        print(f"\nTesting: {params['name']}")
        print(f"  RFF Dimension: {params['rff_dimension']}")
        print(f"  Sigma Bandwidth: {params['sigma_bandwidth']}")
        
        mixer = MMDDataMixer(
            rff_dimension=params['rff_dimension'],
            sigma_bandwidth=params['sigma_bandwidth'],
            ridge_penalty=1e-7,
            random_seed=42
        )
        
        results = mixer.optimize_mixing_weights(train_paths, val_paths)
        results_comparison[params['name']] = results
        
        print("  Weights:", [f"{w:.4f}" for w in results.values()])
    
    # Compare results
    print("\n" + "-"*40)
    print("PARAMETER COMPARISON SUMMARY")
    print("-"*40)
    
    for name, results in results_comparison.items():
        weights = list(results.values())
        print(f"{name}:")
        print(f"  Weights: {[f'{w:.4f}' for w in weights]}")
        # print(f"  Max weight: {max(weights):.4f}")
        # print(f"  Min weight: {min(weights):.4f}")
        # print(f"  Weight std: {np.std(weights):.4f}")
    
    # Clean up sample data
    import shutil
    shutil.rmtree("sample_gradients")

def example_real_world_usage():
    """
    Show how to use with real gradient files (conceptual example).
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Real-world Usage (Conceptual)")
    print("="*60)
    
    # This is how you would use it with real gradient files
    real_train_paths = [
        '/path/to/gradients/dataset1/gradients.pt',
        '/path/to/gradients/dataset2/gradients.pt',
        '/path/to/gradients/dataset3/gradients.pt',
    ]
    
    real_val_paths = [
        '/path/to/gradients/validation1/gradients.pt',
        '/path/to/gradients/validation2/gradients.pt',
    ]
    
    print("Real-world usage example:")
    print("```python")
    print("from mmd_data_mixing import MMDDataMixer")
    print("")
    print("# Initialize mixer")
    print("mixer = MMDDataMixer(")
    print("    rff_dimension=100,")
    print("    sigma_bandwidth=3.0,")
    print("    ridge_penalty=1e-5,")
    print("    random_seed=42")
    print(")")
    print("")
    print("# Define your gradient file paths")
    print("train_paths = [")
    for path in real_train_paths:
        print(f"    '{path}',")
    print("]")
    print("")
    print("val_paths = [")
    for path in real_val_paths:
        print(f"    '{path}',")
    print("]")
    print("")
    print("# Optimize mixing weights")
    print("results = mixer.optimize_mixing_weights(train_paths, val_paths)")
    print("")
    print("# Use the results")
    print("for path, weight in results.items():")
    print("    print(f'Dataset: {path} -> Weight: {weight:.4f}')")
    print("```")

def main():
    """
    Run all examples.
    """
    print("MMD Data Mixing Optimization - Example Usage")
    print("=" * 60)
    
    try:
        # Run examples
        # example_basic_usage()
        # example_auto_sigma()
        example_rbf_vs_rff_comparison()
        # example_parameter_tuning()
        # example_real_world_usage()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()