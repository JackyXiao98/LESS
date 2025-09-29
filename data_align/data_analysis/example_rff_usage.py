#!/usr/bin/env python3
"""
Example usage script for RFF + t-SNE visualization.

This script demonstrates how to use the rff_tsne_visualization.py script
with different configurations and parameters.
"""

import os
import subprocess
import sys

def run_rff_visualization_example():
    """
    Example of running RFF + t-SNE visualization with different configurations.
    """
    
    # Base configuration
    base_config = {
        "experiment_name": "tulu3-Qwen3-8B-p0.05-lora-seed3",
        "output_dir": "./rff_tsne_results",
        "sample_number": 1000,  # Sample 1000 points per dataset
        "random_seed": 42
    }
    
    # Example 1: Single path mode with different RFF dimensions
    print("=" * 60)
    print("Example 1: Testing different RFF dimensions")
    print("=" * 60)
    
    single_path_examples = [
        {
            "name": "RFF-128",
            "base_path": "/mnt/hdfs/selection/yingtai_sft/lora_grads",
            "rff_dimension": 128,
            "sigma_bandwidth": None,  # Auto sigma
        },
        {
            "name": "RFF-256", 
            "base_path": "/mnt/hdfs/selection/yingtai_sft/lora_grads",
            "rff_dimension": 256,
            "sigma_bandwidth": None,  # Auto sigma
        },
        {
            "name": "RFF-512",
            "base_path": "/mnt/hdfs/selection/yingtai_sft/lora_grads", 
            "rff_dimension": 512,
            "sigma_bandwidth": None,  # Auto sigma
        }
    ]
    
    for example in single_path_examples:
        print(f"\nRunning {example['name']} example...")
        
        cmd = [
            "python", "rff_tsne_visualization.py",
            "--base_path", example["base_path"],
            "--experiment_name", base_config["experiment_name"],
            "--output_dir", f"{base_config['output_dir']}/{example['name']}",
            "--sample_number", str(base_config["sample_number"]),
            "--rff_dimension", str(example["rff_dimension"]),
            "--random_seed", str(base_config["random_seed"]),
            "--auto_sigma"
        ]
        
        if example["sigma_bandwidth"] is not None:
            cmd.extend(["--sigma_bandwidth", str(example["sigma_bandwidth"])])
        
        print(f"Command: {' '.join(cmd)}")
        print("Note: This is an example command. Adjust paths according to your setup.")
    
    # Example 2: Dual path mode (train + validation)
    print("\n" + "=" * 60)
    print("Example 2: Dual path mode (train + validation)")
    print("=" * 60)
    
    dual_path_example = {
        "name": "dual-path-RFF-256",
        "train_base_path": "/mnt/hdfs/selection/yingtai_sft/lora_grads",
        "val_base_path": "/mnt/hdfs/selection/yingtai_sft/lora_grads", 
        "rff_dimension": 256,
        "sigma_bandwidth": 1.5,  # Manual sigma
    }
    
    print(f"\nRunning {dual_path_example['name']} example...")
    
    cmd = [
        "python", "rff_tsne_visualization.py",
        "--train_base_path", dual_path_example["train_base_path"],
        "--val_base_path", dual_path_example["val_base_path"],
        "--experiment_name", base_config["experiment_name"],
        "--output_dir", f"{base_config['output_dir']}/{dual_path_example['name']}",
        "--sample_number", str(base_config["sample_number"]),
        "--rff_dimension", str(dual_path_example["rff_dimension"]),
        "--sigma_bandwidth", str(dual_path_example["sigma_bandwidth"]),
        "--random_seed", str(base_config["random_seed"])
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Note: This is an example command. Adjust paths according to your setup.")
    
    # Example 3: Advanced configuration
    print("\n" + "=" * 60)
    print("Example 3: Advanced configuration with custom parameters")
    print("=" * 60)
    
    advanced_example = {
        "name": "advanced-RFF-1024",
        "base_path": "/mnt/hdfs/selection/yingtai_sft/lora_grads",
        "rff_dimension": 1024,
        "sigma_bandwidth": None,  # Auto sigma
        "sigma_sample_size": 2000,
        "pca_components": 100,
        "perplexity": 50,
        "max_iter": 1500
    }
    
    print(f"\nRunning {advanced_example['name']} example...")
    
    cmd = [
        "python", "rff_tsne_visualization.py",
        "--base_path", advanced_example["base_path"],
        "--experiment_name", base_config["experiment_name"],
        "--output_dir", f"{base_config['output_dir']}/{advanced_example['name']}",
        "--sample_number", str(base_config["sample_number"]),
        "--rff_dimension", str(advanced_example["rff_dimension"]),
        "--sigma_sample_size", str(advanced_example["sigma_sample_size"]),
        "--pca_components", str(advanced_example["pca_components"]),
        "--perplexity", str(advanced_example["perplexity"]),
        "--max_iter", str(advanced_example["max_iter"]),
        "--random_seed", str(base_config["random_seed"]),
        "--auto_sigma"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Note: This is an example command. Adjust paths according to your setup.")
    
    # Parameter explanations
    print("\n" + "=" * 60)
    print("Parameter Explanations")
    print("=" * 60)
    
    explanations = {
        "--rff_dimension": "Target dimension for RFF transformation (e.g., 128, 256, 512, 1024)",
        "--sigma_bandwidth": "Bandwidth for Gaussian kernel. Use None for auto-computation",
        "--auto_sigma": "Enable automatic sigma computation using median heuristic",
        "--sigma_sample_size": "Number of samples for sigma computation (default: 1000)",
        "--pca_components": "Number of PCA components before t-SNE (default: 50)",
        "--perplexity": "t-SNE perplexity parameter (default: 30)",
        "--max_iter": "Number of t-SNE iterations (default: 1000)",
        "--sample_number": "Number of samples per dataset. Use -1 for all data",
        "--random_seed": "Random seed for reproducibility"
    }
    
    for param, explanation in explanations.items():
        print(f"{param:20}: {explanation}")
    
    print("\n" + "=" * 60)
    print("Tips for Usage")
    print("=" * 60)
    
    tips = [
        "1. Start with smaller RFF dimensions (128-256) for faster computation",
        "2. Use auto_sigma for automatic bandwidth selection",
        "3. Increase sample_number for better representation but slower computation",
        "4. Adjust perplexity based on dataset size (5-50 is typical range)",
        "5. Use PCA preprocessing for high-dimensional RFF outputs",
        "6. Set random_seed for reproducible results",
        "7. Monitor memory usage with large datasets and high RFF dimensions"
    ]
    
    for tip in tips:
        print(tip)


def show_rff_theory():
    """
    Show theoretical background of RFF transformation.
    """
    print("\n" + "=" * 60)
    print("Random Fourier Features (RFF) Theory")
    print("=" * 60)
    
    theory_text = """
    Random Fourier Features (RFF) is a technique for approximating kernel functions
    using explicit feature maps. This allows us to work with kernel methods in
    linear time complexity.
    
    Key Concepts:
    
    1. Kernel Approximation:
       - RFF approximates the Gaussian (RBF) kernel: k(x,y) = exp(-||x-y||²/(2σ²))
       - Uses Bochner's theorem to represent kernels as Fourier transforms
    
    2. Feature Mapping:
       - Maps input x ∈ ℝᵈ to z(x) ∈ ℝᴰ where D << d typically
       - z(x) = √(2/D) * cos(Ωᵀx + b)
       - Ω ~ N(0, σ⁻²I), b ~ Uniform[0, 2π]
    
    3. Benefits for Gradient Analysis:
       - Reduces dimensionality while preserving kernel relationships
       - Enables efficient similarity computation between gradient vectors
       - Maintains clustering properties in lower-dimensional space
    
    4. Parameter Selection:
       - σ (bandwidth): Controls kernel width, auto-computed using median heuristic
       - D (RFF dimension): Trade-off between approximation quality and efficiency
       - Higher D gives better approximation but increases computation
    
    5. Why RFF for Gradients:
       - Gradient vectors are high-dimensional and sparse
       - RFF captures non-linear relationships between gradients
       - Enables visualization of gradient similarity patterns
       - Preserves dataset-specific gradient characteristics
    """
    
    print(theory_text)


def main():
    """
    Main function to show examples and theory.
    """
    print("RFF + t-SNE Visualization Usage Examples")
    print("=" * 60)
    
    # Show usage examples
    run_rff_visualization_example()
    
    # Show theoretical background
    show_rff_theory()
    
    print("\n" + "=" * 60)
    print("Ready to use RFF + t-SNE visualization!")
    print("Modify the paths and parameters according to your setup.")
    print("=" * 60)


if __name__ == "__main__":
    main()