#!/usr/bin/env python3
"""
Data Mixing Ratio Calculation Script

This script calculates optimal data mixing ratios for training datasets
to best match the distribution of validation datasets using MMD optimization.

Usage:
    python calculate_mixing_ratio.py --train_base_path <path> --val_base_path <path> --experiment_name <name>

Example:
    python calculate_mixing_ratio.py \
        --train_base_path /mnt/hdfs/selection/yingtai_sft/lora_grads \
        --val_base_path /mnt/hdfs/selection/yingtai_sft/lora_val_grads \
        --experiment_name tulu3-Qwen3-8B-p0.05-lora-seed3
"""

import os
import argparse
import glob
import json
from datetime import datetime
from typing import List, Dict
import logging

from mmd_data_mixing import MMDDataMixer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DATASETS = [
            "ai2-adapt-dev_coconot_converted-ckpt368-adam",
            # "ai2-adapt-dev_evol_codealpaca_heval_decontaminated-ckpt368-adam",
            # "ai2-adapt-dev_flan_v2_converted-ckpt368-adam",
            # "ai2-adapt-dev_no_robots_converted-ckpt368-adam",
            # "ai2-adapt-dev_numinamath_tir_math_decontaminated-ckpt368-adam",
            # "ai2-adapt-dev_oasst1_converted-ckpt368-adam",
            "ai2-adapt-dev_personahub_code_v2_34999-ckpt368-adam",
            # "ai2-adapt-dev_personahub_ifdata_manual_seed_v3_29980-ckpt368-adam",
            "ai2-adapt-dev_personahub_math_v5_regen_149960-ckpt368-adam",
            # "ai2-adapt-dev_tulu_hard_coded_repeated_10-ckpt368-adam",
            # "ai2-adapt-dev_tulu_v3.9_aya_100k-ckpt368-adam",
            # "ai2-adapt-dev_tulu_v3.9_open_math_2_gsm8k_50k-ckpt368-adam",
            # "ai2-adapt-dev_tulu_v3.9_personahub_math_interm_algebra_20k-ckpt368-adam",
            "ai2-adapt-dev_tulu_v3.9_sciriff_10k-ckpt368-adam",
            # "ai2-adapt-dev_tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k-ckpt368-adam",
            # "ai2-adapt-dev_tulu_v3.9_table_gpt_5k-ckpt368-adam",
            # "ai2-adapt-dev_tulu_v3.9_wildchat_100k-ckpt368-adam",
            # "ai2-adapt-dev_tulu_v3.9_wildjailbreak_decontaminated_50k-ckpt368-adam",
            # "allenai_tulu-3-sft-personas-math-grade-ckpt368-adam",
            # val datasets
            # "drop-ckpt368-sgd",
            "gsm8k-ckpt368-sgd",
            # "hendrycks_math-ckpt368-sgd",  
            # "humaneval-ckpt368-sgd",  
            "mmlu-ckpt368-sgd",  
            "safety-ckpt368-sgd",  
            "truthfulqa-ckpt368-sgd"
            ]




def extract_dataset_name(file_path: str) -> str:
    """
    Extract dataset name from the file path.
    
    Args:
        file_path: Full path to the gradient file
        
    Returns:
        Dataset name extracted from the path
    """
    # Extract the dataset name from the path structure
    # Example: /path/to/experiment/dataset-name/dim8192/all_origin.pt
    parts = file_path.split('/')
    for i, part in enumerate(parts):
        if part.endswith('.pt') and i >= 3:
            return parts[i-3]  # Get the dataset folder name
    
    # Fallback: use the parent directory of dim folder
    return os.path.basename(os.path.dirname(os.path.dirname(file_path)))


def find_gradient_files(base_path: str, experiment_name: str, dim: str = "dim8192") -> List[str]:
    """
    Find all gradient files under the specified base path.
    
    Args:
        base_path: Base directory path
        experiment_name: Experiment name (e.g., tulu3-Qwen3-8B-p0.05-lora-seed3)
        dim: Dimension folder name (default: dim8192)
        
    Returns:
        List of paths to gradient files
    """
    # First find directories using glob pattern
    dir_pattern = os.path.join(base_path, experiment_name, "*", dim)
    logger.info(f"Searching directory pattern: {dir_pattern}")
    
    # Check if base path exists
    if not os.path.exists(base_path):
        logger.warning(f"Base path does not exist: {base_path}")
        return []
    
    # Check if experiment path exists
    experiment_path = os.path.join(base_path, experiment_name)
    if not os.path.exists(experiment_path):
        logger.warning(f"Experiment path does not exist: {experiment_path}")
        return []
    
    # Find directories first
    gradient_dirs = glob.glob(dir_pattern)
    logger.info(f"Found {len(gradient_dirs)} directories matching pattern")
    
    # Then manually construct file paths
    gradient_files = []
    for grad_dir in gradient_dirs:
        file_path = os.path.join(grad_dir, "all_orig.pt")
        if os.path.exists(file_path):
            dataset_name = extract_dataset_name(file_path)
            if dataset_name in DATASETS:
                gradient_files.append(file_path)
                logger.info(f"Found gradient file: {file_path}")
        else:
            logger.warning(f"Directory found but file missing: {file_path}")
    
    logger.info(f"Found {len(gradient_files)} gradient files in {base_path}")
    
    return sorted(gradient_files)


def save_results(results: Dict[str, float], output_path: str, 
                train_paths: List[str], val_paths: List[str],
                experiment_name: str):
    """
    Save the mixing ratio results to a JSON file.
    
    Args:
        results: Dictionary mapping file paths to mixing weights
        output_path: Path to save the results
        train_paths: List of training gradient file paths
        val_paths: List of validation gradient file paths
        experiment_name: Name of the experiment
    """
    # Create a more readable results format
    readable_results = {}
    total_weight = 0.0
    
    for file_path, weight in results.items():
        dataset_name = extract_dataset_name(file_path)
        readable_results[dataset_name] = {
            "weight": float(weight),
            "percentage": float(weight * 100),
            "file_path": file_path
        }
        total_weight += weight
    
    # Prepare the complete output
    output_data = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "total_weight": float(total_weight),
        "mixing_ratios": readable_results,
        "summary": {
            "num_training_datasets": len(train_paths),
            "num_validation_datasets": len(val_paths),
            "training_datasets": [extract_dataset_name(p) for p in train_paths],
            "validation_datasets": [extract_dataset_name(p) for p in val_paths]
        },
        "file_paths": {
            "training": train_paths,
            "validation": val_paths
        }
    }
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_path}")


def print_results_summary(results: Dict[str, float], train_paths: List[str]):
    """
    Print a summary of the mixing ratio results.
    
    Args:
        results: Dictionary mapping file paths to mixing weights
        train_paths: List of training gradient file paths
    """
    print("\n" + "="*80)
    print("DATA MIXING RATIO RESULTS")
    print("="*80)
    
    total_weight = sum(results.values())
    print(f"Total weight: {total_weight:.6f}")
    print(f"Number of training datasets: {len(train_paths)}")
    print("\nMixing ratios:")
    print("-" * 60)
    
    # Sort by weight (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[0], reverse=False)
    
    for file_path, weight in sorted_results:
        dataset_name = extract_dataset_name(file_path)
        percentage = weight * 100
        print(f"{dataset_name:100} | {weight:8.6f} | {percentage:6.2f}%")
    
    print("-" * 60)
    print(f"{'TOTAL':40} | {total_weight:8.6f} | {total_weight*100:6.2f}%")
    print("="*80)


def main():
    """
    Main function to calculate data mixing ratios.
    """
    parser = argparse.ArgumentParser(description="Calculate optimal data mixing ratios using MMD")
    parser.add_argument("--train_base_path", type=str, required=True,
                       help="Base path for training gradient files")
    parser.add_argument("--val_base_path", type=str, required=True,
                       help="Base path for validation gradient files")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Experiment name (e.g., tulu3-Qwen3-8B-p0.05-lora-seed3)")
    parser.add_argument("--dim", type=str, default="dim8192",
                       help="Dimension folder name (default: dim8192)")
    parser.add_argument("--output_dir", type=str, default="./mixing_results",
                       help="Output directory for results (default: ./mixing_results)")
    parser.add_argument("--rff_dimension", type=int, default=1000,
                       help="RFF dimension (default: 1000)")
    parser.add_argument("--sigma_sample_size", type=int, default=1000,
                       help="Sample size for sigma computation (default: 1000)")
    parser.add_argument("--ridge_penalty", type=float, default=1e-7,
                       help="Ridge penalty (default: 1e-7)")
    parser.add_argument("--regularization_lambda", type=float, default=0.0,
                       help="L2 regularization parameter for QP objective (default: 0.0)")
    parser.add_argument("--sample_number", type=int, default=-1,
                       help="Number of samples to randomly select from each gradient file. If < 0, use all samples (default: -1)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    logger.info("Starting data mixing ratio calculation...")
    logger.info(f"Training base path: {args.train_base_path}")
    logger.info(f"Validation base path: {args.val_base_path}")
    logger.info(f"Experiment name: {args.experiment_name}")
    
    # Find gradient files
    train_paths = find_gradient_files(args.train_base_path, args.experiment_name, args.dim)
    val_paths = find_gradient_files(args.val_base_path, args.experiment_name, args.dim)
    
    if not train_paths:
        logger.error("No training gradient files found!")
        return
    
    if not val_paths:
        logger.error("No validation gradient files found!")
        return
    
    logger.info(f"Found {len(train_paths)} training datasets")
    logger.info(f"Found {len(val_paths)} validation datasets")
    
    # Initialize MMD mixer with specified parameters
    mixer = MMDDataMixer(
        rff_dimension=args.rff_dimension,
        sigma_bandwidth=None,  # Will be computed automatically
        ridge_penalty=args.ridge_penalty,
        regularization_lambda=args.regularization_lambda,
        sample_number=args.sample_number,
        random_seed=args.random_seed,
        auto_sigma=True,
        sigma_sample_size=args.sigma_sample_size
    )
    
    # Calculate optimal mixing weights
    logger.info("Computing optimal mixing weights...")
    results = mixer.optimize_mixing_weights(train_paths, val_paths)
    
    # Print results summary
    print_results_summary(results, train_paths)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"mixing_ratios_{args.experiment_name}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    save_results(results, output_path, train_paths, val_paths, args.experiment_name)
    
    logger.info("Data mixing ratio calculation completed successfully!")


if __name__ == "__main__":
    main()