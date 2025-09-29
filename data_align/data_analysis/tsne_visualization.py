#!/usr/bin/env python3
"""
t-SNE Visualization Script for Gradient Data

This script performs t-SNE visualization on gradient embeddings from different datasets
to visualize the distribution and clustering patterns of the data.

Usage:
    python tsne_visualization.py --base_path <path> --experiment_name <name>

Example:
    python tsne_visualization.py \
        --base_path /mnt/hdfs/selection/yingtai_sft/lora_grads \
        --experiment_name tulu3-Qwen3-8B-p0.05-lora-seed3
"""

import os
import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# Dataset configuration - control which datasets to visualize
TRAIN_DATASETS = [
    "ai2-adapt-dev_coconot_converted-ckpt368-adam",
    "ai2-adapt-dev_evol_codealpaca_heval_decontaminated-ckpt368-adam",
    "ai2-adapt-dev_flan_v2_converted-ckpt368-adam",
    "ai2-adapt-dev_no_robots_converted-ckpt368-adam",
    "ai2-adapt-dev_numinamath_tir_math_decontaminated-ckpt368-adam",
    "ai2-adapt-dev_oasst1_converted-ckpt368-adam",
    "ai2-adapt-dev_personahub_code_v2_34999-ckpt368-adam",
    "ai2-adapt-dev_personahub_ifdata_manual_seed_v3_29980-ckpt368-adam",
    "ai2-adapt-dev_personahub_math_v5_regen_149960-ckpt368-adam",
    # "ai2-adapt-dev_tulu_hard_coded_repeated_10-ckpt368-adam",
    # "ai2-adapt-dev_tulu_v3.9_aya_100k-ckpt368-adam",
    "ai2-adapt-dev_tulu_v3.9_open_math_2_gsm8k_50k-ckpt368-adam",
    "ai2-adapt-dev_tulu_v3.9_personahub_math_interm_algebra_20k-ckpt368-adam",
    "ai2-adapt-dev_tulu_v3.9_sciriff_10k-ckpt368-adam",
    "ai2-adapt-dev_tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k-ckpt368-adam",
    "ai2-adapt-dev_tulu_v3.9_table_gpt_5k-ckpt368-adam",
    "ai2-adapt-dev_tulu_v3.9_wildchat_100k-ckpt368-adam",
    "ai2-adapt-dev_tulu_v3.9_wildjailbreak_decontaminated_50k-ckpt368-adam",
    "allenai_tulu-3-sft-personas-math-grade-ckpt368-adam",
]

VAL_DATASETS = [
    "drop-ckpt368-sgd",
    "gsm8k-ckpt368-sgd",
    "hendrycks_math-ckpt368-sgd",  
    "humaneval-ckpt368-sgd",  
    "mmlu-ckpt368-sgd",  
    "safety-ckpt368-sgd",  
    "truthfulqa-ckpt368-sgd"
]

# For backward compatibility
DATASETS = VAL_DATASETS


def find_gradient_files(base_path: str, experiment_name: str, dim: str = "rank_1/dim8192", 
                       datasets: List[str] = None) -> List[str]:
    """
    Find gradient files for specified datasets.
    
    Args:
        base_path: Base path containing experiment folders
        experiment_name: Experiment name (e.g., tulu3-Qwen3-8B-p0.05-lora-seed3)
        dim: Dimension folder name (e.g., rank_1/dim8192)
        datasets: List of datasets to search for. If None, use DATASETS
        
    Returns:
        List of gradient file paths
    """
    if datasets is None:
        datasets = DATASETS
        
    # First find directories using glob pattern
    dir_pattern = os.path.join(base_path, experiment_name, "*", dim)
    logger.info(f"Searching directory pattern: {dir_pattern}")
    
    # Check if base path exists
    if not os.path.exists(base_path):
        logger.error(f"Base path does not exist: {base_path}")
        return []
    
    # Find all matching directories
    matching_dirs = glob.glob(dir_pattern)
    logger.info(f"Found {len(matching_dirs)} matching directories")
    
    gradient_files = []
    for dir_path in matching_dirs:
        # Extract dataset name from path
        # Path structure: base_path/experiment_name/dataset_name/rank_1/dim8192
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(dir_path)))
        
        # Check if this dataset is in our datasets list
        if dataset_name not in datasets:
            logger.info(f"Skipping dataset not in datasets list: {dataset_name}")
            continue
            
        # Look for .pt files in this directory
        pt_files = glob.glob(os.path.join(dir_path, "*.pt"))
        if pt_files:
            # Use the first .pt file found
            gradient_files.append(pt_files[0])
            logger.info(f"Found gradient file for {dataset_name}: {pt_files[0]}")
        else:
            logger.warning(f"No .pt files found in {dir_path}")
    
    logger.info(f"Total gradient files found: {len(gradient_files)}")
    return gradient_files


def extract_dataset_name(file_path: str) -> str:
    """
    Extract dataset name from file path and clean it for better legend display.
    
    Args:
        file_path: Path to the gradient file
        
    Returns:
        Cleaned dataset name
    """
    # Extract from path structure: 
    # .../experiment_name/dataset_name/rank_1/dim8192/file.pt (with rank_1)
    # or .../experiment_name/dataset_name/dim8192/file.pt (without rank_1)
    path_parts = file_path.split(os.sep)
    
    if len(path_parts) >= 4:
        # Check if rank_1 directory exists in the path
        if len(path_parts) >= 5 and path_parts[-3] == "rank_1":
            # Path structure: .../dataset_name/rank_1/dim8192/file.pt
            dataset_name = path_parts[-4]  # dataset_name is 4 levels up from the file
        else:
            # Path structure: .../dataset_name/dim8192/file.pt
            dataset_name = path_parts[-3]  # dataset_name is 3 levels up from the file
    else:
        dataset_name = os.path.basename(os.path.dirname(file_path))
    
    # Clean the dataset name by removing common prefixes and suffixes
    cleaned_name = clean_dataset_name(dataset_name)
    return cleaned_name


def clean_dataset_name(dataset_name: str) -> str:
    """
    Clean dataset name by removing common prefixes and suffixes.
    
    Args:
        dataset_name: Original dataset name
        
    Returns:
        Cleaned dataset name
    """
    # Remove common prefixes
    prefixes_to_remove = [
        "ai2-adapt-dev_tulu_v3.9_",
        "ai2-adapt-dev_",
        "allenai_tulu-3-sft-"
    ]
    
    # Remove common suffixes
    suffixes_to_remove = [
        "wildguardmixtrain_decontaminated_50k-ckpt368-adam",
        "-ckpt368-adam",
        "-ckpt368-sgd"
    ]
    
    cleaned = dataset_name
    
    # Remove prefixes
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    
    # Remove suffixes
    for suffix in suffixes_to_remove:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break
    
    return cleaned


def load_and_sample_data(file_paths: List[str], sample_number: int = -1, random_seed: int = 42, 
                        data_type: str = "unknown") -> Tuple[np.ndarray, List[str], List[str], List[int]]:
    """
    Load gradient data from files and optionally sample data points.
    
    Args:
        file_paths: List of paths to gradient files
        sample_number: Number of samples per dataset. If < 0, use all data
        random_seed: Random seed for reproducibility
        data_type: Type of data ("train", "val", or "unknown")
        
    Returns:
        Tuple of (combined_data, dataset_labels, data_type_labels, sample_counts)
    """
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    all_data = []
    all_labels = []
    all_data_types = []
    sample_counts = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Loading {data_type} data from: {file_path}")
        
        # Load tensor
        tensor = torch.load(file_path, map_location='cpu')
        
        if not isinstance(tensor, torch.Tensor):
            logger.error(f"Expected tensor in {file_path}, got {type(tensor)}")
            continue
            
        if tensor.dim() != 2:
            logger.error(f"Expected 2D tensor in {file_path}, got shape {tensor.shape}")
            continue
        
        # Convert to numpy
        data = tensor.numpy()
        dataset_name = extract_dataset_name(file_path)
        
        # Apply sampling if specified
        if sample_number > 0 and data.shape[0] > sample_number:
            # Randomly sample data points
            indices = np.random.choice(data.shape[0], sample_number, replace=False)
            data = data[indices]
            logger.info(f"Sampled {sample_number} points from {tensor.shape[0]} total points for {dataset_name} ({data_type})")
        else:
            logger.info(f"Using all {data.shape[0]} points for {dataset_name} ({data_type})")
        
        all_data.append(data)
        all_labels.extend([dataset_name] * data.shape[0])
        all_data_types.extend([data_type] * data.shape[0])
        sample_counts.append(data.shape[0])
    
    if not all_data:
        raise ValueError("No valid data found!")
    
    # Combine all data
    combined_data = np.vstack(all_data)
    logger.info(f"Combined {data_type} data shape: {combined_data.shape}")
    logger.info(f"Total {data_type} samples per dataset: {dict(zip([extract_dataset_name(fp) for fp in file_paths], sample_counts))}")
    
    return combined_data, all_labels, all_data_types, sample_counts


def apply_pca_preprocessing(data: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    Apply PCA preprocessing to reduce dimensionality before t-SNE.
    
    Args:
        data: Input data array
        n_components: Number of PCA components to keep
        
    Returns:
        PCA-transformed data
    """
    logger.info(f"Applying PCA preprocessing: {data.shape[1]} -> {n_components} dimensions")
    
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data)
    
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA explained variance ratio: {explained_variance_ratio:.4f}")
    
    return data_pca


def perform_tsne(data: np.ndarray, perplexity: int = 30, max_iter: int = 1000, random_seed: int = 42) -> np.ndarray:
    """
    Perform t-SNE dimensionality reduction.
    
    Args:
        data: Input data array
        perplexity: t-SNE perplexity parameter
        max_iter: Number of iterations
        random_seed: Random seed for reproducibility
        
    Returns:
        2D t-SNE embedding
    """
    logger.info(f"Performing t-SNE with perplexity={perplexity}, max_iter={max_iter}")
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_seed,
        verbose=1
    )
    
    tsne_result = tsne.fit_transform(data)
    logger.info(f"t-SNE completed. Final KL divergence: {tsne.kl_divergence_:.4f}")
    
    return tsne_result


def create_visualization(tsne_result: np.ndarray, labels: List[str], data_types: List[str] = None, 
                       output_path: str = "", title: str = "t-SNE Visualization"):
    """
    Create and save t-SNE visualization.
    
    Args:
        tsne_result: 2D t-SNE coordinates
        labels: Dataset labels for each point
        data_types: Data type labels for each point ("train", "val", etc.)
        output_path: Path to save the plot
        title: Plot title
    """
    # Create DataFrame for easier plotting
    df_data = {
        'x': tsne_result[:, 0],
        'y': tsne_result[:, 1],
        'dataset': labels
    }
    
    if data_types is not None:
        df_data['data_type'] = data_types
        # Create combined labels for legend
        df_data['combined_label'] = [f"{dataset} ({data_type})" for dataset, data_type in zip(labels, data_types)]
    else:
        df_data['combined_label'] = labels
    
    df = pd.DataFrame(df_data)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    if data_types is not None:
        # Create scatter plot with different colors for datasets and shapes for data types
        unique_datasets = sorted(set(labels))
        unique_data_types = sorted(set(data_types))
        
        # Color palette for datasets - handle large number of datasets
        if len(unique_datasets) <= 10:
            # Use bright, distinct colors for small number of datasets
            colors = sns.color_palette("bright", len(unique_datasets))
        elif len(unique_datasets) <= 20:
            # Use tab20 for medium number of datasets
            colors = sns.color_palette("tab20", len(unique_datasets))
        else:
            # For large number of datasets (>20), combine multiple palettes
            base_colors = (sns.color_palette("tab20", 20) + 
                          sns.color_palette("Set1", 9) + 
                          sns.color_palette("Set2", 8) +
                          sns.color_palette("Set3", 12))
            # Cycle through colors if we have more datasets than colors
            colors = [base_colors[i % len(base_colors)] for i in range(len(unique_datasets))]
        dataset_colors = {dataset: colors[i] for i, dataset in enumerate(unique_datasets)}
        
        # Enhanced markers for data types - more variety for large datasets
        markers = ['o', '*', '^', 'D', 'v', '<', '>', 'p', 's', 'h', 'X', 'P', '8', 'H', '+', 'x', '1', '2', '3', '4']
        data_type_markers = {dt: markers[i % len(markers)] for i, dt in enumerate(unique_data_types)}
        
        # Plot each combination of dataset and data type
        for dataset in unique_datasets:
            for data_type in unique_data_types:
                mask = (df['dataset'] == dataset) & (df['data_type'] == data_type)
                if mask.any():
                    plt.scatter(
                        df[mask]['x'], 
                        df[mask]['y'], 
                        c=[dataset_colors[dataset]], 
                        marker=data_type_markers[data_type],
                        label=f"{dataset} ({data_type})", 
                        alpha=0.7, 
                        s=30,
                        edgecolors='black',
                        linewidth=0.5
                    )
    else:
        # Original single-type visualization
        unique_datasets = df['dataset'].unique()
        # Use more vibrant colors for better distinction - handle large number of datasets
        if len(unique_datasets) <= 10:
            colors = sns.color_palette("bright", len(unique_datasets))
        elif len(unique_datasets) <= 20:
            colors = sns.color_palette("tab20", len(unique_datasets))
        else:
            # For large number of datasets (>20), combine multiple palettes
            base_colors = (sns.color_palette("tab20", 20) + 
                          sns.color_palette("Set1", 9) + 
                          sns.color_palette("Set2", 8) +
                          sns.color_palette("Set3", 12))
            # Cycle through colors if we have more datasets than colors
            colors = [base_colors[i % len(base_colors)] for i in range(len(unique_datasets))]
        
        # Enhanced markers for better distinction with many datasets
        markers = ['o', '*', '^', 'D', 'v', '<', '>', 'p', 's', 'h', 'X', 'P', '8', 'H', '+', 'x', '1', '2', '3', '4']
        
        for i, dataset in enumerate(unique_datasets):
            mask = df['dataset'] == dataset
            plt.scatter(
                df[mask]['x'], 
                df[mask]['y'], 
                c=[colors[i]], 
                marker=markers[i % len(markers)],
                label=dataset, 
                alpha=0.6, 
                s=20
            )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {output_path}")
        
        # Also save as PDF
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        logger.info(f"PDF version saved to: {pdf_path}")
    
    plt.close()


def save_tsne_coordinates(tsne_result: np.ndarray, labels: List[str], output_path: str, data_types: List[str] = None):
    """
    Save t-SNE coordinates to CSV file.
    
    Args:
        tsne_result: 2D t-SNE coordinates
        labels: Dataset labels for each point
        output_path: Path to save the CSV file
        data_types: Data type labels for each point ("train", "val", etc.)
    """
    df_data = {
        'x': tsne_result[:, 0],
        'y': tsne_result[:, 1],
        'dataset': labels
    }
    
    if data_types is not None:
        df_data['data_type'] = data_types
    
    df = pd.DataFrame(df_data)
    
    df.to_csv(output_path, index=False)
    logger.info(f"t-SNE coordinates saved to: {output_path}")


def main():
    """
    Main function for t-SNE visualization.
    """
    parser = argparse.ArgumentParser(description="Perform t-SNE visualization on gradient data")
    parser.add_argument("--base_path", type=str, 
                       help="Base path for gradient files (for single-path mode)")
    parser.add_argument("--train_base_path", type=str,
                       help="Base path for training gradient files (for dual-path mode)")
    parser.add_argument("--val_base_path", type=str,
                       help="Base path for validation gradient files (for dual-path mode)")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Experiment name (e.g., tulu3-Qwen3-8B-p0.05-lora-seed3)")
    parser.add_argument("--dim", type=str, default="rank_1/dim8192",
                       help="Dimension folder name (default: rank_1/dim8192)")
    parser.add_argument("--output_dir", type=str, default="./tsne_results",
                       help="Output directory for results (default: ./tsne_results)")
    parser.add_argument("--sample_number", type=int, default=-1,
                       help="Number of samples per dataset. If < 0, use all data (default: -1)")
    parser.add_argument("--pca_components", type=int, default=50,
                       help="Number of PCA components for preprocessing (default: 50)")
    parser.add_argument("--perplexity", type=int, default=30,
                       help="t-SNE perplexity parameter (default: 30)")
    parser.add_argument("--max_iter", type=int, default=1000,
                       help="Number of t-SNE iterations (default: 1000)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Validate arguments
    dual_path_mode = args.train_base_path is not None or args.val_base_path is not None
    single_path_mode = args.base_path is not None
    
    if not dual_path_mode and not single_path_mode:
        logger.error("Must specify either --base_path (single-path mode) or --train_base_path/--val_base_path (dual-path mode)")
        return
    
    if dual_path_mode and single_path_mode:
        logger.error("Cannot specify both single-path and dual-path arguments")
        return
    
    logger.info("Starting t-SNE visualization...")
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Sample number: {args.sample_number}")
    logger.info(f"Random seed: {args.random_seed}")
    
    all_data = []
    all_labels = []
    all_data_types = []
    
    if dual_path_mode:
        logger.info("Running in dual-path mode (train + validation)")
        
        # Load training data
        if args.train_base_path:
            logger.info(f"Training base path: {args.train_base_path}")
            train_files = find_gradient_files(args.train_base_path, args.experiment_name, args.dim, TRAIN_DATASETS)
            if train_files:
                train_data, train_labels, train_types, train_counts = load_and_sample_data(
                    train_files, sample_number=args.sample_number, random_seed=args.random_seed, data_type="train"
                )
                all_data.append(train_data)
                all_labels.extend(train_labels)
                all_data_types.extend(train_types)
                logger.info(f"Loaded {len(train_labels)} training data points")
            else:
                logger.warning("No training gradient files found!")
        
        # Load validation data
        if args.val_base_path:
            logger.info(f"Validation base path: {args.val_base_path}")
            val_files = find_gradient_files(args.val_base_path, args.experiment_name, args.dim, VAL_DATASETS)
            if val_files:
                val_data, val_labels, val_types, val_counts = load_and_sample_data(
                    val_files, sample_number=args.sample_number, random_seed=args.random_seed, data_type="val"
                )
                all_data.append(val_data)
                all_labels.extend(val_labels)
                all_data_types.extend(val_types)
                logger.info(f"Loaded {len(val_labels)} validation data points")
            else:
                logger.warning("No validation gradient files found!")
        
        if not all_data:
            logger.error("No gradient files found in either training or validation paths!")
            return
        
        # Combine all data
        combined_data = np.vstack(all_data)
        data_types = all_data_types
        
    else:
        logger.info("Running in single-path mode")
        logger.info(f"Base path: {args.base_path}")
        
        # Find gradient files
        gradient_files = find_gradient_files(args.base_path, args.experiment_name, args.dim)
        
        if not gradient_files:
            logger.error("No gradient files found!")
            return
        
        logger.info(f"Found {len(gradient_files)} gradient files")
        
        # Load and sample data
        combined_data, all_labels, data_types, sample_counts = load_and_sample_data(
            gradient_files, 
            sample_number=args.sample_number, 
            random_seed=args.random_seed,
            data_type="unknown"
        )
    
    # Apply PCA preprocessing if data dimension is high
    if combined_data.shape[1] > args.pca_components:
        combined_data = apply_pca_preprocessing(combined_data, n_components=args.pca_components)
    
    # Perform t-SNE
    tsne_result = perform_tsne(
        combined_data, 
        perplexity=args.perplexity, 
        max_iter=args.max_iter, 
        random_seed=args.random_seed
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "dual" if dual_path_mode else "single"
    base_filename = f"tsne_{args.experiment_name}_{mode_suffix}_{timestamp}"
    
    # Create visualization
    plot_path = os.path.join(args.output_dir, f"{base_filename}.png")
    title = f"t-SNE Visualization - {args.experiment_name}"
    if dual_path_mode:
        title += " (Train + Validation)"
    
    create_visualization(tsne_result, all_labels, data_types if dual_path_mode else None, plot_path, title)
    
    # Save coordinates
    csv_path = os.path.join(args.output_dir, f"{base_filename}_coordinates.csv")
    save_tsne_coordinates(tsne_result, all_labels, csv_path, data_types if dual_path_mode else None)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("t-SNE Visualization Summary")
    logger.info("="*50)
    logger.info(f"Mode: {'Dual-path (Train + Validation)' if dual_path_mode else 'Single-path'}")
    logger.info(f"Total data points: {len(all_labels)}")
    logger.info(f"Number of datasets: {len(set(all_labels))}")
    logger.info(f"Datasets: {sorted(set(all_labels))}")
    
    if dual_path_mode:
        train_count = sum(1 for dt in data_types if dt == "train")
        val_count = sum(1 for dt in data_types if dt == "val")
        logger.info(f"Training data points: {train_count}")
        logger.info(f"Validation data points: {val_count}")
    
    logger.info(f"Sample counts by dataset: {dict(zip(sorted(set(all_labels)), [all_labels.count(l) for l in sorted(set(all_labels))]))}")
    logger.info(f"Output files:")
    logger.info(f"  - Visualization: {plot_path}")
    logger.info(f"  - PDF: {plot_path.replace('.png', '.pdf')}")
    logger.info(f"  - Coordinates: {csv_path}")
    logger.info("="*50)
    
    logger.info("t-SNE visualization completed successfully!")


if __name__ == "__main__":
    main()