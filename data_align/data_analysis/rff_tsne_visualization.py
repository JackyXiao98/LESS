#!/usr/bin/env python3
"""
RFF + t-SNE Visualization Script for Gradient Data

This script applies Random Fourier Features (RFF) transformation to gradient embeddings
and then performs t-SNE visualization to visualize the distribution and clustering patterns
of the transformed data.

Usage:
    python rff_tsne_visualization.py --base_path <path> --experiment_name <name>

Example:
    python rff_tsne_visualization.py \
        --base_path /mnt/hdfs/selection/yingtai_sft/lora_grads \
        --experiment_name tulu3-Qwen3-8B-p0.05-lora-seed3 \
        --rff_dimension 512 \
        --sigma_bandwidth 1.0
"""

import os
import sys
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

# Add parent directory to path to import mmd_data_mixing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmd_data_mixing import MMDDataMixer

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


class RFFTransformer:
    """
    Random Fourier Features transformer for gradient data.
    """
    
    def __init__(self, 
                 rff_dimension: int = 512,
                 sigma_bandwidth: Optional[float] = None,
                 random_seed: Optional[int] = 42,
                 auto_sigma: bool = True,
                 sigma_sample_size: int = 1000):
        """
        Initialize RFF transformer.
        
        Args:
            rff_dimension: Target dimension for RFF space
            sigma_bandwidth: Bandwidth for the Gaussian kernel
            random_seed: Random seed for reproducibility
            auto_sigma: Whether to automatically compute sigma using median heuristic
            sigma_sample_size: Number of samples to use for sigma computation
        """
        self.D = rff_dimension
        self.sigma = sigma_bandwidth
        self.random_seed = random_seed
        self.auto_sigma = auto_sigma
        self.sigma_sample_size = sigma_sample_size
        
        # RFF parameters (will be initialized when first used)
        self.Omega = None
        self.b = None
        self.d = None  # Original dimension
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
    
    def _compute_median_heuristic_sigma(self, data_list: List[torch.Tensor]) -> float:
        """
        Compute sigma using median heuristic.
        
        Args:
            data_list: List of data tensors
            
        Returns:
            Computed sigma value
        """
        logger.info("Computing sigma using median heuristic...")
        
        # Sample data for sigma computation
        sampled_data = []
        for tensor in data_list:
            if tensor.shape[0] <= self.sigma_sample_size:
                sampled_data.append(tensor)
            else:
                indices = torch.randperm(tensor.shape[0])[:self.sigma_sample_size]
                sampled_data.append(tensor[indices])
        
        # Combine sampled data
        combined_data = torch.cat(sampled_data, dim=0)
        
        # Compute pairwise distances
        n_samples = min(self.sigma_sample_size, combined_data.shape[0])
        indices = torch.randperm(combined_data.shape[0])[:n_samples]
        sample_data = combined_data[indices]
        
        # Compute squared distances
        distances_squared = torch.cdist(sample_data, sample_data, p=2) ** 2
        
        # Get upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones_like(distances_squared, dtype=torch.bool), diagonal=1)
        distances_squared = distances_squared[mask]
        
        # Compute median
        median_distance_squared = torch.median(distances_squared).item()
        sigma = np.sqrt(median_distance_squared / 2.0)
        
        logger.info(f"Computed sigma using median heuristic: {sigma:.6f}")
        return sigma
    
    def _initialize_rff_parameters(self, dimension: int, data_list: List[torch.Tensor] = None):
        """
        Initialize Random Fourier Features parameters.
        
        Args:
            dimension: Original dimension of the gradient embeddings
            data_list: List of data tensors for sigma computation
        """
        if self.d is not None and self.d != dimension:
            raise ValueError(f"Dimension mismatch: expected {self.d}, got {dimension}")
        
        if self.Omega is None:
            self.d = dimension
            
            # Compute sigma if needed
            if self.sigma is None and self.auto_sigma and data_list is not None:
                self.sigma = self._compute_median_heuristic_sigma(data_list)
            elif self.sigma is None:
                self.sigma = 1.0  # Default value
                logger.warning(f"Using default sigma value: {self.sigma}")
            
            logger.info(f"Initializing RFF parameters: d={self.d}, D={self.D}, sigma={self.sigma}")
            
            # Sample random frequency matrix Omega: (d, D)
            # Each column omega_r ~ N(0, σ^(-2) * I_d)
            self.Omega = torch.randn(self.d, self.D) / self.sigma
            
            # Sample random phase vector b: (D,)
            # Each element b_r ~ Unif[0, 2π]
            self.b = torch.rand(self.D) * 2 * np.pi
            
            logger.info("RFF parameters initialized successfully")
    
    def rff_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply Random Fourier Features transformation.
        
        Args:
            X: Input tensor of shape (num_examples, gradient_dimension)
            
        Returns:
            Transformed tensor of shape (num_examples, rff_dimension)
        """
        # Compute projection: Z = X @ Omega
        Z = torch.mm(X, self.Omega)
        
        # Add phase: Z = Z + b
        Z = Z + self.b.unsqueeze(0)
        
        # Apply cosine: Z = cos(Z)
        Z = torch.cos(Z)
        
        # Scale and return: Z * sqrt(2/D)
        return Z * np.sqrt(2.0 / self.D)


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
                        data_type: str = "unknown") -> Tuple[List[torch.Tensor], List[str], List[str], List[int]]:
    """
    Load gradient data from files and optionally sample data points.
    
    Args:
        file_paths: List of paths to gradient files
        sample_number: Number of samples per dataset. If < 0, use all data
        random_seed: Random seed for reproducibility
        data_type: Type of data ("train", "val", or "unknown")
        
    Returns:
        Tuple of (list_of_tensors, dataset_labels, data_type_labels, sample_counts)
    """
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    tensor_list = []
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
        
        dataset_name = extract_dataset_name(file_path)
        
        # Apply sampling if specified
        if sample_number > 0 and tensor.shape[0] > sample_number:
            # Randomly sample data points
            indices = torch.randperm(tensor.shape[0])[:sample_number]
            tensor = tensor[indices]
            logger.info(f"Sampled {sample_number} points from {tensor.shape[0]} total points for {dataset_name} ({data_type})")
        else:
            logger.info(f"Using all {tensor.shape[0]} points for {dataset_name} ({data_type})")
        
        tensor_list.append(tensor)
        all_labels.extend([dataset_name] * tensor.shape[0])
        all_data_types.extend([data_type] * tensor.shape[0])
        sample_counts.append(tensor.shape[0])
    
    if not tensor_list:
        raise ValueError("No valid data found!")
    
    logger.info(f"Loaded {len(tensor_list)} {data_type} datasets")
    logger.info(f"Total {data_type} samples per dataset: {dict(zip([extract_dataset_name(fp) for fp in file_paths], sample_counts))}")
    
    return tensor_list, all_labels, all_data_types, sample_counts


def apply_rff_transformation(tensor_list: List[torch.Tensor], rff_transformer: RFFTransformer) -> np.ndarray:
    """
    Apply RFF transformation to a list of tensors and combine them.
    
    Args:
        tensor_list: List of tensors to transform
        rff_transformer: RFF transformer instance
        
    Returns:
        Combined RFF-transformed data as numpy array
    """
    logger.info("Applying RFF transformation...")
    
    # Initialize RFF parameters using all data
    if rff_transformer.Omega is None:
        dimension = tensor_list[0].shape[1]
        rff_transformer._initialize_rff_parameters(dimension, tensor_list)
    
    # Transform each tensor and combine
    transformed_tensors = []
    for i, tensor in enumerate(tensor_list):
        logger.info(f"Transforming tensor {i+1}/{len(tensor_list)} with shape {tensor.shape}")
        transformed = rff_transformer.rff_transform(tensor)
        transformed_tensors.append(transformed)
    
    # Combine all transformed data
    combined_data = torch.cat(transformed_tensors, dim=0)
    logger.info(f"Combined RFF-transformed data shape: {combined_data.shape}")
    
    return combined_data.numpy()


def apply_pca_preprocessing(data: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    Apply PCA preprocessing to reduce dimensionality before t-SNE.
    
    Args:
        data: Input data array
        n_components: Number of PCA components to keep
        
    Returns:
        PCA-transformed data
    """
    if data.shape[1] <= n_components:
        logger.info(f"Data dimension ({data.shape[1]}) <= PCA components ({n_components}), skipping PCA")
        return data
    
    logger.info(f"Applying PCA preprocessing: {data.shape[1]} -> {n_components} dimensions")
    
    from sklearn.decomposition import PCA
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
                       output_path: str = "", title: str = "RFF + t-SNE Visualization"):
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
    Main function for RFF + t-SNE visualization.
    """
    parser = argparse.ArgumentParser(description="Perform RFF + t-SNE visualization on gradient data")
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
    parser.add_argument("--output_dir", type=str, default="./rff_tsne_results",
                       help="Output directory for results (default: ./rff_tsne_results)")
    parser.add_argument("--sample_number", type=int, default=-1,
                       help="Number of samples per dataset. If < 0, use all data (default: -1)")
    
    # RFF parameters
    parser.add_argument("--rff_dimension", type=int, default=512,
                       help="RFF dimension (default: 512)")
    parser.add_argument("--sigma_bandwidth", type=float, default=None,
                       help="Sigma bandwidth for RFF. If None, use median heuristic (default: None)")
    parser.add_argument("--auto_sigma", action="store_true", default=True,
                       help="Automatically compute sigma using median heuristic (default: True)")
    parser.add_argument("--sigma_sample_size", type=int, default=1000,
                       help="Number of samples for sigma computation (default: 1000)")
    
    # t-SNE parameters
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
    
    logger.info("Starting RFF + t-SNE visualization...")
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"RFF dimension: {args.rff_dimension}")
    logger.info(f"Sigma bandwidth: {args.sigma_bandwidth}")
    logger.info(f"Sample number: {args.sample_number}")
    logger.info(f"Random seed: {args.random_seed}")
    
    # Create RFF transformer
    rff_transformer = RFFTransformer(
        rff_dimension=args.rff_dimension,
        sigma_bandwidth=args.sigma_bandwidth,
        random_seed=args.random_seed,
        auto_sigma=args.auto_sigma,
        sigma_sample_size=args.sigma_sample_size
    )
    
    all_tensor_lists = []
    all_labels = []
    all_data_types = []
    
    if dual_path_mode:
        logger.info("Running in dual-path mode (train + validation)")
        
        # Load training data
        if args.train_base_path:
            logger.info(f"Training base path: {args.train_base_path}")
            train_files = find_gradient_files(args.train_base_path, args.experiment_name, args.dim, TRAIN_DATASETS)
            if train_files:
                train_tensor_list, train_labels, train_types, train_counts = load_and_sample_data(
                    train_files, sample_number=args.sample_number, random_seed=args.random_seed, data_type="train"
                )
                all_tensor_lists.extend(train_tensor_list)
                all_labels.extend(train_labels)
                all_data_types.extend(train_types)
        
        # Load validation data
        if args.val_base_path:
            logger.info(f"Validation base path: {args.val_base_path}")
            val_files = find_gradient_files(args.val_base_path, args.experiment_name, args.dim, VAL_DATASETS)
            if val_files:
                val_tensor_list, val_labels, val_types, val_counts = load_and_sample_data(
                    val_files, sample_number=args.sample_number, random_seed=args.random_seed, data_type="val"
                )
                all_tensor_lists.extend(val_tensor_list)
                all_labels.extend(val_labels)
                all_data_types.extend(val_types)
    
    else:
        logger.info("Running in single-path mode")
        logger.info(f"Base path: {args.base_path}")
        
        # Load data
        gradient_files = find_gradient_files(args.base_path, args.experiment_name, args.dim, DATASETS)
        if not gradient_files:
            logger.error("No gradient files found!")
            return
        
        tensor_list, labels, data_types, sample_counts = load_and_sample_data(
            gradient_files, sample_number=args.sample_number, random_seed=args.random_seed, data_type="unknown"
        )
        all_tensor_lists = tensor_list
        all_labels = labels
        all_data_types = data_types
    
    if not all_tensor_lists:
        logger.error("No data loaded!")
        return
    
    # Apply RFF transformation
    rff_data = apply_rff_transformation(all_tensor_lists, rff_transformer)
    
    # Apply PCA preprocessing if needed
    if args.pca_components > 0 and rff_data.shape[1] > args.pca_components:
        rff_data = apply_pca_preprocessing(rff_data, args.pca_components)
    
    # Perform t-SNE
    tsne_result = perform_tsne(rff_data, args.perplexity, args.max_iter, args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"rff_tsne_{args.experiment_name}_{timestamp}"
    
    # Create visualization
    plot_path = os.path.join(args.output_dir, f"{base_filename}.png")
    title = f"RFF + t-SNE Visualization\nRFF Dim: {args.rff_dimension}, Sigma: {rff_transformer.sigma:.4f}"
    
    create_visualization(
        tsne_result, 
        all_labels, 
        all_data_types if dual_path_mode else None,
        plot_path, 
        title
    )
    
    # Save coordinates
    csv_path = os.path.join(args.output_dir, f"{base_filename}_coordinates.csv")
    save_tsne_coordinates(
        tsne_result, 
        all_labels, 
        csv_path, 
        all_data_types if dual_path_mode else None
    )
    
    logger.info("RFF + t-SNE visualization completed successfully!")


if __name__ == "__main__":
    main()