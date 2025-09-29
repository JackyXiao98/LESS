#!/usr/bin/env python3
"""
Cluster-based MMD Data Mixing Optimization Script

This script performs MMD-based data mixing optimization using clustered embedding data.
It calculates optimal mixing ratios for each cluster to best match a target distribution.

Author: AI Assistant
Date: 2025-09-25
"""

import sys
import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import MMDDataMixer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_align'))
from mmd_data_mixing import MMDDataMixer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ClusterMMDMixer(MMDDataMixer):
    """
    A specialized MMD Data Mixer for clustered embedding data.
    
    This class extends MMDDataMixer to work with clustered .pkl files
    and calculate optimal mixing ratios for each cluster.
    """
    
    def __init__(self, 
                 rff_dimension: int = 100,
                 sigma_bandwidth: Optional[float] = None,
                 ridge_penalty: float = 1e-7,
                 regularization_lambda: float = 0.0,
                 sample_number: int = -1,
                 random_seed: Optional[int] = 42,
                 auto_sigma: bool = True,
                 sigma_sample_size: int = 1000):
        """
        Initialize the Cluster MMD Mixer.
        
        Args:
            rff_dimension: Target dimension for Random Fourier Features space
            sigma_bandwidth: Bandwidth for the Gaussian kernel
            ridge_penalty: Ridge penalty for numerical stability
            regularization_lambda: L2 regularization parameter
            sample_number: Number of samples to randomly select from each cluster
            random_seed: Random seed for reproducibility
            auto_sigma: Whether to automatically compute sigma using median heuristic
            sigma_sample_size: Number of samples to use for sigma computation
        """
        super().__init__(
            rff_dimension=rff_dimension,
            sigma_bandwidth=sigma_bandwidth,
            ridge_penalty=ridge_penalty,
            regularization_lambda=regularization_lambda,
            sample_number=sample_number,
            random_seed=random_seed,
            auto_sigma=auto_sigma,
            sigma_sample_size=sigma_sample_size
        )
        
        # Store cluster information
        self.cluster_info = {}
        self.cluster_weights = {}
    
    def load_cluster_data(self, cluster_dir: str) -> List[torch.Tensor]:
        """
        Load clustered embedding data from .pkl files.
        
        Args:
            cluster_dir: Directory containing cluster .pkl files
            
        Returns:
            List of loaded cluster embedding tensors
        """
        cluster_dir = Path(cluster_dir)
        if not cluster_dir.exists():
            raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")
        
        # Find all cluster files
        cluster_files = sorted(list(cluster_dir.glob("cluster_*.pkl")))
        if not cluster_files:
            raise FileNotFoundError(f"No cluster files found in: {cluster_dir}")
        
        logger.info(f"Found {len(cluster_files)} cluster files")
        
        tensors = []
        for cluster_file in cluster_files:
            logger.info(f"Loading cluster data from: {cluster_file}")
            
            # Load cluster pickle file
            with open(cluster_file, 'rb') as f:
                cluster_data = pickle.load(f)
            
            # Extract cluster information
            cluster_id = cluster_data['cluster_id']
            embeddings = cluster_data['embeddings']
            source_labels = cluster_data['source_labels']
            cluster_size = cluster_data['size']
            
            # Store cluster information
            self.cluster_info[cluster_id] = {
                'file': cluster_file.name,
                'size': cluster_size,
                'source_distribution': dict(zip(*np.unique(source_labels, return_counts=True))),
                'embeddings_shape': embeddings.shape
            }
            
            # Convert to torch tensor if it's numpy array
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings).float()
            elif not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, dtype=torch.float32)
            
            tensors.append(embeddings)
            logger.info(f"Loaded cluster {cluster_id}: {embeddings.shape}")
        
        return tensors
    
    def load_target_data(self, target_file: str) -> torch.Tensor:
        """
        Load target embedding data from .pkl file.
        
        Args:
            target_file: Path to target embedding .pkl file
            
        Returns:
            Target embedding tensor
        """
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"Target file not found: {target_file}")
        
        logger.info(f"Loading target data from: {target_file}")
        
        # Load target pickle file
        with open(target_file, 'rb') as f:
            target_data = pickle.load(f)
        
        # Extract embeddings
        if isinstance(target_data, dict) and 'embeddings' in target_data:
            embeddings = target_data['embeddings']
        elif isinstance(target_data, (list, tuple)):
            embeddings = np.array(target_data)
        elif isinstance(target_data, np.ndarray):
            embeddings = target_data
        else:
            raise ValueError(f"Unknown target data format: {type(target_data)}")
        
        # Convert to torch tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        elif not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        logger.info(f"Loaded target data: {embeddings.shape}")
        return embeddings
    
    def optimize_cluster_mixing_weights(self, 
                                      cluster_dir: str, 
                                      target_file: str) -> Dict[str, float]:
        """
        Optimize mixing weights for clustered data to match target distribution.
        
        Args:
            cluster_dir: Directory containing cluster .pkl files
            target_file: Path to target embedding .pkl file
            
        Returns:
            Dictionary mapping cluster IDs to optimal weights
        """
        logger.info("Starting cluster mixing weight optimization...")
        
        # Load cluster data
        cluster_tensors = self.load_cluster_data(cluster_dir)
        
        # Load target data
        target_tensor = self.load_target_data(target_file)
        
        # Optimize mixing weights using tensor-based method
        weights = self.optimize_mixing_weights_from_tensors(cluster_tensors, target_tensor)
        
        # Map weights to cluster IDs
        cluster_weights = {}
        for i, (cluster_id, info) in enumerate(sorted(self.cluster_info.items())):
            cluster_weights[f"cluster_{cluster_id:02d}"] = weights[i]
        
        self.cluster_weights = cluster_weights
        
        # Log results
        logger.info("Cluster mixing optimization completed!")
        for cluster_name, weight in cluster_weights.items():
            cluster_id = int(cluster_name.split('_')[1])
            size = self.cluster_info[cluster_id]['size']
            logger.info(f"{cluster_name}: weight={weight:.6f}, size={size}")
        
        return cluster_weights
    
    def optimize_mixing_weights_from_tensors(self, 
                                           train_tensors: List[torch.Tensor], 
                                           val_tensor: torch.Tensor) -> np.ndarray:
        """
        Optimize mixing weights using tensor data directly.
        
        Args:
            train_tensors: List of training tensors (cluster embeddings)
            val_tensor: Validation tensor (target embeddings)
            
        Returns:
            Array of optimal weights
        """
        logger.info("Starting MMD-based data mixing optimization...")
        logger.info(f"Training datasets: {len(train_tensors)}")
        logger.info(f"Validation dataset size: {val_tensor.shape[0]}")
        
        # Compute sigma automatically if needed
        if self.auto_sigma and self.sigma is None:
            self.sigma = self._compute_median_heuristic_sigma(train_tensors)
            logger.info(f"Using automatically computed sigma: {self.sigma:.6f}")
        elif self.sigma is not None:
            logger.info(f"Using provided sigma: {self.sigma:.6f}")
        else:
            raise ValueError("Sigma bandwidth must be provided or auto_sigma must be True")
        
        # Initialize RFF parameters
        if not hasattr(self, 'W') or self.W is None:
            # Get dimension from first tensor
            dimension = train_tensors[0].shape[1]
            self._initialize_rff_parameters(dimension)
        
        # Compute mean features
        train_means, val_mean = self.compute_mean_features(train_tensors, [val_tensor])
        
        # Solve quadratic programming problem
        weights = self.solve_qp(train_means, val_mean)
        
        # Calculate final MMD value
        final_mmd = self.calculate_mmd_value(train_means, val_mean, weights)
        logger.info(f"Final MMD value: {final_mmd:.6f}")
        
        return weights
    
    def create_cluster_mixing_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive cluster mixing analysis visualization.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating cluster mixing analysis visualization...")
        
        if not self.cluster_weights:
            raise ValueError("No cluster weights available. Run optimization first.")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cluster Mixing Weight Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        cluster_ids = []
        weights = []
        sizes = []
        source_distributions = {}
        
        for cluster_name, weight in self.cluster_weights.items():
            cluster_id = int(cluster_name.split('_')[1])
            cluster_ids.append(cluster_id)
            weights.append(weight)
            sizes.append(self.cluster_info[cluster_id]['size'])
            
            # Collect source distributions
            for source, count in self.cluster_info[cluster_id]['source_distribution'].items():
                if source not in source_distributions:
                    source_distributions[source] = []
                source_distributions[source].append(count)
        
        # 1. Cluster weights bar plot
        ax1 = axes[0, 0]
        bars = ax1.bar(cluster_ids, weights, alpha=0.7, color=plt.cm.tab10(cluster_ids))
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Mixing Weight')
        ax1.set_title('Optimal Mixing Weights by Cluster')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(weights)*0.01,
                    f'{weight:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Weight vs Size scatter plot
        ax2 = axes[0, 1]
        scatter = ax2.scatter(sizes, weights, c=cluster_ids, cmap='tab10', alpha=0.7, s=100)
        ax2.set_xlabel('Cluster Size')
        ax2.set_ylabel('Mixing Weight')
        ax2.set_title('Mixing Weight vs Cluster Size')
        ax2.grid(True, alpha=0.3)
        
        # Add cluster ID labels
        for i, (size, weight, cluster_id) in enumerate(zip(sizes, weights, cluster_ids)):
            ax2.annotate(f'C{cluster_id}', (size, weight), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # 3. Source distribution heatmap
        ax3 = axes[1, 0]
        source_matrix = []
        source_names = sorted(source_distributions.keys())
        
        for cluster_id in cluster_ids:
            row = []
            for source in source_names:
                # Find the count for this cluster and source
                cluster_info = self.cluster_info[cluster_id]['source_distribution']
                count = cluster_info.get(source, 0)
                row.append(count)
            source_matrix.append(row)
        
        source_matrix = np.array(source_matrix)
        im = ax3.imshow(source_matrix, cmap='Blues', aspect='auto')
        ax3.set_xticks(range(len(source_names)))
        ax3.set_xticklabels(source_names, rotation=45)
        ax3.set_yticks(range(len(cluster_ids)))
        ax3.set_yticklabels([f'C{cid}' for cid in cluster_ids])
        ax3.set_xlabel('Source Subset')
        ax3.set_ylabel('Cluster ID')
        ax3.set_title('Source Distribution per Cluster')
        
        # Add text annotations
        for i in range(len(cluster_ids)):
            for j in range(len(source_names)):
                text = ax3.text(j, i, source_matrix[i, j], ha="center", va="center", 
                               color="white" if source_matrix[i, j] > source_matrix.max()/2 else "black")
        
        # Add colorbar
        plt.colorbar(im, ax=ax3, label='Count')
        
        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics
        total_weight = sum(weights)
        max_weight = max(weights)
        min_weight = min(weights)
        weight_std = np.std(weights)
        
        stats_text = f"""
Cluster Mixing Statistics:

Total Clusters: {len(cluster_ids)}
Total Data Points: {sum(sizes):,}

Weight Statistics:
• Total Weight: {total_weight:.6f}
• Max Weight: {max_weight:.6f} (Cluster {cluster_ids[weights.index(max_weight)]})
• Min Weight: {min_weight:.6f} (Cluster {cluster_ids[weights.index(min_weight)]})
• Weight Std: {weight_std:.6f}

Top 3 Clusters by Weight:
"""
        
        # Sort clusters by weight
        sorted_clusters = sorted(zip(cluster_ids, weights, sizes), key=lambda x: x[1], reverse=True)
        for i, (cluster_id, weight, size) in enumerate(sorted_clusters[:3]):
            stats_text += f"• Cluster {cluster_id}: {weight:.6f} ({size} points)\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster mixing analysis plot saved to: {save_path}")
        
        return fig
    
    def save_mixing_results(self, output_dir: str):
        """
        Save cluster mixing results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save weights to text file
        weights_file = output_dir / "cluster_mixing_weights.txt"
        with open(weights_file, 'w') as f:
            f.write("Cluster Mixing Weights\n")
            f.write("=" * 50 + "\n\n")
            
            total_weight = sum(self.cluster_weights.values())
            f.write(f"Total weight: {total_weight:.6f}\n\n")
            
            f.write("Cluster Weights:\n")
            f.write("-" * 30 + "\n")
            for cluster_name, weight in sorted(self.cluster_weights.items()):
                cluster_id = int(cluster_name.split('_')[1])
                size = self.cluster_info[cluster_id]['size']
                percentage = weight / total_weight * 100 if total_weight > 0 else 0
                f.write(f"{cluster_name}: {weight:.6f} ({percentage:5.2f}%, {size} points)\n")
            
            # Cluster information
            f.write("\nCluster Information:\n")
            f.write("-" * 40 + "\n")
            for cluster_id, info in sorted(self.cluster_info.items()):
                f.write(f"Cluster {cluster_id:2d}:\n")
                f.write(f"  Size: {info['size']}\n")
                f.write(f"  Shape: {info['embeddings_shape']}\n")
                f.write(f"  Source distribution: {info['source_distribution']}\n")
                f.write("\n")
        
        logger.info(f"Cluster mixing weights saved to: {weights_file}")
        
        # Save weights to pickle file for programmatic use
        weights_pkl = output_dir / "cluster_mixing_weights.pkl"
        with open(weights_pkl, 'wb') as f:
            pickle.dump({
                'cluster_weights': self.cluster_weights,
                'cluster_info': self.cluster_info,
                'optimization_params': {
                    'rff_dimension': getattr(self, 'rff_dimension', None),
                    'sigma_bandwidth': getattr(self, 'sigma_bandwidth', None),
                    'ridge_penalty': getattr(self, 'ridge_penalty', None),
                    'regularization_lambda': getattr(self, 'regularization_lambda', None),
                    'sigma': getattr(self, 'sigma', None)
                }
            }, f)
        
        logger.info(f"Cluster mixing data saved to: {weights_pkl}")


def calculate_cluster_mixing_ratios(cluster_dir: str, 
                                  target_file: str,
                                  output_dir: str = "./mixing_cluster_results",
                                  **mixer_kwargs) -> Dict[str, float]:
    """
    Calculate optimal mixing ratios for clustered embedding data.
    
    Args:
        cluster_dir: Directory containing cluster .pkl files
        target_file: Path to target embedding .pkl file
        output_dir: Directory to save results
        **mixer_kwargs: Additional arguments for ClusterMMDMixer
        
    Returns:
        Dictionary of cluster mixing weights
    """
    logger.info("Starting cluster mixing ratio calculation...")
    
    # Create mixer
    mixer = ClusterMMDMixer(**mixer_kwargs)
    
    # Optimize mixing weights
    weights = mixer.optimize_cluster_mixing_weights(cluster_dir, target_file)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    mixer.save_mixing_results(output_dir)
    
    # Create analysis visualization
    analysis_plot_path = output_path / "cluster_mixing_analysis.png"
    mixer.create_cluster_mixing_analysis(save_path=str(analysis_plot_path))
    
    logger.info(f"Cluster mixing analysis completed! Results saved to: {output_dir}")
    return weights


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate optimal mixing ratios for clustered embedding data')
    parser.add_argument('--cluster_dir', type=str, required=True,
                       help='Directory containing cluster .pkl files')
    parser.add_argument('--target_file', type=str, required=True,
                       help='Path to target embedding .pkl file')
    parser.add_argument('--output_dir', type=str, default='./mixing_cluster_results',
                       help='Directory to save results')
    parser.add_argument('--rff_dimension', type=int, default=100,
                       help='RFF dimension for kernel approximation')
    parser.add_argument('--sigma_bandwidth', type=float, default=None,
                       help='Bandwidth for Gaussian kernel (auto if None)')
    parser.add_argument('--ridge_penalty', type=float, default=1e-7,
                       help='Ridge penalty for numerical stability')
    parser.add_argument('--regularization_lambda', type=float, default=0.0,
                       help='L2 regularization parameter')
    parser.add_argument('--sample_number', type=int, default=-1,
                       help='Number of samples per cluster (-1 for all)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Prepare mixer arguments
    mixer_kwargs = {
        'rff_dimension': args.rff_dimension,
        'sigma_bandwidth': args.sigma_bandwidth,
        'ridge_penalty': args.ridge_penalty,
        'regularization_lambda': args.regularization_lambda,
        'sample_number': args.sample_number,
        'random_seed': args.random_seed
    }
    
    try:
        weights = calculate_cluster_mixing_ratios(
            cluster_dir=args.cluster_dir,
            target_file=args.target_file,
            output_dir=args.output_dir,
            **mixer_kwargs
        )
        
        print("\n" + "="*60)
        print("CLUSTER MIXING RESULTS SUMMARY")
        print("="*60)
        print(f"Cluster directory: {args.cluster_dir}")
        print(f"Target file: {args.target_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"\nOptimal mixing weights:")
        total_weight = sum(weights.values())
        for cluster_name, weight in sorted(weights.items()):
            percentage = weight / total_weight * 100 if total_weight > 0 else 0
            print(f"  {cluster_name}: {weight:.6f} ({percentage:5.2f}%)")
        print(f"\nTotal weight: {total_weight:.6f}")
        print("="*60)
        print("Cluster mixing optimization completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Cluster mixing optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()