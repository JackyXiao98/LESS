#!/usr/bin/env python3
"""
Cluster-based t-SNE Visualization for Embedding Data

This module provides t-SNE visualization based on cluster mixing weights
from MMD optimization results. It samples data from clusters according to
their optimal mixing weights and creates visualizations.

Author: AI Assistant
Date: 2024
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ClusterTSNEVisualizer:
    """
    A t-SNE visualization class for cluster-based embedding data analysis.
    
    This class handles loading cluster data, applying mixing weights from
    MMD optimization, sampling data points, and creating t-SNE visualizations.
    """
    
    def __init__(self, 
                 cluster_weights_file: str,
                 target_file: str,
                 target_samples: int = 1000,
                 tsne_params: Optional[Dict] = None,
                 random_seed: int = 42):
        """
        Initialize the cluster t-SNE visualizer.
        
        Args:
            cluster_weights_file: Path to the cluster mixing weights .pkl file
            target_file: Path to the target yelp_train embedding file
            target_samples: Total number of samples to use for visualization
            tsne_params: Parameters for t-SNE algorithm
            random_seed: Random seed for reproducibility
        """
        self.cluster_weights_file = Path(cluster_weights_file)
        self.target_file = Path(target_file)
        self.target_samples = target_samples
        self.random_seed = random_seed
        
        # Default t-SNE parameters
        self.tsne_params = {
            'n_components': 2,
            'perplexity': 30,
            'learning_rate': 200,
            'max_iter': 1000,
            'random_state': random_seed,
            'init': 'pca'
        }
        if tsne_params:
            self.tsne_params.update(tsne_params)
        
        # Data storage
        self.cluster_weights = {}
        self.cluster_info = {}
        self.cluster_data = {}
        self.target_data = None
        self.sampled_data = None
        self.tsne_results = None
        
        # Color schemes - use distinct colors for clusters
        self.cluster_colors = {
            'cluster_00': '#2C3E50',  # Red'#2C3E50'
            'cluster_01': '#4ECDC4',  # Teal
            'cluster_02': '#45B7D1',  # Blue
            'cluster_03': '#96CEB4',  # Green
            'cluster_04': '#FFEAA7',  # Yellow
            'cluster_05': '#DDA0DD',  # Plum
            'cluster_06': '#98D8C8',  # Mint
            'cluster_07': '#F7DC6F',  # Light yellow
            'cluster_08': '#BB8FCE',  # Light purple
            'cluster_09': '#85C1E9',  # Light blue
            'yelp_train': '#FF6B6B'   # Dark blue-gray for target
        }
        
        # Markers
        self.markers = {
            'cluster': '^',           # Triangle for clusters
            'yelp_train': 'o'         # Circle for target
        }
        
        np.random.seed(random_seed)
    
    def load_cluster_weights(self) -> Dict:
        """
        Load cluster mixing weights from the .pkl file.
        
        Returns:
            Dictionary containing cluster weights and info
        """
        logger.info(f"Loading cluster weights from: {self.cluster_weights_file}")
        
        with open(self.cluster_weights_file, 'rb') as f:
            weights_data = pickle.load(f)
        
        self.cluster_weights = weights_data['cluster_weights']
        self.cluster_info = weights_data['cluster_info']
        
        logger.info(f"Loaded weights for {len(self.cluster_weights)} clusters")
        
        # Log cluster weights
        for cluster_id, weight in self.cluster_weights.items():
            logger.info(f"  {cluster_id}: {weight:.6f}")
        
        return weights_data
    
    def load_cluster_data(self, cluster_dir: str) -> Dict[str, np.ndarray]:
        """
        Load cluster embedding data from .pkl files.
        
        Args:
            cluster_dir: Directory containing cluster .pkl files
            
        Returns:
            Dictionary mapping cluster names to embedding arrays
        """
        logger.info("Loading cluster embedding data...")
        
        cluster_dir = Path(cluster_dir)
        
        # Check if cluster directory exists and has files
        if not cluster_dir.exists():
            logger.warning(f"Cluster directory not found: {cluster_dir}")
            logger.info("Will create synthetic cluster data based on target data for visualization")
            return self._create_synthetic_cluster_data()
        
        cluster_files = list(cluster_dir.glob("cluster_*.pkl"))
        if not cluster_files:
            logger.warning(f"No cluster files found in: {cluster_dir}")
            logger.info("Will create synthetic cluster data based on target data for visualization")
            return self._create_synthetic_cluster_data()
        
        for cluster_id, info in self.cluster_info.items():
            cluster_file = cluster_dir / info['file']
            
            if not cluster_file.exists():
                logger.warning(f"Cluster file not found: {cluster_file}")
                continue
            
            with open(cluster_file, 'rb') as f:
                cluster_data = pickle.load(f)
            
            cluster_name = f"cluster_{cluster_id:02d}"
            self.cluster_data[cluster_name] = cluster_data['embeddings']
            
            logger.info(f"Loaded {cluster_name}: {self.cluster_data[cluster_name].shape}")
        
        return self.cluster_data
    
    def _create_synthetic_cluster_data(self) -> Dict[str, np.ndarray]:
        """
        Create synthetic cluster data based on target data for visualization purposes.
        This is used when actual cluster files are not available.
        
        Returns:
            Dictionary mapping cluster names to synthetic embedding arrays
        """
        logger.info("Creating synthetic cluster data for visualization...")
        
        if self.target_data is None:
            logger.error("Target data must be loaded first to create synthetic clusters")
            return {}
        
        # Use K-means to create clusters from target data
        from sklearn.cluster import KMeans
        
        n_clusters = len(self.cluster_info)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        cluster_labels = kmeans.fit_predict(self.target_data)
        
        # Create cluster data based on K-means results
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = self.target_data[cluster_mask]
            
            cluster_name = f"cluster_{cluster_id:02d}"
            self.cluster_data[cluster_name] = cluster_embeddings
            
            logger.info(f"Created synthetic {cluster_name}: {cluster_embeddings.shape}")
        
        return self.cluster_data
    
    def load_target_data(self) -> np.ndarray:
        """
        Load target yelp_train embedding data.
        
        Returns:
            Target embedding array
        """
        logger.info(f"Loading target data from: {self.target_file}")
        
        with open(self.target_file, 'rb') as f:
            target_data = pickle.load(f)
        
        self.target_data = target_data['embeddings']
        logger.info(f"Loaded target data: {self.target_data.shape}")
        
        return self.target_data
    
    def sample_data_by_cluster_weights(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Sample data points according to cluster mixing weights.
        
        Returns:
            Tuple of (embeddings, labels, dataset_names)
        """
        logger.info("Sampling data according to cluster mixing weights...")
        
        # Calculate number of samples for each cluster based on weights
        sample_counts = {}
        total_weight = sum(max(0, weight) for weight in self.cluster_weights.values())
        
        if total_weight == 0:
            logger.warning("All cluster weights are zero or negative, using equal sampling")
            # Use equal sampling if all weights are zero
            active_clusters = [name for name in self.cluster_data.keys()]
            samples_per_cluster = self.target_samples // len(active_clusters)
            for cluster_name in active_clusters:
                sample_counts[cluster_name] = samples_per_cluster
        else:
             for cluster_id, weight in self.cluster_weights.items():
                 # cluster_id is already in format "cluster_XX", so use it directly
                 cluster_name = cluster_id
                 if cluster_name in self.cluster_data and weight > 0:
                     n_samples = int(np.round(weight / total_weight * self.target_samples))
                     sample_counts[cluster_name] = n_samples
        
        # Add target data samples
        sample_counts['yelp_train'] = self.target_samples
        
        logger.info("Sample counts:")
        for name, count in sample_counts.items():
            logger.info(f"  {name}: {count}")
        
        # Sample data
        all_embeddings = []
        all_labels = []
        all_dataset_names = []
        
        # Sample from clusters
        for cluster_name, n_samples in sample_counts.items():
            if n_samples == 0:
                continue
            
            if cluster_name == 'yelp_train':
                embeddings = self.target_data
            else:
                embeddings = self.cluster_data[cluster_name]
            
            # Sample indices
            if n_samples >= len(embeddings):
                # Use all data if requested samples >= available data
                indices = np.arange(len(embeddings))
            else:
                indices = np.random.choice(len(embeddings), n_samples, replace=False)
            
            sampled_embeddings = embeddings[indices]
            
            all_embeddings.append(sampled_embeddings)
            all_labels.extend([cluster_name] * len(sampled_embeddings))
            all_dataset_names.extend([cluster_name] * len(sampled_embeddings))
        
        # Combine all data
        combined_embeddings = np.vstack(all_embeddings)
        combined_labels = np.array(all_labels)
        
        logger.info(f"Total sampled data: {combined_embeddings.shape}")
        
        self.sampled_data = {
            'embeddings': combined_embeddings,
            'labels': combined_labels,
            'dataset_names': all_dataset_names
        }
        
        return combined_embeddings, combined_labels, all_dataset_names
    
    def compute_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute t-SNE transformation.
        
        Args:
            embeddings: Input embedding data
            
        Returns:
            2D t-SNE coordinates
        """
        logger.info("Computing t-SNE transformation...")
        logger.info(f"t-SNE parameters: {self.tsne_params}")
        
        tsne = TSNE(**self.tsne_params)
        tsne_results = tsne.fit_transform(embeddings)
        
        self.tsne_results = tsne_results
        logger.info(f"t-SNE completed. Output shape: {tsne_results.shape}")
        
        return tsne_results
    
    def create_visualization(self, 
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (14, 10),
                           title: Optional[str] = None,
                           show_legend: bool = True,
                           alpha: float = 0.7,
                           point_size: int = 50) -> plt.Figure:
        """
        Create the cluster-based t-SNE visualization plot.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
            title: Plot title
            show_legend: Whether to show legend
            alpha: Point transparency
            point_size: Size of points
            
        Returns:
            Matplotlib figure object
        """
        if self.tsne_results is None or self.sampled_data is None:
            raise ValueError("Must run sampling and t-SNE computation first")
        
        logger.info("Creating cluster-based t-SNE visualization...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get data
        tsne_coords = self.tsne_results
        labels = self.sampled_data['labels']
        
        # Plot each dataset
        unique_labels = np.unique(labels)
        
        # Sort labels to ensure consistent ordering (target first, then clusters)
        sorted_labels = []
        if 'yelp_train' in unique_labels:
            sorted_labels.append('yelp_train')
        
        cluster_labels = [label for label in unique_labels if label.startswith('cluster_')]
        cluster_labels.sort()
        sorted_labels.extend(cluster_labels)
        
        for label in sorted_labels:
            mask = labels == label
            coords = tsne_coords[mask]
            
            if len(coords) == 0:
                continue
            
            # Determine color and marker
            color = self.cluster_colors.get(label, '#888888')
            
            if label == 'yelp_train':
                marker = self.markers['yelp_train']
                label_name = 'Yelp Train (Target)'
                edgecolor = 'white'
                linewidth = 1.0
                alpha_val = alpha
                zorder = 10  # Plot target on top
            else:
                 marker = self.markers['cluster']
                 # Get cluster weight for label (label is already in cluster_XX format)
                 weight = self.cluster_weights.get(label, 0.0)
                 label_name = f'{label.replace("_", " ").title()} (w={weight:.3f})'
                 edgecolor = 'black'
                 linewidth = 1.0
                 alpha_val = 0.8
                 zorder = 5
            
            # Plot points
            scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                               c=color, marker=marker, s=point_size, 
                               alpha=alpha_val, label=label_name,
                               edgecolors=edgecolor, linewidth=linewidth,
                               zorder=zorder)
        
        # Customize plot
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        
        if title is None:
            title = 'Cluster-based Embedding t-SNE Visualization\n(Weighted Sampling Based on MMD Cluster Optimization)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if show_legend:
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                             frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        return fig
    
    def create_weight_summary_plot(self, 
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a bar plot showing the cluster mixing weights.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating cluster weight summary plot...")
        
        # Prepare data
        cluster_names = []
        weights = []
        colors = []
        
        for cluster_id, weight in self.cluster_weights.items():
            cluster_name = f"Cluster {cluster_id.split('_')[1]}"
            cluster_names.append(cluster_name)
            weights.append(max(0, weight))  # Use 0 for negative weights in visualization
            colors.append(self.cluster_colors.get(cluster_id, '#888888'))
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(cluster_names, weights, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(weights) * 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('Clusters', fontsize=12)
        ax.set_ylabel('Mixing Weight', fontsize=12)
        ax.set_title('MMD Cluster Optimization Results: Optimal Mixing Weights', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(weights) * 1.15 if weights else 1)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Weight plot saved to: {save_path}")
        
        return fig
    
    def run_full_analysis(self, 
                         cluster_dir: str,
                         output_dir: Optional[str] = None,
                         show_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        Run the complete cluster-based t-SNE analysis pipeline.
        
        Args:
            cluster_dir: Directory containing cluster .pkl files
            output_dir: Directory to save plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary of created figures
        """
        logger.info("Starting full cluster-based t-SNE analysis pipeline...")
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data (target data first, as it may be needed for synthetic clusters)
        self.load_cluster_weights()
        self.load_target_data()
        self.load_cluster_data(cluster_dir)
        
        # Sample data
        embeddings, labels, dataset_names = self.sample_data_by_cluster_weights()
        
        # Compute t-SNE
        tsne_coords = self.compute_tsne(embeddings)
        
        # Create visualizations
        figures = {}
        
        # Main t-SNE plot
        tsne_save_path = None
        if output_dir:
            tsne_save_path = output_path / 'cluster_tsne_visualization.png'
        
        figures['tsne'] = self.create_visualization(save_path=tsne_save_path)
        
        # Weight summary plot
        weight_save_path = None
        if output_dir:
            weight_save_path = output_path / 'cluster_mixing_weights_summary.png'
        
        figures['weights'] = self.create_weight_summary_plot(save_path=weight_save_path)
        
        # Show plots if requested
        if show_plots:
            plt.show()
        
        logger.info("Full cluster-based analysis completed successfully!")
        
        return figures
    
    def get_analysis_summary(self) -> Dict:
        """
        Get a summary of the analysis results.
        
        Returns:
            Dictionary containing analysis summary
        """
        if not self.sampled_data:
            return {}
        
        summary = {
            'total_samples': len(self.sampled_data['labels']),
            'datasets': {},
            'cluster_weights': self.cluster_weights.copy(),
            'tsne_params': self.tsne_params.copy()
        }
        
        # Count samples per dataset
        unique_labels, counts = np.unique(self.sampled_data['labels'], return_counts=True)
        for label, count in zip(unique_labels, counts):
            summary['datasets'][label] = {
                'sample_count': int(count),
                'percentage': float(count / len(self.sampled_data['labels']) * 100)
            }
        
        return summary


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create cluster-based t-SNE visualization")
    parser.add_argument("--cluster_weights_file", type=str, required=True,
                       help="Path to cluster mixing weights .pkl file")
    parser.add_argument("--cluster_dir", type=str, required=True,
                       help="Directory containing cluster .pkl files")
    parser.add_argument("--target_file", type=str, required=True,
                       help="Path to target yelp_train embedding file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots")
    parser.add_argument("--target_samples", type=int, default=1000,
                       help="Number of samples for visualization")
    parser.add_argument("--perplexity", type=int, default=30,
                       help="t-SNE perplexity parameter")
    parser.add_argument("--learning_rate", type=int, default=200,
                       help="t-SNE learning rate")
    parser.add_argument("--max_iter", type=int, default=1000,
                       help="Number of t-SNE iterations")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--no_show", action="store_true",
                       help="Don't display plots")
    
    args = parser.parse_args()
    
    # Set up t-SNE parameters
    tsne_params = {
        'perplexity': args.perplexity,
        'learning_rate': args.learning_rate,
        'max_iter': args.max_iter
    }
    
    # Create visualizer
    visualizer = ClusterTSNEVisualizer(
        cluster_weights_file=args.cluster_weights_file,
        target_file=args.target_file,
        target_samples=args.target_samples,
        tsne_params=tsne_params,
        random_seed=args.random_seed
    )
    
    # Run analysis
    figures = visualizer.run_full_analysis(
        cluster_dir=args.cluster_dir,
        output_dir=args.output_dir,
        show_plots=not args.no_show
    )
    
    # Print summary
    summary = visualizer.get_analysis_summary()
    print("\n" + "="*60)
    print("CLUSTER-BASED t-SNE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Datasets:")
    for dataset, info in summary['datasets'].items():
        print(f"  {dataset}: {info['sample_count']} samples ({info['percentage']:.1f}%)")
    print("\nCluster weights:")
    for cluster_id, weight in summary['cluster_weights'].items():
        print(f"  {cluster_id}: {weight:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()