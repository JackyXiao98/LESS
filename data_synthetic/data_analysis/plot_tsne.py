#!/usr/bin/env python3
"""
t-SNE Visualization for Embedding Data

This module provides a comprehensive t-SNE visualization class for analyzing
embedding distributions and data mixing weights from MMD optimization results.

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


class EmbeddingTSNEVisualizer:
    """
    A comprehensive t-SNE visualization class for embedding data analysis.
    
    This class handles loading embedding data, applying mixing weights,
    sampling data points, and creating beautiful t-SNE visualizations
    with proper color coding and legends.
    """
    
    def __init__(self, 
                 embedding_dir: str,
                 weights_file: str,
                 target_samples: int = 1000,
                 tsne_params: Optional[Dict] = None,
                 random_seed: int = 42):
        """
        Initialize the t-SNE visualizer.
        
        Args:
            embedding_dir: Directory containing embedding .pkl files
            weights_file: Path to the mixing weights file
            target_samples: Total number of samples to use for visualization
            tsne_params: Parameters for t-SNE algorithm
            random_seed: Random seed for reproducibility
        """
        self.embedding_dir = Path(embedding_dir)
        self.weights_file = Path(weights_file)
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
        self.embeddings_data = {}
        self.weights = {}
        self.sampled_data = None
        self.tsne_results = None
        
        # Color schemes - darker colors for better contrast
        self.colors = {
            'yelp_train': '#FF6B6B',  # Red for target data
            'subset_1': '#2C7A7B',    # Dark teal
            'subset_2': '#2B6CB0',    # Dark blue
            'subset_3': '#38A169',    # Dark green
            'subset_4': '#D69E2E',    # Dark yellow/orange
            'subset_5': '#9F7AEA'     # Dark purple
        }
        
        # Markers
        self.markers = {
            'yelp_train': 'o',        # Circle for target
            'yelp_huggingface': '^'   # Triangle for huggingface
        }
        
        np.random.seed(random_seed)
    
    def load_weights(self) -> Dict[str, float]:
        """
        Load mixing weights from the weights file.
        
        Returns:
            Dictionary mapping file paths to weights
        """
        logger.info(f"Loading weights from: {self.weights_file}")
        
        weights = {}
        with open(self.weights_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split(',')
                if len(parts) == 2:
                    file_path, weight = parts
                    weights[file_path] = float(weight)
        
        self.weights = weights
        logger.info(f"Loaded weights for {len(weights)} files")
        
        return weights
    
    def load_embedding_data(self) -> Dict[str, np.ndarray]:
        """
        Load all embedding data from .pkl files.
        
        Returns:
            Dictionary mapping dataset names to embedding arrays
        """
        logger.info("Loading embedding data...")
        
        # Load target data (yelp_train)
        target_files = list(self.embedding_dir.glob("*yelp_train*_embeddings.pkl"))
        if not target_files:
            raise FileNotFoundError("No yelp_train embedding file found")
        
        target_file = target_files[0]
        with open(target_file, 'rb') as f:
            target_data = pickle.load(f)
        
        self.embeddings_data['yelp_train'] = target_data['embeddings']
        logger.info(f"Loaded yelp_train: {self.embeddings_data['yelp_train'].shape}")
        
        # Load huggingface subset data
        for file_path in self.weights.keys():
            file_name = Path(file_path).name
            
            # Extract subset number
            if 'subset_' in file_name:
                subset_num = file_name.split('subset_')[1].split('_')[0]
                dataset_name = f'subset_{subset_num}'
                
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.embeddings_data[dataset_name] = data['embeddings']
                logger.info(f"Loaded {dataset_name}: {self.embeddings_data[dataset_name].shape}")
        
        return self.embeddings_data
    
    def sample_data_by_weights(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Sample data points according to mixing weights.
        
        Returns:
            Tuple of (embeddings, labels, dataset_names)
        """
        logger.info("Sampling data according to mixing weights...")
        
        # Calculate number of samples for each dataset
        total_weight = sum(self.weights.values())
        sample_counts = {}
        
        for file_path, weight in self.weights.items():
            file_name = Path(file_path).name
            if 'subset_' in file_name:
                subset_num = file_name.split('subset_')[1].split('_')[0]
                dataset_name = f'subset_{subset_num}'
                
                # Calculate samples based on weight
                n_samples = int(np.round(weight / total_weight * self.target_samples))
                sample_counts[dataset_name] = n_samples
        
        # Add yelp_train samples (same amount as total huggingface samples)
        sample_counts['yelp_train'] = self.target_samples
        
        logger.info("Sample counts:")
        for name, count in sample_counts.items():
            logger.info(f"  {name}: {count}")
        
        # Sample data
        all_embeddings = []
        all_labels = []
        all_dataset_names = []
        
        for dataset_name, n_samples in sample_counts.items():
            if n_samples == 0:
                continue
                
            embeddings = self.embeddings_data[dataset_name]
            
            # Sample indices
            if n_samples >= len(embeddings):
                # Use all data if requested samples >= available data
                indices = np.arange(len(embeddings))
            else:
                indices = np.random.choice(len(embeddings), n_samples, replace=False)
            
            sampled_embeddings = embeddings[indices]
            
            all_embeddings.append(sampled_embeddings)
            all_labels.extend([dataset_name] * len(sampled_embeddings))
            all_dataset_names.extend([dataset_name] * len(sampled_embeddings))
        
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
                           figsize: Tuple[int, int] = (12, 8),
                           title: Optional[str] = None,
                           show_legend: bool = True,
                           alpha: float = 0.7,
                           point_size: int = 50) -> plt.Figure:
        """
        Create the t-SNE visualization plot.
        
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
        
        logger.info("Creating t-SNE visualization...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get data
        tsne_coords = self.tsne_results
        labels = self.sampled_data['labels']
        
        # Plot each dataset
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            coords = tsne_coords[mask]
            
            # Determine color and marker
            color = self.colors.get(label, '#888888')
            
            if label == 'yelp_train':
                marker = self.markers['yelp_train']
                label_name = 'Yelp Train (Target)'
                edgecolor = 'white'
                linewidth = 1.0
                alpha_val = alpha
            else:
                marker = self.markers['yelp_huggingface']
                label_name = f'Huggingface {label.replace("subset_", "Subset ")}'
                edgecolor = 'black'  # Changed to black for better contrast
                linewidth = 1.5      # Increased linewidth for stronger border
                alpha_val = 0.9      # Higher alpha for more solid appearance
            
            # Plot points
            ax.scatter(coords[:, 0], coords[:, 1], 
                      c=color, marker=marker, s=point_size, 
                      alpha=alpha_val, label=label_name,
                      edgecolors=edgecolor, linewidth=linewidth)
        
        # Customize plot
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        
        if title is None:
            title = 'Embedding t-SNE Visualization\n(Weighted Sampling Based on MMD Optimization)'
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
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create a bar plot showing the mixing weights.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating weight summary plot...")
        
        # Prepare data
        dataset_names = []
        weights = []
        colors = []
        
        for file_path, weight in self.weights.items():
            file_name = Path(file_path).name
            if 'subset_' in file_name:
                subset_num = file_name.split('subset_')[1].split('_')[0]
                dataset_name = f'Subset {subset_num}'
                dataset_names.append(dataset_name)
                weights.append(weight)
                colors.append(self.colors[f'subset_{subset_num}'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(dataset_names, weights, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('Huggingface Subsets', fontsize=12)
        ax.set_ylabel('Mixing Weight', fontsize=12)
        ax.set_title('MMD Optimization Results: Optimal Mixing Weights', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(weights) * 1.2)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Weight plot saved to: {save_path}")
        
        return fig
    
    def run_full_analysis(self, 
                         output_dir: Optional[str] = None,
                         show_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        Run the complete analysis pipeline.
        
        Args:
            output_dir: Directory to save plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary of created figures
        """
        logger.info("Starting full t-SNE analysis pipeline...")
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.load_weights()
        self.load_embedding_data()
        
        # Sample data
        embeddings, labels, dataset_names = self.sample_data_by_weights()
        
        # Compute t-SNE
        tsne_coords = self.compute_tsne(embeddings)
        
        # Create visualizations
        figures = {}
        
        # Main t-SNE plot
        tsne_save_path = None
        if output_dir:
            tsne_save_path = output_path / 'tsne_visualization.png'
        
        figures['tsne'] = self.create_visualization(save_path=tsne_save_path)
        
        # Weight summary plot
        weight_save_path = None
        if output_dir:
            weight_save_path = output_path / 'mixing_weights.png'
        
        figures['weights'] = self.create_weight_summary_plot(save_path=weight_save_path)
        
        # Show plots if requested
        if show_plots:
            plt.show()
        
        logger.info("Full analysis completed successfully!")
        
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
            'weights': self.weights.copy(),
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
    
    parser = argparse.ArgumentParser(description="Create t-SNE visualization for embedding data")
    parser.add_argument("--embedding_dir", type=str, required=True,
                       help="Directory containing embedding .pkl files")
    parser.add_argument("--weights_file", type=str, required=True,
                       help="Path to mixing weights file")
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
    visualizer = EmbeddingTSNEVisualizer(
        embedding_dir=args.embedding_dir,
        weights_file=args.weights_file,
        target_samples=args.target_samples,
        tsne_params=tsne_params,
        random_seed=args.random_seed
    )
    
    # Run analysis
    figures = visualizer.run_full_analysis(
        output_dir=args.output_dir,
        show_plots=not args.no_show
    )
    
    # Print summary
    summary = visualizer.get_analysis_summary()
    print("\n" + "="*60)
    print("t-SNE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Datasets:")
    for dataset, info in summary['datasets'].items():
        print(f"  {dataset}: {info['sample_count']} samples ({info['percentage']:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()