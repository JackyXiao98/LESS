#!/usr/bin/env python3
"""
Embedding Clustering Script

This script loads yelp_huggingface_subset embedding files, combines them,
performs K-means clustering, and saves each cluster to separate pkl files.

Author: AI Assistant
Date: 2025-09-25
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EmbeddingClusterer:
    """
    A class for clustering embedding data from multiple yelp_huggingface_subset files.
    """
    
    def __init__(self, 
                 embedding_dir: str,
                 output_dir: str = "./cluster_embeddings",
                 n_clusters: int = 10,
                 random_seed: int = 42):
        """
        Initialize the embedding clusterer.
        
        Args:
            embedding_dir: Directory containing embedding .pkl files
            output_dir: Directory to save clustered results
            n_clusters: Number of clusters for K-means
            random_seed: Random seed for reproducibility
        """
        self.embedding_dir = Path(embedding_dir)
        self.output_dir = Path(output_dir)
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        
        # Data storage
        self.embeddings_data = {}
        self.combined_embeddings = None
        self.combined_labels = None
        self.cluster_labels = None
        self.kmeans_model = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(random_seed)
    
    def load_embedding_files(self) -> Dict[str, np.ndarray]:
        """
        Load all yelp_huggingface_subset embedding files.
        
        Returns:
            Dictionary mapping file names to embedding arrays
        """
        logger.info("Loading embedding files...")
        
        # Find all yelp_huggingface_subset files
        pattern = "yelp_huggingface_subset_*_1000_embeddings.pkl"
        embedding_files = list(self.embedding_dir.glob(pattern))
        
        if not embedding_files:
            raise FileNotFoundError(f"No embedding files found matching pattern: {pattern}")
        
        logger.info(f"Found {len(embedding_files)} embedding files")
        
        for file_path in sorted(embedding_files):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract embeddings from dictionary format
                if isinstance(data, dict) and 'embeddings' in data:
                    embeddings = data['embeddings']
                elif isinstance(data, (list, tuple)):
                    embeddings = np.array(data)
                elif isinstance(data, np.ndarray):
                    embeddings = data
                else:
                    logger.error(f"Unknown data format in {file_path}: {type(data)}")
                    continue
                
                # Ensure it's a numpy array
                if not isinstance(embeddings, np.ndarray):
                    embeddings = np.array(embeddings)
                
                file_key = file_path.stem.replace('_embeddings', '')
                self.embeddings_data[file_key] = embeddings
                
                logger.info(f"Loaded {file_key}: {embeddings.shape}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        return self.embeddings_data
    
    def combine_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine all embedding data into a single array.
        
        Returns:
            Tuple of (combined_embeddings, source_labels)
        """
        logger.info("Combining embeddings...")
        
        if not self.embeddings_data:
            raise ValueError("No embedding data loaded. Call load_embedding_files() first.")
        
        all_embeddings = []
        all_labels = []
        
        for file_key, embeddings in self.embeddings_data.items():
            all_embeddings.append(embeddings)
            # Create labels indicating source file
            subset_num = file_key.split('_')[3]  # Extract subset number
            labels = [f"subset_{subset_num}"] * len(embeddings)
            all_labels.extend(labels)
        
        self.combined_embeddings = np.vstack(all_embeddings)
        self.combined_labels = np.array(all_labels)
        
        logger.info(f"Combined embeddings shape: {self.combined_embeddings.shape}")
        logger.info(f"Total data points: {len(self.combined_labels)}")
        
        return self.combined_embeddings, self.combined_labels
    
    def perform_clustering(self) -> np.ndarray:
        """
        Perform K-means clustering on combined embeddings.
        
        Returns:
            Cluster labels for each data point
        """
        logger.info(f"Performing K-means clustering with {self.n_clusters} clusters...")
        
        if self.combined_embeddings is None:
            raise ValueError("No combined embeddings available. Call combine_embeddings() first.")
        
        # Initialize K-means
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_seed,
            n_init=10,
            max_iter=300
        )
        
        # Fit and predict
        self.cluster_labels = self.kmeans_model.fit_predict(self.combined_embeddings)
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(self.combined_embeddings, self.cluster_labels)
        calinski_score = calinski_harabasz_score(self.combined_embeddings, self.cluster_labels)
        
        logger.info(f"Clustering completed!")
        logger.info(f"Silhouette Score: {silhouette_avg:.4f}")
        logger.info(f"Calinski-Harabasz Score: {calinski_score:.4f}")
        
        # Log cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            logger.info(f"Cluster {cluster_id}: {count} points ({count/len(self.cluster_labels)*100:.1f}%)")
        
        return self.cluster_labels
    
    def save_clusters(self) -> Dict[int, str]:
        """
        Save each cluster to separate pkl files.
        
        Returns:
            Dictionary mapping cluster IDs to saved file paths
        """
        logger.info("Saving clusters to separate files...")
        
        if self.cluster_labels is None:
            raise ValueError("No cluster labels available. Call perform_clustering() first.")
        
        saved_files = {}
        
        for cluster_id in range(self.n_clusters):
            # Get indices for this cluster
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                logger.warning(f"Cluster {cluster_id} is empty, skipping...")
                continue
            
            # Extract embeddings for this cluster
            cluster_embeddings = self.combined_embeddings[cluster_indices]
            cluster_source_labels = self.combined_labels[cluster_indices]
            
            # Create cluster data dictionary
            cluster_data = {
                'embeddings': cluster_embeddings,
                'source_labels': cluster_source_labels,
                'cluster_id': cluster_id,
                'size': len(cluster_embeddings),
                'indices': cluster_indices
            }
            
            # Save to pkl file
            output_file = self.output_dir / f"cluster_{cluster_id:02d}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(cluster_data, f)
            
            saved_files[cluster_id] = str(output_file)
            logger.info(f"Saved cluster {cluster_id}: {len(cluster_embeddings)} points -> {output_file}")
        
        return saved_files
    
    def create_cluster_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive cluster analysis visualization.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating cluster analysis visualization...")
        
        if self.cluster_labels is None:
            raise ValueError("No cluster labels available. Call perform_clustering() first.")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Embedding Clustering Analysis ({self.n_clusters} Clusters)', fontsize=16, fontweight='bold')
        
        # 1. Cluster size distribution
        ax1 = axes[0, 0]
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        bars = ax1.bar(unique, counts, alpha=0.7, color=plt.cm.tab10(unique))
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Points')
        ax1.set_title('Cluster Size Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}', ha='center', va='bottom', fontsize=9)
        
        # 2. Source distribution per cluster
        ax2 = axes[0, 1]
        source_cluster_matrix = pd.crosstab(self.combined_labels, self.cluster_labels)
        sns.heatmap(source_cluster_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Source Subset')
        ax2.set_title('Source Distribution per Cluster')
        
        # 3. PCA visualization
        ax3 = axes[1, 0]
        pca = PCA(n_components=2, random_state=self.random_seed)
        embeddings_2d = pca.fit_transform(self.combined_embeddings)
        
        scatter = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=self.cluster_labels, cmap='tab10', alpha=0.6, s=20)
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax3.set_title('PCA Visualization of Clusters')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Cluster ID')
        
        # 4. Cluster statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics
        silhouette_avg = silhouette_score(self.combined_embeddings, self.cluster_labels)
        calinski_score = calinski_harabasz_score(self.combined_embeddings, self.cluster_labels)
        
        stats_text = f"""
Clustering Statistics:

Total Data Points: {len(self.cluster_labels):,}
Number of Clusters: {self.n_clusters}
Random Seed: {self.random_seed}

Quality Metrics:
• Silhouette Score: {silhouette_avg:.4f}
• Calinski-Harabasz Score: {calinski_score:.2f}

Cluster Sizes:
"""
        
        for cluster_id, count in zip(unique, counts):
            percentage = count / len(self.cluster_labels) * 100
            stats_text += f"• Cluster {cluster_id}: {count} points ({percentage:.1f}%)\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster analysis plot saved to: {save_path}")
        
        return fig
    
    def run_full_clustering(self) -> Dict:
        """
        Run the complete clustering pipeline.
        
        Returns:
            Dictionary with clustering results and file paths
        """
        logger.info("Starting full clustering pipeline...")
        
        # Step 1: Load embedding files
        self.load_embedding_files()
        
        # Step 2: Combine embeddings
        self.combine_embeddings()
        
        # Step 3: Perform clustering
        self.perform_clustering()
        
        # Step 4: Save clusters
        saved_files = self.save_clusters()
        
        # Step 5: Create analysis visualization
        analysis_plot_path = self.output_dir / "cluster_analysis.png"
        self.create_cluster_analysis(save_path=str(analysis_plot_path))
        
        # Step 6: Save clustering summary
        summary_path = self.output_dir / "clustering_summary.txt"
        self.save_clustering_summary(summary_path)
        
        results = {
            'total_points': len(self.cluster_labels),
            'n_clusters': self.n_clusters,
            'saved_files': saved_files,
            'analysis_plot': str(analysis_plot_path),
            'summary_file': str(summary_path),
            'silhouette_score': silhouette_score(self.combined_embeddings, self.cluster_labels),
            'calinski_score': calinski_harabasz_score(self.combined_embeddings, self.cluster_labels)
        }
        
        logger.info("Clustering pipeline completed successfully!")
        return results
    
    def save_clustering_summary(self, save_path: str):
        """Save a text summary of clustering results."""
        with open(save_path, 'w') as f:
            f.write("Embedding Clustering Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total data points: {len(self.cluster_labels)}\n")
            f.write(f"Number of clusters: {self.n_clusters}\n")
            f.write(f"Random seed: {self.random_seed}\n\n")
            
            # Quality metrics
            silhouette_avg = silhouette_score(self.combined_embeddings, self.cluster_labels)
            calinski_score = calinski_harabasz_score(self.combined_embeddings, self.cluster_labels)
            f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
            f.write(f"Calinski-Harabasz Score: {calinski_score:.2f}\n\n")
            
            # Cluster distribution
            f.write("Cluster Distribution:\n")
            f.write("-" * 30 + "\n")
            unique, counts = np.unique(self.cluster_labels, return_counts=True)
            for cluster_id, count in zip(unique, counts):
                percentage = count / len(self.cluster_labels) * 100
                f.write(f"Cluster {cluster_id:2d}: {count:4d} points ({percentage:5.1f}%)\n")
            
            # Source distribution
            f.write("\nSource Distribution per Cluster:\n")
            f.write("-" * 40 + "\n")
            source_cluster_matrix = pd.crosstab(self.combined_labels, self.cluster_labels)
            f.write(str(source_cluster_matrix))
        
        logger.info(f"Clustering summary saved to: {save_path}")


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cluster embedding data from yelp_huggingface_subset files')
    parser.add_argument('--embedding_dir', type=str, default='./embeddings',
                       help='Directory containing embedding .pkl files')
    parser.add_argument('--output_dir', type=str, default='./cluster_embeddings',
                       help='Directory to save clustered results')
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters for K-means')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create clusterer
    clusterer = EmbeddingClusterer(
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed
    )
    
    # Run clustering
    try:
        results = clusterer.run_full_clustering()
        
        print("\n" + "="*60)
        print("CLUSTERING RESULTS SUMMARY")
        print("="*60)
        print(f"Total data points processed: {results['total_points']}")
        print(f"Number of clusters: {results['n_clusters']}")
        print(f"Silhouette Score: {results['silhouette_score']:.4f}")
        print(f"Calinski-Harabasz Score: {results['calinski_score']:.2f}")
        print(f"\nOutput directory: {args.output_dir}")
        print(f"Cluster files saved: {len(results['saved_files'])}")
        print(f"Analysis plot: {results['analysis_plot']}")
        print(f"Summary file: {results['summary_file']}")
        print("="*60)
        print("Clustering completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        raise


if __name__ == "__main__":
    main()