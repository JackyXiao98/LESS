#!/usr/bin/env python3
"""
Test script for cluster-based t-SNE visualization
"""

import sys
import os
from pathlib import Path

# Add the data_analysis directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_analysis'))

from plot_cluster_tsne import ClusterTSNEVisualizer

def main():
    # Configuration
    cluster_dir = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/data_process/cluster_embeddings"
    target_file = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/data_process/embeddings/yelp_train_sampled_1000_embeddings.pkl"

    # equal weights
    # cluster_weights_file = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/mixing_cluster_equal_results/cluster_mixing_weights.pkl"
    # output_dir = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/cluster_tsne_equal_results"

    # mmd weights
    cluster_weights_file = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/mixing_cluster_results/cluster_mixing_weights.pkl"
    output_dir = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/cluster_tsne_results"
    
    print("Starting cluster-based t-SNE visualization test...")
    print(f"Cluster weights file: {cluster_weights_file}")
    print(f"Cluster directory: {cluster_dir}")
    print(f"Target file: {target_file}")
    print(f"Output directory: {output_dir}")
    
    # Create visualizer
    visualizer = ClusterTSNEVisualizer(
        cluster_weights_file=cluster_weights_file,
        target_file=target_file,
        target_samples=3000,
        random_seed=42
    )
    
    # Run analysis
    try:
        figures = visualizer.run_full_analysis(
            cluster_dir=cluster_dir,
            output_dir=output_dir,
            show_plots=False  # Don't show plots in test
        )
        
        print("\n" + "="*60)
        print("SUCCESS: Cluster-based t-SNE analysis completed!")
        print("="*60)
        
        # Print summary
        summary = visualizer.get_analysis_summary()
        print(f"Total samples: {summary['total_samples']}")
        print("Datasets:")
        for dataset, info in summary['datasets'].items():
            print(f"  {dataset}: {info['sample_count']} samples ({info['percentage']:.1f}%)")
        print("\nCluster weights:")
        for cluster_id, weight in summary['cluster_weights'].items():
            print(f"  {cluster_id}: {weight:.6f}")
        
        print(f"\nOutput files saved to: {output_dir}")
        print("- cluster_tsne_visualization.png")
        print("- cluster_mixing_weights_summary.png")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())