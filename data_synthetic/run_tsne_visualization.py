#!/usr/bin/env python3
"""
Test script for running t-SNE visualization analysis.

This script demonstrates how to use the EmbeddingTSNEVisualizer class
to create beautiful t-SNE visualizations of embedding data.
"""

import sys
import os
from pathlib import Path

# Add the data_analysis directory to the Python path
sys.path.append(str(Path(__file__).parent / "data_analysis"))

from plot_tsne import EmbeddingTSNEVisualizer

def main():
    """Run the t-SNE visualization analysis."""
    
    # Configuration
    embedding_dir = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/data_process/embeddings"
    # weights_file = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/mixing_results/embedding_mixing_weights_20250925_135248.txt"
    # output_dir = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/tsne_results_equal_weights"
    
    # unequal weight
    # weights_file = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/mixing_results/embedding_mixing_weights_20250925_133404.txt"
    # output_dir = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/tsne_results_weights"

    # single weight
    weights_file = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/mixing_results/embedding_mixing_weights_20250925_133045.txt"
    output_dir = "/Users/bytedance/Desktop/Github/LESS/data_synthetic/tsne_results_single_weights"

    print("="*60)
    print("t-SNE EMBEDDING VISUALIZATION")
    print("="*60)
    print(f"Embedding directory: {embedding_dir}")
    print(f"Weights file: {weights_file}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Check if files exist
    if not os.path.exists(embedding_dir):
        print(f"Error: Embedding directory not found: {embedding_dir}")
        return
    
    if not os.path.exists(weights_file):
        print(f"Error: Weights file not found: {weights_file}")
        return
    
    # Create visualizer
    visualizer = EmbeddingTSNEVisualizer(
        embedding_dir=embedding_dir,
        weights_file=weights_file,
        target_samples=1000,  # Total samples for huggingface data
        tsne_params={
            'perplexity': 30,
            'learning_rate': 200,
            'max_iter': 2000,
            'random_state': 42
        },
        random_seed=42
    )
    
    try:
        # Run full analysis
        figures = visualizer.run_full_analysis(
            output_dir=output_dir,
            show_plots=False  # Set to True if you want to display plots
        )
        
        # Get and print summary
        summary = visualizer.get_analysis_summary()
        
        print("\n" + "="*60)
        print("ANALYSIS RESULTS SUMMARY")
        print("="*60)
        print(f"Total samples processed: {summary['total_samples']}")
        print(f"Number of datasets: {len(summary['datasets'])}")
        print("\nDataset breakdown:")
        for dataset, info in summary['datasets'].items():
            print(f"  {dataset}: {info['sample_count']} samples ({info['percentage']:.1f}%)")
        
        print("\nMixing weights:")
        for file_path, weight in summary['weights'].items():
            file_name = Path(file_path).name
            print(f"  {file_name}: {weight:.6f}")
        
        print(f"\nt-SNE parameters:")
        for param, value in summary['tsne_params'].items():
            print(f"  {param}: {value}")
        
        print("\nOutput files:")
        output_path = Path(output_dir)
        if output_path.exists():
            for file in output_path.glob("*.png"):
                print(f"  {file}")
        
        print("="*60)
        print("Analysis completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()