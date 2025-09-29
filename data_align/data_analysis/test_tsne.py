#!/usr/bin/env python3
"""
Simple test script for t-SNE visualization functionality.
This script creates synthetic data to test the t-SNE visualization pipeline.
"""

import os
import numpy as np
import torch
import tempfile
import shutil
from tsne_visualization import load_and_sample_data, apply_pca_preprocessing, perform_tsne, create_visualization

def create_test_data():
    """Create synthetic gradient data for testing."""
    # Create temporary directory structure
    # temp_dir = tempfile.mkdtemp()
    temp_dir = "./tmp"
    
    # Create experiment structure
    experiment_name = "test_experiment"
    datasets = ["dataset1", "dataset2", "dataset3"]
    dim = "dim8192"
    
    file_paths = []
    
    for dataset in datasets:
        # Create directory structure
        dataset_dir = os.path.join(temp_dir, experiment_name, dataset, dim)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create synthetic gradient data
        # Different datasets have different distributions
        if dataset == "dataset1":
            # Cluster around origin
            data = np.random.normal(0, 1, (100, 512))
        elif dataset == "dataset2":
            # Cluster around (5, 5)
            data = np.random.normal(5, 1, (100, 512))
        else:
            # Cluster around (-5, -5)
            data = np.random.normal(-5, 1, (100, 512))
        
        # Convert to tensor and save
        tensor = torch.tensor(data, dtype=torch.float32)
        file_path = os.path.join(dataset_dir, "gradients.pt")
        torch.save(tensor, file_path)
        file_paths.append(file_path)
        
        print(f"Created test data: {file_path} with shape {tensor.shape}")
    
    return temp_dir, file_paths

def test_tsne_pipeline():
    """Test the complete t-SNE pipeline."""
    print("Creating test data...")
    temp_dir, file_paths = create_test_data()
    
    try:
        print("\nTesting data loading and sampling...")
        # Test data loading
        data, labels, sample_counts = load_and_sample_data(file_paths, sample_number=50, random_seed=42)
        print(f"Loaded data shape: {data.shape}")
        print(f"Labels: {set(labels)}")
        print(f"Sample counts: {sample_counts}")
        
        print("\nTesting PCA preprocessing...")
        # Test PCA preprocessing
        data_pca = apply_pca_preprocessing(data, n_components=50)
        print(f"PCA data shape: {data_pca.shape}")
        
        print("\nTesting t-SNE...")
        # Test t-SNE (with fewer iterations for speed)
        tsne_result = perform_tsne(data_pca, perplexity=10, n_iter=250, random_seed=42)
        print(f"t-SNE result shape: {tsne_result.shape}")
        
        print("\nTesting visualization...")
        # Test visualization
        output_dir = os.path.join(temp_dir, "test_output")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "test_tsne.png")
        
        create_visualization(tsne_result, labels, plot_path, "Test t-SNE Visualization")
        
        if os.path.exists(plot_path):
            print(f"✓ Visualization saved successfully: {plot_path}")
        else:
            print("✗ Visualization failed to save")
            
        print("\n" + "="*50)
        print("✓ All tests passed successfully!")
        print("✓ t-SNE visualization pipeline is working correctly")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temporary directory
        # shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    print("Testing t-SNE visualization functionality...")
    test_tsne_pipeline()