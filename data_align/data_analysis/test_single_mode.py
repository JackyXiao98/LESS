#!/usr/bin/env python3
"""
Test script for single-path t-SNE visualization functionality
"""

import os
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path

def create_mock_gradient_data(base_path, experiment_name, dim, datasets, data_type="val"):
    """Create mock gradient data for testing"""
    for dataset in datasets:
        # Create directory structure: base_path/experiment_name/dataset/rank_1/dim8192
        dataset_dir = os.path.join(base_path, experiment_name, dataset, dim)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create mock gradient file
        gradient_file = os.path.join(dataset_dir, "gradients.pt")
        
        # Create random gradient data (100 samples, 512 features)
        mock_data = torch.randn(100, 512)
        torch.save(mock_data, gradient_file)
        
        print(f"Created mock {data_type} data: {gradient_file}")

def test_single_mode():
    """Test single-path mode functionality"""
    print("Testing single-path t-SNE visualization...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="tsne_single_test_")
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Setup test parameters
        experiment_name = "test-experiment"
        dim = "rank_1/dim8192"  # Use the correct default format
        output_dir = os.path.join(temp_dir, "output")
        
        # Use validation datasets for single mode test
        val_datasets = ["drop-ckpt368-sgd", "gsm8k-ckpt368-sgd"]
        
        # Create mock validation data
        create_mock_gradient_data(temp_dir, experiment_name, dim, val_datasets, "val")
        
        # Run t-SNE visualization in single-path mode
        cmd = [
            "python", "tsne_visualization.py",
            "--base_path", temp_dir,
            "--experiment_name", experiment_name,
            "--dim", dim,
            "--output_dir", output_dir,
            "--sample_number", "50",  # Use smaller sample for testing
            "--pca_components", "10",  # Reduce PCA components for speed
            "--perplexity", "5",       # Reduce perplexity for small dataset
            "--max_iter", "250",       # Minimum required for t-SNE
            "--random_seed", "42"
        ]
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
        # Check if output files were created
        if os.path.exists(output_dir):
            output_files = os.listdir(output_dir)
            print(f"Output files created: {len(output_files)}")
            for file in output_files:
                print(f"  - {file}")
        else:
            print("No output directory created")
        
    finally:
        print("=" * 50)
        print("Test completed!")
        print("=" * 50)
        
        # Clean up
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_single_mode()