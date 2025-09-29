#!/usr/bin/env python3
"""
Test script to verify dual-path mode functionality
Creates mock data to test the t-SNE visualization with both training and validation data
"""

import os
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path

def create_mock_gradient_data(base_path, experiment_name, dim, datasets, data_type="train"):
    """Create mock gradient data for testing"""
    print(f"Creating mock {data_type} data in {base_path}")
    
    for dataset in datasets:
        # Create directory structure
        dataset_dir = Path(base_path) / experiment_name / dataset / dim
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock gradient file
        gradient_file = dataset_dir / "gradients.pt"
        
        # Generate different patterns for train vs val data
        if data_type == "train":
            # Training data: clustered around origin with some spread
            mock_data = np.random.normal(0, 1, (100, 512)).astype(np.float32)
        else:
            # Validation data: clustered around (2, 2) with some spread
            mock_data = np.random.normal(2, 1, (100, 512)).astype(np.float32)
        
        # Save as PyTorch tensor
        torch.save(torch.from_numpy(mock_data), gradient_file)
        print(f"  Created {gradient_file} with shape {mock_data.shape}")

def test_dual_mode():
    """Test the dual-path mode functionality"""
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix="tsne_test_")
    train_base = os.path.join(temp_dir, "train")
    val_base = os.path.join(temp_dir, "val")
    output_dir = os.path.join(temp_dir, "output")
    
    try:
        print("="*50)
        print("Testing t-SNE Dual-Path Mode")
        print("="*50)
        print(f"Temporary directory: {temp_dir}")
        print(f"Training base: {train_base}")
        print(f"Validation base: {val_base}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Test configuration
        experiment_name = "test-experiment"
        dim = "rank_1/dim8192"
        
        # Create mock datasets (use actual dataset names from the lists)
        train_datasets = ["ai2-adapt-dev_coconot_converted-ckpt368-adam", "ai2-adapt-dev_flan_v2_converted-ckpt368-adam"]
        val_datasets = ["drop-ckpt368-sgd", "gsm8k-ckpt368-sgd"]
        
        # Create mock data
        create_mock_gradient_data(train_base, experiment_name, dim, train_datasets, "train")
        create_mock_gradient_data(val_base, experiment_name, dim, val_datasets, "val")
        
        print("\nRunning t-SNE visualization in dual-path mode...")
        
        # Import and run the visualization
        import sys
        sys.path.append(os.path.dirname(__file__))
        
        # Run the Python script
        cmd = [
            "python", "tsne_visualization.py",
            "--train_base_path", train_base,
            "--val_base_path", val_base,
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
            output_files = list(Path(output_dir).glob("*"))
            print(f"\nOutput files created: {len(output_files)}")
            for f in output_files:
                print(f"  - {f.name}")
        else:
            print("\nNo output directory created")
        
        print("\n" + "="*50)
        print("Test completed!")
        print("="*50)
        
    finally:
        # Clean up
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_dual_mode()