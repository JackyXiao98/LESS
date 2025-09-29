#!/usr/bin/env python3
"""
Script to verify that the dual-path t-SNE visualization correctly distinguishes 
between training and validation data in the output files.
"""

import os
import numpy as np
import torch
import tempfile
import shutil
import pandas as pd
from pathlib import Path

def create_mock_gradient_data(base_path, experiment_name, dim, datasets, data_type="train"):
    """Create mock gradient data for testing"""
    for dataset in datasets:
        # Create directory structure: base_path/experiment_name/dataset/rank_1/dim8192
        dataset_dir = os.path.join(base_path, experiment_name, dataset, dim)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create mock gradient file
        gradient_file = os.path.join(dataset_dir, "gradients.pt")
        
        # Create random gradient data (50 samples, 512 features)
        # Use different random seeds for train vs val to create distinguishable data
        if data_type == "train":
            torch.manual_seed(42)
        else:
            torch.manual_seed(123)
        
        mock_data = torch.randn(50, 512)
        torch.save(mock_data, gradient_file)
        
        print(f"Created mock {data_type} data: {gradient_file}")

def verify_dual_output():
    """Verify that dual-path output correctly distinguishes train and val data"""
    print("Verifying dual-path t-SNE output files...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="tsne_verify_")
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Setup test parameters
        experiment_name = "verify-experiment"
        dim = "rank_1/dim8192"
        output_dir = os.path.join(temp_dir, "output")
        
        # Create separate train and val base paths
        train_base_path = os.path.join(temp_dir, "train")
        val_base_path = os.path.join(temp_dir, "val")
        
        # Use a subset of datasets for faster testing
        train_datasets = ["ai2-adapt-dev_coconot_converted-ckpt368-adam"]
        val_datasets = ["drop-ckpt368-sgd"]
        
        # Create mock data
        create_mock_gradient_data(train_base_path, experiment_name, dim, train_datasets, "train")
        create_mock_gradient_data(val_base_path, experiment_name, dim, val_datasets, "val")
        
        # Run t-SNE visualization in dual-path mode
        cmd = [
            "python", "tsne_visualization.py",
            "--train_base_path", train_base_path,
            "--val_base_path", val_base_path,
            "--experiment_name", experiment_name,
            "--dim", dim,
            "--output_dir", output_dir,
            "--sample_number", "30",  # Use smaller sample for testing
            "--pca_components", "10",  # Reduce PCA components for speed
            "--perplexity", "5",       # Reduce perplexity for small dataset
            "--max_iter", "250",       # Minimum required for t-SNE
            "--random_seed", "42"
        ]
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        print(f"Command return code: {result.returncode}")
        
        if result.returncode != 0:
            print("STDERR:")
            print(result.stderr)
            return False
        
        # Check if output files were created
        if not os.path.exists(output_dir):
            print("ERROR: No output directory created")
            return False
        
        output_files = os.listdir(output_dir)
        print(f"Output files created: {len(output_files)}")
        
        # Find the coordinates CSV file
        csv_files = [f for f in output_files if f.endswith('_coordinates.csv')]
        if not csv_files:
            print("ERROR: No coordinates CSV file found")
            return False
        
        csv_file = os.path.join(output_dir, csv_files[0])
        print(f"Analyzing coordinates file: {csv_files[0]}")
        
        # Load and analyze the coordinates CSV
        df = pd.read_csv(csv_file)
        print(f"CSV columns: {list(df.columns)}")
        print(f"Total data points: {len(df)}")
        
        # Check if data_type column exists and contains both train and val
        if 'data_type' not in df.columns:
            print("ERROR: 'data_type' column not found in coordinates CSV")
            return False
        
        data_types = df['data_type'].unique()
        print(f"Data types found: {data_types}")
        
        if 'train' not in data_types or 'val' not in data_types:
            print("ERROR: Both 'train' and 'val' data types should be present")
            return False
        
        train_count = len(df[df['data_type'] == 'train'])
        val_count = len(df[df['data_type'] == 'val'])
        print(f"Training data points: {train_count}")
        print(f"Validation data points: {val_count}")
        
        # Check if we have reasonable distribution
        if train_count == 0 or val_count == 0:
            print("ERROR: Missing training or validation data points")
            return False
        
        print("✓ SUCCESS: Output file correctly contains both training and validation data types")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
        
    finally:
        print("=" * 50)
        print("Verification completed!")
        print("=" * 50)
        
        # Clean up
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = verify_dual_output()
    exit(0 if success else 1)