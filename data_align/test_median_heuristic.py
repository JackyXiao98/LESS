#!/usr/bin/env python3
"""
Simple test script for median heuristic sigma calculation.
This script tests the core logic without requiring external dependencies.
"""

import os
import sys
import math
import random

def create_test_data():
    """Create simple test data for sigma calculation."""
    # Simulate gradient data as lists of lists (instead of tensors)
    random.seed(42)
    
    # Create 3 training datasets with different characteristics
    train_data = []
    
    # Dataset 1: 100 samples, 50 dimensions
    dataset1 = []
    for i in range(100):
        sample = [random.gauss(0, 1) for _ in range(50)]
        dataset1.append(sample)
    train_data.append(dataset1)
    
    # Dataset 2: 80 samples, 50 dimensions  
    dataset2 = []
    for i in range(80):
        sample = [random.gauss(1, 0.5) for _ in range(50)]
        dataset2.append(sample)
    train_data.append(dataset2)
    
    # Dataset 3: 120 samples, 50 dimensions
    dataset3 = []
    for i in range(120):
        sample = [random.gauss(-0.5, 1.5) for _ in range(50)]
        dataset3.append(sample)
    train_data.append(dataset3)
    
    return train_data

def sample_data_for_sigma(train_data, K):
    """Sample K data points from each training dataset."""
    sampled_data = []
    
    for i, dataset in enumerate(train_data):
        n_samples = len(dataset)
        if n_samples <= K:
            # Use all samples if dataset has K or fewer
            sampled = dataset[:]
            print(f"Training dataset {i+1}: using all {n_samples} samples")
        else:
            # Randomly sample K points
            indices = random.sample(range(n_samples), K)
            sampled = [dataset[idx] for idx in indices]
            print(f"Training dataset {i+1}: sampled {K} out of {n_samples} samples")
        
        sampled_data.extend(sampled)
    
    print(f"Total sampled data: {len(sampled_data)} samples")
    return sampled_data

def compute_squared_distance(sample1, sample2):
    """Compute squared Euclidean distance between two samples."""
    if len(sample1) != len(sample2):
        raise ValueError("Samples must have the same dimension")
    
    squared_dist = 0.0
    for i in range(len(sample1)):
        diff = sample1[i] - sample2[i]
        squared_dist += diff * diff
    
    return squared_dist

def compute_median_heuristic_sigma(train_data, K=50):
    """
    Compute sigma using median heuristic: sigma^2 = median_distance_squared / 2.0
    """
    print("Computing sigma using median heuristic...")
    
    # Sample data points for sigma computation
    sampled_data = sample_data_for_sigma(train_data, K)
    n_samples = len(sampled_data)
    
    print(f"Computing pairwise distances for {n_samples} samples...")
    
    # Compute all pairwise squared distances
    squared_distances = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):  # Only upper triangular part
            squared_dist = compute_squared_distance(sampled_data[i], sampled_data[j])
            squared_distances.append(squared_dist)
    
    # Compute median of squared distances
    squared_distances.sort()
    n_distances = len(squared_distances)
    
    if n_distances % 2 == 0:
        # Even number of distances
        median_squared_distance = (squared_distances[n_distances // 2 - 1] + 
                                 squared_distances[n_distances // 2]) / 2.0
    else:
        # Odd number of distances
        median_squared_distance = squared_distances[n_distances // 2]
    
    # Apply median heuristic: sigma^2 = median_distance_squared / 2.0
    sigma_squared = median_squared_distance / 2.0
    sigma = math.sqrt(sigma_squared)
    
    print(f"Number of pairwise distances: {n_distances}")
    print(f"Median squared distance: {median_squared_distance:.6f}")
    print(f"Computed sigma: {sigma:.6f}")
    
    return sigma

def test_median_heuristic():
    """Test the median heuristic sigma calculation."""
    print("=== Testing Median Heuristic Sigma Calculation ===")
    
    # Create test data
    train_data = create_test_data()
    print(f"Created {len(train_data)} training datasets")
    for i, dataset in enumerate(train_data):
        print(f"  Dataset {i+1}: {len(dataset)} samples, {len(dataset[0])} dimensions")
    
    # Test with different sample sizes
    sample_sizes = [10, 30, 50]
    
    for K in sample_sizes:
        print(f"\n--- Testing with K={K} ---")
        sigma = compute_median_heuristic_sigma(train_data, K)
        print(f"Final sigma for K={K}: {sigma:.6f}")
    
    print("\n=== Test completed successfully! ===")

if __name__ == "__main__":
    test_median_heuristic()