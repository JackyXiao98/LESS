#!/usr/bin/env python3
"""
Embedding-based MMD Data Mixing Optimization Script

This script extends the MMDDataMixer class to work with sentence embeddings
generated from text data. It calculates optimal data mixing ratios for multiple
embedding datasets to best match a target embedding distribution.

Author: AI Assistant
Date: 2024
"""

import sys
import os
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Add parent directory to path to import MMDDataMixer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_align'))
from mmd_data_mixing import MMDDataMixer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingMMDMixer(MMDDataMixer):
    """
    A specialized MMD Data Mixer for sentence embeddings.
    
    This class extends MMDDataMixer to work with .pkl files containing
    sentence embeddings instead of gradient .pt files.
    """
    
    def __init__(self, 
                 rff_dimension: int = 100,
                 sigma_bandwidth: Optional[float] = None,
                 ridge_penalty: float = 1e-7,
                 regularization_lambda: float = 0.0,
                 sample_number: int = -1,
                 random_seed: Optional[int] = 42,
                 auto_sigma: bool = True,
                 sigma_sample_size: int = 1000):
        """
        Initialize the Embedding MMD Mixer.
        
        Args:
            rff_dimension: Target dimension for Random Fourier Features space
            sigma_bandwidth: Bandwidth for the Gaussian kernel
            ridge_penalty: Ridge penalty for numerical stability
            regularization_lambda: L2 regularization parameter
            sample_number: Number of samples to randomly select from each embedding file
            random_seed: Random seed for reproducibility
            auto_sigma: Whether to automatically compute sigma using median heuristic
            sigma_sample_size: Number of samples to use for sigma computation
        """
        super().__init__(
            rff_dimension=rff_dimension,
            sigma_bandwidth=sigma_bandwidth,
            ridge_penalty=ridge_penalty,
            regularization_lambda=regularization_lambda,
            sample_number=sample_number,
            random_seed=random_seed,
            auto_sigma=auto_sigma,
            sigma_sample_size=sigma_sample_size
        )
    
    def load_embedding_data(self, file_paths: List[str]) -> List[torch.Tensor]:
        """
        Load sentence embeddings from .pkl files.
        
        Args:
            file_paths: List of paths to .pkl files containing embedding data
            
        Returns:
            List of loaded embedding tensors
        """
        tensors = []
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            logger.info(f"Loading embedding data from: {path}")
            
            # Load pickle file
            with open(path, 'rb') as f:
                embedding_data = pickle.load(f)
            
            # Extract embeddings tensor
            if 'embeddings' not in embedding_data:
                raise ValueError(f"No 'embeddings' key found in {path}")
            
            embeddings = embedding_data['embeddings']
            
            # Convert to torch tensor if it's numpy array
            if isinstance(embeddings, np.ndarray):
                tensor = torch.from_numpy(embeddings).float()
            elif isinstance(embeddings, torch.Tensor):
                tensor = embeddings.float()
            else:
                raise ValueError(f"Expected numpy array or torch tensor in {path}, got {type(embeddings)}")
            
            if tensor.dim() != 2:
                raise ValueError(f"Expected 2D tensor in {path}, got shape {tensor.shape}")
            
            # Apply sampling if sample_number is specified and positive
            if self.sample_number > 0 and tensor.shape[0] > self.sample_number:
                # Randomly sample sample_number data points
                indices = torch.randperm(tensor.shape[0])[:self.sample_number]
                tensor = tensor[indices]
                logger.info(f"Sampled {self.sample_number} data points from {tensor.shape[0]} total points in {path}")
            elif self.sample_number > 0:
                logger.info(f"Using all {tensor.shape[0]} data points from {path} (less than sample_number={self.sample_number})")
            else:
                logger.info(f"Using all {tensor.shape[0]} data points from {path} (sample_number={self.sample_number})")
            
            tensors.append(tensor)
            logger.info(f"Loaded embedding tensor with shape: {tensor.shape}")
            
            # Log additional metadata if available
            if 'file_type' in embedding_data:
                logger.info(f"  File type: {embedding_data['file_type']}")
            if 'model_name' in embedding_data:
                logger.info(f"  Model: {embedding_data['model_name']}")
            if 'embedding_dim' in embedding_data:
                logger.info(f"  Embedding dimension: {embedding_data['embedding_dim']}")
        
        return tensors
    
    def optimize_embedding_mixing_weights(self, 
                                        train_embedding_paths: List[str], 
                                        target_embedding_path: str) -> Dict[str, float]:
        """
        Optimize mixing weights for embedding datasets to match a target distribution.
        
        Args:
            train_embedding_paths: List of paths to training embedding .pkl files
            target_embedding_path: Path to target embedding .pkl file to match
            
        Returns:
            Dictionary mapping training data paths to optimal weights
        """
        logger.info("Starting embedding-based MMD data mixing optimization...")
        logger.info(f"Training embedding datasets: {len(train_embedding_paths)}")
        logger.info(f"Target embedding dataset: {target_embedding_path}")
        
        # Load data
        train_tensors = self.load_embedding_data(train_embedding_paths)
        target_tensors = self.load_embedding_data([target_embedding_path])
        
        # Compute sigma automatically if needed
        if self.auto_sigma and self.sigma is None:
            self.sigma = self._compute_median_heuristic_sigma(train_tensors + target_tensors)
            logger.info(f"Using automatically computed sigma: {self.sigma:.6f}")
        elif self.sigma is not None:
            logger.info(f"Using provided sigma: {self.sigma:.6f}")
        else:
            # Fallback to default value
            self.sigma = 3.0
            logger.info(f"Using default sigma: {self.sigma:.6f}")
        
        # Compute mean features
        train_means, target_mean = self.compute_mean_features(train_tensors, target_tensors)
        
        # Solve QP
        optimal_weights = self.solve_qp(train_means, target_mean)
        
        # Calculate final MMD value
        mmd_value = self.calculate_mmd_value(train_means, target_mean, optimal_weights)
        logger.info(f"Final MMD value: {mmd_value:.6f}")
        
        # Create result dictionary
        result = {}
        for i, path in enumerate(train_embedding_paths):
            result[path] = float(optimal_weights[i])
        
        return result


def calculate_embedding_mixing_ratios(embedding_dir: str, 
                                    target_file: str,
                                    output_file: str = None,
                                    **mixer_kwargs) -> Dict[str, float]:
    """
    Calculate optimal mixing ratios for embedding files.
    
    Args:
        embedding_dir: Directory containing embedding .pkl files
        target_file: Name of the target embedding file to match
        output_file: Optional output file to save results
        **mixer_kwargs: Additional arguments for EmbeddingMMDMixer
        
    Returns:
        Dictionary of optimal mixing weights
    """
    embedding_dir = Path(embedding_dir)
    
    # Find all embedding files
    all_embedding_files = list(embedding_dir.glob("*_embeddings.pkl"))
    
    # Separate target file from training files
    target_path = None
    train_paths = []
    
    for file_path in all_embedding_files:
        if target_file in file_path.name:
            target_path = str(file_path)
        else:
            train_paths.append(str(file_path))
    
    if target_path is None:
        raise FileNotFoundError(f"Target file containing '{target_file}' not found in {embedding_dir}")
    
    if len(train_paths) == 0:
        raise ValueError(f"No training embedding files found in {embedding_dir}")
    
    logger.info(f"Target embedding file: {target_path}")
    logger.info(f"Training embedding files: {len(train_paths)}")
    for path in train_paths:
        logger.info(f"  - {path}")
    
    # Create mixer and optimize
    mixer = EmbeddingMMDMixer(**mixer_kwargs)
    results = mixer.optimize_embedding_mixing_weights(train_paths, target_path)
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Optimal Embedding Mixing Weights\n")
            f.write(f"# Target: {target_path}\n")
            f.write("# Format: file_path,weight\n\n")
            
            for path, weight in results.items():
                f.write(f"{path},{weight:.6f}\n")
        
        logger.info(f"Results saved to: {output_path}")
    
    return results


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate optimal embedding mixing ratios using MMD")
    parser.add_argument("--embedding_dir", type=str, required=True,
                       help="Directory containing embedding .pkl files")
    parser.add_argument("--target_file", type=str, required=True,
                       help="Name pattern of target embedding file to match")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save results")
    parser.add_argument("--rff_dimension", type=int, default=100,
                       help="RFF dimension")
    parser.add_argument("--sigma_bandwidth", type=float, default=None,
                       help="Sigma bandwidth (auto-computed if not provided)")
    parser.add_argument("--regularization_lambda", type=float, default=0.0,
                       help="L2 regularization parameter")
    parser.add_argument("--sample_number", type=int, default=-1,
                       help="Number of samples per file (-1 for all)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Prepare mixer arguments
    mixer_kwargs = {
        'rff_dimension': args.rff_dimension,
        'sigma_bandwidth': args.sigma_bandwidth,
        'regularization_lambda': args.regularization_lambda,
        'sample_number': args.sample_number,
        'random_seed': args.random_seed,
        'auto_sigma': True,
        'sigma_sample_size': 1000
    }
    
    # Calculate mixing ratios
    results = calculate_embedding_mixing_ratios(
        embedding_dir=args.embedding_dir,
        target_file=args.target_file,
        output_file=args.output_file,
        **mixer_kwargs
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMAL EMBEDDING MIXING WEIGHTS")
    print("="*60)
    
    total_weight = 0
    for path, weight in results.items():
        filename = Path(path).name
        print(f"{filename:<50} {weight:>8.4f}")
        total_weight += weight
    
    print("-"*60)
    print(f"{'Total Weight':<50} {total_weight:>8.4f}")
    print("="*60)


if __name__ == "__main__":
    main()