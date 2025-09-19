#!/usr/bin/env python3
"""
MMD-based Data Mixing Optimization Script

This script calculates optimal data mixing ratios for multiple training datasets
to best match the distribution of a combined validation dataset using Maximum Mean
Discrepancy (MMD) minimization with Random Fourier Features (RFF).

Author: AI Assistant
Date: 2024
"""

import torch
import numpy as np
import cvxpy as cp
import os
import argparse
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MMDDataMixer:
    """
    A class for computing optimal data mixing weights using MMD optimization.
    """
    
    def __init__(self, 
                 rff_dimension: int = 100,
                 sigma_bandwidth: Optional[float] = None,
                 ridge_penalty: float = 1e-7,
                 random_seed: Optional[int] = 42,
                 auto_sigma: bool = True,
                 sigma_sample_size: int = 1000):
        """
        Initialize the MMD Data Mixer.
        
        Args:
            rff_dimension: Target dimension for Random Fourier Features space
            sigma_bandwidth: Bandwidth for the Gaussian kernel (if None and auto_sigma=True, will be computed automatically)
            ridge_penalty: Ridge penalty for numerical stability
            random_seed: Random seed for reproducibility
            auto_sigma: Whether to automatically compute sigma using median heuristic
            sigma_sample_size: Number of samples to use for sigma computation (K parameter)
        """
        self.D = rff_dimension
        self.sigma = sigma_bandwidth
        self.ridge = ridge_penalty
        self.random_seed = random_seed
        self.auto_sigma = auto_sigma
        self.sigma_sample_size = sigma_sample_size
        
        # RFF parameters (will be initialized when first used)
        self.Omega = None
        self.b = None
        self.d = None  # Original dimension
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
    
    def load_gradient_data(self, file_paths: List[str]) -> List[torch.Tensor]:
        """
        Load gradient embeddings from .pt files.
        
        Args:
            file_paths: List of paths to .pt files containing gradient embeddings
            
        Returns:
            List of loaded tensors
        """
        tensors = []
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            logger.info(f"Loading gradient data from: {path}")
            tensor = torch.load(path, map_location='cpu')
            
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Expected tensor in {path}, got {type(tensor)}")
            
            if tensor.dim() != 2:
                raise ValueError(f"Expected 2D tensor in {path}, got shape {tensor.shape}")
            
            tensors.append(tensor)
            logger.info(f"Loaded tensor with shape: {tensor.shape}")
        
        return tensors
    
    def _sample_data_for_sigma(self, train_tensors: List[torch.Tensor], K: int) -> torch.Tensor:
        """
        Sample K data points from each training dataset for sigma computation.
        
        Args:
            train_tensors: List of training dataset tensors
            K: Number of samples to take from each dataset
            
        Returns:
            Combined tensor of sampled data points
        """
        sampled_data = []
        
        for i, tensor in enumerate(train_tensors):
            n_samples = tensor.shape[0]
            if n_samples <= K:
                # If dataset has K or fewer samples, use all of them
                sampled = tensor
                logger.info(f"Training dataset {i+1}: using all {n_samples} samples")
            else:
                # Randomly sample K points
                indices = torch.randperm(n_samples)[:K]
                sampled = tensor[indices]
                logger.info(f"Training dataset {i+1}: sampled {K} out of {n_samples} samples")
            
            sampled_data.append(sampled)
        
        # Combine all sampled data
        # breakpoint()
        combined_samples = torch.cat(sampled_data, dim=0)
        logger.info(f"Total sampled data shape: {combined_samples.shape}")
        
        return combined_samples
    
    def _compute_median_heuristic_sigma(self, train_tensors: List[torch.Tensor]) -> float:
        """
        Compute sigma using median heuristic: sigma^2 = median_distance_squared / 2.0
        
        Args:
            train_tensors: List of training dataset tensors
            
        Returns:
            Computed sigma value
        """
        logger.info("Computing sigma using median heuristic...")
        
        # Sample data points for sigma computation
        sampled_data = self._sample_data_for_sigma(train_tensors, self.sigma_sample_size)
        n_samples = sampled_data.shape[0]
        
        logger.info(f"Computing pairwise distances for {n_samples} samples...")
        
        # Compute pairwise squared distances
        # Using broadcasting to compute all pairwise distances efficiently
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i^T * x_j
        
        # Compute squared norms for each sample
        squared_norms = torch.sum(sampled_data ** 2, dim=1, keepdim=True)  # (n_samples, 1)
        
        # Compute dot products
        dot_products = torch.mm(sampled_data, sampled_data.t())  # (n_samples, n_samples)
        
        # Compute squared distances matrix
        squared_distances = squared_norms + squared_norms.t() - 2 * dot_products
        
        # Extract upper triangular part (excluding diagonal) to avoid duplicates and zeros
        mask = torch.triu(torch.ones(n_samples, n_samples), diagonal=1).bool()
        squared_distances_upper = squared_distances[mask]
        
        # Compute median of squared distances
        median_squared_distance = torch.median(squared_distances_upper).item()
        
        # Apply median heuristic: sigma^2 = median_distance_squared / 2.0
        sigma_squared = median_squared_distance / 2.0
        sigma = np.sqrt(sigma_squared)
        
        logger.info(f"Median squared distance: {median_squared_distance:.6f}")
        logger.info(f"Computed sigma: {sigma:.6f}")
        
        return sigma
    
    def _initialize_rff_parameters(self, dimension: int):
        """
        Initialize Random Fourier Features parameters.
        
        Args:
            dimension: Original dimension of the gradient embeddings
        """
        if self.d is not None and self.d != dimension:
            raise ValueError(f"Dimension mismatch: expected {self.d}, got {dimension}")
        
        if self.Omega is None:
            self.d = dimension
            logger.info(f"Initializing RFF parameters: d={self.d}, D={self.D}, sigma={self.sigma}")
            
            # Sample random frequency matrix Omega: (d, D)
            # Each column omega_r ~ N(0, σ^(-2) * I_d)
            self.Omega = torch.randn(self.d, self.D) / self.sigma
            
            # Sample random phase vector b: (D,)
            # Each element b_r ~ Unif[0, 2π]
            self.b = torch.rand(self.D) * 2 * np.pi
            
            logger.info("RFF parameters initialized successfully")
    
    def rff_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply Random Fourier Features transformation.
        
        Args:
            X: Input tensor of shape (num_examples, gradient_dimension)
            
        Returns:
            Transformed tensor of shape (num_examples, rff_dimension)
        """
        # Initialize RFF parameters if not done yet
        self._initialize_rff_parameters(X.shape[1])
        
        # Compute projection: Z = X @ Omega
        Z = torch.mm(X, self.Omega)
        
        # Add phase: Z = Z + b
        Z = Z + self.b.unsqueeze(0)
        
        # Apply cosine: Z = cos(Z)
        Z = torch.cos(Z)
        
        # Scale and return: Z * sqrt(2/D)
        return Z * np.sqrt(2.0 / self.D)
    
    def compute_mean_features(self, 
                            train_tensors: List[torch.Tensor], 
                            val_tensors: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute mean feature embeddings for training and validation datasets.
        
        Args:
            train_tensors: List of training dataset tensors
            val_tensors: List of validation dataset tensors
            
        Returns:
            Tuple of (list of training mean features, validation mean features)
        """
        logger.info("Computing mean feature embeddings...")
        
        # Combine all validation tensors
        val_combined = torch.cat(val_tensors, dim=0)
        logger.info(f"Combined validation tensor shape: {val_combined.shape}")
        
        # Transform validation data and compute mean
        val_rff = self.rff_transform(val_combined)
        mu_Q = torch.mean(val_rff, dim=0)
        
        # Transform training data and compute means
        train_means = []
        for i, train_tensor in enumerate(train_tensors):
            logger.info(f"Processing training dataset {i+1}/{len(train_tensors)}")
            train_rff = self.rff_transform(train_tensor)
            mu_i = torch.mean(train_rff, dim=0)
            train_means.append(mu_i)
        
        logger.info("Mean feature computation completed")
        return train_means, mu_Q
    
    def solve_qp(self, train_means: List[torch.Tensor], val_mean: torch.Tensor) -> np.ndarray:
        """
        Solve the quadratic programming problem to find optimal mixing weights.
        
        Args:
            train_means: List of mean feature vectors for training datasets
            val_mean: Mean feature vector for validation dataset
            
        Returns:
            Optimal mixing weights as numpy array
        """
        N = len(train_means)
        logger.info(f"Solving QP for {N} training datasets...")
        
        # Convert to numpy for cvxpy
        train_means_np = [mu.detach().numpy() for mu in train_means]
        val_mean_np = val_mean.detach().numpy()
        
        # Construct matrix A (N x N): A_ij = mu_i^T @ mu_j
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                A[i, j] = np.dot(train_means_np[i], train_means_np[j])
        
        # Add ridge penalty for numerical stability
        A = A + self.ridge * np.eye(N)
        
        # Construct vector b (N,): b_i = mu_i^T @ mu_Q
        b = np.zeros(N)
        for i in range(N):
            b[i] = np.dot(train_means_np[i], val_mean_np)
        
        logger.info(f"QP matrix A shape: {A.shape}")
        logger.info(f"QP vector b shape: {b.shape}")
        
        # Solve QP using cvxpy
        w = cp.Variable(N)
        objective = cp.Minimize(cp.quad_form(w, A) - 2 * b.T @ w)
        constraints = [cp.sum(w) == 1, w >= 0]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = w.value
            logger.info("QP solved successfully")
            logger.info(f"Optimal weights: {optimal_weights}")
            return optimal_weights
        else:
            raise RuntimeError(f"QP solver failed with status: {problem.status}")
    
    def calculate_mmd_value(self, 
                          train_means: List[torch.Tensor], 
                          val_mean: torch.Tensor, 
                          weights: np.ndarray) -> float:
        """
        Calculate the final MMD value given the optimal weights.
        
        Args:
            train_means: List of mean feature vectors for training datasets
            val_mean: Mean feature vector for validation dataset
            weights: Optimal mixing weights
            
        Returns:
            MMD value
        """
        N = len(train_means)
        
        # Convert to numpy
        train_means_np = [mu.detach().numpy() for mu in train_means]
        val_mean_np = val_mean.detach().numpy()
        
        # Calculate weighted average of training means
        weighted_train_mean = np.zeros_like(train_means_np[0])
        for i in range(N):
            weighted_train_mean += weights[i] * train_means_np[i]
        
        # Calculate MMD^2 = ||mu_weighted - mu_Q||^2
        mmd_squared = np.sum((weighted_train_mean - val_mean_np) ** 2)
        mmd_value = np.sqrt(mmd_squared)
        
        return mmd_value
    
    def optimize_mixing_weights(self, 
                              train_gradient_paths: List[str], 
                              validation_gradient_paths: List[str]) -> Dict[str, float]:
        """
        Main function to optimize mixing weights.
        
        Args:
            train_gradient_paths: List of paths to training gradient files
            validation_gradient_paths: List of paths to validation gradient files
            
        Returns:
            Dictionary mapping training data paths to optimal weights
        """
        logger.info("Starting MMD-based data mixing optimization...")
        logger.info(f"Training datasets: {len(train_gradient_paths)}")
        logger.info(f"Validation datasets: {len(validation_gradient_paths)}")
        
        # Load data
        train_tensors = self.load_gradient_data(train_gradient_paths)
        val_tensors = self.load_gradient_data(validation_gradient_paths)
        
        # Compute sigma automatically if needed
        if self.auto_sigma and self.sigma is None:
            self.sigma = self._compute_median_heuristic_sigma(train_tensors)
            logger.info(f"Using automatically computed sigma: {self.sigma:.6f}")
        elif self.sigma is not None:
            logger.info(f"Using provided sigma: {self.sigma:.6f}")
        else:
            # Fallback to default value
            self.sigma = 3.0
            logger.info(f"Using default sigma: {self.sigma:.6f}")
        
        # Compute mean features
        train_means, val_mean = self.compute_mean_features(train_tensors, val_tensors)
        
        # Solve QP
        optimal_weights = self.solve_qp(train_means, val_mean)
        
        # Calculate final MMD value
        mmd_value = self.calculate_mmd_value(train_means, val_mean, optimal_weights)
        logger.info(f"Final MMD value: {mmd_value:.6f}")
        
        # Create result dictionary
        result = {}
        for i, path in enumerate(train_gradient_paths):
            result[path] = float(optimal_weights[i])
        
        return result


class MMDDataMixerRBF(MMDDataMixer):
    """
    A subclass that uses RBF kernel directly to compute MMD instead of Random Fourier Features.
    This provides an exact MMD computation for comparison with the RFF approximation.
    """
    
    def __init__(self, 
                 rff_dimension: int = 100,  # Not used in RBF, kept for compatibility
                 sigma_bandwidth: Optional[float] = None,
                 ridge_penalty: float = 1e-7,
                 random_seed: Optional[int] = 42,
                 auto_sigma: bool = True,
                 sigma_sample_size: int = 1000):
        """
        Initialize the RBF-based MMD Data Mixer.
        
        Args:
            rff_dimension: Not used in RBF method, kept for compatibility
            sigma_bandwidth: Bandwidth for the Gaussian kernel
            ridge_penalty: Ridge penalty for numerical stability
            random_seed: Random seed for reproducibility
            auto_sigma: Whether to automatically compute sigma using median heuristic
            sigma_sample_size: Number of samples to use for sigma computation
        """
        super().__init__(rff_dimension, sigma_bandwidth, ridge_penalty, 
                         random_seed, auto_sigma, sigma_sample_size)
        logger.info("Initialized MMDDataMixerRBF (using exact RBF kernel computation)")
    
    def _compute_rbf_kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        """
        Compute RBF kernel matrix between X and Y (or X and X if Y is None).
        
        Args:
            X: First set of data points (n_samples_x, n_features)
            Y: Second set of data points (n_samples_y, n_features), optional
            
        Returns:
            Kernel matrix (n_samples_x, n_samples_y)
        """
        if Y is None:
            Y = X
        
        # Compute pairwise squared distances
        # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * x_i^T * y_j
        X_sqnorms = torch.sum(X ** 2, dim=1, keepdim=True)  # (n_x, 1)
        Y_sqnorms = torch.sum(Y ** 2, dim=1, keepdim=True)  # (n_y, 1)
        
        # Compute dot products
        XY = torch.mm(X, Y.t())  # (n_x, n_y)
        
        # Compute squared distances
        squared_distances = X_sqnorms + Y_sqnorms.t() - 2 * XY
        
        # Compute RBF kernel: k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
        gamma = 1.0 / (2 * self.sigma ** 2)
        kernel_matrix = torch.exp(-gamma * squared_distances)
        # breakpoint()
        return kernel_matrix
    
    def _compute_mmd_squared_rbf(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute squared MMD between two sets of samples using RBF kernel.
        
        MMD^2(X, Y) = E[k(x, x')] - 2*E[k(x, y)] + E[k(y, y')]
        where x, x' ~ X and y, y' ~ Y
        
        Args:
            X: First set of samples (n_samples_x, n_features)
            Y: Second set of samples (n_samples_y, n_features)
            
        Returns:
            Squared MMD value
        """
        # Compute kernel matrices
        K_XX = self._compute_rbf_kernel_matrix(X, X)
        K_YY = self._compute_rbf_kernel_matrix(Y, Y)
        K_XY = self._compute_rbf_kernel_matrix(X, Y)
        
        # Compute MMD^2 = E[k(x, x')] - 2*E[k(x, y)] + E[k(y, y')]
        # Note: We exclude diagonal elements for unbiased estimation
        n_x, n_y = X.shape[0], Y.shape[0]
        
        # E[k(x, x')] - exclude diagonal
        if n_x > 1:
            K_XX_offdiag = K_XX - torch.diag(torch.diag(K_XX))
            term1 = torch.sum(K_XX_offdiag) / (n_x * (n_x - 1))
        else:
            term1 = 0.0
        
        # E[k(y, y')] - exclude diagonal  
        if n_y > 1:
            K_YY_offdiag = K_YY - torch.diag(torch.diag(K_YY))
            term3 = torch.sum(K_YY_offdiag) / (n_y * (n_y - 1))
        else:
            term3 = 0.0
        
        # -2 * E[k(x, y)]
        term2 = -2.0 * torch.mean(K_XY)
        
        mmd_squared = term1 + term2 + term3
        return float(mmd_squared)
    
    def compute_mean_features(self, 
                            train_tensors: List[torch.Tensor], 
                            val_tensors: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        For RBF method, we don't compute mean features in RFF space.
        Instead, we return the original data tensors for direct kernel computation.
        
        Args:
            train_tensors: List of training dataset tensors
            val_tensors: List of validation dataset tensors
            
        Returns:
            Tuple of (train_tensors, combined_val_tensor)
        """
        logger.info("RBF method: Using original data tensors (no RFF transformation)")
        
        # Combine validation tensors
        combined_val = torch.cat(val_tensors, dim=0)
        logger.info(f"Combined validation data shape: {combined_val.shape}")
        
        return train_tensors, combined_val
    
    def solve_qp(self, train_tensors: List[torch.Tensor], val_tensor: torch.Tensor) -> np.ndarray:
        """
        Solve the quadratic programming problem using RBF kernel MMD.
        
        Based on the algorithm from the paper, the objective is to minimize:
        MMD²(w) = w^T A w - 2 b^T w + c + ridge * ||w||^2
        
        where:
        - A[i,j] = (1/(n_i * n_j)) * sum_p sum_q k_σ(g_i^(p), g_j^(q))
        - b[i] = (1/(n_i * m)) * sum_p sum_q k_σ(g_i^(p), h_q)  
        - c = (1/m²) * sum_q sum_r k_σ(h_q, h_r)
        
        Args:
            train_tensors: List of training dataset tensors
            val_tensor: Combined validation tensor
            
        Returns:
            Optimal mixing weights
        """
        logger.info("Solving QP using RBF kernel MMD...")
        
        n_datasets = len(train_tensors)
        
        # Compute matrix A where A[i,j] = (1/(n_i * n_j)) * sum_p sum_q k_σ(g_i^(p), g_j^(q))
        logger.info("Computing kernel matrix A...")
        A = np.zeros((n_datasets, n_datasets))
        
        for i in range(n_datasets):
            for j in range(i, n_datasets):  # Only compute upper triangle
                # Compute kernel matrix between datasets i and j
                K_ij = self._compute_rbf_kernel_matrix(train_tensors[i], train_tensors[j])
                
                # A[i,j] = mean of all kernel values = (1/(n_i * n_j)) * sum_all k(x_p, x_q)
                A[i, j] = float(torch.mean(K_ij))
                A[j, i] = A[i, j]  # Symmetric
                
                logger.info(f"A[{i+1},{j+1}] = {A[i, j]:.6f}")
        
        # Compute vector b where b[i] = (1/(n_i * m)) * sum_p sum_q k_σ(g_i^(p), h_q)
        logger.info("Computing kernel vector b...")
        b = np.zeros(n_datasets)
        
        for i in range(n_datasets):
            # Compute kernel matrix between training dataset i and validation set
            K_iv = self._compute_rbf_kernel_matrix(train_tensors[i], val_tensor)
            
            # b[i] = mean of all kernel values = (1/(n_i * m)) * sum_all k(x_p, h_q)
            b[i] = float(torch.mean(K_iv))
            logger.info(f"b[{i+1}] = {b[i]:.6f}")
        
        # Compute constant term c = (1/m²) * sum_q sum_r k_σ(h_q, h_r)
        # Note: c doesn't affect the optimization result, but we compute it for completeness
        K_vv = self._compute_rbf_kernel_matrix(val_tensor, val_tensor)
        c = float(torch.mean(K_vv))
        logger.info(f"Constant term c = {c:.6f}")
        
        # Add ridge regularization to diagonal
        A_ridge = A + self.ridge * np.eye(n_datasets)
        
        # Set up QP: minimize (1/2) * w^T * A_ridge * w - b^T * w
        # Note: We omit the constant term c as it doesn't affect the optimization
        # subject to: sum(w) = 1, w >= 0
        w = cp.Variable(n_datasets)
        
        # Objective function: minimize w^T A w - 2 b^T w (factor of 2 absorbed into b)
        objective = cp.Minimize(cp.quad_form(w, A_ridge) - 2 * b.T @ w)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0           # Non-negative weights
        ]
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = w.value
            logger.info(f"QP solved successfully. Status: {problem.status}")
            logger.info(f"Optimal weights: {optimal_weights}")
            return optimal_weights
        else:
            logger.error(f"QP solver failed with status: {problem.status}")
            # Return uniform weights as fallback
            uniform_weights = np.ones(n_datasets) / n_datasets
            logger.info(f"Using uniform weights as fallback: {uniform_weights}")
            return uniform_weights
    
    def calculate_mmd_value(self, 
                          train_tensors: List[torch.Tensor], 
                          val_tensor: torch.Tensor, 
                          weights: np.ndarray) -> float:
        """
        Calculate the final MMD value using RBF kernel and the corrected formula.
        
        Uses the formula: MMD²(w) = w^T A w - 2 b^T w + c
        where:
        - A[i,j] = (1/(n_i * n_j)) * sum_p sum_q k_σ(g_i^(p), g_j^(q))
        - b[i] = (1/(n_i * m)) * sum_p sum_q k_σ(g_i^(p), h_q)  
        - c = (1/m²) * sum_q sum_r k_σ(h_q, h_r)
        
        Args:
            train_tensors: List of training dataset tensors
            val_tensor: Combined validation tensor
            weights: Optimal mixing weights
            
        Returns:
            Final MMD value
        """
        n_datasets = len(train_tensors)
        
        # Compute matrix A where A[i,j] = mean kernel values between datasets i and j
        A = np.zeros((n_datasets, n_datasets))
        for i in range(n_datasets):
            for j in range(n_datasets):
                K_ij = self._compute_rbf_kernel_matrix(train_tensors[i], train_tensors[j])
                A[i, j] = float(torch.mean(K_ij))
        
        # Compute vector b where b[i] = mean kernel values between dataset i and validation
        b = np.zeros(n_datasets)
        for i in range(n_datasets):
            K_iv = self._compute_rbf_kernel_matrix(train_tensors[i], val_tensor)
            b[i] = float(torch.mean(K_iv))
        
        # Compute constant term c = mean kernel values within validation set
        K_vv = self._compute_rbf_kernel_matrix(val_tensor, val_tensor)
        c = float(torch.mean(K_vv))
        
        # Calculate MMD² using the formula: w^T A w - 2 b^T w + c
        mmd_squared = np.dot(weights, np.dot(A, weights)) - 2 * np.dot(b, weights) + c
        
        # Return MMD (square root of MMD²), ensuring non-negative
        mmd_value = np.sqrt(max(0, mmd_squared))
        
        return mmd_value


def main():
    """
    Main function with command line interface.
    """
    parser = argparse.ArgumentParser(description='MMD-based Data Mixing Optimization')
    parser.add_argument('--train_paths', nargs='+', required=True,
                       help='Paths to training gradient files')
    parser.add_argument('--val_paths', nargs='+', required=True,
                       help='Paths to validation gradient files')
    parser.add_argument('--rff_dimension', type=int, default=100,
                       help='RFF dimension (default: 100)')
    parser.add_argument('--sigma_bandwidth', type=float, default=3.0,
                       help='Gaussian kernel bandwidth (default: 3.0)')
    parser.add_argument('--ridge_penalty', type=float, default=1e-7,
                       help='Ridge penalty for numerical stability (default: 1e-5)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # Initialize mixer
    mixer = MMDDataMixer(
        rff_dimension=args.rff_dimension,
        sigma_bandwidth=args.sigma_bandwidth,
        ridge_penalty=args.ridge_penalty,
        random_seed=args.random_seed
    )
    
    # Optimize weights
    try:
        results = mixer.optimize_mixing_weights(args.train_paths, args.val_paths)
        
        # Print results
        print("\n" + "="*60)
        print("OPTIMAL DATA MIXING WEIGHTS")
        print("="*60)
        
        total_weight = 0
        for i, (path, weight) in enumerate(results.items()):
            print(f"Dataset {i+1}: {weight:.6f}")
            print(f"  Path: {path}")
            total_weight += weight
        
        print(f"\nTotal weight (should be 1.0): {total_weight:.6f}")
        print("="*60)
        
        # Save results if output file specified
        if args.output_file:
            import json
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise


if __name__ == "__main__":
    main()


