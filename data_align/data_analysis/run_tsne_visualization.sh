#!/bin/bash

# t-SNE Visualization Configuration Script
# This script configures and runs t-SNE visualization for gradient data
# Supports both single-path and dual-path (train + validation) modes

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Base paths for gradient data (consistent with run_mixing_calculation.sh)
TRAIN_BASE_PATH="/mnt/hdfs/selection/yingtai_sft/lora_grads"
VAL_BASE_PATH="/mnt/hdfs/selection/yingtai_sft/lora_val_grads"

# Visualization mode configuration
# Set MODE to "single" for single-path mode or "dual" for dual-path mode
MODE="dual"  # Options: "single", "dual"

# For single-path mode, specify which base path to use
SINGLE_BASE_PATH="$VAL_BASE_PATH"  # Use validation path by default

# Experiment configuration
EXPERIMENT_NAME="tulu3-Qwen3-8B-p0.05-lora-seed3"
DIM="rank_1/dim8192"

# Output configuration
OUTPUT_DIR="./tsne_results"

# t-SNE parameters
SAMPLE_NUMBER=100          # Number of samples per dataset (-1 for all data)
PCA_COMPONENTS=50         # Number of PCA components for preprocessing
PERPLEXITY=30            # t-SNE perplexity parameter
MAX_ITER=1000            # Number of t-SNE iterations (renamed from N_ITER)
RANDOM_SEED=42           # Random seed for reproducibility

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

echo "=============================================="
echo "t-SNE Visualization Configuration"
echo "=============================================="
echo "Mode: $MODE"
if [[ "$MODE" == "dual" ]]; then
    echo "Training Base Path: $TRAIN_BASE_PATH"
    echo "Validation Base Path: $VAL_BASE_PATH"
else
    echo "Single Base Path: $SINGLE_BASE_PATH"
fi
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Dimension: $DIM"
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "t-SNE Parameters:"
echo "  Sample Number: $SAMPLE_NUMBER"
echo "  PCA Components: $PCA_COMPONENTS"
echo "  Perplexity: $PERPLEXITY"
echo "  Max Iterations: $MAX_ITER"
echo "  Random Seed: $RANDOM_SEED"
echo "=============================================="
echo ""

# =============================================================================
# VALIDATION
# =============================================================================

# Validate mode
if [[ "$MODE" != "single" && "$MODE" != "dual" ]]; then
    echo "Error: Invalid MODE '$MODE'. Must be 'single' or 'dual'."
    exit 1
fi

# Check if paths exist (only if they're local paths)
if [[ "$MODE" == "dual" ]]; then
    if [[ "$TRAIN_BASE_PATH" != /mnt/* ]] && [[ ! -d "$TRAIN_BASE_PATH" ]]; then
        echo "Warning: Training base path does not exist locally: $TRAIN_BASE_PATH"
        echo "Assuming it's a remote path that will be accessible during execution."
    fi
    if [[ "$VAL_BASE_PATH" != /mnt/* ]] && [[ ! -d "$VAL_BASE_PATH" ]]; then
        echo "Warning: Validation base path does not exist locally: $VAL_BASE_PATH"
        echo "Assuming it's a remote path that will be accessible during execution."
    fi
else
    if [[ "$SINGLE_BASE_PATH" != /mnt/* ]] && [[ ! -d "$SINGLE_BASE_PATH" ]]; then
        echo "Warning: Single base path does not exist locally: $SINGLE_BASE_PATH"
        echo "Assuming it's a remote path that will be accessible during execution."
    fi
fi

# Check if Python script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/tsne_visualization.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# =============================================================================
# EXECUTION
# =============================================================================

echo "Starting t-SNE visualization in $MODE mode..."

# Build command arguments based on mode
if [[ "$MODE" == "dual" ]]; then
    echo "Command: python $PYTHON_SCRIPT \\"
    echo "    --train_base_path \"$TRAIN_BASE_PATH\" \\"
    echo "    --val_base_path \"$VAL_BASE_PATH\" \\"
    echo "    --experiment_name \"$EXPERIMENT_NAME\" \\"
    echo "    --dim \"$DIM\" \\"
    echo "    --output_dir \"$OUTPUT_DIR\" \\"
    echo "    --sample_number $SAMPLE_NUMBER \\"
    echo "    --pca_components $PCA_COMPONENTS \\"
    echo "    --perplexity $PERPLEXITY \\"
    echo "    --max_iter $MAX_ITER \\"
    echo "    --random_seed $RANDOM_SEED"
    echo ""
    
    # Run the Python script with dual-path parameters
    python "$PYTHON_SCRIPT" \
        --train_base_path "$TRAIN_BASE_PATH" \
        --val_base_path "$VAL_BASE_PATH" \
        --experiment_name "$EXPERIMENT_NAME" \
        --dim "$DIM" \
        --output_dir "$OUTPUT_DIR" \
        --sample_number "$SAMPLE_NUMBER" \
        --pca_components "$PCA_COMPONENTS" \
        --perplexity "$PERPLEXITY" \
        --max_iter "$MAX_ITER" \
        --random_seed "$RANDOM_SEED"
else
    echo "Command: python $PYTHON_SCRIPT \\"
    echo "    --base_path \"$SINGLE_BASE_PATH\" \\"
    echo "    --experiment_name \"$EXPERIMENT_NAME\" \\"
    echo "    --dim \"$DIM\" \\"
    echo "    --output_dir \"$OUTPUT_DIR\" \\"
    echo "    --sample_number $SAMPLE_NUMBER \\"
    echo "    --pca_components $PCA_COMPONENTS \\"
    echo "    --perplexity $PERPLEXITY \\"
    echo "    --max_iter $MAX_ITER \\"
    echo "    --random_seed $RANDOM_SEED"
    echo ""
    
    # Run the Python script with single-path parameters
    python "$PYTHON_SCRIPT" \
        --base_path "$SINGLE_BASE_PATH" \
        --experiment_name "$EXPERIMENT_NAME" \
        --dim "$DIM" \
        --output_dir "$OUTPUT_DIR" \
        --sample_number "$SAMPLE_NUMBER" \
        --pca_components "$PCA_COMPONENTS" \
        --perplexity "$PERPLEXITY" \
        --max_iter "$MAX_ITER" \
        --random_seed "$RANDOM_SEED"
fi

# Check if the command was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=============================================="
    echo "t-SNE visualization completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Error: t-SNE visualization failed!"
    echo "Please check the error messages above."
    echo "=============================================="
    exit 1
fi