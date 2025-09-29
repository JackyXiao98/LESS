#!/bin/bash

# RFF + t-SNE Visualization Configuration Script
# This script configures and runs RFF + t-SNE visualization for gradient data
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
OUTPUT_DIR="./rff_tsne_results"

# RFF (Random Fourier Features) parameters
RFF_DIMENSION=512         # RFF target dimension (128, 256, 512, 1024)
SIGMA_BANDWIDTH=""        # Sigma bandwidth (leave empty for auto-computation)
AUTO_SIGMA="true"         # Enable automatic sigma computation using median heuristic
SIGMA_SAMPLE_SIZE=1000    # Number of samples for sigma computation

# t-SNE parameters
SAMPLE_NUMBER=1000        # Number of samples per dataset (-1 for all data)
PCA_COMPONENTS=50         # Number of PCA components for preprocessing
PERPLEXITY=30            # t-SNE perplexity parameter
MAX_ITER=1000            # Number of t-SNE iterations
RANDOM_SEED=42           # Random seed for reproducibility

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

echo "=============================================="
echo "RFF + t-SNE Visualization Configuration"
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
echo "RFF Parameters:"
echo "  RFF Dimension: $RFF_DIMENSION"
if [[ -n "$SIGMA_BANDWIDTH" ]]; then
    echo "  Sigma Bandwidth: $SIGMA_BANDWIDTH (manual)"
else
    echo "  Sigma Bandwidth: auto-computed using median heuristic"
fi
echo "  Auto Sigma: $AUTO_SIGMA"
echo "  Sigma Sample Size: $SIGMA_SAMPLE_SIZE"
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

# Validate RFF dimension
if [[ ! "$RFF_DIMENSION" =~ ^[0-9]+$ ]] || [[ "$RFF_DIMENSION" -lt 32 ]] || [[ "$RFF_DIMENSION" -gt 2048 ]]; then
    echo "Error: Invalid RFF_DIMENSION '$RFF_DIMENSION'. Must be a number between 32 and 2048."
    exit 1
fi

# Validate sigma bandwidth if provided
if [[ -n "$SIGMA_BANDWIDTH" ]] && [[ ! "$SIGMA_BANDWIDTH" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Error: Invalid SIGMA_BANDWIDTH '$SIGMA_BANDWIDTH'. Must be a positive number."
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
PYTHON_SCRIPT="$SCRIPT_DIR/rff_tsne_visualization.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# =============================================================================
# EXECUTION
# =============================================================================

echo "Starting RFF + t-SNE visualization in $MODE mode..."

# Build base command arguments
BASE_ARGS=(
    "--experiment_name" "$EXPERIMENT_NAME"
    "--dim" "$DIM"
    "--output_dir" "$OUTPUT_DIR"
    "--sample_number" "$SAMPLE_NUMBER"
    "--rff_dimension" "$RFF_DIMENSION"
    "--sigma_sample_size" "$SIGMA_SAMPLE_SIZE"
    "--pca_components" "$PCA_COMPONENTS"
    "--perplexity" "$PERPLEXITY"
    "--max_iter" "$MAX_ITER"
    "--random_seed" "$RANDOM_SEED"
)

# Add sigma bandwidth if specified
if [[ -n "$SIGMA_BANDWIDTH" ]]; then
    BASE_ARGS+=("--sigma_bandwidth" "$SIGMA_BANDWIDTH")
fi

# Add auto sigma flag if enabled
if [[ "$AUTO_SIGMA" == "true" ]]; then
    BASE_ARGS+=("--auto_sigma")
fi

# Build command arguments based on mode
if [[ "$MODE" == "dual" ]]; then
    echo "Command: python $PYTHON_SCRIPT \\"
    echo "    --train_base_path \"$TRAIN_BASE_PATH\" \\"
    echo "    --val_base_path \"$VAL_BASE_PATH\" \\"
    for ((i=0; i<${#BASE_ARGS[@]}; i+=2)); do
        if [[ $((i+1)) -lt ${#BASE_ARGS[@]} ]]; then
            echo "    --${BASE_ARGS[i]#--} \"${BASE_ARGS[i+1]}\" \\"
        else
            echo "    --${BASE_ARGS[i]#--} \\"
        fi
    done
    echo ""
    
    # Run the Python script with dual-path parameters
    python "$PYTHON_SCRIPT" \
        --train_base_path "$TRAIN_BASE_PATH" \
        --val_base_path "$VAL_BASE_PATH" \
        "${BASE_ARGS[@]}"
else
    echo "Command: python $PYTHON_SCRIPT \\"
    echo "    --base_path \"$SINGLE_BASE_PATH\" \\"
    for ((i=0; i<${#BASE_ARGS[@]}; i+=2)); do
        if [[ $((i+1)) -lt ${#BASE_ARGS[@]} ]]; then
            echo "    --${BASE_ARGS[i]#--} \"${BASE_ARGS[i+1]}\" \\"
        else
            echo "    --${BASE_ARGS[i]#--} \\"
        fi
    done
    echo ""
    
    # Run the Python script with single-path parameters
    python "$PYTHON_SCRIPT" \
        --base_path "$SINGLE_BASE_PATH" \
        "${BASE_ARGS[@]}"
fi

# Check if the command was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=============================================="
    echo "RFF + t-SNE visualization completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - PNG visualization: ${OUTPUT_DIR}/*.png"
    echo "  - PDF visualization: ${OUTPUT_DIR}/*.pdf"
    echo "  - CSV coordinates: ${OUTPUT_DIR}/*_coordinates.csv"
    echo ""
    echo "RFF Configuration Used:"
    echo "  - RFF Dimension: $RFF_DIMENSION"
    if [[ -n "$SIGMA_BANDWIDTH" ]]; then
        echo "  - Sigma Bandwidth: $SIGMA_BANDWIDTH (manual)"
    else
        echo "  - Sigma Bandwidth: auto-computed"
    fi
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Error: RFF + t-SNE visualization failed!"
    echo "Please check the error messages above."
    echo ""
    echo "Common troubleshooting tips:"
    echo "1. Verify that gradient files exist in the specified paths"
    echo "2. Check that the experiment name matches your data structure"
    echo "3. Ensure sufficient memory for the specified RFF dimension"
    echo "4. Try reducing sample_number if running out of memory"
    echo "5. Verify that all required Python packages are installed"
    echo "=============================================="
    exit 1
fi

# =============================================================================
# ADDITIONAL INFORMATION
# =============================================================================

echo ""
echo "=============================================="
echo "RFF + t-SNE Visualization Information"
echo "=============================================="
echo ""
echo "What is RFF (Random Fourier Features)?"
echo "  RFF approximates kernel functions using explicit feature maps,"
echo "  enabling efficient computation of kernel similarities in linear time."
echo ""
echo "Benefits for Gradient Analysis:"
echo "  - Reduces high-dimensional gradient vectors to manageable size"
echo "  - Preserves kernel relationships between gradient vectors"
echo "  - Enables efficient similarity computation and clustering"
echo "  - Maintains dataset-specific gradient characteristics"
echo ""
echo "Parameter Tuning Tips:"
echo "  - RFF Dimension: Higher values (512-1024) give better approximation"
echo "  - Sigma Bandwidth: Auto-computation usually works well"
echo "  - Sample Number: Balance between quality and computation time"
echo "  - Perplexity: Adjust based on dataset size (5-50 typical range)"
echo ""
echo "For more examples and advanced usage, see:"
echo "  $SCRIPT_DIR/example_rff_usage.py"
echo "=============================================="