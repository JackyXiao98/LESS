#!/bin/bash

# Embedding-based MMD Data Mixing Optimization Script
# This script calculates optimal mixing ratios for embedding datasets
# to match a target embedding distribution using MMD optimization.


# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMBEDDING_DIR="${SCRIPT_DIR}/data_process/embeddings"
OUTPUT_DIR="${SCRIPT_DIR}/mixing_results"
PYTHON_SCRIPT="${SCRIPT_DIR}/mmd_synthetic.py"

# Target file pattern (the embedding file to match)
TARGET_FILE="yelp_train_sampled_1000"

# MMD optimization parameters
RFF_DIMENSION=500
REGULARIZATION_LAMBDA=0.1
SAMPLE_NUMBER=-1  # Use all samples (-1) or specify a number
RANDOM_SEED=42

# Output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/embedding_mixing_weights_${TIMESTAMP}.txt"
LOG_FILE="${OUTPUT_DIR}/mixing_log_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if required files exist
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if Python script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python script not found: $PYTHON_SCRIPT"
    fi
    
    # Check if embedding directory exists
    if [ ! -d "$EMBEDDING_DIR" ]; then
        print_error "Embedding directory not found: $EMBEDDING_DIR"
    fi
    
    # Check if target embedding file exists
    TARGET_FILES=($(find "$EMBEDDING_DIR" -name "*${TARGET_FILE}*_embeddings.pkl"))
    if [ ${#TARGET_FILES[@]} -eq 0 ]; then
        print_error "Target embedding file not found: *${TARGET_FILE}*_embeddings.pkl in $EMBEDDING_DIR"
    elif [ ${#TARGET_FILES[@]} -gt 1 ]; then
        print_warning "Multiple target files found, using: ${TARGET_FILES[0]}"
    fi
    
    # Check if training embedding files exist
    TRAIN_FILES=($(find "$EMBEDDING_DIR" -name "*huggingface*_embeddings.pkl"))
    if [ ${#TRAIN_FILES[@]} -eq 0 ]; then
        print_error "No training embedding files found: *huggingface*_embeddings.pkl in $EMBEDDING_DIR"
    fi
    
    print_success "Prerequisites check passed"
    print_info "Target file: ${TARGET_FILES[0]}"
    print_info "Training files found: ${#TRAIN_FILES[@]}"
    for file in "${TRAIN_FILES[@]}"; do
        print_info "  - $(basename "$file")"
    done
}

# Function to create output directory
setup_output_dir() {
    print_info "Setting up output directory..."
    mkdir -p "$OUTPUT_DIR"
    print_success "Output directory ready: $OUTPUT_DIR"
}

# Function to run the embedding mixing optimization
run_optimization() {
    print_info "Starting embedding mixing optimization..."
    print_info "Parameters:"
    print_info "  - RFF Dimension: $RFF_DIMENSION"
    print_info "  - Regularization Lambda: $REGULARIZATION_LAMBDA"
    print_info "  - Sample Number: $SAMPLE_NUMBER"
    print_info "  - Random Seed: $RANDOM_SEED"
    print_info "  - Output File: $OUTPUT_FILE"
    print_info "  - Log File: $LOG_FILE"
    
    # Run the Python script
    python3 "$PYTHON_SCRIPT" \
        --embedding_dir "$EMBEDDING_DIR" \
        --target_file "$TARGET_FILE" \
        --output_file "$OUTPUT_FILE" \
        --rff_dimension "$RFF_DIMENSION" \
        --regularization_lambda "$REGULARIZATION_LAMBDA" \
        --sample_number "$SAMPLE_NUMBER" \
        --random_seed "$RANDOM_SEED" \
        2>&1 | tee "$LOG_FILE"
    
    # Check if the script succeeded
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "Optimization completed successfully!"
    else
        print_error "Optimization failed. Check log file: $LOG_FILE"
    fi
}

# Function to display results
display_results() {
    print_info "Displaying results..."
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "=========================================="
        echo "OPTIMAL EMBEDDING MIXING WEIGHTS"
        echo "=========================================="
        cat "$OUTPUT_FILE"
        echo "=========================================="
        echo ""
        print_success "Results saved to: $OUTPUT_FILE"
        print_success "Log saved to: $LOG_FILE"
    else
        print_error "Output file not found: $OUTPUT_FILE"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -d, --embedding-dir     Embedding directory (default: $EMBEDDING_DIR)"
    echo "  -t, --target-file       Target file pattern (default: $TARGET_FILE)"
    echo "  -o, --output-dir        Output directory (default: $OUTPUT_DIR)"
    echo "  -r, --rff-dimension     RFF dimension (default: $RFF_DIMENSION)"
    echo "  -l, --lambda            Regularization lambda (default: $REGULARIZATION_LAMBDA)"
    echo "  -s, --sample-number     Sample number (default: $SAMPLE_NUMBER)"
    echo "  --seed                  Random seed (default: $RANDOM_SEED)"
    echo ""
    echo "Example:"
    echo "  $0 --rff-dimension 200 --lambda 0.01 --sample-number 500"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            ;;
        -d|--embedding-dir)
            EMBEDDING_DIR="$2"
            shift 2
            ;;
        -t|--target-file)
            TARGET_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--rff-dimension)
            RFF_DIMENSION="$2"
            shift 2
            ;;
        -l|--lambda)
            REGULARIZATION_LAMBDA="$2"
            shift 2
            ;;
        -s|--sample-number)
            SAMPLE_NUMBER="$2"
            shift 2
            ;;
        --seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Update output files with new output directory
OUTPUT_FILE="${OUTPUT_DIR}/embedding_mixing_weights_${TIMESTAMP}.txt"
LOG_FILE="${OUTPUT_DIR}/mixing_log_${TIMESTAMP}.log"

# Main execution
main() {
    echo "=========================================="
    echo "EMBEDDING MMD DATA MIXING OPTIMIZATION"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    setup_output_dir
    run_optimization
    display_results
    
    echo ""
    print_success "All tasks completed successfully!"
    echo "=========================================="
}

# Run main function
main "$@"