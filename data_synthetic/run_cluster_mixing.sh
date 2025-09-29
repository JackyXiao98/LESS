#!/bin/bash

# Cluster Data Mixing Analysis Script
# This script performs MMD-based data mixing optimization using clustered embedding data
# to find optimal mixing ratios for each cluster to match the target train_yelp distribution.


# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_DIR="${SCRIPT_DIR}/data_process/cluster_embeddings"
TARGET_FILE="${SCRIPT_DIR}/data_process/embeddings/yelp_train_sampled_1000_embeddings.pkl"
OUTPUT_DIR="${SCRIPT_DIR}/mixing_cluster_equal_results"
PYTHON_SCRIPT="${SCRIPT_DIR}/cluster_mixing.py"

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

# Function to check if file/directory exists
check_path() {
    local path="$1"
    local type="$2"
    
    if [[ "$type" == "file" ]]; then
        if [[ ! -f "$path" ]]; then
            print_error "File not found: $path"
            return 1
        fi
    elif [[ "$type" == "dir" ]]; then
        if [[ ! -d "$path" ]]; then
            print_error "Directory not found: $path"
            return 1
        fi
    fi
    return 0
}

# Main function
main() {
    print_info "Starting Cluster Data Mixing Analysis..."
    echo "=============================================="
    
    # Display configuration
    print_info "Configuration:"
    echo "  Cluster Directory: $CLUSTER_DIR"
    echo "  Target File: $TARGET_FILE"
    echo "  Output Directory: $OUTPUT_DIR"
    echo "  Python Script: $PYTHON_SCRIPT"
    echo ""
    
    # Check prerequisites
    print_info "Checking prerequisites..."
    
    # Check if cluster directory exists
    if ! check_path "$CLUSTER_DIR" "dir"; then
        print_error "Cluster embeddings directory not found!"
        print_info "Please run cluster_embedding.py first to generate cluster files."
        return 1
    fi
    
    # Check if cluster files exist
    cluster_files=$(find "$CLUSTER_DIR" -name "cluster_*.pkl" 2>/dev/null | wc -l)
    if [[ $cluster_files -eq 0 ]]; then
        print_error "No cluster files found in $CLUSTER_DIR"
        print_info "Please run cluster_embedding.py first to generate cluster files."
        return 1
    fi
    print_success "Found $cluster_files cluster files"
    
    # Check if target file exists
    if ! check_path "$TARGET_FILE" "file"; then
        print_error "Target file not found!"
        print_info "Please ensure yelp_train_sampled_1000_embeddings.pkl exists in the embeddings directory."
        return 1
    fi
    print_success "Target file found"
    
    # Check if Python script exists
    if ! check_path "$PYTHON_SCRIPT" "file"; then
        print_error "Python script not found!"
        return 1
    fi
    print_success "Python script found"
    
    # Check Python dependencies
    print_info "Checking Python dependencies..."
    python3 -c "import torch, numpy, sklearn, matplotlib, seaborn, pandas" 2>/dev/null
    if [[ $? -ne 0 ]]; then
        print_warning "Some Python dependencies may be missing. The script will attempt to run anyway."
    else
        print_success "Python dependencies check passed"
    fi
    
    # Create output directory
    print_info "Creating output directory..."
    mkdir -p "$OUTPUT_DIR"
    print_success "Output directory created: $OUTPUT_DIR"
    
    # Run cluster mixing analysis
    print_info "Running cluster mixing analysis..."
    echo "=============================================="
    
    # Change to script directory to ensure relative imports work
    cd "$SCRIPT_DIR"
    
    # Run the Python script with parameters
    python3 "$PYTHON_SCRIPT" \
        --cluster_dir "$CLUSTER_DIR" \
        --target_file "$TARGET_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --rff_dimension 100 \
        --ridge_penalty 1e-7 \
        --regularization_lambda 1 \
        --sample_number -1 \
        --random_seed 42
    
    # Check if the script ran successfully
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "=============================================="
        print_success "Cluster mixing analysis completed successfully!"
        
        # Display output files
        print_info "Generated files:"
        if [[ -f "$OUTPUT_DIR/cluster_mixing_weights.txt" ]]; then
            echo "  ✓ cluster_mixing_weights.txt - Human-readable mixing weights"
        fi
        if [[ -f "$OUTPUT_DIR/cluster_mixing_weights.pkl" ]]; then
            echo "  ✓ cluster_mixing_weights.pkl - Machine-readable mixing data"
        fi
        if [[ -f "$OUTPUT_DIR/cluster_mixing_analysis.png" ]]; then
            echo "  ✓ cluster_mixing_analysis.png - Visualization of mixing analysis"
        fi
        
        # Display summary
        echo ""
        print_info "Results summary:"
        if [[ -f "$OUTPUT_DIR/cluster_mixing_weights.txt" ]]; then
            echo "----------------------------------------"
            head -20 "$OUTPUT_DIR/cluster_mixing_weights.txt"
            echo "----------------------------------------"
            print_info "Full results available in: $OUTPUT_DIR/cluster_mixing_weights.txt"
        fi
        
        echo ""
        print_success "All files saved to: $OUTPUT_DIR"
        print_info "You can view the analysis visualization at: $OUTPUT_DIR/cluster_mixing_analysis.png"
        
    else
        print_error "Cluster mixing analysis failed!"
        return 1
    fi
}

# Help function
show_help() {
    echo "Cluster Data Mixing Analysis Script"
    echo "=================================="
    echo ""
    echo "This script performs MMD-based data mixing optimization using clustered"
    echo "embedding data to find optimal mixing ratios for each cluster."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --verbose  Enable verbose output"
    echo ""
    echo "Configuration (edit script to modify):"
    echo "  CLUSTER_DIR:   Directory containing cluster .pkl files"
    echo "  TARGET_FILE:   Path to target embedding .pkl file (train_yelp)"
    echo "  OUTPUT_DIR:    Directory to save results"
    echo ""
    echo "Prerequisites:"
    echo "  1. Run cluster_embedding.py to generate cluster files"
    echo "  2. Ensure target file (yelp_train_sampled_1000_embeddings.pkl) exists"
    echo "  3. Python dependencies: torch, numpy, sklearn, matplotlib, seaborn, pandas"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            return 0
            ;;
        -v|--verbose)
            set -x  # Enable verbose mode
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            return 1
            ;;
    esac
done

# Run main function
main