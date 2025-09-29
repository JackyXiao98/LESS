#!/bin/bash

# Data Mixing Ratio Calculation Script
# This script runs the MMD-based data mixing optimization

# Set parameters
TRAIN_BASE_PATH="/mnt/hdfs/selection/yingtai_sft/lora_grads"
# TRAIN_BASE_PATH="/mnt/hdfs/selection/yingtai_sft/lora_val_grads"
VAL_BASE_PATH="/mnt/hdfs/selection/yingtai_sft/lora_val_grads"
EXPERIMENT_NAME="tulu3-Qwen3-8B-p0.05-lora-seed3"
DIM="rank_1/dim8192"
# DIM='dim8192"
OUTPUT_DIR="./mixing_results"

# MMD parameters (as specified)
RFF_DIMENSION=1000
SIGMA_SAMPLE_SIZE=200
RIDGE_PENALTY=1e-7
REGULARIZATION_LAMBDA=0
SAMPLE_NUMBER=-1
RANDOM_SEED=42

echo "开始计算数据混合比例..."
echo "训练数据路径: ${TRAIN_BASE_PATH}"
echo "验证数据路径: ${VAL_BASE_PATH}"
echo "实验名称: ${EXPERIMENT_NAME}"
echo "维度文件夹: ${DIM}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "MMD参数:"
echo "  RFF维度: ${RFF_DIMENSION}"
echo "  Sigma采样大小: ${SIGMA_SAMPLE_SIZE}"
echo "  Ridge惩罚: ${RIDGE_PENALTY}"
echo "  正则化Lambda: ${REGULARIZATION_LAMBDA}"
echo "  采样数量: ${SAMPLE_NUMBER}"
echo "  随机种子: ${RANDOM_SEED}"
echo ""

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Run the calculation
python calculate_mixing_ratio.py \
    --train_base_path "${TRAIN_BASE_PATH}" \
    --val_base_path "${VAL_BASE_PATH}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --dim "${DIM}" \
    --output_dir "${OUTPUT_DIR}" \
    --rff_dimension ${RFF_DIMENSION} \
    --sigma_sample_size ${SIGMA_SAMPLE_SIZE} \
    --ridge_penalty ${RIDGE_PENALTY} \
    --regularization_lambda ${REGULARIZATION_LAMBDA} \
    --sample_number ${SAMPLE_NUMBER} \
    --random_seed ${RANDOM_SEED}

if [ $? -eq 0 ]; then
    echo ""
    echo "数据混合比例计算完成！"
    echo "结果已保存到: ${OUTPUT_DIR}"
    # echo ""
    # echo "查看结果文件:"
    # ls -la "${OUTPUT_DIR}"/mixing_ratios_${EXPERIMENT_NAME}_*.json
else
    echo ""
    echo "计算过程中出现错误！"
fi