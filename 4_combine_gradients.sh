#!/bin/bash

# 梯度聚合脚本：对所有checkpoint文件夹运行梯度聚合算法
# 基于 aggregate_gradients.py 脚本

BASE_PATH="/mnt/hdfs/selection/yingtai_sft/lora_grads"
EXPERIMENT_NAME="tulu3-Qwen3-8B-p0.05-lora-seed3"
NUM_GPUS=8
DIM=8192

# 定义所有需要处理的checkpoint文件夹名称
CHECKPOINT_NAMES=(
    "ai2-adapt-dev_coconot_converted-ckpt368-adam"
    "ai2-adapt-dev_evol_codealpaca_heval_decontaminated-ckpt368-adam"
    "ai2-adapt-dev_flan_v2_converted-ckpt368-adam"
    "ai2-adapt-dev_no_robots_converted-ckpt368-adam"
    "ai2-adapt-dev_numinamath_tir_math_decontaminated-ckpt368-adam"
    "ai2-adapt-dev_oasst1_converted-ckpt368-adam"
    "ai2-adapt-dev_personahub_code_v2_34999-ckpt368-adam"
    "ai2-adapt-dev_personahub_ifdata_manual_seed_v3_29980-ckpt368-adam"
    "ai2-adapt-dev_personahub_math_v5_regen_149960-ckpt368-adam"
    "ai2-adapt-dev_tulu_hard_coded_repeated_10-ckpt368-adam"
    "ai2-adapt-dev_tulu_v3.9_aya_100k-ckpt368-adam"
    "ai2-adapt-dev_tulu_v3.9_open_math_2_gsm8k_50k-ckpt368-adam"
    "ai2-adapt-dev_tulu_v3.9_personahub_math_interm_algebra_20k-ckpt368-adam"
    "ai2-adapt-dev_tulu_v3.9_sciriff_10k-ckpt368-adam"
    "ai2-adapt-dev_tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k-ckpt368-adam"
    "ai2-adapt-dev_tulu_v3.9_table_gpt_5k-ckpt368-adam"
    "ai2-adapt-dev_tulu_v3.9_wildchat_100k-ckpt368-adam"
    "ai2-adapt-dev_tulu_v3.9_wildjailbreak_decontaminated_50k-ckpt368-adam"
    "allenai_tulu-3-sft-personas-math-grade-ckpt368-adam"
)

echo "开始处理 ${#CHECKPOINT_NAMES[@]} 个checkpoint文件夹的梯度聚合..."
echo "基础路径: $BASE_PATH"
echo "实验名称: $EXPERIMENT_NAME"
echo "GPU数量: $NUM_GPUS"
echo "梯度维度: $DIM"
echo "=========================================="

# 记录开始时间
START_TIME=$(date)
echo "开始时间: $START_TIME"

# 循环处理每个checkpoint
for i in "${!CHECKPOINT_NAMES[@]}"; do
    CHECKPOINT_NAME="${CHECKPOINT_NAMES[$i]}"
    
    echo ""
    echo "处理进度: $((i+1))/${#CHECKPOINT_NAMES[@]}"
    echo "当前处理: $CHECKPOINT_NAME"
    echo "----------------------------------------"
    
    # 检查输入路径是否存在
    INPUT_PATH="$BASE_PATH/$EXPERIMENT_NAME/$CHECKPOINT_NAME"
    if [ ! -d "$INPUT_PATH" ]; then
        echo "警告: 输入路径不存在，跳过: $INPUT_PATH"
        continue
    fi
    
    # 运行梯度聚合脚本
    echo "开始聚合梯度..."
    python aggregate_gradients.py \
        --base_path "$BASE_PATH" \
        --experiment_name "$EXPERIMENT_NAME" \
        --checkpoint_name "$CHECKPOINT_NAME" \
        --num_gpus $NUM_GPUS \
        --dim $DIM
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 成功完成: $CHECKPOINT_NAME"
    else
        echo "✗ 执行失败: $CHECKPOINT_NAME"
    fi
    
    echo "----------------------------------------"
done

# 记录结束时间
END_TIME=$(date)
echo ""
echo "=========================================="
echo "所有任务完成!"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "总共处理了 ${#CHECKPOINT_NAMES[@]} 个checkpoint文件夹"

# 生成汇总报告
echo ""
echo "生成汇总报告..."
SUMMARY_FILE="gradient_aggregation_summary_$(date +%Y%m%d_%H%M%S).txt"

cat > "$SUMMARY_FILE" << EOF
梯度聚合任务汇总报告
====================

执行时间: $START_TIME - $END_TIME
基础路径: $BASE_PATH
实验名称: $EXPERIMENT_NAME
GPU数量: $NUM_GPUS
梯度维度: $DIM

处理的Checkpoint列表:
EOF

for i in "${!CHECKPOINT_NAMES[@]}"; do
    CHECKPOINT_NAME="${CHECKPOINT_NAMES[$i]}"
    OUTPUT_PATH="$BASE_PATH/$EXPERIMENT_NAME/$CHECKPOINT_NAME/dim$DIM"
    
    if [ -f "$OUTPUT_PATH/all_orig.pt" ] && [ -f "$OUTPUT_PATH/all_ids.pkl" ]; then
        STATUS="✓ 成功"
        # 尝试获取文件大小信息
        GRAD_SIZE=$(ls -lh "$OUTPUT_PATH/all_orig.pt" 2>/dev/null | awk '{print $5}' || echo "未知")
        echo "$((i+1)). $CHECKPOINT_NAME - $STATUS (梯度文件: $GRAD_SIZE)" >> "$SUMMARY_FILE"
    else
        STATUS="✗ 失败或未完成"
        echo "$((i+1)). $CHECKPOINT_NAME - $STATUS" >> "$SUMMARY_FILE"
    fi
done

echo "" >> "$SUMMARY_FILE"
echo "汇总报告已保存到: $SUMMARY_FILE"
echo "脚本执行完成!"