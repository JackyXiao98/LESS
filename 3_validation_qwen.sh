#!/bin/bash

CKPT=368
DATA_DIR=./data_align/build_val/tulu3_validation
GRADIENT_TYPE="sgd"
MODEL_NAME="tulu3-Qwen3-8B-p0.05-lora-seed3"
MODEL_PATH=/mnt/bn/pilab0/yt/github/out/${MODEL_NAME}/checkpoint-${CKPT}
DIMS="8192"

# 获取DATA_DIR下所有子文件夹中的validation.jsonl文件
validation_files=($(find "$DATA_DIR" -name "validation.jsonl" -type f))

echo "找到 ${#validation_files[@]} 个validation.jsonl文件"

# 遍历每个validation.jsonl文件
for i in "${!validation_files[@]}"; do
    TRAINING_DATA_FILE="${validation_files[$i]}"
    
    # 从文件路径中提取文件夹名称用于输出路径
    dataset_dir=$(dirname "$TRAINING_DATA_FILE")
    filename=$(basename "$dataset_dir")
    
    OUTPUT_PATH=/mnt/hdfs/selection/yingtai_sft/lora_val_grads/${MODEL_NAME}/${filename}-ckpt${CKPT}-${GRADIENT_TYPE}
    
    echo "处理文件 $((i+1))/${#validation_files[@]}: $filename"
    echo "输入文件: $TRAINING_DATA_FILE"
    echo "输出路径: $OUTPUT_PATH"
    echo "开始处理..."
    
    # 运行梯度计算脚本
    ./less/scripts/get_info/grad/get_train_lora_grads_multi_gpu.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
    
    echo "文件 $filename 处理完成"
    echo "----------------------------------------"
done

echo "所有 ${#validation_files[@]} 个validation.jsonl文件处理完成！"