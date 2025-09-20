#!/bin/bash

CKPT=296
DATA_DIR=/mnt/hdfs/selection/yingtai_sft/tulu_3_by_source
GRADIENT_TYPE="adam"
MODEL_PATH=/mnt/bn/pilab0/yt/github/out/tulu3-Qwen3-4B-p0.02-lora-seed3/checkpoint-${CKPT}
DIMS="8192"

# 获取DATA_DIR下所有的parquet文件
parquet_files=($(find "$DATA_DIR" -name "*.parquet" -type f))

echo "找到 ${#parquet_files[@]} 个parquet文件"

# 遍历每个parquet文件
for i in "${!parquet_files[@]}"; do
    TRAINING_DATA_FILE="${parquet_files[$i]}"
    
    # 从文件路径中提取文件名（不含路径和扩展名）用于输出路径
    filename=$(basename "$TRAINING_DATA_FILE" .parquet)
    
    OUTPUT_PATH=/mnt/hdfs/selection/yingtai_sft/lora_grads/tulu3-Qwen3-4B-p0.02-lora-seed3/${filename}-ckpt${CKPT}-${GRADIENT_TYPE}
    
    echo "处理文件 $((i+1))/${#parquet_files[@]}: $filename"
    echo "输入文件: $TRAINING_DATA_FILE"
    echo "输出路径: $OUTPUT_PATH"
    echo "开始处理..."
    
    # 运行梯度计算脚本
    ./less/scripts/get_info/grad/get_train_lora_grads_multi_gpu.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
    
    echo "文件 $filename 处理完成"
    echo "----------------------------------------"
done

echo "所有 ${#parquet_files[@]} 个parquet文件处理完成！"