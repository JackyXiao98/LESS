#!/bin/bash

# 验证dolly数据集ID提取的运行脚本
# 使用方法: ./run_validate_dolly_id.sh [GPU数量] [最大样本数]

# 默认参数
NUM_GPUS=${1:-8}
MAX_SAMPLES=${2:-100}
DATA_FILE="../data/train/processed/dolly/dolly_data.jsonl"
OUTPUT_FILE="dolly_id_validation_results.json"
MASTER_PORT=${MASTER_PORT:-17813}

echo "开始验证dolly数据集ID提取"
echo "GPU数量: $NUM_GPUS"
echo "最大样本数: $MAX_SAMPLES"
echo "数据文件: $DATA_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "主端口: $MASTER_PORT"

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "错误: 数据文件不存在: $DATA_FILE"
    echo "请确保数据文件路径正确"
fi

# 检查GPU数量
if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt 8 ]; then
    echo "错误: GPU数量必须在1-8之间"
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "开始验证..."

# 根据GPU数量选择运行方式
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "使用单GPU模式"
    python test/validate_dolly_id.py \
        --data_file "$DATA_FILE" \
        --max_samples "$MAX_SAMPLES" \
        --output_file "$OUTPUT_FILE"
else
    echo "使用多GPU分布式模式"
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        test/validate_dolly_id.py \
        --data_file "$DATA_FILE" \
        --max_samples "$MAX_SAMPLES" \
        --output_file "$OUTPUT_FILE"
fi

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 验证完成！"
    echo "结果文件: $OUTPUT_FILE"
    
    # 如果结果文件存在，显示简要摘要
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "验证摘要:"
        python -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        results = json.load(f)
    
    total_processed = sum(r['processed_batches'] for r in results)
    total_correct = sum(r['correct_id_mappings'] for r in results)
    total_incorrect = sum(r['incorrect_id_mappings'] for r in results)
    
    print(f'  总处理样本: {total_processed}')
    print(f'  正确映射: {total_correct}')
    print(f'  错误映射: {total_incorrect}')
    
    if total_processed > 0:
        rate = total_correct / total_processed
        print(f'  正确率: {rate:.2%}')
        print(f'  验证结果: {\"✓ 通过\" if total_incorrect == 0 else \"✗ 失败\"}')
except Exception as e:
    print(f'  无法解析结果文件: {e}')
"
    fi
else
    echo ""
    echo "✗ 验证失败！"
    echo "请检查错误信息并重试"
fi