#!/bin/bash
# 用法: bash get_train_lora_grads_multi_gpu.sh <train_file> <model> <output_path> <dims> <gradient_type> [num_gpus] [master_port]

train_file=$1          # path/to/train.jsonl
model=$2               # path to model
output_path=$3         # path to output
dims=$4                # dimension of projection, can be a list
gradient_type=$5       # e.g., grads/full 等
NUM_GPUS=${6:-8}       # 可选, 默认使用8个GPU
MASTER_PORT=${7:-17238} # 可选, 默认端口17238

# 检查GPU数量
available_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ $NUM_GPUS -gt $available_gpus ]; then
    echo "Error: Requested $NUM_GPUS GPUs but only $available_gpus available"
    exit 1
fi

# 设置环境变量
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"
echo "Master port: $MASTER_PORT"

# 打印GPU信息
python3 - <<'PY'
import os, torch
print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
PY

# 创建输出目录
if [[ ! -d $output_path ]]; then
    mkdir -p "$output_path"
fi

# 启动多GPU训练 - 使用torchrun
torchrun --nproc_per_node=$NUM_GPUS \
         --master_port=$MASTER_PORT \
         --nnodes=1 \
         --node_rank=0 \
         -m less.data_selection.get_info \
  --train_file "$train_file" \
  --info_type grads \
  --model_path "$model" \
  --output_path "$output_path" \
  --gradient_projection_dimension "$dims" \
  --gradient_type "$gradient_type"

echo "Multi-GPU gradient calculation completed!"
echo "Results saved in: $output_path"
echo "Each GPU's results are in separate rank_* subdirectories"