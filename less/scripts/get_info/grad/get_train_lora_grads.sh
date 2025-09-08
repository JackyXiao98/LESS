#!/bin/bash
# 用法: bash run.sh <train_file> <model> <output_path> <dims> <gradient_type> [gpu_id]

train_file=$1          # path/to/train.jsonl
model=$2               # path to model
output_path=$3         # path to output
dims=$4                # dimension of projection, can be a list
gradient_type=$5       # e.g., grads/full 等
GPU_ID=${6:-0}         # 可选, 默认用 0 号卡

# 只暴露一个 GPU 给进程（注意: 此时 python 里的 cuda:0 就是这张卡）
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID

# (可选) 快速打印一下可见 GPU，确认只剩 1 块
echo "Using GPU_ID=$GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
python3 - <<'PY'
import os, torch
print("Visible:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
PY

# 创建输出目录
if [[ ! -d $output_path ]]; then
    mkdir -p "$output_path"
fi

python3 -u -m less.data_selection.get_info \
  --train_file "$train_file" \
  --info_type grads \
  --model_path "$model" \
  --output_path "$output_path" \
  --gradient_projection_dimension "$dims" \
  --gradient_type "$gradient_type"
