DATA_DIR=/mnt/hdfs/selection/yingtai_sft/tulu_3_by_source
MODEL_PATH=meta-llama/Llama-2-7b-hf
PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=3
JOB_NAME=tulu3-llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo $LD_LIBRARY_PATH

./less/scripts/train/warmup_lora_train_qwen.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"