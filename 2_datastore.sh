CKPT=1692

TRAINING_DATA_NAME=dolly
TRAINING_DATA_FILE=./data/train/processed/dolly/dolly_data.jsonl # when changing data name, change the data path accordingly
GRADIENT_TYPE="adam"
MODEL_PATH=../out/llama2-7b-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-p0.05-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"