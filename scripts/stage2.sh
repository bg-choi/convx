#!/usr/bin/env bash
STAGE=$1
OUTPUT_DIR=$2
MODEL_NAME_OR_PATH=$3
MEM_SIZE=$4

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/$STAGE"
fi
if [ -z "$MODEL_NAME_OR_PATH" ]; then
  MODEL_NAME_OR_PATH="mistralai/Mistral-7B-Instruct-v0.2"
fi
if [ -z "$MEM_SIZE" ]; then
  MEM_SIZE=16
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data_sample/"
fi
if [ -z "$TRAIN_FILE" ]; then
  TRAIN_FILE="train.jsonl" # v6 to Random
fi

mkdir -p "${OUTPUT_DIR}"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
torchrun --nproc_per_node ${PROC_PER_NODE} src/$STAGE/train.py \
    --do_train \
    --seed 1234 \
    --bf16 \
    --stage $STAGE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --train_file "${DATA_DIR}"/"${TRAIN_FILE}" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --mem_size $MEM_SIZE \
    --p_max_len 256 \
    --dataloader_num_workers 0 \
    --learning_rate 2e-4 \
    --lr_scheduler_type 'linear' \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_total_limit 3 \
    --save_strategy steps \
    --save_steps 100 \
    --max_steps 10000 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm False \
    --report_to none "$@"
