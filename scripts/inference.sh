#!/usr/bin/env bash

STAGE=$1
MODEL_NAME_OR_PATH=$2
SEARCH_FILE=$3
N_PASSAGES=$4
BATCH_SIZE=$5

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$N_PASSAGES" ]; then
    N_PASSAGES=5
fi
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=8
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data_sample/"
fi

PYTHONPATH=src/$STAGE/ python -u src/$STAGE/inferences/generate.py \
    --bf16 \
    --stage $STAGE \
    --file_name "${SEARCH_FILE}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_eval_batch_size $BATCH_SIZE \
    --n_passages $N_PASSAGES \
    --p_max_len 256 \
    --dataloader_num_workers 1 \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@"