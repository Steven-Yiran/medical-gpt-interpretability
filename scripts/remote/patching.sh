#! /bin/bash

MODEL_NAME=dmis-lab/meerkat-7b-v1.0
#MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
#DATA_PATH=./data/Mistral-7B-Instruct-v0.2_medqa-official_results.json
DATA_PATH=./data/meerkat-7b-v1.0_medqa-official_results.json
HOOK_POINT=$1

if [ "$HOOK_POINT" = "" ]; then
    # Loop through all module kinds if MODULE_KIND is NONE
    echo "Running patching for all hook points"
    python ./medqa/patching.py \
        --model_tag $MODEL_NAME \
        --data_path $DATA_PATH \
        --patching_batch_size 2 \
        --max_tokens_to_patch 15 \
        --max_layers_to_patch 20 \
        --cache \
        --results_dir ./results/meerkat-cot
else
    # Run with the specified module kind
    python ./medqa/patching.py \
     --model_tag $MODEL_NAME \
     --data_path $DATA_PATH \
     --hook_point $HOOK_POINT \
     --patching_batch_size 16 \
     --max_tokens_to_patch 13 \
     --max_layers_to_patch 20 \
     --cache \
     --plot \
     --results_dir ./results \
     --start_idx 0 \
     --end_idx 20
fi