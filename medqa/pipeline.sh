#! /bin/bash

#MODEL_NAME=dmis-lab/meerkat-7b-v1.0
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
MAX_TOKENS=10
DO_INFERENCE=true
DATASET_NAME=medqa-official

if [ "$DO_INFERENCE" = true ]; then
    INFERENCE_FLAG=--inference
    echo "Inferring with $MODEL_NAME with $MAX_TOKENS tokens"
else
    INFERENCE_FLAG=""
fi

# python evaluation.py \
#  --model_name $MODEL_NAME \
#  --max_tokens $MAX_TOKENS \
#  --dataset_name $DATASET_NAME \
#  $INFERENCE_FLAG

python patching.py \
 --model_name $MODEL_NAME \
 --tokenizer_name $MODEL_NAME \
 --data_path ../data/Mistral-7B-Instruct-v0.2_medqa-official_results.json \
 --cache_patching_results \
 --plot_patching_results