#! /bin/bash

#MODEL_NAME=dmis-lab/meerkat-7b-v1.0
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
MAX_TOKENS=1000
DO_INFERENCE=true
DATASET_NAME=medqa-official

if [ "$DO_INFERENCE" = true ]; then
    INFERENCE_FLAG=--inference
    echo "Inferring with $MODEL_NAME with $MAX_TOKENS tokens"
else
    INFERENCE_FLAG=""
fi

python evaluation.py \
 --model_name $MODEL_NAME \
 --max_tokens $MAX_TOKENS \
 --dataset_name $DATASET_NAME \
 $INFERENCE_FLAG