#! /bin/bash

MODEL_NAME=dmis-lab/meerkat-7b-v1.0
MAX_TOKENS=1000
DO_INFERENCE=false

if [ "$DO_INFERENCE" = true ]; then
    INFERENCE_FLAG=--inference
    echo "Evaluating $MODEL_NAME with $MAX_TOKENS tokens"
else
    INFERENCE_FLAG=""
fi

python evaluation.py \
 --model_name $MODEL_NAME \
 --max_tokens $MAX_TOKENS \
 $INFERENCE_FLAG