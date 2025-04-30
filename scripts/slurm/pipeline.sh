#!/bin/bash
#SBATCH --job-name=medqa
#SBATCH --output=outputs/medqa/slurm_out/log_%j.out
#SBATCH --error=outputs/medqa/slurm_out/log_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

export DEV_HOME=/users/yshi28/dev/medical-gpt-interpretability

source $DEV_HOME/venv/bin/activate

MODEL_NAME=dmis-lab/meerkat-7b-v1.0
#MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
MAX_TOKENS=10
DO_INFERENCE=true
DATASET_NAME=medqa-official

if [ "$DO_INFERENCE" = true ]; then
    INFERENCE_FLAG=--inference
    echo "Inferring with $MODEL_NAME with $MAX_TOKENS tokens"
else
    INFERENCE_FLAG=""
fi

# python $DEV_HOME/medqa/evaluation.py \
#  --model_name $MODEL_NAME \
#  --max_tokens $MAX_TOKENS \
#  --dataset_name $DATASET_NAME \
#  $INFERENCE_FLAG

python $DEV_HOME/medqa/patching.py \
 --model_name $MODEL_NAME \
 --tokenizer_name $MODEL_NAME \
 --data_path GBaker/MedQA-USMLE-4-options-hf \
 --result_dir ./results \
 --cache_patching_results \
 --plot_patching_results