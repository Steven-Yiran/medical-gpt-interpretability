#!/bin/bash
#SBATCH --job-name=medqa
#SBATCH --output=outputs/medqa/slurm_out/log_%j.out
#SBATCH --error=outputs/medqa/slurm_out/log_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

export DEV_HOME=/users/yshi28/dev/medical-gpt-interpretability

source $DEV_HOME/venv/bin/activate

#MODEL_NAME=dmis-lab/meerkat-7b-v1.0
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
DATA_PATH=$DEV_HOME/data/Mistral-7B-Instruct-v0.2_medqa-official_results.json
MODULE_KIND=attn
ENTRY_IDX=$1

python $DEV_HOME/medqa/patching.py \
 --model_tag $MODEL_NAME \
 --data_path $DATA_PATH \
 --module_kind $MODULE_KIND \
 --patching_batch_size 1 \
 --max_tokens_to_patch 4 \
 --max_layers_to_patch 17 \
 --cache \
 --plot \
 --results_dir $DEV_HOME/results \
 --start_idx $ENTRY_IDX \
 --end_idx $((ENTRY_IDX + 1))