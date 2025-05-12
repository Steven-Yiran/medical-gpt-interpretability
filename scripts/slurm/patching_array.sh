#!/bin/bash
#SBATCH --job-name=medqa_patch
#SBATCH --output=outputs/medqa/slurm_out/patch_%A_%a.out
#SBATCH --error=outputs/medqa/slurm_out/patch_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --array=0-1%2  # Run 2 jobs, max 2 at a time

export DEV_HOME=/users/yshi28/dev/medical-gpt-interpretability

source $DEV_HOME/venv/bin/activate

MODEL_NAME=dmis-lab/meerkat-7b-v1.0
MODULE_KIND=attn
DATA_PATH=$DEV_HOME/data/Mistral-7B-Instruct-v0.2_medqa-official_results.json
RESULTS_DIR=$DEV_HOME/results

# Calculate the number of examples per job
TOTAL_EXAMPLES=$(python -c "import json; f=open('$DATA_PATH'); data=json.load(f); print(len(data))")
EXAMPLES_PER_JOB=$((TOTAL_EXAMPLES / 10))  # Divide into 10 jobs
START_IDX=$((SLURM_ARRAY_TASK_ID * EXAMPLES_PER_JOB))
END_IDX=$((START_IDX + EXAMPLES_PER_JOB))

# For the last job, make sure we process all remaining examples
if [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    END_IDX=$TOTAL_EXAMPLES
fi

echo "Processing examples from $START_IDX to $END_IDX"

python $DEV_HOME/medqa/patching.py \
    --model_tag $MODEL_NAME \
    --data_path $DATA_PATH \
    --module_kind $MODULE_KIND \
    --patching_batch_size 4 \
    --max_tokens_to_patch 11 \
    --max_layers_to_patch 32 \
    --cache \
    --plot \
    --results_dir $RESULTS_DIR \
    --start_idx $START_IDX \
    --end_idx $END_IDX 