#!/bin/bash
#SBATCH --job-name=medqa
#SBATCH --output=outputs/medqa/slurm_out/log_%a.out
#SBATCH --error=outputs/medqa/slurm_out/log_%a.err
#SBATCH --array=0-35%36
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --cpus-per-task=1

export DEV_HOME=/users/yshi28/dev/medical-gpt-interpretability

source $DEV_HOME/venv/bin/activate

python $DEV_HOME/medqa/evaluation.py --model_name dmis-lab/meerkat-7b-v1.0 --inference --max_tokens 1000