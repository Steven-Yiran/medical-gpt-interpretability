# Medical LM Interpretability

## Background

- Meerkat-7B
- Mistral-7B

## Run Experiments

Activate environment and install required dependencies.
```bash
pip install -r requirements.txt
```
If running on cluster managed by Slurm
```bash
sbatch scripts/slurm/pipeline.sh
```
otherwise,
```bash
bash scripts/slurm/pipeline.sh
```