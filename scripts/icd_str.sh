#!/bin/bash

# Start from parent directory of script
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -m experiments.icd_replacement --model_name "microsoft/BioGPT-Large-PubMedQA" --noise_level 0.1 --fact_data "icd-disease"

#python -m experiments.causal_trace --model_name "microsoft/BioGPT-Large-PubMedQA" --noise_level 0.1 --fact_data "pubmedqa"