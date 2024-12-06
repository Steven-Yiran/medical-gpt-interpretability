#!/bin/bash

# BioGPT Large PubMedQA
python -m experiments.activation_patching --model_name "microsoft/BioGPT-Large-PubMedQA" --fact_data "icd-disease" --method "STR"

# GPT2 Large
#python -m experiments.activation_patching --model_name "openai-community/gpt2-large" --fact_data "icd-disease" --method "STR"

# microsoft/BioGPT-Large
#python -m experiments.activation_patching --model_name "microsoft/BioGPT-Large" --fact_data "icd-disease" --method "STR"