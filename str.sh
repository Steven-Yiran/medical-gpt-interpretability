#!/bin/bash

python -m experiments.activation_patching --model_name "microsoft/BioGPT-Large-PubMedQA" --fact_data "icd-disease" --method "GN"