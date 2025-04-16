import os
import sys
import re
import argparse
import json
import random
from functools import partial
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt


import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix
import transformer_lens.patching as patching

from prompts import prompt_eval_bare, meerkat_medqa_system_prompt
from plotting import plot_patching_heatmap
from utils import *


def get_logit_diff(logits, answer_token_indices):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1:])
    # get the min of the incorrect logits
    incorrect_logits = incorrect_logits.mean(dim=1)
    return (correct_logits - incorrect_logits).mean()


def run_activation_patching(
    model, 
    tokenizer, 
    baseline_data, 
    max_tokens,
    cache_patching_results=False,
    plot_patching_results=False,
    modelname="Meerkat-7B"
):
    gender_condition_filter = GenderConditionFilter()
    patient_gender_filter = PatientInfoFilter()

    skipped_incorrect_answer = 0
    skipped_long_prompt = 0
    skipped_gender_condition = 0
    skipped_no_patient_info = 0
    skipped_small_logit_diff = 0
    skipped_negative_clean_logit_diff = 0

    for i in range(len(baseline_data)):
        if os.path.exists(f"../results/patching_results_{i}.pdf"):
            print(f"Skipping example {i} because results already exist")
            continue
        print(f"Processing example {i}")

        baseline_prompt = baseline_data[i]['generated_response']
        baseline_prompt = setup_prompt(baseline_prompt)

        if baseline_prompt is None:
            print(f"Skipping example {i} because the prompt is None")
            continue

        patient_gender = patient_gender_filter.extract_gender(baseline_prompt)
        if patient_gender is None:
            print(f"Skipping example {i} because did not find patient information")
            skipped_no_patient_info += 1
            continue
        # if gender_condition_filter.filter_text(baseline_prompt):
        #     print(f"Skipping example {i} because it contains gender-specific medical conditions")
        #     skipped_gender_condition += 1
        #     continue

        counterfactual_prompt = generate_counterfactual_patient_info(baseline_prompt, patient_gender, swap_gender=True, swap_pronouns=True)       
        answer = chr(ord('A') + baseline_data[i]['gold'])
        counterfactual_answer = ["A", "B", "C", "D"]
        counterfactual_answer.remove(answer)
        answers = [answer] + counterfactual_answer
        answer_token_indices = torch.tensor([[model.to_single_token(answer) for answer in answers]], device="cuda")

        if baseline_prompt is None or counterfactual_prompt is None:
            print(f"Skipping example {i} because the answer is not in the prompt")
            skipped_incorrect_answer += 1
            continue

        clean_tokens = tokenizer.encode(baseline_prompt, return_tensors="pt").to("cuda")
        corrupted_tokens = tokenizer.encode(counterfactual_prompt, return_tensors="pt").to("cuda")

        assert len(clean_tokens[0]) == len(corrupted_tokens[0]), "length mismatch, corrupted tokens are not the same length as clean tokens"

        if len(clean_tokens[0]) > max_tokens:
            print(f"Skipping example {i} because the prompt is too long")
            skipped_long_prompt += 1
            continue

        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits = model(corrupted_tokens)

        clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
        corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
        print(f"Clean logit diff: {clean_logit_diff}")
        print(f"Corrupted logit diff: {corrupted_logit_diff}")

        if clean_logit_diff < 0:
            print(f"Skipping example {i} because the clean logit diff is negative, meaning wrong answer is more likely")
            skipped_negative_clean_logit_diff += 1
            continue

        if abs(clean_logit_diff - corrupted_logit_diff) < 1e-3:
            print(f"Skipping example {i} because the logit diff is too small")
            skipped_small_logit_diff += 1
            continue

        def corruption_metric(logits, answer_token_indices=answer_token_indices):
            return get_logit_diff(logits, answer_token_indices)

        def residual_stream_patching_hook(
            resid_pre,
            hook,
            position
        ):
            clean_resid_pre = clean_cache[hook.name]
            resid_pre[:, position, :] = clean_resid_pre[:, position, :]
            return resid_pre

        patching_results = torch.zeros((model.cfg.n_layers, len(clean_tokens[0])), device="cuda")

        # Process each position for each layer
        for layer in tqdm(range(model.cfg.n_layers)):
            for position in range(0, len(clean_tokens[0])):
                temp_hook_fn = partial(residual_stream_patching_hook, position=position)
                patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[
                    (utils.get_act_name("resid_pre", layer), temp_hook_fn)
                ])
                patched_logit_diff = corruption_metric(patched_logits, answer_token_indices).detach()
                patching_results[layer, position] = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

        # normalize the patching results
        patching_results = (patching_results - patching_results.min(dim=1, keepdim=True).values) / (patching_results.max(dim=1, keepdim=True).values - patching_results.min(dim=1, keepdim=True).values)

        if cache_patching_results:
            torch.save(patching_results, f"../results/patching_results_{i}.pt")

        # Plot the heatmap
        if plot_patching_results:
            plot_patching_heatmap(
                patching_results,
                clean_tokens,
                tokenizer,
                answer=answer,
                filepath=f"../results/patching_results_{i}.pdf",
                modelname=modelname
            )

    print(f"Skipped due to incorrect answer: {skipped_incorrect_answer}")
    print(f"Skipped due to long prompt: {skipped_long_prompt}")
    print(f"Skipped due to gender condition: {skipped_gender_condition}")
    print(f"Skipped due to no patient info: {skipped_no_patient_info}")
    print(f"Skipped due to small logit diff: {skipped_small_logit_diff}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="dmis-lab/meerkat-7b-v1.0")
    parser.add_argument("--tokenizer_name", type=str, default="dmis-lab/meerkat-7b-v1.0")
    parser.add_argument("--data_path", type=str, default="../data/meerkat-7b-v1.0_medqa-original_results.json")
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--cache_patching_results", type=bool, default=True)
    parser.add_argument("--plot_patching_results", type=bool, default=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    model = HookedTransformer.from_pretrained(
        "mistral-7b",
        hf_model=hf_model,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=torch.bfloat16,
        device="cuda",
        tokenizer=tokenizer,
        use_checkpointing=True,  # Enable gradient checkpointing
    )

    model = model.to("cuda")

    with open(args.data_path, "r") as f:
        baseline_data = json.load(f)
    print(f"Baseline data: {len(baseline_data)}")

    run_activation_patching(model, tokenizer, baseline_data, args.max_tokens, cache_patching_results=args.cache_patching_results, plot_patching_results=args.plot_patching_results)
    
if __name__ == "__main__":
    main()
