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

from plotting import plot_patching_heatmap, plot_patching_heatmap_normalized
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


def run_patching_experiment(
    model, 
    tokenizer, 
    baseline_data,
    batch_size,
    max_tokens_to_patch,
    max_layers_to_patch,
    cache_patching_results=False,
    plot_patching_results=False,
    normalize_patching_results=False,
    modelname=None,
    module_kind=None,
    results_dir=None,
    low_memory=False,
    start_idx=0,
    end_idx=None,
):
    gender_condition_filter = GenderConditionFilter()
    patient_gender_filter = PatientInfoFilter()

    skipped_incorrect_answer = 0
    skipped_gender_condition = 0
    skipped_no_patient_info = 0
    skipped_none_prompt = 0
    skipped_small_logit_diff = 0
    skipped_negative_clean_logit_diff = 0

    if end_idx is None:
        end_idx = len(baseline_data)
    
    print(f"Processing examples from {start_idx} to {end_idx}")
    
    i = start_idx
    while i < end_idx:
        # Clear CUDA cache at the start of each iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        result_filepath = f"{results_dir}/{i}/{modelname}_{module_kind}"
        if os.path.exists(f"{result_filepath}.pdf"):
            print(f"Skipping example {i} because results already exist")
            i += 1
            continue
        print(f"\nProcessing example {i}")

        try:
            baseline_prompt = baseline_data[i]['generated_response']
            baseline_prompt = setup_prompt(baseline_prompt, modelname)

            if baseline_prompt is None:
                print(f"Skipping example {i} because the prompt is None")
                skipped_none_prompt += 1
                i += 1
                continue

            patient_gender = patient_gender_filter.extract_gender(baseline_prompt)
            if patient_gender is None:
                print(f"Skipping example {i} because did not find patient information")
                skipped_no_patient_info += 1
                i += 1
                continue
            # if gender_condition_filter.filter_text(baseline_prompt):
            #     print(f"Skipping example {i} because it contains gender-specific medical conditions")
            #     skipped_gender_condition += 1
            #     i += 1
            #     continue

            counterfactual_prompt = generate_counterfactual_patient_info(
                baseline_prompt,
                patient_gender,
                swap_gender=True,
                swap_pronouns=False
            )
            answer = chr(ord('A') + baseline_data[i]['gold'])
            counterfactual_answer = ["A", "B", "C", "D"]
            counterfactual_answer.remove(answer)
            answers = [answer] + counterfactual_answer
            answer_token_indices = torch.tensor([[model.to_single_token(answer) for answer in answers]], device="cuda")

            if baseline_prompt is None or counterfactual_prompt is None:
                print(f"Skipping example {i} because the answer is not in the prompt")
                skipped_incorrect_answer += 1
                i += 1
                continue
            
            clean_tokens = tokenizer.encode(baseline_prompt, return_tensors="pt").to("cuda")
            corrupted_tokens = tokenizer.encode(counterfactual_prompt, return_tensors="pt").to("cuda")

            assert len(clean_tokens[0]) == len(corrupted_tokens[0]), "length mismatch, corrupted tokens are not the same length as clean tokens"

            def names_filter(name):
                if module_kind == "resid":
                    return "hook_resid_pre" in name
                elif module_kind == "mlp":
                    return "mlp.hook_out" in name
                elif module_kind == "attn":
                    return "hook_attn_out" in name
                else:
                    return (
                        "hook_resid_pre" in name or
                        "mlp.hook_out" in name or
                        "hook_attn_out" in name
                    )

            # Run with cache and immediately delete unused tensors
            clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=names_filter)
            corrupted_logits = model(corrupted_tokens)

            # Clear cache after getting initial results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
            corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
            print(f"Total patching effect: {clean_logit_diff - corrupted_logit_diff}")

            if clean_logit_diff < 0:
                print(f"Skipping example {i} because the clean logit diff is negative, meaning wrong answer is more likely")
                skipped_negative_clean_logit_diff += 1
                del clean_logits, clean_cache, corrupted_logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                i += 1
                continue

            if abs(clean_logit_diff - corrupted_logit_diff) < 1e-4:
                print(f"Skipping example {i} because the total patching effect is too small")
                skipped_small_logit_diff += 1
                del clean_logits, clean_cache, corrupted_logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                i += 1
                continue

            def corruption_metric(logits, answer_token_indices=answer_token_indices):
                return get_logit_diff(logits, answer_token_indices)

            def patching_hook(
                activations,
                hook,
                position
            ):
                clean_activations = clean_cache[hook.name]
                activations[:, position, :] = clean_activations[:, position, :]
                return activations

            patching_results = torch.zeros((max_layers_to_patch, max_tokens_to_patch), device="cuda")
            
            # Process layers in smaller batches to manage memory
            for layer_batch_start in tqdm(range(0, max_layers_to_patch, batch_size)):
                layer_batch_end = min(layer_batch_start + batch_size, max_layers_to_patch)
                for layer_idx in range(layer_batch_start, layer_batch_end):            
                    for position in range(0, max_tokens_to_patch):
                        fwd_hooks = []
                        temp_hook_fn = partial(patching_hook, position=position)
                        if module_kind == "resid":
                            hook_point = f"blocks.{layer_idx}.hook_resid_pre"
                            fwd_hooks.append((hook_point, temp_hook_fn))
                        elif module_kind == "mlp":
                            hook_point = f"blocks.{layer_idx}.mlp.hook_out"
                            fwd_hooks.append((hook_point, temp_hook_fn))
                        elif module_kind == "attn":
                            hook_point = f"blocks.{layer_idx}.hook_attn_out"
                            fwd_hooks.append((hook_point, temp_hook_fn))

                        patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=fwd_hooks)
                        patched_logit_diff = corruption_metric(patched_logits, answer_token_indices).detach()
                        patching_results[layer_idx, position] = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                        
                        # Clear memory after each position
                        del patched_logits
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                # Clear memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            if normalize_patching_results:
                min_val = patching_results.min()
                max_val = patching_results.max()
                patching_results = (patching_results - min_val) / (max_val - min_val)
            
            os.makedirs(f"{results_dir}/{i}", exist_ok=True)

            if cache_patching_results:
                torch.save(patching_results, f"{result_filepath}.pt")

            # Plot the heatmap
            if plot_patching_results:
                if normalize_patching_results:
                    plot_patching_heatmap_normalized(
                        patching_results,
                        clean_tokens,
                        tokenizer,
                        module_kind=module_kind,
                        answer=answer,
                        filepath=f"{result_filepath}.pdf",
                        modelname=modelname
                    )
                else:
                    plot_patching_heatmap(
                        patching_results,
                        clean_tokens,
                        tokenizer,
                        module_kind=module_kind,
                        answer=answer,
                        filepath=f"{result_filepath}.pdf",
                        modelname=modelname
                    )

            # Clean up all tensors at the end of each iteration
            del clean_logits, clean_cache, corrupted_logits, patching_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            i += 1

    print(f"Skipped due to incorrect answer: {skipped_incorrect_answer}")
    print(f"Skipped due to gender condition: {skipped_gender_condition}")
    print(f"Skipped due to no patient info: {skipped_no_patient_info}")
    print(f"Skipped due to small logit diff: {skipped_small_logit_diff}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--patching_batch_size", type=int, default=4)
    parser.add_argument("--max_tokens_to_patch", type=int, default=1500)
    parser.add_argument("--max_layers_to_patch", type=int, default=None)
    parser.add_argument("--module_kind", type=str, default="resid", choices=["resid", "attn", "mlp"])
    parser.add_argument("--cache", "-c", action="store_true")
    parser.add_argument("--plot", "-p", action="store_true")
    parser.add_argument("--normalize", "-n", action="store_true")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--low_memory", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_tag)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_tag, torch_dtype=torch.bfloat16)

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
    if args.max_layers_to_patch is None:
        args.max_layers_to_patch = model.cfg.n_layers
    model_name = args.model_tag.split("/")[-1].split("-")[0]

    model = model.to("cuda")

    with open(args.data_path, "r") as f:
        baseline_data = json.load(f)
    print(f"Baseline data: {len(baseline_data)}")

    run_patching_experiment(
        model,
        tokenizer,
        baseline_data,
        batch_size=args.patching_batch_size,
        max_tokens_to_patch=args.max_tokens_to_patch,
        max_layers_to_patch=args.max_layers_to_patch,
        cache_patching_results=args.cache,
        plot_patching_results=args.plot,
        normalize_patching_results=args.normalize,
        modelname=model_name,
        module_kind=args.module_kind,
        results_dir=args.results_dir,
        low_memory=args.low_memory,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    

if __name__ == "__main__":
    main()
