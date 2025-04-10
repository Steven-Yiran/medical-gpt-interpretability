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
from filter import *


def assert_logits_match(model, hf_model, tokenizer):
    prompts = [
        "The capital of Germany is",
        "2 * 42 = ", 
        "My favorite", 
        "aosetuhaosuh aostud aoestuaoentsudhasuh aos tasat naostutshaosuhtnaoe usaho uaotsnhuaosntuhaosntu haouaoshat u saotheu saonuh aoesntuhaosut aosu thaosu thaoustaho usaothusaothuao sutao sutaotduaoetudet uaosthuao uaostuaoeu aostouhsaonh aosnthuaoscnuhaoshkbaoesnit haosuhaoe uasotehusntaosn.p.uo ksoentudhao ustahoeuaso usant.hsa otuhaotsi aostuhs",
    ]

    model.eval()
    hf_model.eval()
    prompt_ids = [tokenizer.encode(prompt, return_tensors="pt") for prompt in prompts]

    tl_logits = [model(prompt_ids).detach().cpu() for prompt_ids in tqdm(prompt_ids)]
    logits = [hf_model(prompt_ids).logits.detach().cpu() for prompt_ids in tqdm(prompt_ids)]

    for i in range(len(prompts)):
        assert torch.allclose(logits[i], tl_logits[i], atol=1, rtol=1e-1)


def select_random_answer(correct_answer):
    wrong_answers = ['A', 'B', 'C', 'D']
    wrong_answers.remove(correct_answer)
    return random.choice(wrong_answers)

def get_logit_diff(logits, answer_token_indices):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

def truncate_answer_text(prompt):
    # truncate the prompt until the end of the pattern "the answer is ("
    # since there are multiple instances of this pattern in the prompt, we need to find the last one
    pattern = "the answer is ("
    # find the last instance of the pattern
    last_instance = prompt.rfind(pattern)
    if last_instance == -1:
        return None
    return prompt[:last_instance + len(pattern)]

def find_assistant_response(prompt, pattern="ASSISTANT:"):
    # find the last instance of the pattern
    last_instance = prompt.rfind(pattern)
    if last_instance == -1:
        return None
    return prompt[last_instance + len(pattern):]

def setup_prompt(prompt):
    prompt = truncate_answer_text(prompt)
    prompt = find_assistant_response(prompt)
    return prompt

def generate_counterfactual_patient_info(prompt, patient_gender, swap_gender=False, swap_pronouns=False):
    """
    Generate a counterfactual version of patient information by changing gender.
    
    Args:
        prompt (str): The original prompt containing patient information
        
    Returns:
        str: A counterfactual version of the prompt with gender changed
    """
    pronoun_map = {
        "male": {"he": "she", "his": "her", "him": "her", "He": "She", "His": "Her", "Him": "Her"},
        "female": {"she": "he", "her": "his", "hers": "his", "She": "He", "Her": "His", "Hers": "His"},
    }
    if swap_gender:
        if patient_gender == "male":
            prompt = re.sub(r"\bman\b", "woman", prompt)
            prompt = re.sub(r"\bmale\b", "female", prompt)
        else:
            prompt = re.sub(r"\bwoman\b", "man", prompt)
            prompt = re.sub(r"\bfemale\b", "male", prompt)
    if swap_pronouns:
        replace_map = pronoun_map[patient_gender]
        for old, new in replace_map.items():
            prompt = re.sub(r'\b' + re.escape(old) + r'\b', new, prompt)
    return prompt

def run_activation_patching(
    model, 
    tokenizer, 
    baseline_data, 
    max_tokens,
    cache_patching_results=False,
    modelname="Meerkat-7B"
):
    gender_condition_filter = GenderConditionFilter()
    patient_gender_filter = PatientInfoFilter()

    for i in range(len(baseline_data)):
        if os.path.exists(f"../results/patching_results_{i}.pdf"):
            print(f"Skipping example {i} because results already exist")
            continue
        print(f"Processing example {i}")

        baseline_prompt = baseline_data[i]['generated_response']
        baseline_prompt = setup_prompt(baseline_prompt)

        patient_gender = patient_gender_filter.extract_gender(baseline_prompt)
        if patient_gender is None:
            print(f"Skipping example {i} because did not find patient gender")
            continue
        if gender_condition_filter.filter_text(baseline_prompt):
            print(f"Skipping example {i} because it contains gender-specific medical conditions")
            continue

        counterfactual_prompt = generate_counterfactual_patient_info(baseline_prompt, patient_gender, swap_gender=True, swap_pronouns=True)       
        answer = chr(ord('A') + baseline_data[i]['gold'])
        counterfactual_answer = select_random_answer(answer)
        answers = [answer, counterfactual_answer]
        answer_token_indices = torch.tensor(
            [
                [model.to_single_token(answer) for answer in answers]
            ],
            device="cuda"
        )

        if baseline_prompt is None or counterfactual_prompt is None:
            print(f"Skipping example {i} because the answer is not in the prompt")
            continue

        clean_tokens = tokenizer.encode(baseline_prompt, return_tensors="pt").to("cuda")
        corrupted_tokens = tokenizer.encode(counterfactual_prompt, return_tensors="pt").to("cuda")

        assert len(clean_tokens[0]) == len(corrupted_tokens[0])

        if len(clean_tokens[0]) > max_tokens:
            print(f"Skipping example {i} because the prompt is too long")
            continue

        clean_logits, clean_cache = model.run_with_cache(clean_tokens)   
        corrupted_logits = model(corrupted_tokens)

        # Check that the top choice for the answer token is the same as the answer
        clean_logits_last_token = clean_logits[0, -1, :]
        top_token_id = torch.argmax(clean_logits_last_token).item()
        top_token = tokenizer.decode([top_token_id])
        
        if top_token != answer:
            print(f"Skipping example {i} because the top token '{top_token}' doesn't match the answer '{answer}'")
            continue

        clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
        corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
        print(f"Clean logit diff: {clean_logit_diff}")
        print(f"Corrupted logit diff: {corrupted_logit_diff}")

        if abs(clean_logit_diff - corrupted_logit_diff) < 1e-2:
            print(f"Skipping example {i} because the logit diff is too small")
            continue
        if corrupted_logit_diff > clean_logit_diff:
            print(f"Skipping example {i} because the corrupted logit diff is greater than the clean logit diff")
            continue

        def corruption_metric(logits, answer_token_indices=answer_token_indices):
            return (get_logit_diff(logits, answer_token_indices) - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

        patching_results = patching.get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, corruption_metric)   

        # def residual_stream_patching_hook(
        #     resid_pre,
        #     hook,
        #     position
        # ):
        #     clean_resid_pre = clean_cache[hook.name]
        #     resid_pre[:, position, :] = clean_resid_pre[:, position, :]
        #     return resid_pre

        # patching_results = torch.zeros((model.cfg.n_layers, len(clean_tokens[0])), device="cuda")

        # # Process each position for each layer
        # for layer in tqdm(range(model.cfg.n_layers)):
        #     for position in range(0, len(clean_tokens[0])):
        #         temp_hook_fn = partial(residual_stream_patching_hook, position=position)
        #         patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[
        #             (utils.get_act_name("resid_pre", layer), temp_hook_fn)
        #         ])
        #         patched_logit_diff = corruption_metric(patched_logits, answer_token_indices).detach()
        #         patching_results[layer, position] = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
        
        if patching_results.abs().max() > 1.0:
            print(f"Example {i} has absolute value greater than 1.0")

        if cache_patching_results:
            torch.save(patching_results, f"../results/patching_results_{i}.pt")
        # Plot the heatmap
        plot_patching_heatmap(
            patching_results,
            clean_tokens,
            tokenizer,
            answer=answer,
            filepath=f"../results/patching_results_{i}.pdf",
            modelname=modelname
        )

        break
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="dmis-lab/meerkat-7b-v1.0")
    parser.add_argument("--tokenizer_name", type=str, default="dmis-lab/meerkat-7b-v1.0")
    parser.add_argument("--data_path", type=str, default="../data/meerkat-7b-v1.0_medqa-original_results.json")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--cache_patching_results", type=bool, default=True)
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

    run_activation_patching(model, tokenizer, baseline_data, args.max_tokens, cache_patching_results=args.cache_patching_results)
    
if __name__ == "__main__":
    main()
