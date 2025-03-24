import os
import sys
import argparse
import json
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

#import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix

from prompts import prompt_eval_bare, meerkat_medqa_system_prompt



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


def convert_to_tokens(tokenizer, question, choices):
    content = prompt_eval_bare.format(question=question, **choices)
    messages = [
        {"role": "system", "content": meerkat_medqa_system_prompt},
        {"role": "user", "content": content},
    ]
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt")
    return tokens
    #generated_ids = model.generate(encodeds, max_new_tokens=max_tokens, do_sample=False)
    #decoded = tokenizer.batch_decode(generated_ids)
    #return decoded[0]


def select_random_answer(correct_answer):
    wrong_answers = ['A', 'B', 'C', 'D']
    wrong_answers.remove(correct_answer)
    return random.choice(wrong_answers)


def setup_patching(model, tokenizer, baseline_data, counterfactual_data, max_tokens):
    for i in range(len(baseline_data)):
        question = baseline_data[i]['sent1']
        choices = {'choice_A': baseline_data[i]['ending0'], 'choice_B': baseline_data[i]['ending1'], 'choice_C': baseline_data[i]['ending2'], 'choice_D': baseline_data[i]['ending3']}
        answer = chr(ord('A') + baseline_data[i]['label'])
        counterfactual_question = counterfactual_data[i]['sent1']
        counterfactual_choices = {'choice_A': counterfactual_data[i]['ending0'], 'choice_B': counterfactual_data[i]['ending1'], 'choice_C': counterfactual_data[i]['ending2'], 'choice_D': counterfactual_data[i]['ending3']}
        counterfactual_answer = select_random_answer(answer)
        answers = [answer, counterfactual_answer]

        clean_tokens = convert_to_tokens(tokenizer, question, choices)
        corrupted_tokens = convert_to_tokens(tokenizer, counterfactual_question, counterfactual_choices)
        answer_token_indices = torch.tensor(
            [model.to_single_token(ans) for ans in answers]
        )
        print(answer_token_indices)
        break

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="dmis-lab/meerkat-7b-v1.0")
    parser.add_argument("--tokenizer_name", type=str, default="dmis-lab/meerkat-7b-v1.0")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--model_path", type=str, default=None)
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
    )

    model = model.to("cuda")
    #hf_model = hf_model.to("cuda")
    #assert_logits_match(model, hf_model, tokenizer)

    with open("patient_medqa.json", "r") as f:
        baseline_data = json.load(f)
    
    with open("male_medqa.json", "r") as f:
        counterfactual_data = json.load(f)

    print(f"Baseline data: {len(baseline_data)}")
    print(f"Counterfactual data: {len(counterfactual_data)}")

    setup_patching(model, tokenizer, baseline_data, counterfactual_data, args.max_tokens)

    
if __name__ == "__main__":
    main()
