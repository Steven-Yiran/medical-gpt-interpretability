import os
import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

#import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="dmis-lab/meerkat-7b-v1.0")
    parser.add_argument("--tokenizer_name", type=str, default="dmis-lab/meerkat-7b-v1.0")
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

    assert_logits_match(model, hf_model, tokenizer)
    
if __name__ == "__main__":
    main()
