import argparse
import re
import os
import json
from collections import defaultdict
from typing import Tuple

import numpy
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from clinical_knowns import ClinicalKnownsDataset

def main():
    parser = argparse.ArgumentParser(description="Causal Trace")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="microsoft/biogpt",
        choices=[
            "microsoft/biogpt",
        ]
    )
    aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=None, type=int)
    args = parser.parse_args()

    model_dir = f"r{args.replace}_{args.model_name.replace('/', '_')}"
    model_dir = f"n{args.noise_level}_{model_dir}"
    output_dir = args.output_dir.format(model_name=model_dir)
    result_dir = output_dir / "cases"
    os.makedirs(result_dir, exist_ok=True)

    mt = ModelAndTokenizer(args.model_name)

    if args.fact_file is None:
        knowns = ClinicalKnownsDataset("experiments")
    else:
        with open(args.fact_file, "r") as f:
            knowns = json.load(f)

    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian noise
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * 0.05 # TODO: temporary
            print(f"Using noise_level {noise_level} to match model times {factor}")

    for knowledge in tqdm(knowns):
        known_id = knowledge["known_id"]
        for kind in [None, "mlp", "attn"]:
            kind_suffix = f"_{kind}" if kind is not None else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            if not os.path.exists(filename):
                result = calculate_hidden_flow(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    expect=knowledge["attribute"],
                    module_kind=kind,
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                )
                print(result)
            #     numpy_result = {
            #         k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
            #         for k, v in result.items()
            #     }
            #     numpy.savez(filename, **numpy_result)
            # else:
            #     numpy_result = numpy.load(filename, allow_pickle=True)
            # if not numpy_result["correct_prediction"]:
            #     tqdm.write(f"Skipping {knowledge['prompt']}")
            #     continue


    # Start of causal tracing of Factual Association Experiment

    clean_prompt = "Alzheimer's disease is characterized by progressive cognitive decline, particularly in"
    corrupted_prompt = "Alzheimer's disease is characterized by progressive cognitive decline, particularly in"


def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples: int = 10,
    noise: float = 0.0,
    token_range: Tuple = None,
    uniform_noise: bool = False,
    replace: int = None,
    window=10,
    module_kind: str = None,
    expect: str = None,
):
    """
    Runs causal tracing experiment over every token/layer combination in the network
    and returns a dictionary that numerically summarizing the results.
    """
    inputs = make_inputs(mt.tokenizer, [prompt] * (samples + 1), device="cuda")
    return inputs


class ModelAndTokenizer:
    """
    A wrapper class for GPT-style models and tokenizers.
    Automatically downloads and gather relevent metadata.
    """

    def __init__(
        self,
        model_name: str = None,
        model: torch.nn.Module = None,
        tokenizer: AutoTokenizer = None,
        low_cpu_mem_usage: bool = False,
        torch_dtype: torch.dtype = None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            name
            for name, _ in model.named_modules()
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(attention_mask, device=device),
    }


if __name__ == "__main__":
    main()
