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

from dsets import ClinicalKnownsDataset
from util import nethook

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
    aa("--replace", default=False, type=bool)
    args = parser.parse_args()

    model_dir = f"r{args.replace}_{args.model_name.replace('/', '_')}"
    model_dir = f"n{args.noise_level}_{model_dir}"
    output_dir = args.output_dir.format(model_name=model_dir)
    result_dir = f"{output_dir}/cases"
    os.makedirs(result_dir, exist_ok=True)

    mt = ModelAndTokenizer(args.model_name)

    if args.fact_file is None:
        knowns = ClinicalKnownsDataset("data")
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


def run_activation_patching_experiment(
    model,
    inputs,
    states_to_patch,
    answer_t,
    tokens_to_currupt: Tuple[int, int],
    noise: int = 0.1,
    uniform_noise=False,
    replace: bool = False,
    trace_layers=None, 
):
    """
    Runs a single causal trace experiment. Given a GPT-style model and a batch of n inputs,
    performs n + 1 runs where the 0th input is used for a clean run and [1...n] inputs are used for
    corrupted run.

    Two methods can be used to corrupt the input:
        - Gaussian noising (GN): Adds a large Gaussian noise to the token embeddings of the input 
          specified by tokens_to_currupt.
        - Symmetric token replacement (STR): Swap input tokens with semantically related ones.

    Args:
        states_to_patch: list of (token_index, layernam). Specifies the states to be patched by restoring 
            the original values in the clean run.

        tokens_to_currupt: specifies a range of tokens (begin, end) to be corrupted.
    """
    rs = numpy.random.RandomState(42) # Fixed seed for reproducibility
    if uniform_noise:
        generator = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        generator = lambda *shape: rs.normal(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = get_layer_name(model, 0, component="embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define patch rules
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rule(x, layer):
        """
        Define rules to execute corrupted run or patched run for a given layer.
        """
        if layer == embed_layername:
            if tokens_to_currupt is not None:
                start, end = tokens_to_currupt
                noise_embed = noise_fn(
                    torch.from_numpy(generator(x.shape[0] - 1, end - start, x.shape[2]))
                ).cuda()
                x[1:, start:end] += noise_embed
            return x

        if layer not in patch_spec:
            return x

        # patch states
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # patched runs
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rule,
    ) as td:
        out = model(**inputs)

    # report the probability of object token
    probs = torch.softmax(out.logits[1:, -1, :], dim=1).mean(dim=0)[answer_t]

    return probs


def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples: int = 10,
    noise: float = 0.0,
    token_range: Tuple = None,
    uniform_noise: bool = False,
    replace: bool = False,
    window=10,
    module_kind: str = None,
    expect: str = None,
):
    """
    Runs causal tracing experiment over every token/layer combination in the network
    and returns a dictionary that numerically summarizing the results.
    """
    inputs = make_inputs(mt.tokenizer, [prompt] * (samples + 1), device="cuda")
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inputs)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    if expect is not None and answer.strip() != expect:
        return dict(correct_prediction=False)
    corrupt_range = find_token_range(mt.tokenizer, inputs["input_ids"][0], subject)
    low_score = run_activation_patching_experiment(
        mt.model, inputs, [], answer_t, corrupt_range, noise=noise, uniform_noise=uniform_noise
    ).item()
    return low_score


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


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = " ".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def predict_from_input(model, inputs):
    out = model(**inputs)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


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


def get_layer_name(model, layer_id, component=None):
    comp_names = {
        "biogpt": {
            "embed": "embed_tokens",
            "mlp": "fc2",
            "attn": "self_attn",
        }
    }

    if hasattr(model, "biogpt"):
        if component == "embed":
            return f"biogpt.{comp_names['biogpt']['embed']}"
        elif component == "mlp":
            return f"biogpt.layers.{layer_id}.{comp_names['biogpt']['mlp']}"
        elif component == "attn":
            return f"biogpt.layers.{layer_id}.{comp_names['biogpt']['attn']}"
            
    assert False, "unknown model architecture or layer type"


if __name__ == "__main__":
    main()
