import argparse
import re
import os
import json
from collections import defaultdict
from typing import Tuple

import numpy
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import ClinicalKnownsDataset, ClinicalAgeGroupDataset
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
            "microsoft/BioGPT-Large-PubMedQA"
        ]
    )
    aa("--fact_data", default=None, type=str)
    aa("--output_dir", default="results/{model_name}/causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=False, type=bool)
    args = parser.parse_args()

    model_dir = f"r{args.replace}_{args.model_name.replace('/', '_')}"
    model_dir = f"n{args.noise_level}_{model_dir}"
    output_dir = args.output_dir.format(model_name=model_dir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdf"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    mt = ModelAndTokenizer(args.model_name)

    if args.fact_data == "knowns":
        knowns = ClinicalKnownsDataset("data")
    elif args.fact_data == "pubmedqa":
        knowns = ClinicalAgeGroupDataset("data")
    else:
        raise ValueError(f"Unknown fact_data: {args.fact_data}")

    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian noise
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * 0.05 # TODO: temporary
            print(f"Using noise_level {noise_level} to match emperical SD of model embedding times {factor}")

    for knowledge in tqdm(knowns):
        kid = knowledge["id"]
        #orid = knowledge["original_id"]
        for module_kind in None, "mlp", "attn":
            kind_suffix = f"_{module_kind}" if module_kind is not None else ""
            filename = f"{result_dir}/knowledge_{kid}{kind_suffix}.npz"
            if not os.path.exists(filename):
                result = run_patching_analysis(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    knowledge["attribute"],
                    module_kind=module_kind,
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                ) 
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                numpy.savez(filename, **numpy_result)
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            if not numpy_result["correct_prediction"]:
                tqdm.write(f"Skipping {kid}, prediction: {numpy_result['answer']}, expected: {numpy_result['expect']}")
                continue
            plot_result = dict(numpy_result)
            plot_result["module_kind"] = module_kind
            pdfname = f'{pdf_dir}/{str(numpy_result["answer"]).strip()}_{kid}{kind_suffix}.pdf'
            plot_trace_heatmap(plot_result, filepath=pdfname, modelname=args.model_name)


def get_prob_interv(
    model,
    inputs,
    states_to_patch,
    answer_t,
    corrupt_range: Tuple[int, int],
    noise: int = 0.1,
    uniform_noise=False,
    replace: bool = False,
    trace_layers=None, 
):
    """
    Compute the probability of the output while performing intervention on model's states.
    
    Given a GPT-style model and a batch of n inputs, performs n + 1 runs where the 0th input is
    used for a clean run and [1...n] inputs are used for corrupted run.

    Two methods can be used for corruption:
        - Gaussian noising (GN): Adds a large Gaussian noise to the token embeddings of the input 
          specified by corrupt_range.
        - Symmetric token replacement (STR): Swap input tokens with semantically related ones.

    Args:
        states_to_patch: list of (token_index, layernam). Specifies the states to be patched by restoring 
            the original values in the clean run.

        corrupt_range: specifies a range of tokens (begin, end) to be corrupted.

    Returns:
        probs: the probability of emitting the answer token (answer_t)
    """
    rs = numpy.random.RandomState(42) # Fixed seed for reproducibility
    if uniform_noise:
        generator = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        generator = lambda *shape: rs.normal(*shape)

    patch_spec = defaultdict(list)
    for token, layer in states_to_patch:
        patch_spec[layer].append(token)

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
        This function execute state patching depending on the layer.
        """
        if layer == embed_layername:
            if corrupt_range is not None:
                start, end = corrupt_range
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


def run_patching_analysis(
    mt,
    prompt: str,
    subject: str,
    expect: str,
    samples: int = 10,
    noise: float = 0.0,
    token_range: Tuple = None,
    uniform_noise: bool = False,
    replace: bool = False,
    window=10,
    module_kind: str = None,
):
    """
    Runs an activation patching experiment over the provided model, subject, object, 
    and relation prompt. Activation patching quantifies the contribution of each 
    state or a group of neighboring states in the model towards a correct answer prediction.
    To do this, we observe model's internal activations during three runs: a clean run, a corrupted run,
    and a corrupted run with restoration that tests the ability of state(s) to restore the correct prediction.

    Args:
        mt: ModelAndTokenizer object containing the model and tokenizer
        prompt: the prompt to be used for the experiment
        subject: the subject to be corrupted in the experiment
        expect: the expected answer to the prompt

    Returns:
        A dictionary containing the following
        - scores: a table of shape (ntokens, num_layers) containing the each state's 
            causal impact on output probability
        - low_score: the probability of the output after corrupting the subject
        - high_score: the probability of the output in the clean run

    """
    # clean run
    inputs = make_inputs(mt.tokenizer, [prompt] * (samples + 1), device="cuda")
    with torch.no_grad():
        answer_id, clean_prob = [d[0] for d in predict_from_input(mt.model, inputs)]
    [answer_tok] = decode_tokens(mt.tokenizer, [answer_id])
    if expect is not None and answer_tok.strip() != expect:
        return dict(
            answer=answer_tok,
            expect=expect,
            correct_prediction=False
        )
    corrupt_range = find_token_range(mt.tokenizer, inputs["input_ids"][0], subject)
    # corrupted run
    corrupt_prob = get_prob_interv(
        mt.model,
        inputs,
        [],
        answer_id,
        corrupt_range=corrupt_range,
        noise=noise,
        uniform_noise=uniform_noise
    ).item()
    # find trace token range: (i.e. After "Questions:" token Before "the answer of the question is" token)
    # if token_range is None:
    #     start_range = find_token_range(mt.tokenizer, inputs["input_ids"][0], "Question:")
    #     end_range = find_token_range(mt.tokenizer, inputs["input_ids"][0], "the answer of the question is")
    #     token_range = (start_range[1], end_range[0])
    # corrupted-with-restoration run
    if not module_kind:
        probs = trace_significant_states(
            mt.model,
            mt.num_layers,
            inputs,
            corrupt_range,
            answer_id,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        probs = trace_significant_window(
            mt.model,
            mt.num_layers,
            inputs,
            corrupt_range,
            answer_id,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            module_kind=module_kind,
            token_range=token_range,
        )
    probs = probs.detach().cpu()
    return dict(
        scores=probs,
        low_score=corrupt_prob,
        high_score=clean_prob,
        input_ids=inputs["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inputs["input_ids"][0]),
        subject_range=corrupt_range,
        answer=answer_tok,
        window=window,
        correct_prediction=True,
        module_kind=module_kind or "",
        token_range=token_range,
    )


def trace_significant_states(
    model,
    num_layers,
    inputs,
    corrupt_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    """
    Traces the important states in the model by running the activation patching experiment
    over every token/layer combination in the network.

    Returns:
        A tensor of shape (ntokens, num_layers) containing the difference in the probability 
        of the correct answer between the corrupted and restored runs when the state at the
        token/layer combination is patched.
    """
    ntokens = inputs["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntokens)
    elif isinstance(token_range, Tuple):
        token_range = range(*token_range)
    
    for tid in token_range:
        row = []
        for layer in range(num_layers):
            r = get_prob_interv(
                model,
                inputs,
                [(tid, get_layer_name(model, layer))],
                answer_t,
                corrupt_range=corrupt_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,  
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_significant_window(
    model,
    num_layers,
    inputs,
    corrupt_range,
    answer_t,
    module_kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntokens = inputs["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntokens)
    elif isinstance(token_range, Tuple):
        token_range = range(*token_range)

    for tid in token_range:
        row = []
        for layer in range(num_layers):
            layerlist = [
                (tid, get_layer_name(model, nei, component=module_kind))
                for nei in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = get_prob_interv(
                model,
                inputs,
                layerlist,
                answer_t,
                corrupt_range=corrupt_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


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
            if (re.match(r"^(transformer|biogpt)\.(layers)\.\d+$", name))
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
    #remove all whitespace in substring
    substring = "".join(substring.split())
    whole_string = "".join(toks)
    try:
        char_loc = whole_string.index(substring)
    except ValueError:
        print(f"Could not find substring {substring} in {whole_string}")
        raise ValueError
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
        else:
            return f"biogpt.layers.{layer_id}"
            
    assert False, f"unknown model architecture or layer type"


def plot_trace_heatmap(result, filepath=None, modelname=None):
    """
    Plots the causal impact on output probability on the prediction for
        1. each hidden state
        2. only MLP activations
        3. only attention activations
    """
    if modelname is None:
        modelname = "BioGPT-PubMedQA"

    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    module_kind = result["module_kind"]
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    # cut labels
    # start, end = result["token_range"]
    # labels = labels[start:end]

    with plt.rc_context():
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                module_kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not module_kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            ax.set_title(f"Impact of restoring state after corrupted input ({module_kind})")
            ax.set_xlabel(f"center of interval of {window} layers within {module_kind} layers")
        cb = plt.colorbar(h)
        if answer is not None:
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    main()
