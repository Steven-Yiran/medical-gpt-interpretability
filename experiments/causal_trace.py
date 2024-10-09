import argparse
import re

import torch
from nnsight import LanguageModel, util
from nnsight.tracing.Proxy import Proxy
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Causal Trace")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--model_name",
        default="microsoft/biogpt",
        choices=[
            "microsoft/biogpt",
        ]
    )
    args = parser.parse_args()

    #mt = ModelAndTokenizer(args.model_name)

    model = LanguageModel(args.model_name, device_map='cuda:0')

    clean_prompt = "Alzheimer's disease is characterized by progressive cognitive decline, particularly in"
    corrupted_prompt = "Alzheimer's disease is characterized by progressive cognitive decline, particularly in"
    
    N_LAYERS = model.config.n_layer

    with model.trace() as tracer:

        # Clean run
        with tracer.invoke(clean_prompt) as invoker:
            clean_tokens = invoker.inputs[0]['input_ids']

            clean_hs = [
                model.transformer.h[layer_idx].output[0]
                for layer_idx in range(N_LAYERS)
            ]


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
    
if __name__ == "__main__":
    main()
