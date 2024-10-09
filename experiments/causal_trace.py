import argparse
import re

import torch
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

    mt = ModelAndTokenizer(args.model_name)
    print(mt)


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
