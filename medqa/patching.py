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

from prompts import prompt_eval_bare, meerkat_medqa_system_prompt

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

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


def truncate_prompt(prompt):
    # truncate the prompt until the end of the pattern "the answer is ("
    # since there are multiple instances of this pattern in the prompt, we need to find the last one
    pattern = "the answer is ("
    # find the last instance of the pattern
    last_instance = prompt.rfind(pattern)
    if last_instance == -1:
        return None
    return prompt[:last_instance + len(pattern)]

def save_plot_to_pdf(fig, filename):
    fig.write_image(filename, format="pdf")

def plot_patching_heatmap(patching_results, clean_tokens, tokenizer, answer=None, filepath=None, modelname="Meerkat-7B"):
    """
    Plots the activation patching results as a heatmap.
    
    Args:
        patching_results: Tensor of shape (n_layers, n_positions) containing patching scores
        clean_tokens: Original token IDs
        tokenizer: Tokenizer used for decoding
        filepath: Where to save the plot
        modelname: Name of the model for the plot title
    """
    # Convert results to numpy for plotting
    differences = patching_results.cpu().numpy()
    differences = differences.T
    low_score = patching_results.min().item()
    module_kind = None
    window = 10
    labels = [tokenizer.decode([t]) for t in clean_tokens[0]]
    
    num_tokens = differences.shape[0]
    num_layers = differences.shape[1]
    plot_height = max(num_tokens / 7, 20)
    plot_width = num_layers / 10
    with plt.rc_context():
        fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=200)
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


"""
Generate a counterfactual version of patient information by changing gender references.

This function takes a prompt containing patient information and creates a counterfactual
version by changing all references from female to male (or vice versa).

Args:
    prompt (str): The original prompt containing patient information

Returns:
    str: A modified version of the prompt with gender references changed
"""
def change_gender_references(text):
    # Dictionary of gender-specific terms to replace
    replacements = {
        "woman": "man",
        "women": "men",
        "female": "male",
        "she": "he",
        "her": "his",
        "hers": "his",
        "herself": "himself",
        "girl": "boy",
        "girls": "boys",
        "lady": "gentleman",
        "ladies": "gentlemen",
        "mother": "father",
        "mom": "dad",
        "sister": "brother",
        "daughter": "son",
        "wife": "husband",
        "girlfriend": "boyfriend",
        "fiancée": "fiancé",
        "widow": "widower",
        "Ms.": "Mr.",
        "Mrs.": "Mr.",
        "Miss": "Mr.",
    }
    
    # Create case-insensitive pattern
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in replacements.keys()) + r')\b', 
                         re.IGNORECASE)
    
    # Function to replace with correct case
    def replace(match):
        matched = match.group(0)
        replacement = replacements[matched.lower()]
        
        # Preserve capitalization
        if matched.islower():
            return replacement
        elif matched.isupper():
            return replacement.upper()
        elif matched[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement
    
    return pattern.sub(replace, text)

def generate_counterfactual_patient_info(prompt):
    """
    Generate a counterfactual version of patient information by changing gender.
    
    Args:
        prompt (str): The original prompt containing patient information
        
    Returns:
        str: A counterfactual version of the prompt with gender changed
    """
    return change_gender_references(prompt)


def run_activation_patching(model, tokenizer, baseline_data, counterfactual_data, max_tokens):
    average_clean_logit_diff = 0
    average_corrupted_logit_diff = 0
    for i in range(len(baseline_data)):
        if os.path.exists(f"../results/patching_results_{i}.pdf"):
            continue
        print(f"Processing example {i}")

        baseline_prompt = baseline_data[i]['generated_response']
        baseline_prompt = truncate_prompt(baseline_prompt)
        baseline_prompt = baseline_prompt[-max_tokens:]
        answer = chr(ord('A') + baseline_data[i]['gold'])
        counterfactual_prompt = generate_counterfactual_patient_info(baseline_prompt)
        counterfactual_answer = select_random_answer(answer)
        answers = [(answer, counterfactual_answer)]
        answer_token_indices = torch.tensor(
            [
                [model.to_single_token(answers[i][j]) for j in range(2)]
                for i in range(len(answers))
            ],
            device="cuda"
        )

        if baseline_prompt is None or counterfactual_prompt is None:
            print(f"Skipping example {i} because the answer is not in the prompt")
            continue

        clean_tokens = tokenizer.encode(baseline_prompt, return_tensors="pt").to("cuda")
        corrupted_tokens = tokenizer.encode(counterfactual_prompt, return_tensors="pt").to("cuda")

        clean_logits, clean_cache = model.run_with_cache(clean_tokens)   
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

        clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
        corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
        print(f"Clean logit diff: {clean_logit_diff}")
        print(f"Corrupted logit diff: {corrupted_logit_diff}")

        if abs(clean_logit_diff - corrupted_logit_diff) < 1e-2:
            print(f"Skipping example {i} because the logit diff is too small")
            continue

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
                patched_logit_diff = get_logit_diff(patched_logits, answer_token_indices).detach()
                patching_results[layer, position] = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
            
        # Plot the heatmap
        plot_patching_heatmap(
            patching_results,
            clean_tokens,
            tokenizer,
            answer=answer,
            filepath=f"../results/patching_results_{i}.pdf",
            modelname="Mistral-7B"
        )

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

    temp_content = "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\".\n\nUSER: A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions? (A) Inhibition of proteasome (B) Hyperstabilization of microtubules (C) Generation of free radicals (D) Cross-linking of DNA ASSISTANT: \n\nStep 1: Summarize available information. The patient is a 67-year-old man with transitional cell carcinoma of the bladder. He has a 2-day history of a ringing sensation in his ear, which began after he received neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB.\n\nStep 2: Relate the symptoms to chemotherapy. The patient's symptoms of a ringing sensation in the ear and sensorineural hearing loss are likely to be an adverse effect of chemotherapy.\n\nStep 3: Identify the drug likely responsible. Common chemotherapy agents known to cause ototoxicity (damage to the ear leading to hearing loss) include platinum compounds such as cisplatin.\n\nStep 4: Determine the mechanism of action of the drug. Cisplatin, a platinum-containing drug, is known to cause ototoxicity. Its mechanism of action involves the formation of free radicals that lead to cell death primarily through apoptosis.\n\nStep 5: Match the drug's action with the options provided. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to the generation of free radicals, as this is the mechanism of action by which drugs like cisplatin exert their anticancer effects.\n\nTherefore, the answer is ("
    messages = [
        {"role": "user", "content": temp_content},
    ]
    #tokens = tokenizer.apply_chat_template(messages, return_tensors="pt")
    tokens = tokenizer.encode(temp_content, return_tensors="pt")
    tokens = tokens.to("cuda")

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

    with open("meerkat-7b-v1.0_medqa-original_results.json", "r") as f:
        baseline_data = json.load(f)
    
    with open("meerkat-7b-v1.0_medqa-female_results.json", "r") as f:
        counterfactual_data = json.load(f)

    print(f"Baseline data: {len(baseline_data)}")
    print(f"Counterfactual data: {len(counterfactual_data)}")

    run_activation_patching(model, tokenizer, baseline_data, counterfactual_data, args.max_tokens)
    
if __name__ == "__main__":
    main()
