from transformers import AutoModelForCausalLM, AutoTokenizer
#from utils import Metrics
import json
from datasets import load_dataset
from prompts import prompt_eval_bare_fully
import torch
import re
from tqdm import tqdm


# Define prompt template
prompt_template = """{question} (A) {choice_A} (B) {choice_B} (C) {choice_C} (D) {choice_D}"""

def format_choices(choices):
    a = zip(list(choices.keys()), choices.values())
    final_answers = []
    for x,y in a:
        final_answers.append(f'[{x}] : {y}')
    return "\n".join(final_answers)

def check_answer(model, tokenizer, question, choices):
    content = prompt_template.format(question=question, **choices)
    #content += " The answer is\n"
    messages = [
        {"role": "system", "content": "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\".\n\n"},
        {"role": "user", "content": content},
    ]

    # messages = [
    #     {"role": "system", "content": "Answer the multiple-choice question about medical knowledge. Always answer in the form of [A], [B], [C], or [D].\n\n"},
    #     {"role": "user", "content": content},
    # ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda:0")
    generated_ids = model.generate(encodeds, max_new_tokens=500, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)

    # Extract the generated response after "ASSISTANT"
    assistant_index = decoded[0].find("ASSISTANT:")
    if assistant_index != -1:
        generated_response = decoded[0][assistant_index + len("ASSISTANT:"):].strip()
    else:
        generated_response = decoded[0].strip()
    #print(generated_response)
    return generated_response


def main():
    # Load the model and tokenizer
    #model_name = "KrithikV/MedMobile"
    model_name = "dmis-lab/meerkat-7b-v1.0"
    #model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")    
    # Create a 100 subset of the dataset
    results = []
    for item in tqdm(dataset):
        question = item['sent1']
        choices = {'choice_A': item['ending0'], 'choice_B': item['ending1'], 'choice_C': item['ending2'], 'choice_D': item['ending3']}
        correct_answer = item['label'] 

        generated_response = check_answer(model, tokenizer, question, choices)
        
        results.append({
            "id": item['id'],
            "gold": item['label'],
            "generated_response": generated_response
        })

    model_name = model_name.split("/")[-1]
    with open(f"{model_name}_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()