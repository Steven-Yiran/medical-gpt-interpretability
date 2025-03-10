from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datasets import load_dataset
from prompts import prompt_eval_bare_fully
import torch
import re
from tqdm import tqdm
import argparse
import os

from filter import MultiChoiceFilter

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


def evaluate(model_name, results):
    filter = MultiChoiceFilter()

    choice_tally = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'invalid': 0}
    correct_choice_tally = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    incorrect_choice_tally = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    answer_type_tally = {}
    invalid_ids = []

    for result in results:
        id = result["id"]
        generated_response = result["generated_response"]
        gold_idx = result["gold"]

        answer, answer_type = filter.extract_answer(generated_response)
        if answer_type not in answer_type_tally:
            answer_type_tally[answer_type] = 0
        answer_type_tally[answer_type] += 1

        if answer not in choice_tally:
            choice_tally["invalid"] += 1
            invalid_ids.append(id)
        else:
            if answer == "invalid":
                choice_tally["invalid"] += 1
                invalid_ids.append(id)
            else:
                choice_tally[answer] += 1
                answer_idx = ord(answer) - ord('A')
                if answer_idx == gold_idx:
                    correct_choice_tally[answer] += 1
                else:
                    incorrect_choice_tally[answer] += 1

    correct = sum(correct_choice_tally.values())
    incorrect = sum(incorrect_choice_tally.values())
    print(f"Correct: {correct}, Incorrect: {incorrect}")
    print(f"Accuracy: {correct / (len(results))}")
    print(f"Accuracy (without invalid): {correct / (correct + incorrect)}")
    print("Choice Tally:")
    for choice, count in choice_tally.items():
        print(f"  {choice}: {count}")
    
    print("\nCorrect Choice Tally:")
    for choice, count in correct_choice_tally.items():
        print(f"  {choice}: {count}")

    print("\nIncorrect Choice Tally:")
    for choice, count in incorrect_choice_tally.items():
        print(f"  {choice}: {count}")
    
    print("\nAnswer Type Tally:")
    for answer_type, count in answer_type_tally.items():
        print(f"  {answer_type}: {count}")

    # save invalid ids
    with open(f"{model_name}_invalid_ids.json", "w") as f:
        json.dump(invalid_ids, f)
    print(f"Invalid IDs saved to {model_name}_invalid_ids.json")

def inference(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")    

    results = []
    for item in tqdm(dataset):
        question = item['sent1']
        choices = {'choice_A': item['ending0'], 'choice_B': item['ending1'], 'choice_C': item['ending2'], 'choice_D': item['ending3']}

        generated_response = check_answer(model, tokenizer, question, choices)
        
        results.append({
            "id": item['id'],
        "gold": item['label'],
        "generated_response": generated_response
    })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--inference", action="store_true")
    args = parser.parse_args()

    # Load the model and tokenizer
    model_name = args.model_name
    model_out_name = model_name.split("/")[-1]

    if not os.path.exists(f"{model_out_name}_results.json") or args.inference:
        print(f"Running inference for {model_out_name}...")
        results = inference(args)
        with open(f"{model_out_name}_results.json", "w") as f:
            json.dump(results, f)
    else:
        print(f"Loading results for {model_out_name}...")
        with open(f"{model_out_name}_results.json", "r") as f:
            results = json.load(f)

    evaluate(model_out_name, results)

if __name__ == "__main__":
    main()