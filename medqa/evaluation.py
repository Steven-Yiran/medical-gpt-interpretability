from transformers import AutoModelForCausalLM, AutoTokenizer
import json
#from datasets import load_dataset
import torch
import re
from tqdm import tqdm
import argparse
import os
import sys
from filter import MultiChoiceFilter, PatientInfoFilter

from prompts import prompt_eval_bare

def format_choices(choices):
    a = zip(list(choices.keys()), choices.values())
    final_answers = []
    for x,y in a:
        final_answers.append(f'[{x}] : {y}')
    return "\n".join(final_answers)

def check_answer(model, tokenizer, question, choices, max_tokens):
    content = prompt_eval_bare.format(question=question, **choices)
    messages = [
        {"role": "system", "content": "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\".\n\n"},
        {"role": "user", "content": content},
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda:0")
    generated_ids = model.generate(encodeds, max_new_tokens=max_tokens, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0]



def evaluate(model_name, results):
    filter = MultiChoiceFilter()
    patient_info_filter = PatientInfoFilter()

    choice_tally = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'invalid': 0}
    correct_choice_tally = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    incorrect_choice_tally = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    answer_type_tally = {}
    invalid_ids = []

    patient_info_total = 0
    man_correct = 0
    man_total = 0
    woman_correct = 0
    woman_total = 0
    for result in results:
        id = result["id"]
        generated_response = result["generated_response"]
        gold_idx = result["gold"]

        patient_info = patient_info_filter.filter_text(generated_response)
        gender = None
        if patient_info:
            patient_info_total += 1
            matched_text = patient_info.group(0).lower()
            if "woman" in matched_text:
                gender = "female"
                woman_total += 1
            elif "man" in matched_text:
                gender = "male"
                man_total += 1
            
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
                    if gender == "male":
                        man_correct += 1
                    elif gender == "female":
                        woman_correct += 1
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

    print(f"Patient Info Total: {patient_info_total}")
    print(f"Man Correct: {man_correct}, Man Total: {man_total}")
    print(f"Woman Correct: {woman_correct}, Woman Total: {woman_total}")
    print(f"Man Accuracy: {man_correct / man_total}")
    print(f"Woman Accuracy: {woman_correct / woman_total}")

    # save invalid ids
    with open(f"{model_name}_invalid_ids.json", "w") as f:
        json.dump(invalid_ids, f)
    print(f"Invalid IDs saved to {model_name}_invalid_ids.json")

def inference(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    filter = MultiChoiceFilter()

    if args.dataset_name == "medqa-official":
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")    
    elif args.dataset_name == "medqa-male":
        data_path = "male_medqa.json"
        dataset = json.load(open(data_path))
    elif args.dataset_name == "medqa-female":
        data_path = "female_medqa.json"
        dataset = json.load(open(data_path))
    else:
        raise ValueError(f"Dataset {args.dataset_name} not found")
    
    

    print(f"Loaded dataset from {data_path} with {len(dataset)} questions")

    results = []
    total = 0
    correct = 0
    invalid = 0
    progress_bar = tqdm(dataset)
    for item in progress_bar:
        question = item['sent1']
        choices = {'choice_A': item['ending0'], 'choice_B': item['ending1'], 'choice_C': item['ending2'], 'choice_D': item['ending3']}
        gold_idx = item['label']

        generated_response = check_answer(model, tokenizer, question, choices, args.max_tokens)

        answer, answer_type = filter.extract_answer(generated_response)

        total += 1
        if answer not in ['A', 'B', 'C', 'D']:
            print(f"Invalid answer: {answer}")
            print(f"--------------------------------")
            invalid += 1
        else:
            answer_idx = ord(answer) - ord('A')
            if answer_idx == gold_idx:
                correct += 1
        
        # Update progress bar with current accuracy
        if total > 0:
            accuracy = correct / total
            invalid_rate = invalid / total
            progress_bar.set_description(f"Accuracy: {accuracy:.4f}, Invalid: {invalid_rate:.4f}")
        
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
    parser.add_argument("--dataset_name", type=str, default="medqa-official")
    parser.add_argument("--inference", action="store_true")
    args = parser.parse_args()

    # Load the model and tokenizer
    model_name = args.model_name
    model_out_name = model_name.split("/")[-1]

    if not os.path.exists(f"{model_out_name}_{args.dataset_name}_results.json") or args.inference:
        print(f"Running inference for {model_out_name}...")
        results = inference(args)
        with open(f"{model_out_name}_{args.dataset_name}_results.json", "w") as f:
            json.dump(results, f)
    else:
        print(f"Loading results for {model_out_name}...")
        with open(f"{model_out_name}_{args.dataset_name}_results.json", "r") as f:
            results = json.load(f)

    evaluate(model_out_name, results)

if __name__ == "__main__":
    main()