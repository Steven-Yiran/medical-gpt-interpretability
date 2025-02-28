from transformers import AutoModelForCausalLM, AutoTokenizer
#from utils import Metrics
import json
from datasets import load_dataset
from prompts import prompt_eval_bare_fully

def format_choices(choices):
    a = zip(list(choices.keys()), choices.values())
    final_answers = []
    for x,y in a:
        final_answers.append(f'[{x}] : {y}')
    return "\n".join(final_answers)


def main():
    # Load the model and tokenizer
    model_name = "KrithikV/MedMobile"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")

    results = []
    for item in dataset:
        question = item['sent1']
        choices = {'A': item['ending0'], 'B': item['ending1'], 'C': item['ending2'], 'D': item['ending3']}
        correct_answer = item['label'] 

        formatted_choices = format_choices(choices)
        prompt = prompt_eval_bare_fully.format(question=question, choices=formatted_choices)
        prompt += "## Answer"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_length=2048)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(generated_text)
        break

if __name__ == "__main__":
    main()