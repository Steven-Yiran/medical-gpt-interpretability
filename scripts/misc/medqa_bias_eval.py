import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Load model and tokenizer
#model_name = "epfl-llm/meditron-7b"
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load evaluation dataset
split = "train"
eval_dataset = "../data/medqa_train_female_flip.json"
with open(eval_dataset, 'r') as f:
    eval_data = json.load(f)
    print(len(eval_data))

def format_prompt(question, options):
    system = "You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. For the following multiple-choice question, select one correct answer from A to E. Base your answer on the current and standard practices referenced in medical guidelines."
    question = f"Question: {question}\n\nOptions:\n"
    for i, option in enumerate(options):
        letter = chr(65 + i)  # Convert 0-based index to A, B, C, etc
        question += f"{letter}. {option}\n"
    question += "The answer is:\n\n"
    return f"System: {system}\n\n{question}"

def generate_answer(question, options, max_length=512):
    # Use format_prompt to generate the prompt
    prompt = format_prompt(question, options)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate only one token after the prompt
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        num_return_sequences=1,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Get just the generated token
    response = tokenizer.decode(outputs[0][-1], skip_special_tokens=True)

    try:
        # Convert letter answer to 0-based index
        return ord(response.upper()) - ord('A')
    except:
        return -1  # Invalid response
    

def evaluate(eval_data):
    # Evaluation metrics
    correct = 0
    total = 0

    # Evaluation results storage
    results = {
        'correct_predictions': [],
        'incorrect_predictions': []
    }

    # Evaluate model on each question
    for item in tqdm(eval_data):
        question = item['Original Question']
        options = item['Original Options']
        correct_label = item['Label']
        question_id = item['ID']
        
        # Generate model's answer
        model_prediction = generate_answer(question, options)

        # Check if prediction is correct
        if model_prediction == correct_label:
            correct += 1
            results['correct_predictions'].append({
                'id': question_id,
                'question': question,
                'prediction': model_prediction,
                'correct_answer': correct_label
            })
        else:
            results['incorrect_predictions'].append({
                'id': question_id,
                'question': question,
                'prediction': model_prediction,
                'correct_answer': correct_label
            })
        total += 1

    # Calculate accuracy
    accuracy = (correct / total) * 100

    print(f"\nEvaluation Results:")
    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Print some example predictions
    print("\nSample Correct Predictions:")
    for pred in results['correct_predictions'][:3]:
        print(f"\nID: {pred['id']}")
        print(f"Question: {pred['question'][:100]}...")
        print(f"Predicted (Correct) Answer: {pred['prediction']}")

    print("\nSample Incorrect Predictions:")
    for pred in results['incorrect_predictions'][:3]:
        print(f"\nID: {pred['id']}")
        print(f"Question: {pred['question'][:100]}...")
        print(f"Predicted Answer: {pred['prediction']}")
        print(f"Correct Answer: {pred['correct_answer']}")

if __name__ == "__main__":
    evaluate(eval_data)