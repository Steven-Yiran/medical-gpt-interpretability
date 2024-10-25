import argparse
import os
import json

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

labels = ["Disease and Disorders", "Medication and Drugs"]
required_context_labels = ['BACKGROUND', 'PATIENTS AND METHODS', 'RESULTS']

def main():
    parser = argparse.ArgumentParser(description="Preprocess PubMed data into disease and medicine samples")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--with-context", action="store_false", help="Include context in the samples")
    aa("--min-samples", type=int, default=100, help="Minimum number of samples to collect")
    aa("--original-file", type=str, default="data/ori_pqal.json", help="Original PubMed QA data file")
    aa("--disease-file", type=str, default="data/disease_pqal.json", help="Disease samples output file")
    aa("--medicine-file", type=str, default="data/medicine_pqal.json", help="Medicine samples output file")
    args = parser.parse_args()

    with_context = args.with_context
    min_samples = args.min_samples
    original_file = args.original_file
    disease_file = args.disease_file
    medicine_file = args.medicine_file

    with open(original_file, "r") as f:
        data = json.load(f)

    disease_data = []
    medicine_data = []

    for entry_id, entry_data in data.items():
        text = get_full_text(entry_data)
        response = run_openai_task(text)
        response = json.loads(response)

        disease_keywords = response.get("Disease and Disorders", [])
        medicine_keywords = response.get("Medication and Drugs", [])

        if len(disease_keywords) == 0 and len(medicine_keywords) == 0:
            print(f"Skipping entry {entry_id} with no keywords")
            continue

        if len(disease_keywords) > len(medicine_keywords):
            subjects = disease_keywords
            category = "Disease"
            disease_data.append(make_sample(entry_data, subjects, category, with_context))
        else:
            subjects = medicine_keywords
            category = "Medicine"
            medicine_data.append(make_sample(entry_data, subjects, category, with_context))

        if len(disease_data) >= min_samples and len(medicine_data) >= min_samples:
            break

    with open(disease_file, "w") as f:
        json.dump(disease_data, f, indent=4)

    with open(medicine_file, "w") as f:
        json.dump(medicine_data, f, indent=4)

    print(f"Saved {len(disease_data)} disease samples to {disease_file}")
    print(f"Saved {len(medicine_data)} medicine samples to {medicine_file}")


def system_message():
    return f"""
You are an expert in clinical NLP. Your task is to identify common clinical Named Entities (NER) in a given text.
The possible common Named Entities (NER) are exclusively: ({", ".join(labels)}).
"""

def assistant_message():
    return f"""
EXAMPLE:
    Text: 'Syncope during bathing in infants, a pediatric form of water-induced urticaria?'
        {{
            "Disease and Disorders": ["Syncope", "water-induced urticaria"],
            "Medication and Drugs": []
        }}
--"""

def user_message(question):
    return f"""
TASK:
    Text: {question}
"""


def run_openai_task(text):
    messages = [
        {"role": "system", "content": system_message()},
        {"role": "assistant", "content": assistant_message()},
        {"role": "user", "content": user_message(text)}
    ]

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )

    return response.choices[0].message.content


def get_full_text(entry_data, with_context=False):

    question = entry_data.get("QUESTION", "")

    if not with_context:
        return question

    contexts = entry_data.get("CONTEXT", "")
    labels = entry_data.get("LABELS", "")
    # combined_context = "".join(
    #     [contexts[i].lower() for i in range(len(contexts)) if labels[i].upper() in required_context_labels]
    # )

    return question + " " + contexts


def make_sample(entry_data, subjects, category, with_context=False):
    question = entry_data.get("QUESTION", "")
    answer = entry_data.get("final_decision", "")
    prompt = "Question: " + question + " the answer to the question is"

    return {
        "category": category,
        "subjects": subjects,
        "prompt": prompt,
        "attribute": answer
    }

    
if __name__ == "__main__":
    main()



