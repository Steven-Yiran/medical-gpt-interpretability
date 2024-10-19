from openai import OpenAI
import os
import json
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def system_message():
    return f"""
You are an expert in Natural Language Processing. Your task is to identify words related to the provided MeSH disease keywords.
First, you need to identify diseases MeSH keywords from the provided MeSH keywords.
Then, you need to identify the words in the text that are related to the diseases MeSH keywords.
You need to return the identified words in a common separated string.
If you can't find any related words, return an string.
The MeSH keywords are exclusively user provided."""

def assistant_message():
    return f"""
EXAMPLE:
    MeSH: Baths, Histamine, Humans, Infant, Syncope, Urticaria, Water
    Text: 'Apparent life-threatening events in infants are a difficult and frequent problem in pediatric practice. / 
    The prognosis is uncertain because of risk of sudden infant death syndrome. / 
    Syncope during bathing in infants, a pediatric form of water-induced urticaria?'
    {"syncope, water-induced urticaria"}
--"""

def user_message(meshes, text):
    return f"""
TASK:
    MeSH: {", ".join(meshes)}
    Text: {text}
"""


def run_openai_task(meshes, text):
    messages = [
        {"role": "system", "content": system_message()},
        {"role": "assistant", "content": assistant_message()},
        {"role": "user", "content": user_message(meshes, text)}
    ]

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )

    return response.choices[0].message.content


def contain_disease_meshes(meshes):
    """
    Filter meshes that contain *disease*, *syndrome*, *infections*, *neoplasms*, *disorders*, *conditions*
    """
    for mesh in meshes:
        if re.search(r'disease|syndrome|infections|neoplasms|disorders|conditions', mesh, re.IGNORECASE):
            return True
    return False


def main():
    original_file = 'data/ori_pqal.json'
    output_file = 'data/pqa_disease.json'

    with open(original_file, 'r') as f:
        data = json.load(f)

    required_context_labels = ['BACKGROUND', 'PATIENTS AND METHODS', 'RESULTS']

    filtered_data = []
    all_disease_keywords = set()
    new_id = 0

    for entry_id, entry_data in data.items():
        meshes = entry_data.get("MESHES", [])
        if not contain_disease_meshes(meshes):
            continue

        question = entry_data.get("QUESTION", "").lower()
        contexts = entry_data.get("CONTEXTS", [])
        labels = entry_data.get("LABELS", [])
        combined_context = "".join(
            [contexts[i].lower() for i in range(len(contexts)) if labels[i].upper() in required_context_labels]
        )
        answer = entry_data.get("final_decision", "")


        response = run_openai_task(meshes, combined_context + question)
        disease_keywords = response.split(", ")
        all_disease_keywords.update(disease_keywords)

        filtered_data.append({
            "id": new_id,
            "subject": disease_keywords,
            "attribute": answer,
            "question": question,
            "context": combined_context,
            "original_id": entry_id,
        })

        new_id += 1

        # save first 300 samples
        if new_id >= 10:
            break

    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Filtered data saved to {output_file} with {len(filtered_data)} samples")


if __name__ == "__main__":
    main()
