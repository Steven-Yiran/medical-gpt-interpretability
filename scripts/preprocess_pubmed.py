from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def system_message():
    return f"""
You are an expert in clinical NLP. Your task is to classify medical questions into one of the three categories:
'Medicine', 'Disease', 'Other'.
"""

def assistant_message():
    return f"""
EXAMPLE:
    Question: Syncope during bathing in infants, a pediatric form of water-induced urticaria?
    {"Disease"}
--"""

def user_message(question):
    return f"""
TASK:
    Question: {question}
"""


def run_openai_task(question):
    messages = [
        {"role": "system", "content": system_message()},
        {"role": "assistant", "content": assistant_message()},
        {"role": "user", "content": user_message(question)}
    ]

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )

    return response.choices[0].message.content


def main():
    original_file = "data/ori_pqal.json"
    disease_file = "data/disease_pqal.json"
    medicine_file = "data/medicine_pqal.json"

    with open(original_file, "r") as f:
        data = json.load(f)

    disease_data = []
    medicine_data = []

    for entry_id, entry_data in data.items():
        question = entry_data.get("QUESTION", "").lower()
        response = run_openai_task(question)

        if response == "Disease":
            disease_data.append(entry_data)
        elif response == "Medicine":
            medicine_data.append(entry_data)

        if len(disease_data) >= 5 and len(medicine_data) >= 5:
            break

    with open(disease_file, "w") as f:
        json.dump(disease_data, f, indent=4)

    with open(medicine_file, "w") as f:
        json.dump(medicine_data, f, indent=4)

    
if __name__ == "__main__":
    main()



