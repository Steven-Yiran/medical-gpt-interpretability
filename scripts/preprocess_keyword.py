from openai import OpenAI
import os
import json
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def system_message():
    return f"""
You are an expert in Natural Language Processing. Your task is to identify the Named Entities (NER) in a given text.
The possible Named Entities (NER) types are exclusively user-defined.
"""

def assistant_message():
    return f"""
EXAMPLE:
    Keyword: Hirschsprung Disease
    Text: 'Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through? / 
        The transanal endorectal pull-through (TERPT) is becoming the most popular procedure in the treatment of Hirschsprung disease (HD), /
        but overstretching of the anal sphincters remains a critical issue that may impact the continence. This study examined the long-term /
        outcome of TERPT versus conventional transabdominal (ABD) pull-through for HD. / 
    {{
        "Hirschsprung Disease": ["Hirschsprung Disease", "HD", "HD"]
    }}
--"""


def user_message(keyword, text):
    return f"""
TASK:
    Keyword: {keyword}
    Text: {text}
"""


def run_openai_task(keyword, text):
    messages = [
        {"role": "system", "content": system_message()},
        {"role": "assistant", "content": assistant_message()},
        {"role": "user", "content": user_message(keyword, text)}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content


def test_openai_task():
    text = "Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through? / " \
           "The transanal endorectal pull-through (TERPT) is becoming the most popular procedure in the treatment of Hirschsprung disease (HD), " \
           "but overstretching of the anal sphincters remains a critical issue that may impact the continence. This study examined the long-term " \
           "outcome of TERPT versus conventional transabdominal (ABD) pull-through for HD. "
    keyword = "Hirschsprung Disease"
    response = run_openai_task(keyword, text)
    
    # load as dict
    response_dict = json.loads(response)
    print(response_dict[keyword])


def contain_disease_meshes(meshes):
    """
    Filter meshes that contain *disease*, *syndrome*, *infections*, *neoplasms*, *disorders*, *conditions*
    """
    for mesh in meshes:
        if re.search(r'disease|syndrome|infections|neoplasms|disorders|conditions', mesh, re.IGNORECASE):
            return True
    return False


def split_disease_entries():
    num_samples = 50
    original_file = 'data/ori_pqal.json'
    output_file = 'data/pqa_disease.json'
    output_keywords_file = 'data/disease_keywords.json'

    with open(original_file, 'r') as f:
        data = json.load(f)

    required_context_labels = ['BACKGROUND', 'PATIENTS AND METHODS', 'RESULTS']

    filtered_data = []
    all_disease_keywords = {}
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


        response = run_openai_task(meshes, question + " " + combined_context)
        disease_keywords = response.lower().split(", ")
        all_disease_keywords[new_id] = disease_keywords

        filtered_data.append({
            "id": new_id,
            "subject": disease_keywords,
            "attribute": answer,
            "question": question,
            "context": combined_context,
            "original_id": entry_id,
        })

        new_id += 1

        # save first x samples
        if new_id >= 50:
            break

    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    with open(output_keywords_file, 'w') as f:
        json.dump(all_disease_keywords, f, indent=2)

    print(f"Filtered data saved to {output_file} with {len(filtered_data)} samples")
    print(f"Disease keywords saved to {output_keywords_file}")


def main():
    test_openai_task()


if __name__ == "__main__":
    main()
