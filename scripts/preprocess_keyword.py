from openai import OpenAI
import os
import json
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def system_message():
#     return f"""
# You are an expert in Natural Language Processing. Your task is to identify wo.
# First, you need to identify MeSH keywords that are related to a specific disease, symptom, or disorder.
# Then, you need to identify the words in the text that are related to the diseases MeSH keywords.
# You need to return the identified words in a common separated string.
# If you can't find any related words, return an empty string.
# """

def system_message(labels):
    return f"""
You are an expert in Natural Language Processing. Your task is to identify common biomedicine Named Entity in a given test.
The possible Named Entities (NER) are exclusively: ({", ".join(labels)})
"""

# def assistant_message():
#     return f"""
# EXAMPLE:
#     MeSH: Baths, Histamine, Humans, Infant, Syncope, Urticaria, Water
#     Text: 'Apparent life-threatening events in infants are a difficult and frequent problem in pediatric practice. / 
#         The prognosis is uncertain because of risk of sudden infant death syndrome. / 
#         Syncope during bathing in infants, a pediatric form of water-induced urticaria?'
#     {"syncope, water-induced urticaria"}
# --"""


def assistant_message():
    return f"""
EXAMPLE:
    Text: 'Apparent life-threatening events in infants are a difficult and frequent problem in pediatric practice. /
        The prognosis is uncertain because of risk of sudden infant death syndrome. /
        Syncope during bathing in infants, a pediatric form of water-induced urticaria?'
        {{
            "Disease": ["Syncope", "Urticaria"],
        }}
--"""

# def user_message(meshes, text):
#     return f"""
# TASK:
#     MeSH: {", ".join(meshes)}
#     Text: {text}
# """

def user_message(text):
    return f"""
TASK:
    Text: {text}
"""


# def run_openai_task(meshes, text):
#     messages = [
#         {"role": "system", "content": system_message()},
#         {"role": "assistant", "content": assistant_message()},
#         {"role": "user", "content": user_message(meshes, text)}
#     ]

#     response = client.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=messages
#     )

#     return response.choices[0].message.content


# def contain_disease_meshes(meshes):
#     """
#     Filter meshes that contain *disease*, *syndrome*, *infections*, *neoplasms*, *disorders*, *conditions*
#     """
#     for mesh in meshes:
#         if re.search(r'disease|syndrome|infections|neoplasms|disorders|conditions', mesh, re.IGNORECASE):
#             return True
#     return False


def run_openai_task(text, type):
    if type == "Disease":
        labels = ["Disease"]
    elif type == "Medicine":
        labels = ["Medicine"]

    messages = [
        {"role": "system", "content": system_message(labels)},
        {"role": "assistant", "content": assistant_message()},
        {"role": "user", "content": user_message(text)}
    ]

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )

    return response.choices[0].message.content


# def main():
#     num_samples = 50
#     original_file = 'data/ori_pqal.json'
#     output_file = 'data/pqa_disease.json'
#     output_keywords_file = 'data/disease_keywords.json'

#     with open(original_file, 'r') as f:
#         data = json.load(f)

#     required_context_labels = ['BACKGROUND', 'PATIENTS AND METHODS', 'RESULTS']

#     filtered_data = []
#     all_disease_keywords = {}
#     new_id = 0

#     for entry_id, entry_data in data.items():
#         meshes = entry_data.get("MESHES", [])
#         if not contain_disease_meshes(meshes):
#             continue

#         question = entry_data.get("QUESTION", "").lower()
#         contexts = entry_data.get("CONTEXTS", [])
#         labels = entry_data.get("LABELS", [])
#         combined_context = "".join(
#             [contexts[i].lower() for i in range(len(contexts)) if labels[i].upper() in required_context_labels]
#         )
#         answer = entry_data.get("final_decision", "")


#         response = run_openai_task(meshes, question + " " + combined_context)
#         disease_keywords = response.lower().split(", ")
#         all_disease_keywords[new_id] = disease_keywords

#         filtered_data.append({
#             "id": new_id,
#             "subject": disease_keywords,
#             "attribute": answer,
#             "question": question,
#             "context": combined_context,
#             "original_id": entry_id,
#         })

#         new_id += 1

#         # save first x samples
#         if new_id >= 50:
#             break

#     with open(output_file, 'w') as f:
#         json.dump(filtered_data, f, indent=2)

#     with open(output_keywords_file, 'w') as f:
#         json.dump(all_disease_keywords, f, indent=2)

#     print(f"Filtered data saved to {output_file} with {len(filtered_data)} samples")
#     print(f"Disease keywords saved to {output_keywords_file}")


def main():
    keyword_type = "Disease"
    original_file = f"data/{keyword_type.lower()}_pqal.json"
    output_file = f"data/{keyword_type.lower()}_subjects.json"

    with open(original_file, "r") as f:
        data = json.load(f)

    required_context_labels = ['BACKGROUND', 'PATIENTS AND METHODS', 'RESULTS']
    
    filtered_data = []
    new_id = 0

    for entry in data:
        question = entry.get("QUESTION", "").lower()
        contexts = entry.get("CONTEXTS", [])
        labels = entry.get("LABELS", [])
        combined_context = "".join(
            [contexts[i].lower() for i in range(len(contexts)) if labels[i].upper() in required_context_labels]
        )
        answer = entry.get("final_decision", "")


        response = run_openai_task(question + " " + combined_context, keyword_type)
        # parse resonse as json
        response = json.loads(response)
        entities = response.get(keyword_type, [])

        if not entities:
            print(f"Empty entities for question: {question}")
            continue
        
        filtered_data.append({
            "id": new_id,
            "subject": entities,
            "attribute": answer,
            "question": question,
            "context": combined_context
        })

        new_id += 1

        # # save first x samples
        if new_id >= 5:
            break

    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Filtered data saved to {output_file} with {len(filtered_data)} samples")



if __name__ == "__main__":
    main()
