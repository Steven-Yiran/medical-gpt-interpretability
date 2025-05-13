"""
Preprocess the pubmedQA dataset to be used for the experiments.
Specifically, we will:
- Filter only questions that contains age group keywords
- Combine the Background, Patients and methods, and Results sections into one text
- Save only the question, context, and answer fields in a new json file
"""

import json

original_file = 'data/ori_pqal.json'
output_file = 'data/pqa.json'

age_keywords = ['children', 'adult', 'infant', 'adolescent', 'elderly', 'neonate', 'newborn']
required_fields = ['BACKGROUND', 'PATIENTS AND METHODS', 'RESULTS']

# Load the dataset
with open(original_file, 'r') as f:
    data = json.load(f)

filtered_data = []
new_id = 0

for entry_id, entry_data in data.items():
    question = entry_data.get("QUESTION", "").lower()
    keywords = [word for word in age_keywords if word in question]
    if len(keywords) > 0:
        contexts = entry_data.get("CONTEXTS", [])
        labels = entry_data.get("LABELS", [])
        combined_context = "".join(
            [contexts[i] for i in range(len(contexts)) if labels[i].upper() in required_fields]
        )
        answer = entry_data.get("final_decision", "")
        
        prompt = f"Question: {question} the answer of the question is"

        filtered_data.append({
            "id": new_id,
            "subject": keywords[0],
            "attribute": answer,
            "prompt": prompt,
            "original_id": entry_id,
        })

        new_id += 1

# Save the filtered data
with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"Filtered data saved to {output_file} with {len(filtered_data)} samples")