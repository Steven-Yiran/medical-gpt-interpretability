import requests
import json
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm

from preprocess_keyword import run_openai_task

disease_name_file = '../data/diseases.txt'
base_url = "http://id.who.int/icd/entity/search"
token_endpoint = 'https://icdaccessmanagement.who.int/connect/token'
client_id = "c10af678-4f8c-4f4f-b8a2-679c86429f6c_2b5f9b6b-557a-4b17-a059-7f97a3722e8e"
client_secret = "wqP5FKmVE52ogozR7iByQKLlnWZ6hoRhFZN78XsTdDY="
scope = "icdapi_access"
grant_type = "client_credentials"
disease_pattern = r"Disease|Syndrome|Disorders"
required_context_labels = ['BACKGROUND', 'PATIENTS AND METHODS', 'RESULTS']

def get_token():
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
        "grant_type": grant_type,
    }
    res = requests.post(token_endpoint, data=payload, verify=False).json()
    token = res['access_token']
    return token

def get_disease_names():
    with open(disease_name_file, 'r') as f:
        return f.read().splitlines()
    

def get_icd_codes(disease_name, api_key=None):
    if not api_key:
        api_key = get_token()

    headers = {
        "Authorization": f'Bearer {api_key}',
        "Accept": "application/json",
        'Accept-Language': 'en',
        'API-Version': 'v2',
    }
    params = {
        "q": disease_name,
        "linearization": "mms",  # 'mms' is for morbidity and mortality statistics
        "lang": "en",
        "size": 1  # Return only the top result
    }
    response = requests.get(base_url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        # Check if any matches found
        if 'destinationEntities' in data and data['destinationEntities']:
            entity = data['destinationEntities'][0]
            code = entity.get("chapter", "N/A")
            # code = entity.get("stemId", "N/A")
            # title = entity.get("title", {}).get("@value", "Unknown Title")
            return code
        else:
            return "N/A"
    else:
        return "N/A", f"Error {response.status_code}"
    

def plot_chapters_distribution(chapters):
    fig, ax = plt.subplots()
    ax.hist(chapters, bins=range(1, 22), align='left', rwidth=0.8)
    ax.set_xticks(range(1, 22))
    ax.set_xlabel('Chapter')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of ICD-10 Chapters')
    plt.show()


def get_icd_chapters():
    diseases = get_disease_names()
    chapters = []
    api_key = get_token()
    
    for disease in diseases:
        chapter = get_icd_codes(disease, api_key)
        if chapter != "N/A":
            chapters.append(chapter)

    return chapters


def group_by_icd_chapters(data):
    icd_groups = {}
    for entry in tqdm(data, desc="Grouping data by ICD codes"):
        disease_mesh = entry["disease_mesh"]
        icd_code = get_icd_codes(disease_mesh)
        if icd_code == "N/A":
            continue
        if icd_code not in icd_groups:
            icd_groups[icd_code] = []
        icd_groups[icd_code].append(entry)

    return icd_groups


def find_disease_ner(entry):
    keyword = entry["disease_mesh"]
    question = entry.get("QUESTION", "").lower()
    contexts = entry.get("CONTEXTS", [])
    labels = entry.get("LABELS", [])
    combined_context = "".join(
        [contexts[i].lower() for i in range(len(contexts)) if labels[i].upper() in required_context_labels]
    )

    response = run_openai_task(keyword, question + " " + combined_context)
    # make sure the response is a valid JSON
    try:
        response_dict = json.loads(response)
    except json.JSONDecodeError:
        print("Invalid JSON response for", keyword)
        return None

    ner = response_dict[keyword]

    return ner


def filter_disease_entries(data):
    filtered_data = []
    for id, entry in tqdm(data.items(), desc="Filtering disease data"):
        meshes = entry["MESHES"]

        for mesh in meshes:
            if re.search(disease_pattern, mesh):
                entry["disease_mesh"] = mesh
                entry["ori_id"] = id

                disease_ner = find_disease_ner(entry)
                if not disease_ner:
                    print("No disease NER found for", entry["disease_mesh"])
                    break

                entry["subject"] = disease_ner

                filtered_data.append(entry)
                break
    
    return filtered_data


def main():
    data_path = "../data/ori_pqal.json"
    output_path = "../data/pqal_grouped_by_icd.json"
    disease_file = "../data/diseases.txt"
    chapter_file = "../data/icd_chapters.txt"

    with open(data_path, 'r') as f:
        data = json.load(f)    

    if os.path.exists(chapter_file):
        with open(chapter_file, 'r') as f:
            chapters = f.read().splitlines()
    else:
        print("Getting ICD-10 chapters...")
        chapters = get_icd_chapters()
    
    if os.path.exists(disease_file):
        with open(disease_file, 'r') as f:
            diseases = f.read().splitlines()
    else:
        print("Missing disease names file")

    filtered_data = filter_disease_entries(data)
    icd_groups = group_by_icd_chapters(filtered_data)

    with open(output_path, 'w') as f:
        json.dump(icd_groups, f, indent=4)

    print("Data grouped by ICD codes saved to", output_path)
    


if __name__ == '__main__':
    main()