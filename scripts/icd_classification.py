import requests
import json
import matplotlib.pyplot as plt
import os

disease_name_file = '../data/diseases.txt'
base_url = "http://id.who.int/icd/entity/search"
token_endpoint = 'https://icdaccessmanagement.who.int/connect/token'
client_id = "c10af678-4f8c-4f4f-b8a2-679c86429f6c_2b5f9b6b-557a-4b17-a059-7f97a3722e8e"
client_secret = "wqP5FKmVE52ogozR7iByQKLlnWZ6hoRhFZN78XsTdDY="
scope = "icdapi_access"
grant_type = "client_credentials"

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

def main():
    diseases = get_disease_names()
    chapters = []
    api_key = get_token()
    
    for disease in diseases:
        chapter = get_icd_codes(disease, api_key)
        chapters.append(chapter)

    plot_chapters_distribution(chapters)

    # Save the chapters to a file
    with open('../data/icd_chapters.txt', 'w') as f:
        f.write('\n'.join(chapters))



if __name__ == '__main__':
    main()